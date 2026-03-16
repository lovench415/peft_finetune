from __future__ import annotations

import gc
import math
import os
from pathlib import Path

import torch
import torchaudio
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import collate_fn
from f5_tts.model.samplers import (
    DynamicBatchSampler,
    BucketDynamicBatchSampler,
    SpeakerAwareBucketDynamicBatchSampler,
    SpeakerBalancedDynamicBatchSampler,
)
from f5_tts.model.utils import default, exists
from f5_tts.config.metadata import save_checkpoint_metadata

# trainer


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        adapter_config,
        view_training_procedure=False,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,  # -1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        bucket_batching: bool = False,
        speaker_aware_batching: bool = False,
        speaker_balanced_batching: bool = False,
        bucket_size: int = 512,
        max_speakers_per_batch: int = 8,
        max_samples_per_speaker: int = 8,
        speakers_per_batch: int = 8,
        samples_per_speaker: int = 4,
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = None,
        ema_kwargs: dict = None,
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        cfg_dict: dict = None,  # training config
        app_config=None,
        checkpoint_metadata_extra: dict | None = None,
    ):
        # BUG-30 FIX: avoid mutable default arguments
        if accelerate_kwargs is None:
            accelerate_kwargs = {}
        if ema_kwargs is None:
            ema_kwargs = {}
        if cfg_dict is None:
            cfg_dict = {}
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb":
            try:
                import wandb
                if not wandb.api.api_key:
                    logger = None
            except ImportError:
                logger = None
        print(f"Using logger: {logger}")
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                print(wandb_resume_id)
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}


            if not cfg_dict:
                cfg_dict = {
                    "epochs" : epochs,
                    "learning_rate" : learning_rate,
                    "num_warmup_updates" : num_warmup_updates,
                    "batch_size_per_gpu" : batch_size_per_gpu,
                    "batch_size_type" : batch_size_type,
                    "max_samples" : max_samples,
                    "grad_accumulation_steps" : grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=cfg_dict,
            )

            """
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler,
                },
            )
            """

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger : {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkingpoint with per_updates now, old logic per_steps used with before f992c4e"
                )
        """
        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)
        """

        self.adapter_config = adapter_config if adapter_config else None
 
        


        self.view_training_procedure = view_training_procedure
        self.ema_kwargs =  ema_kwargs
        #self.batch_size = batch_size
        self.learning_rate = learning_rate


        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")



        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.bucket_batching = bucket_batching
        self.speaker_aware_batching = speaker_aware_batching
        self.speaker_balanced_batching = speaker_balanced_batching
        self.bucket_size = bucket_size
        self.max_speakers_per_batch = max_speakers_per_batch
        self.max_samples_per_speaker = max_samples_per_speaker
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        self.max_samples = max_samples
        if self.speaker_aware_batching or self.speaker_balanced_batching:
            if not hasattr(self.model, 'transformer'):
                pass
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler
        self.app_config = app_config
        self.checkpoint_metadata_extra = checkpoint_metadata_extra or {}

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb
            print("bnb_optimzer -> True")
            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            print("bnb_optimizer -> False")
            print(f'learning_rate : {learning_rate}')

            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        # BUG-10 FIX: defer prepare to load_checkpoint to avoid double-wrapping
        # self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self._prepared = False

    @property
    def is_main(self):
        return self.accelerator.is_main_process


    def save_checkpoint(self, update, last=False):
        print(f"enter save_checkpoint at update {update}")
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            if last:
                ckpt_path = f"{self.checkpoint_path}/model_last.pt"
                self.accelerator.save(checkpoint, ckpt_path)
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                ckpt_path = f"{self.checkpoint_path}/model_{update}.pt"
                self.accelerator.save(checkpoint, ckpt_path)

                if self.keep_last_n_checkpoints > 0:
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startswith("model_")
                        and not f.startswith("pretrained_")
                        and f.endswith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        oldest_path = os.path.join(self.checkpoint_path, oldest_checkpoint)
                        os.remove(oldest_path)
                        meta_path = Path(oldest_path).with_suffix(".meta.json")
                        if meta_path.exists():
                            meta_path.unlink()
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

            if self.app_config is not None:
                save_checkpoint_metadata(ckpt_path, self.app_config, extra=self.checkpoint_metadata_extra)

    def set_trainable_parameters(self):
        if self.adapter_config:
            for param in self.model.parameters():
                param.requires_grad = False

            for module, mode in self.adapter_config.items():
                print(f"module : {module} , mode : {mode}")
                for name, param in self.model.named_parameters():
                    if module in name:
                        if mode == "full":
                            param.requires_grad = True
                        elif mode == "freeze":
                            param.requires_grad = False
                        elif mode == "adapter":
                            if 'conv_adapt' in name:
                                param.requires_grad = True
                            elif 'lora_layers' in name:
                                param.requires_grad = True

            # Always freeze pretrained embedding
            self.model.transformer.text_embed.text_embed.weight.requires_grad = False
            # BUG-34 FIX: only freeze text_embed_ko when it's not the active embedding
            if self.model.transformer.text_embed.alpha == 1:
                self.model.transformer.text_embed.text_embed_ko.weight.requires_grad = True
            else:
                self.model.transformer.text_embed.text_embed_ko.weight.requires_grad = False

            for name, param in self.model.named_parameters():
                # AdaLayerNorm gates — modulate each block via timestep
                if 'attn_norm.linear' in name and 'transformer_blocks' in name:
                    param.requires_grad = True
                # PROSODY: ConvPositionEmbedding — local temporal patterns (rhythm, tempo)
                if 'conv_pos_embed' in name:
                    param.requires_grad = True
                # PROSODY: TimestepEmbedding MLP — global conditioning for all blocks
                if 'time_embed' in name:
                    param.requires_grad = True
                # PROSODY: FeedForward LoRA — per-position nonlinear transforms
                if 'ff_lora' in name:
                    param.requires_grad = True

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            return 0

        self.accelerator.wait_for_everyone()

        if "model_last.pt" in os.listdir(self.checkpoint_path):
            print("Found model_last.pt")
            latest_checkpoint = "model_last.pt"
        else:
            print("No model_last.pt found")
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            print(f"training_checkpoints : {training_checkpoints}")

            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
                print(f"latest_checkpoint : {latest_checkpoint}")
            else:
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"):
            from safetensors.torch import load_file
            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu")
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")
            print(f"Loaded checkpoint: {latest_checkpoint}")

        # patch for backward compatibility
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint.get("ema_model_state_dict", {}):
                del checkpoint["ema_model_state_dict"][key]

        if self.view_training_procedure:
            print("Keys in checkpoint:", checkpoint.keys())

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        # Resuming from a training checkpoint
        if "update" in checkpoint or "step" in checkpoint:
            print("Resuming from training checkpoint")

            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print("WARNING: Converting old per_steps checkpoint to per_updates.")

            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint.get("model_state_dict", {}):
                    del checkpoint["model_state_dict"][key]

            if self.adapter_config:
                # strict=False: allows resume from older checkpoints missing new adapter keys (e.g. ff_lora)
                # New adapter params (ff_lora) will be initialized from scratch (zero-init → no-op at start)
                load_result = self.accelerator.unwrap_model(self.model).load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                )
                if load_result.missing_keys and self.is_main:
                    print(f"Resume: {len(load_result.missing_keys)} new params initialized fresh: "
                          f"{load_result.missing_keys[:5]}{'...' if len(load_result.missing_keys) > 5 else ''}")
                self.set_trainable_parameters()
                trainable_params = [param for param in self.model.parameters() if param.requires_grad]
                self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Scheduler is created after load_checkpoint; position restored via scheduler.step() loop
            update = checkpoint["update"]

        # First training from pretrained weights
        else:
            print("First training from pretrained weights")
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)

            if self.adapter_config:
                self.set_trainable_parameters()
                if self.is_main:
                    self.ema_model = EMA(self.model, include_online_model=False, **self.ema_kwargs)
                    self.ema_model.to(self.accelerator.device)

                trainable_params = [param for param in self.model.parameters() if param.requires_grad]
                self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

            update = 0

        if self.view_training_procedure:
            for name, param in self.model.named_parameters():
                print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
            exit()

        del checkpoint
        gc.collect()
        return update

    def _create_dataloader(self, train_dataset, num_workers, generator):
        """Create the appropriate dataloader based on batching configuration."""
        if self.batch_size_type == "sample":
            return DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )

        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)

            if self.speaker_balanced_batching:
                batch_sampler = SpeakerBalancedDynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    speakers_per_batch=self.speakers_per_batch,
                    samples_per_speaker=self.samples_per_speaker,
                    max_samples=self.max_samples,
                    random_seed=None,
                    drop_residual=False,
                )
            elif self.speaker_aware_batching:
                batch_sampler = SpeakerAwareBucketDynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    bucket_size=self.bucket_size,
                    max_speakers_per_batch=self.max_speakers_per_batch,
                    max_samples_per_speaker=self.max_samples_per_speaker,
                    random_seed=None,
                    drop_residual=False,
                )
            elif self.bucket_batching:
                batch_sampler = BucketDynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    bucket_size=self.bucket_size,
                    random_seed=None,
                    drop_residual=False,
                )
            else:
                batch_sampler = DynamicBatchSampler(
                    sampler,
                    self.batch_size_per_gpu,
                    max_samples=self.max_samples,
                    random_seed=None,
                    drop_residual=False,
                )

            return DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )

        else:
            raise ValueError(f"batch_size_type must be 'sample' or 'frame', got '{self.batch_size_type}'")

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        view_training_procedure = self.view_training_procedure

        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef
            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        train_dataloader = self._create_dataloader(train_dataset, num_workers, generator)

        # Scheduler setup
        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = max(1, total_updates - warmup_updates)

        # Prepare dataloader
        train_dataloader = self.accelerator.prepare(train_dataloader)

        # Load checkpoint (may replace optimizer)
        start_update = self.load_checkpoint()

        # Create scheduler AFTER load_checkpoint so it binds to the final optimizer
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        self.scheduler = self.accelerator.prepare(self.scheduler)

        # Advance scheduler to match resumed position
        for _ in range(start_update):
            self.scheduler.step()

        global_update = start_update
        print(f"Starting from update: {global_update}")

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch+1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)

                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                if self.accelerator.is_local_main_process:
                    self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update)
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        ref_audio_len = mel_lengths[0]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(self.model).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=[text_inputs[0] + [" "] + text_inputs[0]],
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                            ref_mel_spec = batch["mel"][0].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()

                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav", gen_audio, target_sample_rate
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav", ref_audio, target_sample_rate
                        )

                if global_update % self.last_per_updates == 0 and self.accelerator.sync_gradients:
                    self.save_checkpoint(global_update, last=True)

        self.save_checkpoint(global_update, last=True)
        self.accelerator.end_training()

