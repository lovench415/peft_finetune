from __future__ import annotations

import gc
import math
import os
import sys

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists

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
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
        is_local_vocoder: bool = False,  # use local path vocoder
        local_vocoder_path: str = "",  # local vocoder path
        cfg_dict: dict = dict(),  # training config
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
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
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # mel vocoder config
        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb
            print("bnb_optimzer -> True")
            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            print("bnb_optimizer -> False")
            print(f'learning_rate : {learning_rate}')

            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process


    def save_checkpoint(self, update, last=False):
        print("enter save_checkpoint with {step} steps")
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
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
                if self.keep_last_n_checkpoints > 0 :
                    # Update logic to exclude pretrained model from rotation
                    checkpoints = [
                        f
                        for f in os.listdir(self.checkpoint_path)
                        if f.startwith("model_")
                        and not f.startwith("pretrained_") # Exclude pretrained models
                        and f.endwith(".pt")
                        and f != "model_last.pt"
                    ]
                    checkpoints.sort(key=lambda x : int(x.split("_")[1].split(".")[0]))
                    while len(checkpoints) > self.keep_last_n_checkpoints:
                        oldest_checkpoint = checkpoints.pop(0)
                        os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                        print(f"Removed old checkpoint: {oldest_checkpoint}")

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

            
            self.model.transformer.text_embed.text_embed.weight.requires_grad = False
            self.model.transformer.text_embed.text_embed_ko.weight.requires_grad = False
            
            for name, param in self.model.named_parameters():
                if 'attn_norm.linear' in name and 'transformer_blocks' in name:
                    param.requires_grad = True

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not any(filename.endswith((".pt", ".safetensors")) for filename in os.listdir(self.checkpoint_path))
        ):
            return 0
        

        self.accelerator.wait_for_everyone()
        # if model_last.pt in checkpoint_path -> use it
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            print("there's model_last.pt")
            latest_checkpoint = "model_last.pt"

        
        # if model_last.pt is not in checkpoint_path -> sort the rest of the checkpoints
        else:
            print("there's no model_last.pt")
            all_checkpoints = [
                f
                for f in os.listdir(self.checkpoint_path)
                if (f.startswith("model_") or f.startswith("pretrained_")) and f.endswith((".pt", ".safetensors"))
            ]

            # First try to find regular training checkpoints
            training_checkpoints = [f for f in all_checkpoints if f.startswith("model_") and f != "model_last.pt"]
            print(f"training_checkpoints : {training_checkpoints}")


            if training_checkpoints:
                latest_checkpoint = sorted(
                    training_checkpoints,
                    #[f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt") and f.startswith("model")],
                    key=lambda x: int("".join(filter(str.isdigit, x))),
                )[-1]
                print(f"latest_checkpoint : {latest_checkpoint}")

            else : 
                # If no training checkpoints, use pretrained_model
                latest_checkpoint = next(f for f in all_checkpoints if f.startswith("pretrained_"))

        if latest_checkpoint.endswith(".safetensors"): # always a pretrained checkpoint
            from safetensors.torch import load_file

            checkpoint = load_file(f"{self.checkpoint_path}/{latest_checkpoint}", device="cpu") 
            checkpoint = {"ema_model_state_dict": checkpoint}
        elif latest_checkpoint.endswith(".pt"):
            # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
            checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")
            print(f"the latest_checkpoint of checkpoint path : {latest_checkpoint}")

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        # print checkpoint
        if self.view_training_procedure:
            print("Keys in checkpoint:", checkpoint.keys())

        if self.is_main:
            
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            #self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)
            """
            old_weight = checkpoint["ema_model_state_dict"]['ema_model.transformer.text_embed.text_embed.weight'].to(device)
            new_weight = self.ema_model.ema_model.transformer.text_embed.text_embed.weight.to(device)



            old_dim, embed_dim = old_weight.shape
            new_dim , embed_dim = new_weight.shape
            if old_dim != new_dim:
                print("The Vocab size is different with pretrained model")
                print(f"Now excluding the text embedding size {old_dim} -> {new_dim}")
                new_weight[:old_dim,:] = old_weight
                checkpoint["ema_model_state_dict"]['ema_model.transformer.text_embed.text_embed.weight'] = new_weight

                checkpoint["ema_model_state_dict"]['ema_model.transformer.text_embed.text_embed_ko.weight'] = new_weight
            """
            #checkpoint["ema_model_state_dict"]['ema_model.transformer.text_embed.text_embed_ko.weight']
 
            
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        # is not the first trial to train the model with the checkpoint or the model_config
        if "update" in checkpoint or "step" in checkpoint:
            # patch for backward compatibility, with before f992c4e
            print("there's 'step' in checkpoint")

            if "step" in checkpoint:
                checkpoint["update"] = checkpoint["step"] // self.grad_accumulation_steps
                if self.grad_accumulation_steps > 1 and self.is_main:
                    print(
                        "F5-TTS WARNING: Loading checkpoint waved with per_steps logic (before f99c43), will conver to per_updates according to grad_accumulation_steps setting, may have unexpected behavior."
                    )
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]
            
            
            if self.adapter_config:
                self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=True)
                #self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
                self.set_trainable_parameters()

                trainable_params = [param for param in self.model.parameters() if param.requires_grad]
                self.optimizer = AdamW(trainable_params, lr=self.learning_rate)
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)  
            else:
                self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            #self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                print("there's scheduler")
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            update = checkpoint["update"]
            #step = checkpoint["step"]
            #print(f"step : {step}")
        # first training procedure
        else:
            print("there's no 'step' in checkpoint") 
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "update", "step"]
            }
                        
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"],strict=False)
            #update = 0

            if self.adapter_config: # {'text_embed': 'full', 'input_embed': 'adapter', 'transformer_blocks': 'adpater'}
                self.set_trainable_parameters()
                if self.is_main:
                    self.ema_model = EMA(self.model, include_online_model=False, **self.ema_kwargs)
                    self.ema_model.to(self.accelerator.device)


                trainable_params = [param for param in self.model.parameters() if param.requires_grad]

                
                self.optimizer = AdamW(trainable_params, lr=self.learning_rate)


                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

            #step = 0
            update = 0

        if self.view_training_procedure :
            for name, param in self.model.named_parameters():
                print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
            exit()
        
        del checkpoint
        gc.collect()
        return update

    def train(self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None):
        #print(self.model)
        
        view_training_procedure = self.view_training_procedure
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            #vocoder = load_vocoder(vocoder_name=self.vocoder_name)
            vocoder = load_vocoder(
                vocoder_name= self.vocoder_name, is_local=self.is_local_vocoder, local_path=self.local_vocoder_path
            )
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=self.batch_size,
                shuffle=True,
                generator=generator,
            )
            print(f"train_dataloader.batch_size :{train_dataloader.batch_size}" )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(train_dataset)

            #batch_sampler = DynamicBatchSampler(
            #    sampler, self.batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_last=False
            #)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples= self.max_samples,
                random_seed= resumable_with_seed, # This enables reproducible shuffling
                drop_residual= False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
            #print(f"train_dataloader.batch_size :{train_dataloader.batch_size}" )
            if view_training_procedure:
                pass
                #for batch_idx, batch in enumerate(train_dataloader):
                #    print(f"Batch {batch_idx + 1}: {len(batch)} samples")

        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")




        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_updates = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        #total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        #decay_steps = total_steps - warmup_steps

        total_updates = math.ceil(len(train_dataloader) / self.grad_accumulation_steps) * self.epochs
        decay_updates = total_updates - warmup_updates
        


        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)
        #constant_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=0)

        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )


        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual steps = 1 gpu steps / gpus

        ## LOADING CHECKPOINT -> ADAPTING LORA HERE
        start_update = self.load_checkpoint()

        global_update = start_update
        print(f'global_step : {global_update}')

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        
        ##### model.train
        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(skipped_batch / self.grad_accumulation_steps)
                current_dataloader = skipped_dataloader
                """
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                )
                """
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader
                """
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                )
                """

            # Set epoch for batch sampler if it exits
            if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch+1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            #for batch in progress_bar:
            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    #if view_traning_procedure :
                    #    print(f"batch[text].shape : {batch['text'].shape}")
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_update)





                    # compute loss in model
                    loss, cond, pred = self.model(
                        mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
                    )
            

                    # Backpropagation (Compute Gradient)
                    self.accelerator.backward(loss)

                    """
                    # LoRA 레이어에 대해 gradient를 0.05배로 스케일링
                    for name, param in self.model.named_parameters():
                        if "lora_layers" in name:
                            if param.grad is not None: 
                                param.grad *= 0.05  
                    """
                    if view_training_procedure:
                        # Checking Gradient is whether is flooding
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                print(f"Gradient for {name}: {param.grad is not None} (Mean: {param.grad.abs().mean().item() if param.grad is not None else 'None'})")
                        


                    # Gradient Clipping
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer step 전 파라미터 저장
                    if view_training_procedure:
                        print("=== Optimizer State ===")
                        optimizer_state = self.optimizer.state_dict()




                        before_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        zhen_embedding = self.model.transformer.text_embed.text_embed.weight[:,:].clone()

                        ko_embedding = self.model.transformer.text_embed.text_embed_ko.weight[:,:].clone()

                        


                        # Optimizer Step
                        self.optimizer.step()
                        self.scheduler.step()
                        #self.scaling_scheduler.step()
                        self.optimizer.zero_grad()

                        # Comparing parameters after Optimizer step
                        # check if parameters are updated or not
                        
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                is_updated = not torch.equal(before_params[name], param)
                                print(f"Parameter {name} updated: {is_updated}")
                        
                        # Create a mapping of parameter tensors to their names
                        param_to_name = {param: name for name, param in self.model.named_parameters()}

                        # Iterate over optimizer parameter groups and print their information
                        for group in self.optimizer.param_groups:
                            for param in group['params']:
                                param_name = param_to_name.get(param, "Unnamed Parameter")
                                print(f"Name: {param_name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")

                        # check if origianl embedding ahs changed
                        
                        zhen_updated_embedding = self.model.transformer.text_embed.text_embed.weight[:,:]
                        #ko_updated_embedding = self.model.transformer.text_embed.new_vocab_embed.weight[:,:]
                        ko_updated_embedding = self.model.transformer.text_embed.text_embed_ko.weight[:,:]

                        
                        if torch.equal(zhen_embedding[:,:], zhen_updated_embedding[:,:]):
                            print(f"No Changed Zh & EN!")
                        else: 
                            print(f"changed Zh & EN")

                        if torch.equal(ko_embedding[:,:], ko_updated_embedding[:,:]):
                            print(f"No Chnaged KO!")
                        else:
                            print(f"changed KO")


                    else :
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(update=str(global_update), loss=loss.item())

                """        
                if self.is_main:
                    self.ema_model.update()
                    if view_training_procedure:
                        print(f"global_step : {global_step}")

                        exit()
                        #pass
                
                global_step += 1
                """


                if self.accelerator.is_local_main_process:
                    self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_update)
                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_update)

                #progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                #if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                if global_update % self.save_per_updates == 0 and self.accelerator.sync_gradients :
                    self.save_checkpoint(global_update)

                    if self.log_samples and self.accelerator.is_local_main_process:
                        """
                        ref_audio, ref_audio_len = vocoder.decode(batch["mel"][0].unsqueeze(0)), mel_lengths[0]
                        
                        torchaudio.save(
                            f"{log_samples_path}/step_{global_step}_ref.wav", ref_audio.cpu(), target_sample_rate
                        )
                        """
                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0] + ([" "] if isinstance(text_inputs[0], list) else " ") + text_inputs[0]
                        ]

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
                        
                        #gen_audio = vocoder.decode(
                        #    generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                        #)
                        
                            gen_mel_spec = generated[:, ref_audio_len:, :].permute(0,2,1).to(self.accelerator.device)
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
