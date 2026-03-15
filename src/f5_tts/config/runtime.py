
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from f5_tts.model import DiT, UNetT


def resolve_model_cls(backbone: str):
    if backbone == "UNetT":
        return UNetT
    return DiT


def default_vocoder_local_path(vocoder_name: str) -> str:
    project_root = Path(__file__).resolve().parents[3]
    if vocoder_name == "vocos":
        return str(project_root / "checkpoints" / "vocos-mel-24khz")
    return str(project_root / "checkpoints" / "bigvgan_v2_24khz_100band_256x")


def resolve_checkpoint_dir(dataset_name: str, checkpoint_subdir: str | None, model_name: str) -> str:
    checkpoint_subdir = checkpoint_subdir or model_name
    return str(files("f5_tts").joinpath(f"../../ckpts/{dataset_name}/{checkpoint_subdir}"))


def trainer_kwargs_from_config(app_cfg, checkpoint_path: str, mel_spec_type: str) -> dict:
    train_cfg = app_cfg.train
    batching_cfg = app_cfg.batching
    return {
        "adapter_config": app_cfg.model.trainable_map,
        "view_training_procedure": train_cfg.view_training_procedure,
        "num_warmup_updates": train_cfg.num_warmup_updates,
        "save_per_updates": train_cfg.save_per_updates,
        "keep_last_n_checkpoints": train_cfg.keep_last_n_checkpoints,
        "checkpoint_path": checkpoint_path,
        "batch_size_per_gpu": batching_cfg.batch_size_per_gpu,
        "batch_size_type": batching_cfg.batch_size_type,
        "bucket_batching": batching_cfg.mode == "bucket",
        "speaker_aware_batching": batching_cfg.mode == "speaker_aware",
        "speaker_balanced_batching": batching_cfg.mode == "speaker_balanced",
        "bucket_size": batching_cfg.bucket_size,
        "max_speakers_per_batch": batching_cfg.max_speakers_per_batch,
        "max_samples_per_speaker": batching_cfg.max_samples_per_speaker,
        "speakers_per_batch": batching_cfg.speakers_per_batch,
        "samples_per_speaker": batching_cfg.samples_per_speaker,
        "max_samples": batching_cfg.max_samples,
        "grad_accumulation_steps": train_cfg.grad_accumulation_steps,
        "max_grad_norm": train_cfg.max_grad_norm,
        "logger": train_cfg.logger,
        "wandb_project": train_cfg.dataset_name,
        "wandb_run_name": train_cfg.exp_name,
        "wandb_resume_id": None,
        "log_samples": train_cfg.log_samples,
        "last_per_updates": train_cfg.last_per_updates,
        "bnb_optimizer": train_cfg.bnb_optimizer,
        "mel_spec_type": mel_spec_type,
        "prosody_loss_weight": train_cfg.prosody_loss_weight,
        "app_config": app_cfg,
        "checkpoint_metadata_extra": metadata_extra_from_train_config(app_cfg, mel_spec_type),
    }


def metadata_extra_from_train_config(app_cfg, mel_spec_type: str) -> dict:
    train_cfg = app_cfg.train
    return {
        "vocab_file": train_cfg.tokenizer_path if train_cfg.tokenizer == "custom" else None,
        "tokenizer": train_cfg.tokenizer,
        "mel_spec_type": mel_spec_type,
        "dataset_name": train_cfg.dataset_name,
    }


def print_train_summary(app_cfg) -> None:
    train_cfg = app_cfg.train
    batching_cfg = app_cfg.batching
    print("This is Fine-Tuning of F5-TTS")
    print(f"Dataset: {train_cfg.dataset_name}")
    print(f"Learning rate: {train_cfg.learning_rate}")
    print(f"Training epochs: {train_cfg.epochs}")
    print(f"Batch size per GPU: {batching_cfg.batch_size_per_gpu}")
    print(f"Batch size type: {batching_cfg.batch_size_type}")
    print(f"Grad accumulation steps: {train_cfg.grad_accumulation_steps}")
    print(f"Training model version: {train_cfg.exp_name}")
    print(f"Batching mode: {batching_cfg.mode}")


def print_infer_summary(app_cfg) -> None:
    infer_cfg = app_cfg.infer
    print(f"Using model      : {app_cfg.model.name}")
    print(f"Using checkpoint : {infer_cfg.ckpt_file}")
    print(f"Using vocab      : {infer_cfg.vocab_file}")
    print(f"Using vocoder    : {infer_cfg.vocoder_name}")
