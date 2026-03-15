
from __future__ import annotations

from pathlib import Path

from .types import AppConfig


def validate_train_config(cfg: AppConfig) -> None:
    if cfg.train is None or cfg.batching is None:
        raise ValueError("Train config and batching config are required for training")

    batching = cfg.batching
    if batching.mode == "speaker_balanced":
        required = batching.speakers_per_batch * batching.samples_per_speaker
        if required > batching.max_samples:
            raise ValueError(
                "speaker_balanced batching requires speakers_per_batch * samples_per_speaker <= max_samples"
            )

    if cfg.train.tokenizer == "custom" and not cfg.train.tokenizer_path:
        raise ValueError("Custom tokenizer selected, but tokenizer_path is not provided")

    if cfg.train.pretrained_ckpt and not Path(cfg.train.pretrained_ckpt).exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {cfg.train.pretrained_ckpt}")


def validate_infer_config(cfg: AppConfig) -> None:
    if cfg.infer is None:
        raise ValueError("Infer config is required for inference")

    infer = cfg.infer
    for field_name in ["ckpt_file", "vocab_file", "ref_audio"]:
        value = getattr(infer, field_name)
        if not Path(value).exists():
            raise FileNotFoundError(f"{field_name} not found: {value}")

    if infer.gen_file and not Path(infer.gen_file).exists():
        raise FileNotFoundError(f"gen_file not found: {infer.gen_file}")

    if not infer.gen_text and not infer.gen_file:
        raise ValueError("Either gen_text or gen_file must be provided for inference")
