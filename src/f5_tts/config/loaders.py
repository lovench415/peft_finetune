
from __future__ import annotations

import codecs
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from importlib.resources import files

import tomli
from cached_path import cached_path
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    cfg_strength as default_cfg_strength,
    cross_fade_duration as default_cross_fade_duration,
    fix_duration as default_fix_duration,
    mel_spec_type as default_mel_spec_type,
    nfe_step as default_nfe_step,
    speed as default_speed,
    sway_sampling_coef as default_sway_sampling_coef,
    target_rms as default_target_rms,
)
from f5_tts.train.hparam import get_model_cfg

from .types import AppConfig, BatchingConfig, InferConfig, ModelConfig, TrainConfig
from .metadata import load_checkpoint_metadata


MODEL_ALIASES = {
    "F5TTS_base": "F5TTS_Base",
    "PEFT-TTS_Base": "PEFT-TTS_base",
    "PEFTTTS_Base": "PEFT-TTS_base",
}

def _config_get(cli_value, config: dict, key: str, default):
    return cli_value if cli_value is not None else config.get(key, default)


def _bool_from_sources(cli_flag: bool, config: dict, key: str, default: bool = False) -> bool:
    return True if cli_flag else bool(config.get(key, default))


def normalize_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def _patch_example_path(value: str | None) -> str | None:
    if not value:
        return value
    if "infer/examples/" in value:
        return str(files("f5_tts").joinpath(value))
    return value


def _load_yaml_cfg(path: str) -> dict:
    return dict(OmegaConf.load(path))


def build_model_config(model_name: str, explicit_model_cfg: str | None = None) -> ModelConfig:
    model_name = normalize_model_name(model_name)

    if explicit_model_cfg:
        inferred_backbone = "UNetT" if "E2" in model_name.upper() or "UNET" in model_name.upper() else "DiT"
        return ModelConfig(
            name=model_name,
            backbone=inferred_backbone,
            transformer_kwargs=_load_yaml_cfg(explicit_model_cfg),
        )

    if model_name == "F5TTS_v1_Base":
        base_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                        conv_dilations=[1, 1, 2, 4])  # multi-scale for prosody
        model_cfg, adapter_components, trainable_map = get_model_cfg(base_cfg)
        return ModelConfig(
            name=model_name,
            backbone="DiT",
            transformer_kwargs=model_cfg,
            adapter_components=adapter_components,
            trainable_map=trainable_map,
            default_checkpoint="hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
            checkpoint_subdir="PEFT-TTS_v1",
        )

    if model_name in {"F5TTS_Base", "PEFTTTS_Base", "PEFT-TTS_base"}:
        base_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            conv_dilations=[1, 1, 2, 4],  # multi-scale for prosody
            pe_attn_head=1,
        )
        model_cfg, adapter_components, trainable_map = get_model_cfg(base_cfg)
        checkpoint_subdir = "PEFT-TTS_base"
        return ModelConfig(
            name=model_name,
            backbone="DiT",
            transformer_kwargs=model_cfg,
            adapter_components=adapter_components,
            trainable_map=trainable_map,
            default_checkpoint="hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt",
            checkpoint_subdir=checkpoint_subdir,
        )

    if model_name == "PEFT-TTS_v1":
        base_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                        conv_dilations=[1, 1, 2, 4])  # multi-scale for prosody
        model_cfg, adapter_components, trainable_map = get_model_cfg(base_cfg)
        return ModelConfig(
            name=model_name,
            backbone="DiT",
            transformer_kwargs=model_cfg,
            adapter_components=adapter_components,
            trainable_map=trainable_map,
            default_checkpoint="hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
            checkpoint_subdir="PEFT-TTS_v1",
        )

    if model_name == "E2TTS_Base":
        base_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        model_cfg, adapter_components, trainable_map = get_model_cfg(base_cfg)
        return ModelConfig(
            name=model_name,
            backbone="UNetT",
            transformer_kwargs=model_cfg,
            adapter_components=adapter_components,
            trainable_map=trainable_map,
            default_checkpoint="hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt",
            checkpoint_subdir="E2TTS_Base",
        )

    raise ValueError(f"Unknown model '{model_name}'. Pass an explicit model config.")


def _resolve_vocab_file(vocab_file: str | None) -> str:
    if vocab_file:
        return vocab_file
    return str(files("f5_tts").joinpath("infer/examples/vocab_ko.txt"))


def _resolve_train_pretrained_ckpt(model_cfg: ModelConfig, user_ckpt: str | None) -> str | None:
    if user_ckpt:
        return user_ckpt
    if not model_cfg.default_checkpoint:
        return None
    return str(cached_path(model_cfg.default_checkpoint))


def _resolve_infer_ckpt(model_cfg: ModelConfig, ckpt_file: str | None, ckpt_update: int) -> str:
    if ckpt_file:
        return str(Path(ckpt_file))
    project_root = Path(__file__).resolve().parents[3]
    subdir = model_cfg.checkpoint_subdir or model_cfg.name
    local_candidate = project_root / "ckpts" / "KSS" / subdir / f"model_{ckpt_update}.pt"
    if local_candidate.exists():
        return str(local_candidate)
    if model_cfg.default_checkpoint:
        return str(cached_path(model_cfg.default_checkpoint))
    raise FileNotFoundError(
        f"Checkpoint not found: {local_candidate}. Pass --ckpt_file explicitly."
    )


def build_train_app_config_from_args(args) -> AppConfig:
    model_cfg = build_model_config(args.exp_name)
    batching_mode = "dynamic"
    if getattr(args, "speaker_balanced_batching", False):
        batching_mode = "speaker_balanced"
    elif getattr(args, "speaker_aware_batching", False):
        batching_mode = "speaker_aware"
    elif getattr(args, "bucket_batching", False):
        batching_mode = "bucket"

    batching = BatchingConfig(
        batch_size_type=args.batch_size_type,
        batch_size_per_gpu=args.batch_size_per_gpu,
        max_samples=args.max_samples,
        mode=batching_mode,
        bucket_size=args.bucket_size,
        max_speakers_per_batch=args.max_speakers_per_batch,
        max_samples_per_speaker=args.max_samples_per_speaker,
        speakers_per_batch=args.speakers_per_batch,
        samples_per_speaker=args.samples_per_speaker,
    )

    train = TrainConfig(
        dataset_name=args.dataset_name,
        exp_name=args.exp_name,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        last_per_updates=args.last_per_updates,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        finetune=args.finetune,
        pretrained_ckpt=_resolve_train_pretrained_ckpt(model_cfg, args.pretrain),
        tokenizer=args.tokenizer,
        tokenizer_path=args.tokenizer_path,
        log_samples=args.log_samples,
        logger=args.logger,
        bnb_optimizer=args.bnb_optimizer,
        view_training_procedure=args.view_training_procedure2,
        prosody_loss_weight=args.prosody_loss_weight,
    )
    return AppConfig(model=model_cfg, batching=batching, train=train)


def build_infer_app_config_from_args(args) -> AppConfig:
    config: dict = {}
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, "rb") as f:
                config = tomli.load(f)
        elif not args.ckpt_file:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    checkpoint_meta = None
    metadata_app = None
    metadata_extra = {}

    if args.ckpt_file:
        ckpt_path = Path(args.ckpt_file)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        checkpoint_meta = load_checkpoint_metadata(ckpt_path)
        metadata_app = AppConfig.from_dict(checkpoint_meta["app_config"]) if checkpoint_meta else None
        metadata_extra = checkpoint_meta.get("extra", {}) if checkpoint_meta else {}

    requested_model_name = args.model
    if requested_model_name is None:
        requested_model_name = config.get("model")
    if metadata_app and not requested_model_name and not args.model_cfg:
        model_cfg = metadata_app.model
    else:
        requested_model_name = requested_model_name or "PEFT-TTS_base"
        model_cfg = build_model_config(requested_model_name, args.model_cfg)

    resolved_ckpt = _resolve_infer_ckpt(model_cfg, args.ckpt_file, args.ckpt_update)

    if checkpoint_meta is None:
        checkpoint_meta = load_checkpoint_metadata(resolved_ckpt)
        metadata_app = AppConfig.from_dict(checkpoint_meta["app_config"]) if checkpoint_meta else None
        metadata_extra = checkpoint_meta.get("extra", {}) if checkpoint_meta else {}

    if metadata_app and not args.model_cfg and not args.model:
        model_cfg = metadata_app.model

    inferred_vocab_file = _config_get(args.vocab_file, config, "vocab_file", None)
    if not inferred_vocab_file:
        inferred_vocab_file = metadata_extra.get("vocab_file")
    vocab_file = _resolve_vocab_file(inferred_vocab_file)

    inferred_vocoder_name = _config_get(args.vocoder_name, config, "vocoder_name", None)
    if inferred_vocoder_name is None:
        inferred_vocoder_name = metadata_extra.get("mel_spec_type", default_mel_spec_type)

    infer = InferConfig(
        model_name=model_cfg.name,
        ckpt_file=resolved_ckpt,
        vocab_file=vocab_file,
        ref_audio=_patch_example_path(_config_get(args.ref_audio, config, "ref_audio", None)),
        ref_text=_config_get(args.ref_text, config, "ref_text", ""),
        gen_text=_config_get(args.gen_text, config, "gen_text", ""),
        gen_file=_patch_example_path(_config_get(args.gen_file, config, "gen_file", "")),
        output_dir=_config_get(args.output_dir, config, "output_dir", "tests"),
        output_file=_config_get(
            args.output_file,
            config,
            "output_file",
            f"infer_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
        ),
        save_chunk=_bool_from_sources(args.save_chunk, config, "save_chunk", False),
        remove_silence=_bool_from_sources(args.remove_silence, config, "remove_silence", False),
        load_vocoder_from_local=_bool_from_sources(
            args.load_vocoder_from_local, config, "load_vocoder_from_local", False
        ),
        vocoder_name=inferred_vocoder_name,
        target_rms=_config_get(args.target_rms, config, "target_rms", default_target_rms),
        cross_fade_duration=_config_get(
            args.cross_fade_duration, config, "cross_fade_duration", default_cross_fade_duration
        ),
        nfe_step=_config_get(args.nfe_step, config, "nfe_step", default_nfe_step),
        cfg_strength=_config_get(args.cfg_strength, config, "cfg_strength", default_cfg_strength),
        sway_sampling_coef=_config_get(
            args.sway_sampling_coef, config, "sway_sampling_coef", default_sway_sampling_coef
        ),
        speed=_config_get(args.speed, config, "speed", default_speed),
        fix_duration=_config_get(args.fix_duration, config, "fix_duration", default_fix_duration),
    )

    if infer.gen_file:
        infer.gen_text = codecs.open(infer.gen_file, "r", "utf-8").read()

    app = AppConfig(model=model_cfg, infer=infer)
    setattr(app, "_raw_infer_config", dict(config))
    setattr(app, "_checkpoint_metadata", checkpoint_meta)
    return app

