
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _reconstruct_config_objects(d: dict[str, Any]) -> dict[str, Any]:
    """BUG-42 FIX: reconstruct LoraConfig/ConvAdapterConfig from serialized dicts.

    When metadata is saved, _ConfigEncoder converts config objects to dicts
    with a '__class__' key. This function converts them back to proper objects
    so that DiT can access .method, .r, .lora_alpha etc.
    """
    from f5_tts.model.modules import LoraConfig, ConvAdapterConfig

    _CLASS_MAP = {
        "LoraConfig": LoraConfig,
        "ConvAdapterConfig": ConvAdapterConfig,
    }

    result = {}
    for k, v in d.items():
        if isinstance(v, dict) and "__class__" in v:
            cls_name = v["__class__"]
            cls = _CLASS_MAP.get(cls_name)
            if cls is not None:
                kwargs = {kk: vv for kk, vv in v.items() if kk != "__class__"}
                try:
                    result[k] = cls(**kwargs)
                except TypeError:
                    # If constructor signature doesn't match saved keys,
                    # create instance and set attrs directly
                    obj = object.__new__(cls)
                    for kk, vv in kwargs.items():
                        setattr(obj, kk, vv)
                    result[k] = obj
            else:
                result[k] = v
        else:
            result[k] = v
    return result


@dataclass
class ModelConfig:
    name: str
    backbone: str
    transformer_kwargs: dict[str, Any]
    adapter_components: dict[str, Any] = field(default_factory=dict)
    trainable_map: dict[str, str] = field(default_factory=dict)
    default_checkpoint: str | None = None
    checkpoint_subdir: str | None = None

    def to_transformer_kwargs(self) -> dict[str, Any]:
        return dict(self.transformer_kwargs)


@dataclass
class BatchingConfig:
    batch_size_type: str = "frame"
    batch_size_per_gpu: int = 1600
    max_samples: int = 64
    mode: str = "dynamic"  # dynamic | bucket | speaker_aware | speaker_balanced
    bucket_size: int = 512
    max_speakers_per_batch: int = 8
    max_samples_per_speaker: int = 8
    speakers_per_batch: int = 8
    samples_per_speaker: int = 4


@dataclass
class TrainConfig:
    dataset_name: str
    exp_name: str
    learning_rate: float = 1e-5
    epochs: int = 300
    num_warmup_updates: int = 14200
    save_per_updates: int = 14200
    keep_last_n_checkpoints: int = -1
    last_per_updates: int = 71000
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    finetune: bool = True
    pretrained_ckpt: str | None = None
    tokenizer: str = "pinyin"
    tokenizer_path: str | None = None
    log_samples: bool = False
    logger: str | None = None
    bnb_optimizer: bool = False
    view_training_procedure: bool = False
    prosody_loss_weight: float = 0.0


@dataclass
class InferConfig:
    model_name: str
    ckpt_file: str
    vocab_file: str
    ref_audio: str
    ref_text: str = ""
    gen_text: str = ""
    gen_file: str | None = None
    output_dir: str = "tests"
    output_file: str = "infer_cli.wav"
    save_chunk: bool = False
    remove_silence: bool = False
    load_vocoder_from_local: bool = False
    vocoder_name: str = "vocos"
    target_rms: float = 0.1
    cross_fade_duration: float = 0.15
    nfe_step: int = 32
    cfg_strength: float = 2.0
    sway_sampling_coef: float = -1.0
    speed: float = 1.0
    fix_duration: float | None = None


@dataclass
class AppConfig:
    model: ModelConfig
    batching: BatchingConfig | None = None
    train: TrainConfig | None = None
    infer: InferConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        model_data = dict(data["model"])
        # BUG-42 FIX: reconstruct config objects in transformer_kwargs and adapter_components
        if "transformer_kwargs" in model_data and isinstance(model_data["transformer_kwargs"], dict):
            model_data["transformer_kwargs"] = _reconstruct_config_objects(model_data["transformer_kwargs"])
        if "adapter_components" in model_data and isinstance(model_data["adapter_components"], dict):
            model_data["adapter_components"] = _reconstruct_config_objects(model_data["adapter_components"])
        model = ModelConfig(**model_data)
        batching = BatchingConfig(**data["batching"]) if data.get("batching") else None
        train = TrainConfig(**data["train"]) if data.get("train") else None
        infer = InferConfig(**data["infer"]) if data.get("infer") else None
        return cls(model=model, batching=batching, train=train, infer=infer)
