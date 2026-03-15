
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import AppConfig


METADATA_VERSION = 1


class _ConfigEncoder(json.JSONEncoder):
    """BUG-38 FIX: handle non-serializable config objects (LoraConfig, ConvAdapterConfig, etc.)."""
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            return {"__class__": type(obj).__name__, **obj.__dict__}
        return super().default(obj)


def checkpoint_meta_path(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix:
        return checkpoint_path.with_suffix(".meta.json")
    return checkpoint_path.parent / f"{checkpoint_path.name}.meta.json"


def save_checkpoint_metadata(
    checkpoint_path: str | Path,
    app_config: AppConfig,
    extra: dict[str, Any] | None = None,
) -> Path:
    meta_path = checkpoint_meta_path(checkpoint_path)
    payload = {
        "metadata_version": METADATA_VERSION,
        "app_config": app_config.to_dict(),
        "extra": extra or {},
    }
    meta_path.write_text(json.dumps(payload, cls=_ConfigEncoder, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any] | None:
    meta_path = checkpoint_meta_path(checkpoint_path)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_app_config_from_checkpoint(checkpoint_path: str | Path) -> AppConfig | None:
    payload = load_checkpoint_metadata(checkpoint_path)
    if not payload:
        return None
    return AppConfig.from_dict(payload["app_config"])
