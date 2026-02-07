"""Shared helpers for evaluation output formatting."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _sanitize_tag(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return safe.strip("_")


def build_model_tag(
    *,
    model_type: Optional[str],
    model_weights: Optional[str] = None,
    model_path: Optional[str] = None,
    ciip_epoch: Optional[int] = None,
) -> str:
    parts = [part for part in [model_type, model_weights, model_path] if part]
    if model_type == "ciip_checkpoint" and ciip_epoch is not None:
        parts.append(f"epoch{ciip_epoch}")
    tag = "_".join(parts)
    return _sanitize_tag(tag) if tag else "model"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(_to_jsonable(payload), indent=2))
    return path


def write_run_manifest(
    output_dir: Path,
    *,
    task_name: str,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest = {
        "task": task_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": _to_jsonable(config),
    }
    if extra:
        manifest["extra"] = _to_jsonable(extra)
    return write_json(output_dir / "run_manifest.json", manifest)
