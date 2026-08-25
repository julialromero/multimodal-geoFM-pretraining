"""Shared helpers for evaluation output formatting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
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

RESULT_SCHEMA = "ciip.evaluation/v1"


@dataclass(frozen=True)
class EvaluationResult:
    """Portable record from which evaluation tables and plots can be rebuilt."""

    checkpoint: Optional[str]
    dataset: str
    split: str
    modality: str
    bands: tuple[str, ...]
    feature_space: str
    seed: int
    arguments: Dict[str, Any]
    metrics: Dict[str, Any]
    schema: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESULT_SCHEMA:
            raise ValueError(f"unsupported evaluation schema: {self.schema}")
        if not self.dataset or not self.split or not self.modality or not self.feature_space:
            raise ValueError("dataset, split, modality, and feature_space must be non-empty")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationResult":
        data = dict(payload)
        data["bands"] = tuple(data.get("bands", ()))
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


def write_evaluation_result(path: Path, result: EvaluationResult) -> Path:
    """Persist one evaluation result using the stable, versioned schema."""
    return write_json(path, result.to_dict())


def read_evaluation_result(path: Path) -> EvaluationResult:
    """Load and validate an evaluation result."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation result must be a JSON object: {path}")
    return EvaluationResult.from_dict(payload)


def discover_evaluation_results(root: Path) -> list[tuple[Path, EvaluationResult]]:
    """Find valid result records recursively, ignoring unrelated JSON files."""
    records = []
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema") == RESULT_SCHEMA:
            records.append((path, EvaluationResult.from_dict(payload)))
    return records
