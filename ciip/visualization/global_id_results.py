#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path("/home/juro4948/ciip/diagnostics/global_id_table1")
DOWNSTREAM_DIR = BASE_DIR / "id_downstream"
OUTPUT_PATH = BASE_DIR / "overall_results"
PRESET_ALIASES = {
    "resnet50_s2_all_dino": "dino",
    "croma": "croma_optical",
}

ID_METRIC_SPECS = [
    ("fisher_s", "FisherS ID"),
    ("mle", "MLE ID"),
    ("mom", "MoM ID"),
    ("tle", "TLE ID"),
    ("effective_rank", "Effective Rank"),
]


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[merge] Skipping {path}: {exc}")
        return None


def _normalize_checkpoint(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    try:
        return str(Path(path_str).expanduser().resolve(strict=False))
    except Exception:
        return path_str.strip()

def _canonical_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = str(value).strip()
    if "/" in raw or raw.endswith(".pt"):
        normalized = _normalize_checkpoint(raw)
        return normalized.lower() if normalized else None
    lowered = raw.lower()
    return PRESET_ALIASES.get(lowered, lowered)


def _extract_base_entries(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                yield entry
        return
    if isinstance(data, dict):
        yield data


def _checkpoint_from_base_entry(entry: Dict[str, Any]) -> Optional[str]:
    model_spec = entry.get("model_spec") or {}
    checkpoint = model_spec.get("checkpoint_path") or entry.get("checkpoint_path")
    return _normalize_checkpoint(checkpoint)


def _checkpoint_from_cache(cache: Dict[str, Any]) -> Optional[str]:
    meta = cache.get("meta") or {}
    preset = meta.get("model_preset")
    if preset:
        return str(preset)
    checkpoint = meta.get("checkpoint") or meta.get("checkpoint_path") or cache.get("checkpoint_path")
    return _normalize_checkpoint(checkpoint)


def _collect_base_entries(base_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(base_dir.glob("*.json")):
        data = _load_json(path)
        if data is None:
            continue
        for entry in _extract_base_entries(data):
            checkpoint = _checkpoint_from_base_entry(entry)
            if not checkpoint:
                model_spec = entry.get("model_spec") or {}
                checkpoint = model_spec.get("name") or entry.get("name")
                if not checkpoint:
                    print(f"[merge] Missing checkpoint_path in {path}")
                    continue
            canonical = _canonical_key(checkpoint)
            if not canonical:
                print(f"[merge] Unable to canonicalize key from {path}")
                continue
            out.setdefault(canonical, []).append(entry)
    return out


def _collect_downstream_caches(downstream_dir: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if not downstream_dir.exists():
        print(f"[merge] Downstream dir not found: {downstream_dir}")
        return out
    for path in sorted(downstream_dir.glob("*.json")):
        cache = _load_json(path)
        if not isinstance(cache, dict):
            continue
        checkpoint = _checkpoint_from_cache(cache)
        if not checkpoint:
            checkpoint = (cache.get("meta") or {}).get("model_preset")
            if not checkpoint:
                print(f"[merge] Missing meta.checkpoint in {path}")
                continue
        canonical = _canonical_key(checkpoint)
        if not canonical:
            print(f"[merge] Unable to canonicalize key from {path}")
            continue
        preprocess = (cache.get("meta") or {}).get("preprocess", "unknown")
        out.setdefault(canonical, {}).setdefault(str(preprocess), []).append(
            {"cache_file": str(path), "cache": cache}
        )
    return out


def _merge_results(
    base_entries: Dict[str, List[Dict[str, Any]]],
    downstream_caches: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    for checkpoint in sorted(base_entries.keys(), key=str.lower):
        downstream = downstream_caches.get(checkpoint)
        if not downstream:
            continue
        merged.append({
            "checkpoint": checkpoint,
            "base_entries": base_entries[checkpoint],
            "downstream": downstream,
        })
    return merged


def _select_s2_entry(entries: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for entry in entries:
        dataset = str(entry.get("dataset", "")).strip().upper()
        if dataset == "S2":
            candidates.append(entry)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for entry in candidates:
        id_results = entry.get("id_results")
        if isinstance(id_results, dict) and id_results:
            return entry
    for entry in candidates:
        if entry.get("dims_for_eval"):
            return entry
        model_spec = entry.get("model_spec") or {}
        if model_spec.get("eval_dims"):
            return entry
    return candidates[0]


def _select_cache(downstream: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    return _select_cache_with_preproc(downstream, None)


def _normalize_preproc(value: str) -> str:
    return str(value).strip().lower()


def _select_cache_with_preproc(
    downstream: Dict[str, List[Dict[str, Any]]],
    preferred_preproc: Optional[Iterable[str]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(downstream, dict):
        return None
    if preferred_preproc:
        normalized = { _normalize_preproc(key): key for key in downstream.keys() }
        matched = False
        for preproc in preferred_preproc:
            actual = normalized.get(_normalize_preproc(preproc))
            if not actual:
                continue
            matched = True
            caches = downstream.get(actual) or []
            if caches:
                return caches[0].get("cache")
        if matched:
            return None
    if "scaled" in downstream:
        caches = downstream.get("scaled") or []
        if caches:
            return caches[0].get("cache")
    for preprocess in sorted(downstream.keys()):
        caches = downstream.get(preprocess) or []
        if caches:
            return caches[0].get("cache")
    return None


def _performance_dict(
    cache: Dict[str, Any],
    preferred_preproc: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    perf_by_preprocess = cache.get("performance_by_preprocess")
    if preferred_preproc:
        if isinstance(perf_by_preprocess, dict) and perf_by_preprocess:
            normalized = { _normalize_preproc(key): key for key in perf_by_preprocess.keys() }
            for preproc in preferred_preproc:
                actual = normalized.get(_normalize_preproc(preproc))
                if not actual:
                    continue
                candidate = perf_by_preprocess.get(actual)
                if isinstance(candidate, dict) and candidate:
                    return candidate
            return None
        return None
    perf_results = cache.get("performance_results")
    if isinstance(perf_results, dict) and perf_results:
        return perf_results
    if isinstance(perf_by_preprocess, dict) and perf_by_preprocess:
        scaled = perf_by_preprocess.get("scaled")
        if isinstance(scaled, dict) and scaled:
            return scaled
        for value in perf_by_preprocess.values():
            if isinstance(value, dict) and value:
                return value
    return None


def _best_preproc_for_cache(
    cache: Dict[str, Any],
    embedding_dim: Optional[int],
    exclude_tasks: Optional[Iterable[str]] = None,
    allowed_preproc: Optional[Iterable[str]] = None,
) -> Optional[str]:
    perf_by_preprocess = cache.get("performance_by_preprocess")
    if not isinstance(perf_by_preprocess, dict) or not perf_by_preprocess:
        return None
    exclude = _exclude_set(exclude_tasks)
    allowed = None
    if allowed_preproc:
        allowed = {_normalize_preproc(p) for p in allowed_preproc if p}
    best_preproc = None
    best_score = None
    for preproc, perf_results in perf_by_preprocess.items():
        if allowed and _normalize_preproc(preproc) not in allowed:
            continue
        if not isinstance(perf_results, dict):
            continue
        per_dim = None
        if embedding_dim is not None:
            per_dim = perf_results.get(str(embedding_dim))
            if per_dim is None:
                per_dim = perf_results.get(embedding_dim)
        if per_dim is None and perf_results:
            per_dim = next(iter(perf_results.values()))
        if not isinstance(per_dim, dict):
            continue
        values: List[float] = []
        for key, val in per_dim.items():
            if exclude and _task_key(str(key)) in exclude:
                continue
            if isinstance(val, (int, float)) and math.isfinite(val):
                values.append(float(val))
        if not values:
            continue
        avg = sum(values) / len(values)
        if best_score is None or avg > best_score:
            best_score = avg
            best_preproc = preproc
    return best_preproc


_TASK_SUFFIX_RE = re.compile(r"\\[[^\\]]+\\]$")


def _normalize_task_name(task: str) -> str:
    return _TASK_SUFFIX_RE.sub("", str(task)).strip()


def _task_key(task: str) -> str:
    return _normalize_task_name(task).lower()


def _exclude_set(tasks: Optional[Iterable[str]]) -> set[str]:
    if not tasks:
        return set()
    out = set()
    for task in tasks:
        if task is None:
            continue
        if isinstance(task, str):
            for part in task.split(","):
                part = part.strip()
                if part:
                    out.add(_task_key(part))
    return out


def _model_label(entry: Dict[str, Any], base_entry: Dict[str, Any]) -> str:
    return str(
        (base_entry.get("model_spec") or {}).get("name")
        or base_entry.get("name")
        or Path(entry.get("checkpoint", "model")).stem
    )


def _log_missing_tasks(
    label: str,
    chosen_tasks: Iterable[str],
    available_tasks: Iterable[str],
    logged: set[str],
) -> None:
    if label in logged:
        return
    chosen = {_task_key(t) for t in chosen_tasks}
    available = {_task_key(t) for t in available_tasks}
    if chosen and not (chosen & available):
        missing = ", ".join(sorted(chosen))
        print(f"[plot] Missing downstream tasks for {label}: {missing}")
        logged.add(label)


def _set_task_ylim(ax: plt.Axes, task: str) -> None:
    base_task = _normalize_task_name(task)
    if base_task in (
        "neuco_heatisland_mean",
        "neuco_heatisland_std",
        "neuco_biomass_mean",
        "neuco_biomass_std",
    ):
        ax.set_ylim(-0.2, 0.6)
    else:
        ax.set_ylim(0.2, 0.8)


def _set_auto_ylim(ax: plt.Axes, ys: List[float]) -> None:
    if not ys:
        return
    y_min = min(ys)
    y_max = max(ys)
    if math.isclose(y_min, y_max):
        pad = 0.05 if y_min == 0 else abs(y_min) * 0.05
    else:
        pad = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - pad, y_max + pad)


def _downstream_task_key(task: str, keys: Iterable[str]) -> Optional[str]:
    if task in keys:
        return task
    base = _normalize_task_name(task)
    if base in keys:
        return base
    if base.startswith("eurosat_") and "eurosat_train" in keys:
        return "eurosat_train"
    return None


def _select_cache_for_dims(
    downstream: Dict[str, List[Dict[str, Any]]],
    dims: List[int],
    preferred_preproc: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(downstream, dict):
        return None
    cache_lists: List[List[Dict[str, Any]]] = []
    if preferred_preproc:
        normalized = { _normalize_preproc(key): key for key in downstream.keys() }
        matched = False
        for preproc in preferred_preproc:
            actual = normalized.get(_normalize_preproc(preproc))
            if not actual:
                continue
            matched = True
            cache_lists.append(downstream.get(actual) or [])
        if matched and not any(cache_lists):
            return None
        if not matched:
            cache_lists = []
            if "scaled" in downstream:
                cache_lists.append(downstream.get("scaled") or [])
            for key in sorted(downstream.keys()):
                if key == "scaled":
                    continue
                cache_lists.append(downstream.get(key) or [])
    else:
        if "scaled" in downstream:
            cache_lists.append(downstream.get("scaled") or [])
        for key in sorted(downstream.keys()):
            if key == "scaled":
                continue
            cache_lists.append(downstream.get(key) or [])
    best_cache: Optional[Dict[str, Any]] = None
    best_match = -1
    best_total = -1
    for caches in cache_lists:
        for item in caches:
            cache = item.get("cache") if isinstance(item, dict) else None
            if not isinstance(cache, dict):
                continue
            perf_dict = _performance_dict(cache, preferred_preproc)
            if not perf_dict:
                continue
            match = sum(1 for d in dims if str(d) in perf_dict or d in perf_dict)
            total = len(perf_dict)
            if match > best_match or (match == best_match and total > best_total):
                best_cache = cache
                best_match = match
                best_total = total
    return best_cache or _select_cache_with_preproc(downstream, preferred_preproc)


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _is_matryoshka_entry(entry: Dict[str, Any]) -> bool:
    name = (
        (entry.get("model_spec") or {}).get("name")
        or entry.get("name")
        or ""
    )
    if "matryoshka" in str(name).lower():
        return True
    return bool(_matryoshka_dims_from_entry(entry))


def _matryoshka_dims_from_entry(
    entry: Dict[str, Any],
    *,
    include_embedding_dim: bool = False,
) -> List[int]:
    dims: List[int] = []
    id_results = entry.get("id_results")
    if isinstance(id_results, dict):
        for key in id_results.keys():
            val = _coerce_int(key)
            if val is not None:
                dims.append(val)
    if not dims:
        for source in (
            (entry.get("model_spec") or {}).get("eval_dims"),
            entry.get("dims_for_eval"),
        ):
            if isinstance(source, (list, tuple)):
                for item in source:
                    val = _coerce_int(item)
                    if val is not None:
                        dims.append(val)
    if not include_embedding_dim:
        embedding_dim = _coerce_int(entry.get("embedding_dim"))
        if embedding_dim is not None:
            dims = [d for d in dims if d != embedding_dim]
    dims = sorted({d for d in dims if d > 0})
    return dims


def _id_metric_by_dim(
    entry: Dict[str, Any],
    dim: int,
    metric_key: str,
) -> Optional[float]:
    id_results = entry.get("id_results")
    if not isinstance(id_results, dict):
        return None
    dim_entry = id_results.get(str(dim))
    if dim_entry is None:
        dim_entry = id_results.get(dim)
    if not isinstance(dim_entry, dict):
        return None
    dataset_key = str(entry.get("dataset", "")).strip().lower()
    metrics = dim_entry.get(dataset_key)
    if metrics is None and len(dim_entry) == 1:
        metrics = next(iter(dim_entry.values()))
    if not isinstance(metrics, dict):
        return None
    metric_value = metrics.get(metric_key)
    if isinstance(metric_value, (int, float)) and math.isfinite(metric_value):
        return float(metric_value)
    return None


def _average_performance(
    cache: Dict[str, Any],
    embedding_dim: Optional[int],
    exclude_tasks: Optional[Iterable[str]] = None,
    preferred_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
) -> Optional[float]:
    if best_preproc:
        chosen = _best_preproc_for_cache(cache, embedding_dim, exclude_tasks, preferred_preproc)
        if not chosen:
            return None
        perf_results = _performance_dict(cache, [chosen])
    else:
        perf_results = _performance_dict(cache, preferred_preproc)
    if not isinstance(perf_results, dict):
        return None
    per_dim = None
    if embedding_dim is not None:
        per_dim = perf_results.get(str(embedding_dim))
        if per_dim is None:
            per_dim = perf_results.get(embedding_dim)
    if per_dim is None and perf_results:
        per_dim = next(iter(perf_results.values()))
    if not isinstance(per_dim, dict):
        return None
    exclude = _exclude_set(exclude_tasks)
    values: List[float] = []
    for key, val in per_dim.items():
        if exclude and _task_key(str(key)) in exclude:
            continue
        if isinstance(val, (int, float)) and math.isfinite(val):
            values.append(float(val))
    if not values:
        return None
    return sum(values) / len(values)


def _update_task_minmax(
    task_minmax: Dict[str, Tuple[float, float]],
    perf_map: Dict[str, float],
) -> None:
    for task, val in perf_map.items():
        key = _task_key(task)
        current = task_minmax.get(key)
        if current is None:
            task_minmax[key] = (val, val)
        else:
            task_minmax[key] = (min(current[0], val), max(current[1], val))


def _task_minmax_for_entries(
    merged: List[Dict[str, Any]],
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
) -> Dict[str, Tuple[float, float]]:
    task_minmax: Dict[str, Tuple[float, float]] = {}
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry:
            continue
        embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
        cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
        if not cache:
            continue
        perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
        if not perf_map:
            continue
        _update_task_minmax(task_minmax, perf_map)
    return task_minmax


def _task_minmax_for_matryoshka_entries(
    merged: List[Dict[str, Any]],
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
) -> Dict[str, Tuple[float, float]]:
    task_minmax: Dict[str, Tuple[float, float]] = {}
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry or not _is_matryoshka_entry(base_entry):
            continue
        dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
        if not dims:
            continue
        cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
        if not cache:
            continue
        for dim in dims:
            perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map:
                continue
            _update_task_minmax(task_minmax, perf_map)
    return task_minmax


def _normalized_average_performance(
    perf_map: Dict[str, float],
    task_minmax: Dict[str, Tuple[float, float]],
) -> Optional[float]:
    values: List[float] = []
    for task, val in perf_map.items():
        normalized = _normalized_task_value(task, val, task_minmax)
        if normalized is None:
            continue
        values.append(normalized)
    if not values:
        return None
    return sum(values) / len(values)


def _normalized_task_value(
    task: str,
    value: float,
    task_minmax: Dict[str, Tuple[float, float]],
) -> Optional[float]:
    stats = task_minmax.get(_task_key(task))
    if not stats:
        return None
    min_val, max_val = stats
    if not math.isfinite(min_val) or not math.isfinite(max_val):
        return None
    if math.isclose(min_val, max_val):
        return 1.0
    return (value - min_val) / (max_val - min_val)


def _performance_by_task(
    cache: Dict[str, Any],
    embedding_dim: Optional[int],
    exclude_tasks: Optional[Iterable[str]] = None,
    preferred_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
) -> Optional[Dict[str, float]]:
    if best_preproc:
        chosen = _best_preproc_for_cache(cache, embedding_dim, exclude_tasks, preferred_preproc)
        if not chosen:
            return None
        perf_results = _performance_dict(cache, [chosen])
    else:
        perf_results = _performance_dict(cache, preferred_preproc)
    if not isinstance(perf_results, dict):
        return None
    per_dim = None
    if embedding_dim is not None:
        per_dim = perf_results.get(str(embedding_dim))
        if per_dim is None:
            per_dim = perf_results.get(embedding_dim)
    if per_dim is None and perf_results:
        per_dim = next(iter(perf_results.values()))
    if not isinstance(per_dim, dict):
        return None
    exclude = _exclude_set(exclude_tasks)
    task_map: Dict[str, float] = {}
    for key, val in per_dim.items():
        if exclude and _task_key(str(key)) in exclude:
            continue
        if isinstance(val, (int, float)) and math.isfinite(val):
            task_map[str(key)] = float(val)
    return task_map or None


def _downstream_fisher(cache: Dict[str, Any], embedding_dim: Optional[int], task: str) -> Optional[float]:
    id_results = cache.get("id_results")
    if not isinstance(id_results, dict):
        return None
    if embedding_dim is None:
        return None
    per_dim = id_results.get(str(embedding_dim))
    if per_dim is None:
        per_dim = id_results.get(embedding_dim)
    if not isinstance(per_dim, dict):
        return None
    task_key = _downstream_task_key(task, per_dim.keys())
    if task_key is None:
        return None
    metrics = per_dim.get(task_key)
    if not isinstance(metrics, dict):
        return None
    for key in ("fishers", "fisher_s"):
        val = metrics.get(key)
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
    return None


def _downstream_effective_rank(cache: Dict[str, Any], embedding_dim: Optional[int], task: str) -> Optional[float]:
    er_results = cache.get("er_results")
    if not isinstance(er_results, dict):
        return None
    if embedding_dim is None:
        return None
    per_dim = er_results.get(str(embedding_dim))
    if per_dim is None:
        per_dim = er_results.get(embedding_dim)
    if not isinstance(per_dim, dict):
        return None
    task_key = _downstream_task_key(task, per_dim.keys())
    if task_key is None:
        return None
    val = per_dim.get(task_key)
    if isinstance(val, (int, float)) and math.isfinite(val):
        return float(val)
    return None


def _downstream_id_metric(
    cache: Dict[str, Any],
    embedding_dim: Optional[int],
    task: str,
    metric_key: str,
) -> Optional[float]:
    if metric_key == "effective_rank":
        return _downstream_effective_rank(cache, embedding_dim, task)
    id_results = cache.get("id_results")
    if not isinstance(id_results, dict):
        return None
    if embedding_dim is None:
        return None
    per_dim = id_results.get(str(embedding_dim))
    if per_dim is None:
        per_dim = id_results.get(embedding_dim)
    if not isinstance(per_dim, dict):
        return None
    task_key = _downstream_task_key(task, per_dim.keys())
    if task_key is None:
        return None
    metrics = per_dim.get(task_key)
    if not isinstance(metrics, dict):
        return None
    if metric_key == "fisher_s":
        for key in ("fishers", "fisher_s"):
            val = metrics.get(key)
            if isinstance(val, (int, float)) and math.isfinite(val):
                return float(val)
        return None
    val = metrics.get(metric_key)
    if isinstance(val, (int, float)) and math.isfinite(val):
        return float(val)
    return None


def _s2_id_metric(
    base_entry: Dict[str, Any],
    cache: Dict[str, Any],
    embedding_dim: Optional[int],
    metric_key: str,
) -> Optional[float]:
    metric_value = (base_entry.get("metrics") or {}).get(metric_key)
    if isinstance(metric_value, (int, float)) and math.isfinite(metric_value):
        return float(metric_value)
    return None


def _effective_rank_value(cache: Dict[str, Any], embedding_dim: Optional[int]) -> Optional[float]:
    er_results = cache.get("er_results")
    if not isinstance(er_results, dict):
        return None
    per_dim = None
    if embedding_dim is not None:
        per_dim = er_results.get(str(embedding_dim))
        if per_dim is None:
            per_dim = er_results.get(embedding_dim)
    if per_dim is None and er_results:
        per_dim = next(iter(er_results.values()))
    if not isinstance(per_dim, dict):
        return None
    if "ssl4eo_val" in per_dim and isinstance(per_dim["ssl4eo_val"], (int, float)):
        val = float(per_dim["ssl4eo_val"])
        return val if math.isfinite(val) else None
    for val in per_dim.values():
        if isinstance(val, (int, float)) and math.isfinite(val):
            return float(val)
    return None


def _add_regression_line(ax: plt.Axes, xs: List[float], ys: List[float]) -> None:
    if len(xs) < 2:
        return
    x_vals = np.asarray(xs, dtype=float)
    y_vals = np.asarray(ys, dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    if x_vals.size < 2:
        return
    x_min = float(np.min(x_vals))
    x_max = float(np.max(x_vals))
    if math.isclose(x_min, x_max):
        return
    try:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
    except Exception:
        return
    xs_line = np.array([x_min, x_max], dtype=float)
    ys_line = slope * xs_line + intercept
    ax.plot(xs_line, ys_line, linestyle="--", linewidth=1.5, color="#444444", alpha=0.8)


def _rotation_by_x_proximity(
    xs: List[float],
    ys: List[float],
    *,
    x_threshold: float = 0.5,
    y_threshold: float = 0.05,
    pos_angle: float = 20.0,
    neg_angle: float = -20.0,
) -> List[float]:
    rotations = [0.0 for _ in xs]
    n = len(xs)
    if n < 2:
        return rotations
    if x_threshold <= 0 or y_threshold < 0:
        return rotations
    adjacency: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        xi = xs[i]
        yi = ys[i]
        for j in range(i + 1, n):
            if abs(xs[j] - xi) <= x_threshold and abs(ys[j] - yi) <= y_threshold:
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = [False] * n
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        component: List[int] = []
        visited[i] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in adjacency[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        if len(component) < 2:
            continue
        max_idx = max(component, key=lambda idx: ys[idx])
        min_idx = min(component, key=lambda idx: ys[idx])
        if max_idx == min_idx:
            continue
        rotations[max_idx] = pos_angle
        rotations[min_idx] = neg_angle
    return rotations


def _generate_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
    normalize_task_performance: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("fisher_s", "FisherS ID"),
        ("mle", "MLE ID"),
        ("mom", "MoM ID"),
        ("tle", "TLE ID"),
        ("effective_rank", "Effective Rank"),
    ]

    task_minmax: Dict[str, Tuple[float, float]] = {}
    if normalize_task_performance:
        task_minmax = _task_minmax_for_entries(
            merged,
            exclude_tasks,
            downstream_preproc,
            best_preproc,
        )
        if not task_minmax:
            print("[merge] No downstream performance tasks found for normalization; using raw averages.")
            normalize_task_performance = False

    metric_series: List[tuple[str, str, List[float], List[float], List[str]]] = []
    for metric_key, metric_label in metrics:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            embedding_dim = base_entry.get("embedding_dim")
            if isinstance(embedding_dim, float):
                embedding_dim = int(embedding_dim)
            elif not isinstance(embedding_dim, int):
                embedding_dim = None

            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            if normalize_task_performance:
                perf_map = _performance_by_task(
                    cache,
                    embedding_dim,
                    exclude_tasks,
                    downstream_preproc,
                    best_preproc,
                )
                if not perf_map:
                    continue
                avg_perf = _normalized_average_performance(perf_map, task_minmax)
            else:
                avg_perf = _average_performance(
                    cache,
                    embedding_dim,
                    exclude_tasks,
                    downstream_preproc,
                    best_preproc,
                )
            if avg_perf is None:
                continue

            metric_value = (base_entry.get("metrics") or {}).get(metric_key)

            if not isinstance(metric_value, (int, float)) or not math.isfinite(metric_value):
                continue

            label = (
                (base_entry.get("model_spec") or {}).get("name")
                or base_entry.get("name")
                or Path(entry.get("checkpoint", "model")).stem
            )
            xs.append(float(metric_value))
            ys.append(float(avg_perf))
            labels.append(str(label))

        if not xs:
            print(f"[merge] No points for {metric_key} scatterplot")
            continue

        metric_series.append((metric_key, metric_label, xs, ys, labels))

    if not metric_series:
        print("[merge] No points for S2 ID vs avg performance scatterplots")
        return

    n_metrics = len(metric_series)
    cols = 3 if n_metrics > 3 else n_metrics
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (metric_key, metric_label, xs, ys, labels) in enumerate(metric_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        x_window = 0.5
        if metric_key in ("mle", "mom", "tle"):
            x_window = 2.0
        elif metric_key == "effective_rank":
            x_window = 100.0
        rotations = _rotation_by_x_proximity(xs, ys, x_threshold=x_window)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=11,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )

        ax.set_xlabel(f"S2 {metric_label}" if metric_key != "effective_rank" else "Effective Rank")
        ax.set_ylabel(
            "Average normalized task performance"
            if normalize_task_performance
            else "Average downstream performance"
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"{metric_label} (S2) vs Avg Normalized Task Performance"
            if normalize_task_performance
            else f"{metric_label} (S2) vs Avg Downstream Performance"
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(wspace=0.55)
    out_path = plots_dir / "s2_id_vs_avg_performance_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote scatterplot grid {out_path}")


def _generate_dataset_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    s2_id_metric: str = "fisher_s",
    auto_task_ylim: bool = False,
    best_preproc: bool = False,
    normalize_task_performance: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_names: List[str] = []
    task_set = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry:
            continue
        embedding_dim = base_entry.get("embedding_dim")
        if isinstance(embedding_dim, float):
            embedding_dim = int(embedding_dim)
        elif not isinstance(embedding_dim, int):
            embedding_dim = None
        cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
        if not cache:
            continue
        perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
        if not perf_map:
            continue
        for task in perf_map.keys():
            if task not in task_set:
                task_set.add(task)
                task_names.append(task)

    if not task_names:
        print("[merge] No downstream performance tasks found for dataset scatterplots")
        return

    task_minmax: Dict[str, Tuple[float, float]] = {}
    if normalize_task_performance:
        task_minmax = _task_minmax_for_entries(
            merged,
            exclude_tasks,
            downstream_preproc,
            best_preproc,
        )
        if not task_minmax:
            print("[merge] No downstream performance tasks found for normalization; using raw values.")
            normalize_task_performance = False

    missing_logged: set[str] = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry:
            continue
        embedding_dim = base_entry.get("embedding_dim")
        if isinstance(embedding_dim, float):
            embedding_dim = int(embedding_dim)
        elif not isinstance(embedding_dim, int):
            embedding_dim = None
        cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
        if not cache:
            continue
        perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
        label = _model_label(entry, base_entry)
        available = perf_map.keys() if perf_map else []
        _log_missing_tasks(label, task_names, available, missing_logged)

    task_series: List[tuple[str, List[float], List[float], List[str]]] = []
    for task in task_names:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            embedding_dim = base_entry.get("embedding_dim")
            if isinstance(embedding_dim, float):
                embedding_dim = int(embedding_dim)
            elif not isinstance(embedding_dim, int):
                embedding_dim = None
            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            metric_value = _s2_id_metric(base_entry, cache, embedding_dim, s2_id_metric)
            if metric_value is None:
                continue
            perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map or task not in perf_map:
                continue
            perf_value = perf_map[task]
            if normalize_task_performance:
                normalized = _normalized_task_value(task, perf_value, task_minmax)
                if normalized is None:
                    continue
                perf_value = normalized
            label = (
                (base_entry.get("model_spec") or {}).get("name")
                or base_entry.get("name")
                or Path(entry.get("checkpoint", "model")).stem
            )
            xs.append(float(metric_value))
            ys.append(float(perf_value))
            labels.append(str(label))

        if xs:
            task_series.append((task, xs, ys, labels))

    if not task_series:
        print("[merge] No valid points found for dataset scatterplots")
        return

    n_tasks = len(task_series)
    cols = 3 if n_tasks <= 9 else 4
    rows = math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (task, xs, ys, labels) in enumerate(task_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        rotations = _rotation_by_x_proximity(xs, ys)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=10,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )
        metric_label = dict(ID_METRIC_SPECS).get(s2_id_metric, s2_id_metric)
        ax.set_xlabel(f"S2 {metric_label}" if s2_id_metric != "effective_rank" else "Effective Rank")
        ax.set_ylabel("Normalized task performance" if normalize_task_performance else "Performance")
        if auto_task_ylim:
            _set_auto_ylim(ax, ys)
        elif normalize_task_performance:
            ax.set_ylim(0.0, 1.0)
        else:
            _set_task_ylim(ax, task)
        ax.set_title(
            f"{task} (Normalized task performance)"
            if normalize_task_performance
            else task
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_tasks:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    if s2_id_metric == "fisher_s":
        out_name = "s2_fishers_vs_performance_by_task.png"
    else:
        out_name = f"s2_{s2_id_metric}_vs_performance_by_task.png"
    out_path = plots_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote dataset scatterplot grid {out_path}")


def _generate_downstream_fisher_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
    normalize_task_performance: bool = False,
    auto_task_ylim: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_names: List[str] = []
    task_set = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry:
            continue
        embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
        if embedding_dim is None:
            continue
        cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
        if not cache:
            continue
        perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
        if not perf_map:
            continue
        for task in perf_map.keys():
            if task not in task_set:
                task_set.add(task)
                task_names.append(task)

    if not task_names:
        print("[merge] No downstream performance tasks found for downstream FisherS scatterplots")
        return

    task_minmax: Dict[str, Tuple[float, float]] = {}
    if normalize_task_performance:
        task_minmax = _task_minmax_for_entries(
            merged,
            exclude_tasks,
            downstream_preproc,
            best_preproc,
        )
        if not task_minmax:
            print("[merge] No downstream performance tasks found for normalization; using raw values.")
            normalize_task_performance = False

    missing_logged: set[str] = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry:
            continue
        embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
        if embedding_dim is None:
            continue
        cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
        if not cache:
            continue
        perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
        label = _model_label(entry, base_entry)
        available = perf_map.keys() if perf_map else []
        _log_missing_tasks(label, task_names, available, missing_logged)

    task_series: List[tuple[str, List[float], List[float], List[str]]] = []
    for task in task_names:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
            if embedding_dim is None:
                continue
            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map or task not in perf_map:
                continue
            perf_value = perf_map[task]
            if normalize_task_performance:
                normalized = _normalized_task_value(task, perf_value, task_minmax)
                if normalized is None:
                    continue
                perf_value = normalized
            fisher = _downstream_fisher(cache, embedding_dim, task)
            if fisher is None:
                continue
            label = (
                (base_entry.get("model_spec") or {}).get("name")
                or base_entry.get("name")
                or Path(entry.get("checkpoint", "model")).stem
            )
            xs.append(float(fisher))
            ys.append(float(perf_value))
            labels.append(str(label))

        if xs:
            task_series.append((task, xs, ys, labels))

    if not task_series:
        print("[merge] No valid points found for downstream FisherS scatterplots")
        return

    n_tasks = len(task_series)
    cols = 3 if n_tasks <= 9 else 4
    rows = math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (task, xs, ys, labels) in enumerate(task_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        rotations = _rotation_by_x_proximity(xs, ys)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=10,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )
        ax.set_xlabel("Downstream FisherS ID")
        ax.set_ylabel("Normalized task performance" if normalize_task_performance else "Performance")
        if auto_task_ylim:
            _set_auto_ylim(ax, ys)
        elif normalize_task_performance:
            ax.set_ylim(0.0, 1.0)
        else:
            _set_task_ylim(ax, task)
        ax.set_title(
            f"{task} (Normalized task performance)"
            if normalize_task_performance
            else task
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_tasks:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    out_path = plots_dir / "downstream_fishers_vs_performance_by_task.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote downstream FisherS scatterplot grid {out_path}")


def _generate_downstream_id_metric_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
    normalize_task_performance: bool = False,
    auto_task_ylim: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_minmax: Dict[str, Tuple[float, float]] = {}
    if normalize_task_performance:
        task_minmax = _task_minmax_for_entries(
            merged,
            exclude_tasks,
            downstream_preproc,
            best_preproc,
        )
        if not task_minmax:
            print("[merge] No downstream performance tasks found for normalization; using raw values.")
            normalize_task_performance = False

    for metric_key, metric_label in ID_METRIC_SPECS:
        task_names: List[str] = []
        task_set = set()
        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
            if embedding_dim is None:
                continue
            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map:
                continue
            for task in perf_map.keys():
                if task not in task_set:
                    task_set.add(task)
                    task_names.append(task)

        if not task_names:
            print(f"[merge] No downstream tasks found for {metric_label} scatterplots")
            continue

        missing_logged: set[str] = set()
        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
            if embedding_dim is None:
                continue
            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
            label = _model_label(entry, base_entry)
            available = perf_map.keys() if perf_map else []
            _log_missing_tasks(label, task_names, available, missing_logged)

        task_series: List[tuple[str, List[float], List[float], List[str]]] = []
        for task in task_names:
            xs: List[float] = []
            ys: List[float] = []
            labels: List[str] = []

            for entry in merged:
                base_entry = _select_s2_entry(entry.get("base_entries", []))
                if not base_entry:
                    continue
                embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
                if embedding_dim is None:
                    continue
                cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
                if not cache:
                    continue
                perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
                if not perf_map or task not in perf_map:
                    continue
                perf_value = perf_map[task]
                if normalize_task_performance:
                    normalized = _normalized_task_value(task, perf_value, task_minmax)
                    if normalized is None:
                        continue
                    perf_value = normalized
                metric_value = _downstream_id_metric(cache, embedding_dim, task, metric_key)
                if metric_value is None:
                    continue
                label = _model_label(entry, base_entry)
                xs.append(float(metric_value))
                ys.append(float(perf_value))
                labels.append(label)

            if xs:
                task_series.append((task, xs, ys, labels))

        if not task_series:
            print(f"[merge] No valid points found for {metric_label} scatterplots")
            continue

        n_tasks = len(task_series)
        cols = 3 if n_tasks <= 9 else 4
        rows = math.ceil(n_tasks / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (task, xs, ys, labels) in enumerate(task_series):
            ax = axes_list[idx]
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect(1)

            ax.scatter(xs, ys, alpha=0.85)
            _add_regression_line(ax, xs, ys)
            rotations = _rotation_by_x_proximity(xs, ys)
            for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 0),
                    fontsize=10,
                    rotation=rotation,
                    ha="left",
                    va="center",
                    rotation_mode="anchor",
                    clip_on=False,
                )
            ax.set_xlabel(f"Downstream {metric_label}" if metric_key != "effective_rank" else "Effective Rank")
            ax.set_ylabel("Normalized task performance" if normalize_task_performance else "Performance")
            if auto_task_ylim:
                _set_auto_ylim(ax, ys)
            elif normalize_task_performance:
                ax.set_ylim(0.0, 1.0)
            else:
                _set_task_ylim(ax, task)
            ax.set_title(
                f"{task} (Normalized task performance)"
                if normalize_task_performance
                else task
            )
            ax.grid(True, alpha=0.3, linewidth=0.5)

        for ax in axes_list[n_tasks:]:
            ax.set_visible(False)

        fig.tight_layout(pad=0.4)
        out_path = plots_dir / f"downstream_{metric_key}_vs_performance_by_task.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[merge] Wrote downstream {metric_label} scatterplot grid {out_path}")


def _generate_downstream_fishers_vs_s2_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_names: List[str] = []
    task_set = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry:
            continue
        embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
        if embedding_dim is None:
            continue
        cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
        if not cache:
            continue
        perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
        if not perf_map:
            continue
        for task in perf_map.keys():
            if task not in task_set:
                task_set.add(task)
                task_names.append(task)

    if not task_names:
        print("[merge] No downstream tasks found for S2-vs-downstream FisherS scatterplots")
        return

    missing_logged: set[str] = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry:
            continue
        embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
        if embedding_dim is None:
            continue
        cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
        if not cache:
            continue
        perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
        label = _model_label(entry, base_entry)
        available = perf_map.keys() if perf_map else []
        _log_missing_tasks(label, task_names, available, missing_logged)

    task_series: List[tuple[str, List[float], List[float], List[str]]] = []
    for task in task_names:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            s2_fisher = (base_entry.get("metrics") or {}).get("fisher_s")
            if not isinstance(s2_fisher, (int, float)) or not math.isfinite(s2_fisher):
                continue
            embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
            if embedding_dim is None:
                continue
            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            downstream_fisher = _downstream_fisher(cache, embedding_dim, task)
            if downstream_fisher is None:
                continue
            label = (
                (base_entry.get("model_spec") or {}).get("name")
                or base_entry.get("name")
                or Path(entry.get("checkpoint", "model")).stem
            )
            xs.append(float(s2_fisher))
            ys.append(float(downstream_fisher))
            labels.append(str(label))

        if xs:
            task_series.append((task, xs, ys, labels))

    if not task_series:
        print("[merge] No valid points found for S2-vs-downstream FisherS scatterplots")
        return

    n_tasks = len(task_series)
    cols = 3 if n_tasks <= 9 else 4
    rows = math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (task, xs, ys, labels) in enumerate(task_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        rotations = _rotation_by_x_proximity(xs, ys)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=10,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )
        ax.set_xlabel("S2 FisherS ID")
        ax.set_ylabel("Downstream FisherS ID")
        ax.set_title(task)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_tasks:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    out_path = plots_dir / "s2_fishers_vs_downstream_fishers_by_task.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote S2-vs-downstream FisherS scatterplot grid {out_path}")


def _generate_s2_vs_downstream_id_metric_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_label in ID_METRIC_SPECS:
        task_names: List[str] = []
        task_set = set()
        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
            if embedding_dim is None:
                continue
            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map:
                continue
            for task in perf_map.keys():
                if task not in task_set:
                    task_set.add(task)
                    task_names.append(task)

        if not task_names:
            print(f"[merge] No downstream tasks found for S2-vs-downstream {metric_label} scatterplots")
            continue

        missing_logged: set[str] = set()
        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry:
                continue
            embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
            if embedding_dim is None:
                continue
            cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
            if not cache:
                continue
            perf_map = _performance_by_task(cache, embedding_dim, exclude_tasks, downstream_preproc, best_preproc)
            label = _model_label(entry, base_entry)
            available = perf_map.keys() if perf_map else []
            _log_missing_tasks(label, task_names, available, missing_logged)

        task_series: List[tuple[str, List[float], List[float], List[str]]] = []
        for task in task_names:
            xs: List[float] = []
            ys: List[float] = []
            labels: List[str] = []
            for entry in merged:
                base_entry = _select_s2_entry(entry.get("base_entries", []))
                if not base_entry:
                    continue
                embedding_dim = _coerce_int(base_entry.get("embedding_dim"))
                if embedding_dim is None:
                    continue
                cache = _select_cache_with_preproc(entry.get("downstream", {}), downstream_preproc)
                if not cache:
                    continue
                s2_metric = _s2_id_metric(base_entry, cache, embedding_dim, metric_key)
                if s2_metric is None:
                    continue
                downstream_metric = _downstream_id_metric(cache, embedding_dim, task, metric_key)
                if downstream_metric is None:
                    continue
                label = _model_label(entry, base_entry)
                xs.append(float(s2_metric))
                ys.append(float(downstream_metric))
                labels.append(label)
            if xs:
                task_series.append((task, xs, ys, labels))

        if not task_series:
            print(f"[merge] No valid points found for S2-vs-downstream {metric_label} scatterplots")
            continue

        n_tasks = len(task_series)
        cols = 3 if n_tasks <= 9 else 4
        rows = math.ceil(n_tasks / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (task, xs, ys, labels) in enumerate(task_series):
            ax = axes_list[idx]
            if hasattr(ax, "set_box_aspect"):
                ax.set_box_aspect(1)

            ax.scatter(xs, ys, alpha=0.85)
            _add_regression_line(ax, xs, ys)
            rotations = _rotation_by_x_proximity(xs, ys)
            for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 0),
                    fontsize=10,
                    rotation=rotation,
                    ha="left",
                    va="center",
                    rotation_mode="anchor",
                    clip_on=False,
                )
            if metric_key == "effective_rank":
                ax.set_xlabel("Effective Rank")
                ax.set_ylabel("Downstream Effective Rank")
            else:
                ax.set_xlabel(f"S2 {metric_label}")
                ax.set_ylabel(f"Downstream {metric_label}")
            ax.set_title(task)
            ax.grid(True, alpha=0.3, linewidth=0.5)

        for ax in axes_list[n_tasks:]:
            ax.set_visible(False)

        fig.tight_layout(pad=0.4)
        out_path = plots_dir / f"s2_vs_downstream_{metric_key}_by_task.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[merge] Wrote S2-vs-downstream {metric_label} scatterplot grid {out_path}")


def _generate_matryoshka_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
    normalize_task_performance: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("fisher_s", "FisherS ID"),
        ("mle", "MLE ID"),
        ("mom", "MoM ID"),
        ("tle", "TLE ID"),
        ("effective_rank", "Effective Rank"),
    ]

    task_minmax: Dict[str, Tuple[float, float]] = {}
    if normalize_task_performance:
        task_minmax = _task_minmax_for_matryoshka_entries(
            merged,
            exclude_tasks,
            downstream_preproc,
            best_preproc,
        )
        if not task_minmax:
            print("[merge] No downstream performance tasks found for normalization; using raw averages.")
            normalize_task_performance = False

    metric_series: List[tuple[str, str, List[float], List[float], List[str]]] = []
    for metric_key, metric_label in metrics:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry or not _is_matryoshka_entry(base_entry):
                continue
            dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
            if not dims:
                continue
            cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
            if not cache:
                continue

            base_label = (
                (base_entry.get("model_spec") or {}).get("name")
                or base_entry.get("name")
                or Path(entry.get("checkpoint", "model")).stem
            )

            for dim in dims:
                if normalize_task_performance:
                    perf_map = _performance_by_task(
                        cache,
                        dim,
                        exclude_tasks,
                        downstream_preproc,
                        best_preproc,
                    )
                    if not perf_map:
                        continue
                    avg_perf = _normalized_average_performance(perf_map, task_minmax)
                else:
                    avg_perf = _average_performance(
                        cache,
                        dim,
                        exclude_tasks,
                        downstream_preproc,
                        best_preproc,
                    )
                if avg_perf is None:
                    continue
                if metric_key == "effective_rank":
                    metric_value = _id_metric_by_dim(base_entry, dim, "effective_rank")
                else:
                    metric_value = _id_metric_by_dim(base_entry, dim, metric_key)
                if metric_value is None:
                    continue
                if dim == 8:
                    print(
                        f"[merge][debug] Matryoshka dim=8 included for {base_label}: "
                        f"{metric_key}={metric_value:.6f}, avg_perf={avg_perf:.6f}"
                    )
                xs.append(float(metric_value))
                ys.append(float(avg_perf))
                labels.append(str(dim))

        if not xs:
            print(f"[merge] No matryoshka points for {metric_key} scatterplot")
            continue

        metric_series.append((metric_key, metric_label, xs, ys, labels))

    if not metric_series:
        print("[merge] No matryoshka scatterplots generated; no valid points.")
        return

    n_metrics = len(metric_series)
    cols = 3 if n_metrics > 3 else n_metrics
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (metric_key, metric_label, xs, ys, labels) in enumerate(metric_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        x_window = 0.5
        if metric_key in ("mle", "mom", "tle"):
            x_window = 2.0
        elif metric_key == "effective_rank":
            x_window = 100.0
        rotations = _rotation_by_x_proximity(xs, ys, x_threshold=x_window)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=10,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )

        ax.set_xlabel(
            f"S2 {metric_label}" if metric_key != "effective_rank" else "Effective Rank"
        )
        if idx % cols == 0:
            ax.set_ylabel(
                "Average normalized task performance"
                if normalize_task_performance
                else "Average downstream performance"
            )
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        if ys:
            y_min = min(ys)
            y_max = max(ys)
            if math.isclose(y_min, y_max):
                pad = 0.05 if y_min == 0 else abs(y_min) * 0.05
            else:
                pad = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_title(
            f"{metric_label} (S2) vs Avg Normalized Task Performance (Matryoshka)"
            if normalize_task_performance
            else f"{metric_label} (S2) vs Avg Downstream Performance (Matryoshka)"
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    out_path = plots_dir / "s2_id_vs_avg_performance_grid_matryoshka.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote matryoshka scatterplot grid {out_path}")


def _generate_matryoshka_dataset_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    s2_id_metric: str = "fisher_s",
    auto_task_ylim: bool = False,
    best_preproc: bool = False,
    normalize_task_performance: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_names: List[str] = []
    task_set = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry or not _is_matryoshka_entry(base_entry):
            continue
        dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
        if not dims:
            continue
        cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
        if not cache:
            continue
        for dim in dims:
            perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map:
                continue
            for task in perf_map.keys():
                if task not in task_set:
                    task_set.add(task)
                    task_names.append(task)

    if not task_names:
        print("[merge] No downstream performance tasks found for matryoshka dataset scatterplots")
        return

    task_minmax: Dict[str, Tuple[float, float]] = {}
    if normalize_task_performance:
        task_minmax = _task_minmax_for_matryoshka_entries(
            merged,
            exclude_tasks,
            downstream_preproc,
            best_preproc,
        )
        if not task_minmax:
            print("[merge] No downstream performance tasks found for normalization; using raw values.")
            normalize_task_performance = False

    missing_logged: set[str] = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry or not _is_matryoshka_entry(base_entry):
            continue
        dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
        if not dims:
            continue
        cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
        if not cache:
            continue
        available: set[str] = set()
        for dim in dims:
            perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
            if perf_map:
                available.update(perf_map.keys())
        label = _model_label(entry, base_entry)
        _log_missing_tasks(label, task_names, available, missing_logged)

    task_series: List[tuple[str, List[float], List[float], List[str]]] = []
    for task in task_names:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry or not _is_matryoshka_entry(base_entry):
                continue
            dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
            if not dims:
                continue
            cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
            if not cache:
                continue
            base_label = (
                (base_entry.get("model_spec") or {}).get("name")
                or base_entry.get("name")
                or Path(entry.get("checkpoint", "model")).stem
            )
            for dim in dims:
                if s2_id_metric == "effective_rank":
                    metric_value = _id_metric_by_dim(base_entry, dim, "effective_rank")
                else:
                    metric_value = _id_metric_by_dim(base_entry, dim, s2_id_metric)
                if metric_value is None:
                    continue
                perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
                if not perf_map or task not in perf_map:
                    continue
                perf_value = perf_map[task]
                if normalize_task_performance:
                    normalized = _normalized_task_value(task, perf_value, task_minmax)
                    if normalized is None:
                        continue
                    perf_value = normalized
                xs.append(float(metric_value))
                ys.append(float(perf_value))
                labels.append(str(dim))

        if xs:
            task_series.append((task, xs, ys, labels))

    if not task_series:
        print("[merge] No valid points found for matryoshka dataset scatterplots")
        return

    n_tasks = len(task_series)
    cols = 3 if n_tasks <= 9 else 4
    rows = math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (task, xs, ys, labels) in enumerate(task_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        rotations = _rotation_by_x_proximity(xs, ys)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=9,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )
        metric_label = dict(ID_METRIC_SPECS).get(s2_id_metric, s2_id_metric)
        ax.set_xlabel(f"S2 {metric_label}" if s2_id_metric != "effective_rank" else "Effective Rank")
        ax.set_ylabel("Normalized task performance" if normalize_task_performance else "Performance")
        if auto_task_ylim:
            _set_auto_ylim(ax, ys)
        elif normalize_task_performance:
            ax.set_ylim(0.0, 1.0)
        else:
            _set_task_ylim(ax, task)
        ax.set_title(
            f"{task} (Normalized task performance, Matryoshka)"
            if normalize_task_performance
            else f"{task} (Matryoshka)"
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_tasks:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    if s2_id_metric == "fisher_s":
        out_name = "s2_fishers_vs_performance_by_task_matryoshka.png"
    else:
        out_name = f"s2_{s2_id_metric}_vs_performance_by_task_matryoshka.png"
    out_path = plots_dir / out_name
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote matryoshka dataset scatterplot grid {out_path}")


def _generate_matryoshka_downstream_fisher_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
    normalize_task_performance: bool = False,
    auto_task_ylim: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_names: List[str] = []
    task_set = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry or not _is_matryoshka_entry(base_entry):
            continue
        dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
        if not dims:
            continue
        cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
        if not cache:
            continue
        for dim in dims:
            perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map:
                continue
            for task in perf_map.keys():
                if task not in task_set:
                    task_set.add(task)
                    task_names.append(task)

    if not task_names:
        print("[merge] No downstream performance tasks found for matryoshka downstream FisherS scatterplots")
        return

    task_minmax: Dict[str, Tuple[float, float]] = {}
    if normalize_task_performance:
        task_minmax = _task_minmax_for_matryoshka_entries(
            merged,
            exclude_tasks,
            downstream_preproc,
            best_preproc,
        )
        if not task_minmax:
            print("[merge] No downstream performance tasks found for normalization; using raw values.")
            normalize_task_performance = False

    missing_logged: set[str] = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry or not _is_matryoshka_entry(base_entry):
            continue
        dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
        if not dims:
            continue
        cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
        if not cache:
            continue
        available: set[str] = set()
        for dim in dims:
            perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
            if perf_map:
                available.update(perf_map.keys())
        label = _model_label(entry, base_entry)
        _log_missing_tasks(label, task_names, available, missing_logged)

    task_series: List[tuple[str, List[float], List[float], List[str]]] = []
    for task in task_names:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry or not _is_matryoshka_entry(base_entry):
                continue
            dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
            if not dims:
                continue
            cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
            if not cache:
                continue
            for dim in dims:
                perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
                if not perf_map or task not in perf_map:
                    continue
                perf_value = perf_map[task]
                if normalize_task_performance:
                    normalized = _normalized_task_value(task, perf_value, task_minmax)
                    if normalized is None:
                        continue
                    perf_value = normalized
                fisher = _downstream_fisher(cache, dim, task)
                if fisher is None:
                    continue
                xs.append(float(fisher))
                ys.append(float(perf_value))
                labels.append(str(dim))

        if xs:
            task_series.append((task, xs, ys, labels))

    if not task_series:
        print("[merge] No valid points found for matryoshka downstream FisherS scatterplots")
        return

    n_tasks = len(task_series)
    cols = 3 if n_tasks <= 9 else 4
    rows = math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (task, xs, ys, labels) in enumerate(task_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        rotations = _rotation_by_x_proximity(xs, ys)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=9,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )
        ax.set_xlabel("Downstream FisherS ID")
        ax.set_ylabel("Normalized task performance" if normalize_task_performance else "Performance")
        if auto_task_ylim:
            _set_auto_ylim(ax, ys)
        elif normalize_task_performance:
            ax.set_ylim(0.0, 1.0)
        else:
            _set_task_ylim(ax, task)
        ax.set_title(
            f"{task} (Normalized task performance, Matryoshka)"
            if normalize_task_performance
            else f"{task} (Matryoshka)"
        )
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_tasks:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    out_path = plots_dir / "downstream_fishers_vs_performance_by_task_matryoshka.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote matryoshka downstream FisherS scatterplot grid {out_path}")


def _generate_matryoshka_s2_vs_downstream_fishers_scatterplots(
    merged: List[Dict[str, Any]],
    output_dir: Path,
    exclude_tasks: Optional[Iterable[str]] = None,
    downstream_preproc: Optional[Iterable[str]] = None,
    best_preproc: bool = False,
) -> None:
    plots_dir = output_dir / "scatterplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    task_names: List[str] = []
    task_set = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry or not _is_matryoshka_entry(base_entry):
            continue
        dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
        if not dims:
            continue
        cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
        if not cache:
            continue
        for dim in dims:
            perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
            if not perf_map:
                continue
            for task in perf_map.keys():
                if task not in task_set:
                    task_set.add(task)
                    task_names.append(task)

    if not task_names:
        print("[merge] No downstream tasks found for matryoshka S2-vs-downstream FisherS scatterplots")
        return

    missing_logged: set[str] = set()
    for entry in merged:
        base_entry = _select_s2_entry(entry.get("base_entries", []))
        if not base_entry or not _is_matryoshka_entry(base_entry):
            continue
        dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
        if not dims:
            continue
        cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
        if not cache:
            continue
        available: set[str] = set()
        for dim in dims:
            perf_map = _performance_by_task(cache, dim, exclude_tasks, downstream_preproc, best_preproc)
            if perf_map:
                available.update(perf_map.keys())
        label = _model_label(entry, base_entry)
        _log_missing_tasks(label, task_names, available, missing_logged)

    task_series: List[tuple[str, List[float], List[float], List[str]]] = []
    for task in task_names:
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        for entry in merged:
            base_entry = _select_s2_entry(entry.get("base_entries", []))
            if not base_entry or not _is_matryoshka_entry(base_entry):
                continue
            dims = _matryoshka_dims_from_entry(base_entry, include_embedding_dim=True)
            if not dims:
                continue
            cache = _select_cache_for_dims(entry.get("downstream", {}), dims, downstream_preproc)
            if not cache:
                continue
            for dim in dims:
                s2_fisher = _id_metric_by_dim(base_entry, dim, "fisher_s")
                if s2_fisher is None:
                    continue
                downstream_fisher = _downstream_fisher(cache, dim, task)
                if downstream_fisher is None:
                    continue
                xs.append(float(s2_fisher))
                ys.append(float(downstream_fisher))
                labels.append(str(dim))

        if xs:
            task_series.append((task, xs, ys, labels))

    if not task_series:
        print("[merge] No valid points found for matryoshka S2-vs-downstream FisherS scatterplots")
        return

    n_tasks = len(task_series)
    cols = 3 if n_tasks <= 9 else 4
    rows = math.ceil(n_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (task, xs, ys, labels) in enumerate(task_series):
        ax = axes_list[idx]
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(1)

        ax.scatter(xs, ys, alpha=0.85)
        _add_regression_line(ax, xs, ys)
        rotations = _rotation_by_x_proximity(xs, ys)
        for (x, y, label), rotation in zip(zip(xs, ys, labels), rotations):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 0),
                fontsize=9,
                rotation=rotation,
                ha="left",
                va="center",
                rotation_mode="anchor",
                clip_on=False,
            )
        ax.set_xlabel("S2 FisherS ID")
        ax.set_ylabel("Downstream FisherS ID")
        ax.set_title(f"{task} (Matryoshka)")
        ax.grid(True, alpha=0.3, linewidth=0.5)

    for ax in axes_list[n_tasks:]:
        ax.set_visible(False)

    fig.tight_layout(pad=0.4)
    out_path = plots_dir / "s2_fishers_vs_downstream_fishers_by_task_matryoshka.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[merge] Wrote matryoshka S2-vs-downstream FisherS scatterplot grid {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge global ID JSONs with downstream cache JSONs.")
    parser.add_argument("--base-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--downstream-dir", type=Path, default=DOWNSTREAM_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    base_entries = _collect_base_entries(args.base_dir)
    downstream_caches = _collect_downstream_caches(args.downstream_dir)

    merged = _merge_results(base_entries, downstream_caches)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print(f"[merge] Wrote {len(merged)} merged entries to {args.output}")

    missing_downstream = sorted(set(base_entries.keys()) - set(downstream_caches.keys()))
    if missing_downstream:
        print(f"[merge] Missing downstream caches for {len(missing_downstream)} checkpoints")
        for key in missing_downstream:
            print(f"[merge] Missing downstream cache for: {key}")
    print("[merge] Run plot_global_id_downstream.py to generate scatterplots.")


if __name__ == "__main__":
    main()
