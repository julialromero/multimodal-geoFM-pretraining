"""Run downstream evaluation scripts with shared model/config handling."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


TASK_SCRIPTS = {
    "eurosat_fewshot_1nn": "eurosat_fewshot_1nn.py",
    "eurosat_fewshot_1nn_multi": "eurosat_fewshot_1nn_multi.py",
    "neuco_fewshot_benchmark": "neuco_fewshot_benchmark.py",
    "neuco_fewshot_benchmark_multi": "neuco_fewshot_benchmark_multi.py",
    "neuco_fewshot_episodic": "neuco_fewshot_episodic.py",
    "unified_evaluation": "unified_evaluation.py",
    "linearprobe_comparison": "linearprobe_comparison.py",
}


def _load_model_specs(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        models = data.get("models")
        defaults = data.get("defaults", {})
        if not isinstance(models, list):
            raise ValueError("Expected 'models' to be a list in the JSON file.")
        return models, defaults if isinstance(defaults, dict) else {}
    if isinstance(data, list):
        return data, {}
    raise ValueError("Models JSON must be a list or an object with a 'models' list.")


def _load_config(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object.")
    return data


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    merged.update({k: v for k, v in override.items() if v is not None})
    return merged


def _to_flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _iter_cli_args(params: Dict[str, Any]) -> Iterable[str]:
    for key, value in params.items():
        if value is None:
            continue
        flag = _to_flag(key)
        if isinstance(value, bool):
            if value:
                yield flag
            continue
        if isinstance(value, (list, tuple)):
            yield flag
            for item in value:
                yield str(item)
            continue
        yield flag
        yield str(value)


def _run_command(cmd: List[str], *, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run downstream evaluation scripts.")
    parser.add_argument("--task", choices=sorted(TASK_SCRIPTS), help="Downstream task to run.")
    parser.add_argument("--config", type=Path, help="JSON config with task/defaults/models.")
    parser.add_argument("--models-json", type=Path, help="JSON file listing models to evaluate.")
    parser.add_argument("--defaults", type=Path, help="Optional JSON file with default args.")
    parser.add_argument("--script-args", type=Path, help="Optional JSON file with shared script args.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    if args.config:
        config = _load_config(args.config)
        task = config.get("task")
        if not task:
            raise ValueError("Config must include 'task'.")
        defaults = config.get("defaults", {})
        script_args = config.get("script_args", {})
        models = config.get("models", [])
    else:
        if not args.task:
            raise ValueError("--task is required when --config is not provided.")
        task = args.task
        defaults = {}
        script_args = {}
        models = []

        if args.defaults:
            defaults = _load_config(args.defaults)
        if args.script_args:
            script_args = _load_config(args.script_args)
        if args.models_json:
            models, model_defaults = _load_model_specs(args.models_json)
            defaults = _merge_dicts(defaults, model_defaults)

    script = TASK_SCRIPTS.get(task)
    if not script:
        raise ValueError(f"Unknown task '{task}'.")
    script_path = Path(__file__).resolve().parent / script

    if not models:
        params = _merge_dicts(defaults, script_args)
        cmd = [args.python, str(script_path), *_iter_cli_args(params)]
        _run_command(cmd, dry_run=args.dry_run)
        return

    for idx, model_spec in enumerate(models, start=1):
        params = _merge_dicts(defaults, script_args)
        params = _merge_dicts(params, model_spec)
        print(f"Running model {idx}/{len(models)}")
        cmd = [args.python, str(script_path), *_iter_cli_args(params)]
        _run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
