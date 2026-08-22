#!/usr/bin/env python3
"""Audit Hydra entry-point bindings and statically accessed config keys."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    Path("ciip/open_clip_train/run_train_val.py"),
    Path("ciip/open_clip_train/run_train_val_distributed.py"),
    Path("ciip/open_clip_train/dataparallel/run_train_val_dataparallel.py"),
)


def constants(tree: ast.Module) -> dict[str, object]:
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def dotted_attribute(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def yaml_key_paths(path: Path) -> set[str]:
    """Read mapping key paths from the repository's indentation-based YAML configs."""
    stack: list[tuple[int, str]] = []
    paths = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        content = raw_line.split("#", 1)[0].rstrip()
        if not content.strip() or ":" not in content:
            continue
        indent = len(content) - len(content.lstrip())
        key = content.lstrip().split(":", 1)[0].strip()
        if not key or key.startswith("-"):
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        full = ".".join([item[1] for item in stack] + [key])
        paths.add(full)
        stack.append((indent, key))
    return paths


def audit_entrypoint(path: Path) -> dict[str, object]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=str(path))
    names = constants(tree)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    hydra_call = next(
        decorator
        for decorator in main.decorator_list
        if isinstance(decorator, ast.Call) and dotted_attribute(decorator.func) == "hydra.main"
    )
    kwargs = {item.arg: item.value for item in hydra_call.keywords}
    config_path = ast.literal_eval(kwargs["config_path"])
    config_name_node = kwargs["config_name"]
    config_name = (
        names[config_name_node.id] if isinstance(config_name_node, ast.Name) else ast.literal_eval(config_name_node)
    )
    config_file = ((ROOT / path).parent / config_path / f"{config_name}.yaml").resolve()
    available = yaml_key_paths(config_file)
    raw_accessed = {
        dotted.removeprefix("args.")
        for node in ast.walk(main)
        if isinstance(node, ast.Attribute)
        and (dotted := dotted_attribute(node))
        and dotted.startswith("args.")
    }

    def config_key(key: str) -> str:
        if key in available:
            return key
        leaf_matches = [
            candidate
            for candidate in available
            if key.startswith(f"{candidate}.")
            and not any(other.startswith(f"{candidate}.") for other in available)
        ]
        return max(leaf_matches, key=len) if leaf_matches else key

    accessed = sorted({config_key(key) for key in raw_accessed})
    assigned = sorted(
        {
            config_key(dotted.removeprefix("args."))
            for node in ast.walk(main)
            if isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and (dotted := dotted_attribute(node))
            and dotted.startswith("args.")
        }
    )
    missing = sorted(key for key in accessed if key not in available and key not in assigned)
    return {
        "entrypoint": str(path),
        "config_path": str(config_file.relative_to(ROOT)),
        "config_exists": config_file.is_file(),
        "accessed_keys": accessed,
        "runtime_assigned_keys": assigned,
        "statically_missing_keys": missing,
    }


def build_audit() -> dict[str, object]:
    digest = hashlib.sha256()
    for path in ENTRYPOINTS:
        digest.update(str(path).encode() + b"\0" + (ROOT / path).read_bytes() + b"\0")
    audits = [audit_entrypoint(path) for path in ENTRYPOINTS]
    config_paths = sorted({item["config_path"] for item in audits})
    for path in config_paths:
        digest.update(path.encode() + b"\0" + (ROOT / path).read_bytes() + b"\0")
    return {
        "schema_version": 1,
        "source_fingerprint": f"sha256:{digest.hexdigest()}",
        "status": "static Hydra binding audit; runtime composition review required",
        "entrypoints": audits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/phase1-config-contracts.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_audit(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
