#!/usr/bin/env python3
"""Snapshot model-facing Python signatures before structural cleanup."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    Path("ciip/lorentz.py"),
    Path("ciip/loss.py"),
    Path("ciip/mae_decoder.py"),
    Path("ciip/masking.py"),
    Path("ciip/model.py"),
    Path("ciip/model_ciip.py"),
    Path("ciip/open_clip_train/dataparallel/model_arch.py"),
    Path("clip/model.py"),
)


def fingerprint(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        content = (ROOT / path).read_bytes()
        digest.update(str(path).encode() + b"\0" + content + b"\0")
    return f"sha256:{digest.hexdigest()}"


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return f"{node.name}({ast.unparse(node.args)})"


def checkpoint_contracts(tree: ast.AST) -> list[dict[str, object]]:
    """Capture literal checkpoint namespaces used by loader functions."""
    contracts = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if "load" not in node.name.lower() and "checkpoint" not in node.name.lower():
            continue
        prefixes: set[str] = set()
        container_keys: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                value = child.value
                if (
                    any(isinstance(target, ast.Name) and target.id == "prefixes" for target in targets)
                    and isinstance(value, (ast.List, ast.Tuple))
                ):
                    prefixes.update(
                        item.value
                        for item in value.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    )
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "get"
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                container_keys.add(child.args[0].value)
        if prefixes or container_keys:
            contracts.append(
                {
                    "loader": node.name,
                    "prefixes": sorted(prefixes),
                    "container_keys": sorted(container_keys),
                }
            )
    return contracts


def file_contract(path: Path) -> dict[str, object]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=str(path))
    functions = []
    classes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            functions.append(signature(node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [
                signature(child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (not child.name.startswith("_") or child.name == "__init__")
            ]
            classes.append(
                {
                    "name": node.name,
                    "bases": [ast.unparse(base) for base in node.bases],
                    "methods": methods,
                }
            )
    return {
        "path": str(path),
        "functions": functions,
        "classes": classes,
        "checkpoint_contracts": checkpoint_contracts(tree),
    }


def build_contracts() -> dict[str, object]:
    missing = [str(path) for path in MODEL_FILES if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"model contract inputs are missing: {', '.join(missing)}")
    return {
        "schema_version": 1,
        "source_fingerprint": fingerprint(MODEL_FILES),
        "status": "structural baseline only; runtime validation required",
        "files": [file_contract(path) for path in MODEL_FILES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/phase2-model-contracts.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_contracts(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
