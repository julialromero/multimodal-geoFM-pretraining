#!/usr/bin/env python3
"""Audit tracked notebooks, launchers, config fragments, and remote launchers."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_SUFFIXES = {".sh", ".slurm", ".sbatch"}
COMMAND_PATTERN = re.compile(r"^\s*(?:!|%run\s+|python(?:3)?\s+(?:-m\s+)?).+$")


def git(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, text=True, capture_output=True).stdout


def notebook_signals(path: Path) -> dict[str, object]:
    notebook = json.loads((ROOT / path).read_text(encoding="utf-8"))
    commands = set()
    imports = set()
    python_paths = set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        command_lines = [line.strip() for line in source.splitlines() if COMMAND_PATTERN.match(line)]
        commands.update(command_lines)
        for line in command_lines:
            python_paths.update(re.findall(r"[A-Za-z0-9_./-]+\.py\b", line))
        python_source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
        try:
            tree = ast.parse(python_source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
    return {
        "path": str(path),
        "commands": sorted(commands),
        "imports": sorted(imports),
        "python_paths": sorted(python_paths),
    }


def remote_launchers() -> list[dict[str, object]]:
    paths_to_refs: dict[str, list[str]] = defaultdict(list)
    refs = git("for-each-ref", "--format=%(refname:short)", "refs/remotes").splitlines()
    for ref in refs:
        for path in git("ls-tree", "-r", "--name-only", ref).splitlines():
            if Path(path).suffix.lower() in LAUNCHER_SUFFIXES:
                paths_to_refs[path].append(ref)
    return [{"path": path, "refs": sorted(refs)} for path, refs in sorted(paths_to_refs.items())]


def build_audit() -> dict[str, object]:
    tracked = [Path(path) for path in git("ls-files").splitlines()]
    notebooks = [notebook_signals(path) for path in tracked if path.suffix == ".ipynb"]
    launchers = sorted(str(path) for path in tracked if path.suffix.lower() in LAUNCHER_SUFFIXES)
    configs = sorted(str(path) for path in tracked if path.suffix.lower() in {".yaml", ".yml"})
    bound_configs = {
        "ciip/open_clip_train/configs/prod_default.yaml",
        "ciip/open_clip_train/configs/local_default.yaml",
    }
    return {
        "schema_version": 1,
        "status": "tracked callers audited; external untracked launchers require operator confirmation",
        "tracked_notebooks": notebooks,
        "tracked_launchers": launchers,
        "remote_branch_launchers": remote_launchers(),
        "config_files": configs,
        "unbound_config_fragments": sorted(set(configs) - bound_configs - {"environment.yml", "temp.yml"}),
        "limitations": [
            "untracked launchers on external HPC systems are not visible to Git",
            "notebook commands assembled dynamically are not discoverable statically",
            "unbound config fragments may still be selected by command-line Hydra overrides",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/phase1-external-audit.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_audit(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
