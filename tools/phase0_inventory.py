#!/usr/bin/env python3
"""Create a deterministic repository inventory for cleanup audits.

This tool is intentionally read-only apart from its explicit output file.  It
records Git refs, tracked-file sizes, and Python import relationships so that
cleanup candidates can be reviewed before any file is removed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATED_PATHS = {
    Path("docs/phase0-inventory.json"),
    Path("docs/phase1-runtime-audit.md"),
    Path("docs/phase1-config-contracts.json"),
    Path("docs/phase1-external-audit.json"),
    Path("docs/phase2-model-contracts.json"),
}
REFERENCE_SUFFIXES = {".ini", ".json", ".md", ".sh", ".slurm", ".txt", ".yaml", ".yml"}


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout


def source_fingerprint(paths: list[Path]) -> str:
    """Hash audited paths and contents without including generated artifacts."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        encoded_path = str(path).encode()
        content = (ROOT / path).read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def refs(prefix: str, normalized_ref: str | None = None) -> list[dict[str, str]]:
    """Return refs, normalizing the current branch's moving remote ref."""
    fields = "%(refname:short)%00%(objectname)%00%(creatordate:iso-strict)%00%(subject)"
    rows = []
    for line in git("for-each-ref", f"--format={fields}", prefix).splitlines():
        name, commit, date, subject = line.split("\0", 3)
        if name == normalized_ref:
            commit = date = subject = "CURRENT_CHECKOUT"
        rows.append({"name": name, "commit": commit, "date": date, "subject": subject})
    return rows


def upstream_ref() -> str | None:
    """Return the current branch's short upstream ref, if one is configured."""
    result = subprocess.run(
        ("git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def module_name(path: Path) -> str:
    parts = list(path.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_from_import(path: Path, module: str | None, level: int) -> str:
    """Resolve an ImportFrom node to the repository-style dotted module name."""
    if level == 0:
        return module or ""
    current = module_name(path).split(".")
    if path.name != "__init__.py":
        current.pop()
    keep = max(0, len(current) - level + 1)
    return ".".join([*current[:keep], *([] if module is None else module.split("."))])


def literal_references(target: Path, sources: list[Path]) -> list[dict[str, object]]:
    """Find explicit path/module mentions in non-Python text files."""
    module = module_name(target)
    tokens = {str(target), str(target.with_suffix("")), module}
    matches = []
    for source in sources:
        try:
            content = (ROOT / source).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        found = sorted(token for token in tokens if token and token in content)
        if found:
            matches.append({"path": str(source), "matched": found})
    return matches


def dotted_call_name(call: ast.Call) -> str:
    parts = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def runtime_signals(tree: ast.AST) -> dict[str, object]:
    """Collect entry-point and checkpoint signals that require runtime review."""
    calls = {dotted_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and any(isinstance(value, ast.Constant) and value.value == "__main__" for value in node.test.comparators)
        for node in ast.walk(tree)
    )
    return {
        "main_guard": has_main_guard,
        "argument_parser": "argparse.ArgumentParser" in calls,
        "dynamic_import_calls": sorted(calls & {"__import__", "importlib.import_module"}),
        "checkpoint_load_calls": sorted(
            name for name in calls if name in {"torch.load", "load_state_dict"} or name.endswith(".load_state_dict")
        ),
        "checkpoint_save_calls": sorted(
            name for name in calls if name in {"torch.save", "state_dict"} or name.endswith(".state_dict")
        ),
    }


def python_inventory(
    paths: list[Path], reference_sources: list[Path]
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    known = {module_name(path) for path in paths}
    known_roots = {name.split(".", 1)[0] for name in known}
    imports_by_path: dict[Path, set[str]] = {}
    signals_by_path: dict[Path, dict[str, object]] = {}
    errors: list[dict[str, str]] = []
    for path in paths:
        imports: set[str] = set()
        try:
            tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = resolve_from_import(path, node.module, node.level)
                if base:
                    imports.add(base)
                if node.module is None:
                    imports.update(f"{base}.{alias.name}".lstrip(".") for alias in node.names)
        imports_by_path[path] = imports
        signals_by_path[path] = runtime_signals(tree)

    inbound: dict[str, set[str]] = defaultdict(set)
    for source, imports in imports_by_path.items():
        for imported in imports:
            matches = [name for name in known if imported == name or imported.startswith(f"{name}.")]
            if matches:
                inbound[max(matches, key=len)].add(str(source))

    inventory: list[dict[str, object]] = []
    for path, imports in imports_by_path.items():
        name = module_name(path)
        inventory.append(
            {
                "path": str(path),
                "module": name,
                "internal_imports": sorted(i for i in imports if i.split(".", 1)[0] in known_roots),
                "external_imports": sorted(i for i in imports if i.split(".", 1)[0] not in known_roots),
                "imported_by": sorted(inbound[name]),
                "literal_references": literal_references(path, reference_sources),
                "runtime_signals": signals_by_path[path],
            }
        )
    return inventory, errors


def build_inventory() -> dict[str, object]:
    all_tracked = [Path(line) for line in git("ls-files").splitlines()]
    tracked = [path for path in all_tracked if path not in GENERATED_PATHS]
    python_paths = [path for path in tracked if path.suffix == ".py"]
    reference_sources = [path for path in tracked if path.suffix.lower() in REFERENCE_SUFFIXES]
    python_files, parse_errors = python_inventory(python_paths, reference_sources)
    extensions = Counter(path.suffix or "[no extension]" for path in tracked)
    return {
        "schema_version": 3,
        "source_fingerprint": source_fingerprint(tracked),
        "remote_branches": refs("refs/remotes", normalized_ref=upstream_ref()),
        "tags": refs("refs/tags"),
        "tracked_files": {
            "count": len(tracked),
            "bytes": sum((ROOT / path).stat().st_size for path in tracked),
            "by_extension": dict(sorted(extensions.items())),
            "excluded_generated_paths": sorted(map(str, GENERATED_PATHS)),
        },
        "python": {
            "file_count": len(python_paths),
            "parse_errors": parse_errors,
            "files": python_files,
        },
        "audit_scope": {
            "covered": [
                "tracked files and sizes",
                "fetched remote branches and tags",
                "absolute and relative static Python imports",
                "static inbound Python import relationships",
                "literal module and path references in tracked text/configuration files",
                "Python main guards, argument parsers, dynamic imports, and checkpoint calls",
                "Python syntax parsing",
            ],
            "requires_manual_review": [
                "dynamic imports and runtime path construction",
                "Hydra targets and configuration composition",
                "shell, Slurm, notebook, and documentation entry points",
                "checkpoint and state-dict compatibility",
                "runtime behavior for each model family",
            ],
        },
    }


def render_runtime_audit(inventory: dict[str, object]) -> str:
    """Render the reviewed Phase 1 signals as a compact Markdown worklist."""
    files = inventory["python"]["files"]
    sections = [
        ("Executable entry points", "main_guard"),
        ("Argument parser construction", "argument_parser"),
        ("Dynamic import sites", "dynamic_import_calls"),
        ("Checkpoint loading sites", "checkpoint_load_calls"),
        ("Checkpoint/state-dict production sites", "checkpoint_save_calls"),
    ]
    lines = [
        "# Phase 1 runtime and checkpoint audit",
        "",
        "This generated worklist summarizes static audit signals; it does not mark any file",
        "as unused and does not authorize removal. Regenerate it with",
        "`python tools/phase0_inventory.py` after fetching remote refs.",
        "",
        f"Source fingerprint: `{inventory['source_fingerprint']}`",
        "",
    ]
    for heading, key in sections:
        matches = []
        for item in files:
            value = item["runtime_signals"][key]
            if value:
                detail = ", ".join(value) if isinstance(value, list) else "yes"
                matches.append((item["path"], detail))
        lines.extend([f"## {heading} ({len(matches)})", "", "| File | Signal |", "| --- | --- |"])
        lines.extend(f"| `{path}` | {detail} |" for path, detail in matches)
        lines.append("")
    lines.extend(
        [
            "## Required follow-up",
            "",
            "Before reorganizing any listed file, validate its CLI/config callers and add the",
            "applicable model-construction, forward-pass, and checkpoint round-trip baseline.",
            "Dynamic paths, Hydra composition, notebooks, and external HPC launchers still",
            "require manual review.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/phase0-inventory.json")
    parser.add_argument(
        "--summary-output", type=Path, default=ROOT / "docs/phase1-runtime-audit.md"
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    summary_output = (
        args.summary_output if args.summary_output.is_absolute() else ROOT / args.summary_output
    )
    inventory = build_inventory()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(render_runtime_audit(inventory), encoding="utf-8")
    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(display_path)
    try:
        display_summary = summary_output.relative_to(ROOT)
    except ValueError:
        display_summary = summary_output
    print(display_summary)


if __name__ == "__main__":
    main()
