import ast
import unittest
from pathlib import Path
from unittest import mock

from tools.phase0_inventory import (
    GENERATED_PATHS,
    literal_references,
    python_inventory,
    refs,
    render_runtime_audit,
    resolve_from_import,
    runtime_signals,
)


class ResolveFromImportTests(unittest.TestCase):
    def test_current_upstream_ref_is_stably_normalized(self) -> None:
        row = "\0".join(("origin/work", "abc123", "2026-08-23T00:00:00+00:00", "subject"))
        with mock.patch("tools.phase0_inventory.git", return_value=row):
            remote_refs = refs("refs/remotes", normalized_ref="origin/work")
        current = remote_refs[0]
        self.assertEqual(current["commit"], "CURRENT_CHECKOUT")
        self.assertEqual(current["date"], "CURRENT_CHECKOUT")
        self.assertEqual(current["subject"], "CURRENT_CHECKOUT")

    def test_runtime_audit_is_a_non_removal_worklist(self) -> None:
        report = render_runtime_audit(
            {
                "source_fingerprint": "sha256:test",
                "python": {
                    "files": [
                        {
                            "path": "train.py",
                            "runtime_signals": {
                                "main_guard": True,
                                "argument_parser": False,
                                "dynamic_import_calls": [],
                                "checkpoint_load_calls": ["torch.load"],
                                "checkpoint_save_calls": [],
                            },
                        }
                    ]
                },
            }
        )
        self.assertIn("Executable entry points (1)", report)
        self.assertIn("Checkpoint loading sites (1)", report)
        self.assertIn("does not authorize removal", report)

    def test_runtime_entrypoint_and_checkpoint_signals(self) -> None:
        tree = ast.parse(
            """\
import argparse
import importlib
import torch
if __name__ == "__main__":
    argparse.ArgumentParser()
    importlib.import_module("plugin")
    model.load_state_dict(torch.load("model.pt"))
"""
        )
        signals = runtime_signals(tree)
        self.assertTrue(signals["main_guard"])
        self.assertTrue(signals["argument_parser"])
        self.assertEqual(signals["dynamic_import_calls"], ["importlib.import_module"])
        self.assertEqual(signals["checkpoint_load_calls"], ["model.load_state_dict", "torch.load"])

    def test_generated_inventory_excludes_itself(self) -> None:
        self.assertIn(Path("docs/phase0-inventory.json"), GENERATED_PATHS)

    def test_literal_references_reports_matching_tokens(self) -> None:
        references = literal_references(Path("tools/phase0_inventory.py"), [Path("docs/phase0-cleanup.md")])
        self.assertEqual(references[0]["path"], "docs/phase0-cleanup.md")
        self.assertIn("tools/phase0_inventory.py", references[0]["matched"])

    def resolve(self, path: str, statement: str) -> str:
        node = ast.parse(statement).body[0]
        self.assertIsInstance(node, ast.ImportFrom)
        return resolve_from_import(Path(path), node.module, node.level)

    def test_absolute_import(self) -> None:
        self.assertEqual(self.resolve("ciip/model.py", "from clip.model import CLIP"), "clip.model")

    def test_sibling_import(self) -> None:
        self.assertEqual(self.resolve("ciip/model.py", "from .loss import ClipLoss"), "ciip.loss")

    def test_parent_import(self) -> None:
        self.assertEqual(
            self.resolve("ciip/evaluation/model_utils.py", "from ..model import CIIP"), "ciip.model"
        )

    def test_package_init_import(self) -> None:
        self.assertEqual(self.resolve("ciip/__init__.py", "from .ciip import *"), "ciip.ciip")

    def test_repository_relative_imports_are_internal(self) -> None:
        files, errors = python_inventory(
            [
                Path("ciip/model_ciip.py"),
                Path("ciip/mae_decoder.py"),
                Path("ciip/masking.py"),
                Path("ciip/model.py"),
                Path("ciip/lorentz.py"),
            ],
            [],
        )
        self.assertEqual(errors, [])
        model = next(item for item in files if item["path"] == "ciip/model_ciip.py")
        self.assertTrue(
            {"ciip.mae_decoder", "ciip.masking", "ciip.model", "ciip.lorentz"}.issubset(
                model["internal_imports"]
            )
        )


if __name__ == "__main__":
    unittest.main()
