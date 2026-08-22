import tempfile
import unittest
from pathlib import Path

from tools.phase1_config_contracts import ENTRYPOINTS, audit_entrypoint, build_audit, yaml_key_paths


class ConfigContractTests(unittest.TestCase):
    def test_yaml_key_paths_tracks_nested_mappings(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as config:
            config.write("model:\n  width: 32\ntrain:\n  epochs: 2\n")
            config.flush()
            self.assertEqual(yaml_key_paths(Path(config.name)), {"model", "model.width", "train", "train.epochs"})

    def test_all_hydra_entrypoints_resolve_existing_configs(self) -> None:
        audit = build_audit()
        self.assertEqual([item["entrypoint"] for item in audit["entrypoints"]], list(map(str, ENTRYPOINTS)))
        self.assertTrue(all(item["config_exists"] for item in audit["entrypoints"]))

    def test_distributed_entrypoint_binding(self) -> None:
        contract = audit_entrypoint(Path("ciip/open_clip_train/run_train_val_distributed.py"))
        self.assertEqual(contract["config_path"], "ciip/open_clip_train/configs/prod_default.yaml")
        self.assertIn("model.framework", contract["accessed_keys"])
        self.assertNotIn("model.framework.lower", contract["accessed_keys"])

    def test_runtime_assignment_satisfies_local_device_key(self) -> None:
        contract = audit_entrypoint(Path("ciip/open_clip_train/run_train_val.py"))
        self.assertIn("train.device", contract["runtime_assigned_keys"])
        self.assertNotIn("train.device", contract["statically_missing_keys"])


if __name__ == "__main__":
    unittest.main()
