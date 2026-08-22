import unittest

from tools.phase1_external_audit import build_audit


class ExternalAuditTests(unittest.TestCase):
    def test_tracked_notebooks_and_launchers_are_inventoried(self) -> None:
        audit = build_audit()
        self.assertIn("tracked_notebooks", audit)
        self.assertIn("tracked_launchers", audit)
        self.assertIn("remote_branch_launchers", audit)

    def test_hydra_fragments_are_not_classified_as_unused(self) -> None:
        audit = build_audit()
        self.assertIn(
            "ciip/open_clip_train/configs/train/high_learning_rate.yaml",
            audit["unbound_config_fragments"],
        )
        self.assertTrue(any("Hydra overrides" in item for item in audit["limitations"]))


if __name__ == "__main__":
    unittest.main()
