import ast
import unittest

from tools.phase2_model_contracts import MODEL_FILES, build_contracts, checkpoint_contracts, signature


class ModelContractTests(unittest.TestCase):
    def test_checkpoint_prefixes_and_container_keys_are_preserved(self) -> None:
        tree = ast.parse(
            """\
def load_checkpoint(path):
    prefixes = ["module.encoder.", "encoder.", ""]
    state = checkpoint.get("state_dict", checkpoint)
"""
        )
        self.assertEqual(
            checkpoint_contracts(tree),
            [
                {
                    "loader": "load_checkpoint",
                    "prefixes": ["", "encoder.", "module.encoder."],
                    "container_keys": ["state_dict"],
                }
            ],
        )

    def test_signature_preserves_defaults_and_annotations(self) -> None:
        node = ast.parse("def forward(self, image: Tensor, normalize: bool = True): pass").body[0]
        self.assertEqual(signature(node), "forward(self, image: Tensor, normalize: bool=True)")

    def test_repository_contract_covers_every_declared_model_file(self) -> None:
        contracts = build_contracts()
        self.assertEqual([item["path"] for item in contracts["files"]], list(map(str, MODEL_FILES)))
        self.assertTrue(contracts["source_fingerprint"].startswith("sha256:"))

    def test_core_forward_contracts_are_present(self) -> None:
        contracts = build_contracts()
        classes = {
            item["name"]: item
            for file_contract in contracts["files"]
            for item in file_contract["classes"]
        }
        self.assertTrue(any(method.startswith("forward(") for method in classes["CIIP"]["methods"]))
        self.assertTrue(any(method.startswith("forward(") for method in classes["CLIP"]["methods"]))

    def test_ciip_checkpoint_namespaces_are_baselined(self) -> None:
        contracts = build_contracts()
        model_arch = next(
            item
            for item in contracts["files"]
            if item["path"] == "ciip/open_clip_train/dataparallel/model_arch.py"
        )
        prefixes = {
            prefix
            for contract in model_arch["checkpoint_contracts"]
            for prefix in contract["prefixes"]
        }
        self.assertIn("module.encoder_s2.", prefixes)
        self.assertIn("module.model.encoder_text.", prefixes)


if __name__ == "__main__":
    unittest.main()
