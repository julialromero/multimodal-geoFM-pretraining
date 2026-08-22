"""Dependency-backed Phase 2 model behavior baselines.

These tests skip when PyTorch is unavailable, allowing the static audit to run
in lightweight environments while remaining executable in the declared ML
environment.
"""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for Phase 2 runtime baselines")
class ModelRuntimeBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        cls.clip_model = load_module("phase2_clip_model", "clip/model.py")
        cls.lorentz = load_module("phase2_lorentz", "ciip/lorentz.py")
        cls.masking = load_module("phase2_masking", "ciip/masking.py")

    def build_tiny_clip(self):
        return self.clip_model.CLIP(
            embed_dim=8,
            image_resolution=8,
            vision_layers=1,
            vision_width=64,
            vision_patch_size=4,
            context_length=4,
            vocab_size=16,
            transformer_width=64,
            transformer_heads=1,
            transformer_layers=1,
        ).eval()

    def test_clip_forward_shapes_and_transpose_contract(self) -> None:
        torch = self.torch
        torch.manual_seed(0)
        model = self.build_tiny_clip()
        image = torch.randn(2, 3, 8, 8)
        text = torch.tensor([[1, 2, 3, 15], [4, 5, 6, 15]])
        with torch.no_grad():
            logits_image, logits_text = model(image, text)
        self.assertEqual(tuple(logits_image.shape), (2, 2))
        torch.testing.assert_close(logits_text, logits_image.T)

    def test_clip_checkpoint_round_trip(self) -> None:
        torch = self.torch
        torch.manual_seed(1)
        source = self.build_tiny_clip()
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save({"state_dict": source.state_dict()}, checkpoint.name)
            restored = self.build_tiny_clip()
            restored.load_state_dict(torch.load(checkpoint.name, map_location="cpu")["state_dict"])
        for expected, actual in zip(source.state_dict().values(), restored.state_dict().values()):
            torch.testing.assert_close(actual, expected)

    def test_clip_build_model_accepts_existing_state_dict(self) -> None:
        source = self.build_tiny_clip()
        restored = self.clip_model.build_model(dict(source.state_dict())).eval()
        self.assertEqual(set(restored.state_dict()), set(source.state_dict()))
        for expected, actual in zip(source.state_dict().values(), restored.state_dict().values()):
            self.torch.testing.assert_close(actual, expected)

    def test_masking_shapes_and_restore_indices(self) -> None:
        torch = self.torch
        torch.manual_seed(2)
        tokens = torch.randn(2, 16, 8)
        visible, mask, ids_restore = self.masking.random_masking(tokens, mask_ratio=0.5)
        self.assertEqual(tuple(visible.shape), (2, 8, 8))
        self.assertEqual(tuple(mask.shape), (2, 16))
        self.assertEqual(tuple(ids_restore.shape), (2, 16))
        torch.testing.assert_close(mask.sum(dim=1), torch.tensor([8.0, 8.0]))

    def test_lorentz_exp_log_round_trip(self) -> None:
        tangent = self.torch.tensor([[0.1, -0.2, 0.05]], dtype=self.torch.float64)
        embedded = self.lorentz.exp_map0(tangent, curv=0.7)
        restored = self.lorentz.log_map0(embedded, curv=0.7)
        self.torch.testing.assert_close(restored, tangent, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
