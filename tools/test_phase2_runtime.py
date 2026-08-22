"""Dependency-backed Phase 2 model behavior baselines.

These tests skip when PyTorch is unavailable, allowing the static audit to run
in lightweight environments while remaining executable in the declared ML
environment.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHGEO = importlib.util.find_spec("torchgeo") is not None


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
            # build_model intentionally applies CLIP's mixed-precision conversion.
            self.torch.testing.assert_close(actual, expected.to(dtype=actual.dtype))
        self.assertEqual(restored.visual.conv1.weight.dtype, self.torch.float16)

    def test_masking_shapes_and_restore_indices(self) -> None:
        torch = self.torch
        torch.manual_seed(2)
        tokens = torch.randn(2, 16, 8)
        visible, mask, ids_restore, ids_keep = self.masking.random_masking(tokens, mask_ratio=0.5)
        self.assertEqual(tuple(visible.shape), (2, 8, 8))
        self.assertEqual(tuple(mask.shape), (2, 16))
        self.assertEqual(tuple(ids_restore.shape), (2, 16))
        self.assertEqual(tuple(ids_keep.shape), (2, 8))
        torch.testing.assert_close(mask.sum(dim=1), torch.tensor([8.0, 8.0]))
        expected_visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, 8))
        torch.testing.assert_close(visible, expected_visible)

    def test_lorentz_exp_log_round_trip(self) -> None:
        tangent = self.torch.tensor([[0.1, -0.2, 0.05]], dtype=self.torch.float64)
        embedded = self.lorentz.exp_map0(tangent, curv=0.7)
        restored = self.lorentz.log_map0(embedded, curv=0.7)
        self.torch.testing.assert_close(restored, tangent, rtol=1e-6, atol=1e-8)


@unittest.skipUnless(
    HAS_TORCH and HAS_TORCHGEO,
    "PyTorch and TorchGeo are required for CIIP runtime baselines",
)
class CIIPRuntimeBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        # Load the model modules without executing ciip/__init__.py, whose dataset
        # imports are unrelated to model compatibility and require data tooling.
        package = types.ModuleType("phase2_ciip")
        package.__path__ = [str(ROOT / "ciip")]
        sys.modules[package.__name__] = package
        load_module("phase2_ciip.masking", "ciip/masking.py")
        load_module("phase2_ciip.mae_decoder", "ciip/mae_decoder.py")
        load_module("phase2_ciip.model", "ciip/model.py")
        load_module("phase2_ciip.lorentz", "ciip/lorentz.py")
        cls.model_ciip = load_module("phase2_ciip.model_ciip", "ciip/model_ciip.py")
        cls.torch = torch

    def build_representative_ciip(self):
        """Build a small model with the production modality/framework contract."""
        return self.model_ciip.CIIP(
            embed_dim=8,
            s1_resolution=8,
            s1_layers=1,
            s1_width=64,
            s1_patch_size=4,
            s1_bands=2,
            s2_resolution=8,
            s2_layers=1,
            s2_width=64,
            s2_patch_size=4,
            s2_bands=12,
            framework="transformer",
            pretrain=False,
        ).eval()

    def test_ciip_transformer_forward_contract(self) -> None:
        torch = self.torch
        torch.manual_seed(3)
        model = self.build_representative_ciip()
        with torch.no_grad():
            output = model(torch.randn(2, 2, 8, 8), torch.randn(2, 12, 8, 8))
        self.assertEqual(set(output), {"s1_features", "s2_features", "logit_scale"})
        self.assertEqual(tuple(output["s1_features"].shape), (2, 8))
        self.assertEqual(tuple(output["s2_features"].shape), (2, 8))
        torch.testing.assert_close(output["s1_features"].norm(dim=-1), torch.ones(2))
        torch.testing.assert_close(output["s2_features"].norm(dim=-1), torch.ones(2))

    def test_ciip_checkpoint_round_trip(self) -> None:
        torch = self.torch
        torch.manual_seed(4)
        source = self.build_representative_ciip()
        restored = self.build_representative_ciip()
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save({"state_dict": source.state_dict()}, checkpoint.name)
            state_dict = torch.load(checkpoint.name, map_location="cpu")["state_dict"]
            restored.load_state_dict(state_dict, strict=True)
        self.assertEqual(set(restored.state_dict()), set(source.state_dict()))
        for expected, actual in zip(source.state_dict().values(), restored.state_dict().values()):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
