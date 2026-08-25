"""Dependency-backed model runtime compatibility checks.

The checks skip when PyTorch is unavailable, so discovery remains usable in
lightweight environments while the full suite runs in the declared ML environment.
"""

import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TORCHGEO = importlib.util.find_spec("torchgeo") is not None


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for model runtime checks")
class ModelRuntimeBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch
        cls.clip_model = load_module("runtime_clip_model", "clip/model.py")
        cls.lorentz = load_module("runtime_lorentz", "ciip/lorentz.py")
        cls.masking = load_module("runtime_masking", "ciip/masking.py")
        package = types.ModuleType("runtime_loss_package")
        package.__path__ = [str(ROOT / "ciip")]
        sys.modules[package.__name__] = package
        load_module("runtime_loss_package.lorentz", "ciip/lorentz.py")
        cls.losses = load_module("runtime_loss_package.loss", "ciip/loss.py")

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

    def test_siglip_loss_forward_and_backward(self) -> None:
        s1 = self.torch.randn(3, 8, requires_grad=True)
        s2 = self.torch.randn(3, 8, requires_grad=True)
        loss = self.losses.SigLipLoss()(s1, s2, self.torch.tensor(2.0), None)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(self.torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(s1.grad)
        self.assertIsNotNone(s2.grad)

    def test_shared_optimizer_applies_text_learning_rate_scale(self) -> None:
        from ciip.open_clip_train.optimizer import create_optimizer

        class TwoTower(self.torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_s1 = self.torch.nn.Linear(4, 4)
                self.encoder_text = self.torch.nn.Linear(4, 4)

        config = SimpleNamespace(lr=1e-3, wd=0.1, beta1=0.9, beta2=0.99, eps=1e-8)
        optimizer = create_optimizer(
            TwoTower(), self.torch.nn.Identity(), config, text_lr_scale=0.1
        )

        learning_rates = {group["lr"] for group in optimizer.param_groups}
        self.assertEqual(learning_rates, {1e-3, 1e-4})

    def test_shared_optimizer_step_updates_parameters(self) -> None:
        from ciip.open_clip_train.optimizer import step_optimizer

        model = self.torch.nn.Linear(2, 1)
        optimizer = self.torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()
        model(self.torch.ones(1, 2)).sum().backward()

        step_optimizer(model, optimizer, grad_clip_norm=1.0)

        self.assertFalse(self.torch.equal(model.weight, before))

    def test_checkpoint_round_trip_restores_training_state(self) -> None:
        from ciip.open_clip_train.checkpointing import (
            build_training_checkpoint,
            restore_training_checkpoint,
            save_checkpoint,
        )

        model = self.torch.nn.Linear(4, 2)
        loss = self.torch.nn.Linear(2, 1)
        optimizer = self.torch.optim.AdamW(
            [*model.parameters(), *loss.parameters()], lr=1e-3
        )
        expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        checkpoint = build_training_checkpoint(
            model, optimizer, loss, epoch=3, name="test"
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, path)
            loaded = self.torch.load(path, map_location="cpu", weights_only=False)
            with self.torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            epoch = restore_training_checkpoint(
                loaded, model, optimizer=optimizer, loss=loss, strict=True
            )

        self.assertEqual(epoch, 3)
        for name, value in model.state_dict().items():
            self.torch.testing.assert_close(value, expected[name])

    def test_training_batch_maps_supported_encoder_pairs(self) -> None:
        from ciip.open_clip_train.batches import prepare_training_batch

        batch = {
            "s1": self.torch.randn(2, 2, 4, 4),
            "s2": self.torch.randn(2, 12, 4, 4),
            "text": self.torch.ones(2, 8, dtype=self.torch.long),
        }
        paired = prepare_training_batch(
            batch, encoder_pair="s1s2", device="cpu", input_dtype=self.torch.float16
        )
        optical_text = prepare_training_batch(
            batch, encoder_pair="s2_text", device="cpu", input_dtype=self.torch.float32
        )
        triple = prepare_training_batch(
            batch, encoder_pair="s1s2_text", device="cpu", input_dtype=self.torch.float32
        )

        self.assertEqual(paired.s1.dtype, self.torch.float16)
        self.assertEqual(tuple(optical_text.s1.shape), tuple(batch["s2"].shape))
        self.assertEqual(optical_text.s2.dtype, self.torch.long)
        self.assertEqual(len(triple.model_inputs), 3)

    def test_training_objective_composes_reconstruction(self) -> None:
        from ciip.open_clip_train.objectives import compose_training_loss

        class PrimaryLoss:
            def __call__(self, feature, output_dict):
                self.assert_output_dict = output_dict
                return {"contrastive_loss": feature.square().mean()}

        primary = PrimaryLoss()
        feature = self.torch.tensor([1.0, 2.0], requires_grad=True)
        reconstruction = self.torch.tensor(3.0, requires_grad=True)
        config = SimpleNamespace(warmup_steps=2, warmup_epochs=0)
        setattr(config, "lambda", 0.5)
        losses, total = compose_training_loss(
            primary,
            {"feature": feature},
            reconstruction=reconstruction,
            reconstruction_config=config,
            epoch=0,
            step=2,
        )

        self.assertTrue(primary.assert_output_dict)
        self.torch.testing.assert_close(losses["recon_loss"], self.torch.tensor(1.5))
        self.torch.testing.assert_close(total, self.torch.tensor(4.0))
        total.backward()
        self.assertIsNotNone(feature.grad)
        self.assertIsNotNone(reconstruction.grad)

    def test_shared_training_step_normalizes_scale_and_backpropagates(self) -> None:
        from ciip.open_clip_train.objectives import run_training_step

        class Model(self.torch.nn.Module):
            def forward(self, value):
                return {
                    "feature": value * 2,
                    "logit_scale": self.torch.tensor([1.0, 3.0]),
                }

        class Loss:
            def __call__(self, feature, logit_scale, output_dict):
                assert output_dict
                return {"contrastive_loss": feature.mean() * logit_scale}

        value = self.torch.tensor([1.0, 2.0], requires_grad=True)
        output, losses, total = run_training_step(
            Model(),
            (value,),
            Loss(),
            autocast=nullcontext,
            reconstruction_config=None,
            epoch=0,
            step=0,
        )

        self.torch.testing.assert_close(output["logit_scale"], self.torch.tensor(2.0))
        self.torch.testing.assert_close(losses["loss"], total)
        self.assertIsNotNone(value.grad)

    def test_accumulated_inputs_replace_only_current_microbatch(self) -> None:
        from ciip.open_clip_train.accumulation import (
            build_accumulated_loss_inputs,
            cache_model_features,
        )

        cached = {}
        first = {
            "s1_features": self.torch.tensor([[1.0, 2.0]]),
            "logit_scale": self.torch.tensor(2.0),
        }
        second = {
            "s1_features": self.torch.tensor([[3.0, 4.0]]),
            "logit_scale": self.torch.tensor(2.0),
        }
        cache_model_features(cached, first)
        cache_model_features(cached, second)
        tracked = self.torch.tensor([[5.0, 6.0]], requires_grad=True)
        inputs = build_accumulated_loss_inputs(
            cached,
            {"s1_features": tracked, "logit_scale": self.torch.tensor([1.0, 3.0])},
            1,
        )

        self.torch.testing.assert_close(
            inputs["s1_features"], self.torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        )
        self.torch.testing.assert_close(inputs["logit_scale"], self.torch.tensor(2.0))
        inputs["s1_features"].sum().backward()
        self.assertIsNotNone(tracked.grad)


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
        package = types.ModuleType("runtime_ciip")
        package.__path__ = [str(ROOT / "ciip")]
        sys.modules[package.__name__] = package
        load_module("runtime_ciip.masking", "ciip/masking.py")
        load_module("runtime_ciip.mae_decoder", "ciip/mae_decoder.py")
        load_module("runtime_ciip.model", "ciip/model.py")
        load_module("runtime_ciip.lorentz", "ciip/lorentz.py")
        cls.model_ciip = load_module("runtime_ciip.model_ciip", "ciip/model_ciip.py")
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

    def test_shared_model_factory_builds_euclidean_model(self) -> None:
        from ciip.open_clip_train.utils import create_model

        model_config = SimpleNamespace(
            embed_dim=8,
            framework="transformer",
            width=64,
            s1_resolution=8,
            s2_resolution=8,
            s1_layers=1,
            s2_layers=1,
            s1_patch_size=4,
            s2_patch_size=4,
            s1_bands=[1, 2],
            s2_bands=list(range(12)),
            patch_masking=False,
        )
        config = SimpleNamespace(
            model=model_config,
            loss=SimpleNamespace(hyperbolic=False),
        )

        model = create_model(config, device="cpu")
        self.assertEqual(type(model).__name__, "CIIP")

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
