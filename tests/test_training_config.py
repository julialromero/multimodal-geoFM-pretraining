"""Contract tests for maintained training option names."""

from pathlib import Path

import pytest

from ciip.open_clip_train.config_validation import validate_training_config
from ciip.open_clip_train.params import parse_args


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "ciip" / "open_clip_train" / "configs"


def valid_config() -> dict:
    return {
        "model": {
            "encoder_pair": "s1s2",
            "framework": "transformer",
            "embed_dim": 512,
            "s1_resolution": 224,
            "s2_resolution": 224,
            "s1_patch_size": 16,
            "s2_patch_size": 16,
            "patch_mask_ratio": 0.5,
        },
        "loss": {"matryoshka_enabled": False, "hyperbolic": False},
        "datamodule": {"batch_size": 4},
        "train": {"accum_freq": 1, "epochs": 2, "val_frac": 0.1},
    }


def test_vc_cli_uses_canonical_destination() -> None:
    assert parse_args(["--vc-regularization"]).vc_reg_enabled is True
    assert parse_args(["--no-vc-regularization"]).vc_reg_enabled is False


def test_tracked_configs_do_not_use_removed_option_names() -> None:
    contents = "\n".join(path.read_text() for path in CONFIGS.rglob("*.yaml"))
    assert "vc_enabled:" not in contents
    assert "apply_orthogonal_mapping:" not in contents


def test_dataparallel_accepts_text_pairings() -> None:
    config = valid_config()
    config["model"]["encoder_pair"] = "s1s2_text"
    validate_training_config(config, runner="dataparallel")


def test_single_runner_rejects_text_pairings() -> None:
    config = valid_config()
    config["model"]["encoder_pair"] = "s2_text"
    with pytest.raises(ValueError, match="not supported by the single runner"):
        validate_training_config(config, runner="single")


def test_unsupported_distillation_fails_before_model_setup() -> None:
    config = valid_config()
    config["distill_model"] = "teacher-model"
    with pytest.raises(ValueError, match="distillation is not supported"):
        validate_training_config(config, runner="distributed")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "patch_mask_ratio"), 1.0, "patch_mask_ratio"),
        (("train", "accum_freq"), 0, "accum_freq"),
        (("train", "val_frac"), -0.1, "val_frac"),
    ],
)
def test_invalid_numeric_settings_fail_early(path, value, message) -> None:
    config = valid_config()
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=message):
        validate_training_config(config, runner="single")


def test_matryoshka_dimensions_must_fit_embedding() -> None:
    config = valid_config()
    config["loss"].update(
        matryoshka_enabled=True,
        matryoshka_dims=[128, 1024],
        matryoshka_weights=[0.5, 0.5],
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_training_config(config, runner="distributed")


def test_matryoshka_and_hyperbolic_are_mutually_exclusive() -> None:
    config = valid_config()
    config["loss"].update(
        matryoshka_enabled=True,
        matryoshka_dims=[128, 512],
        matryoshka_weights=[0.5, 0.5],
        hyperbolic=True,
    )
    with pytest.raises(ValueError, match="cannot both be true"):
        validate_training_config(config, runner="distributed")
