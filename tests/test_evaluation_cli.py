from pathlib import Path

import pytest

from ciip.evaluation.unified_cli import build_parser, config_from_args


def test_unified_cli_builds_explicit_config() -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            "out",
            "--eurosat-root",
            "eurosat",
            "--neuco-root",
            "neuco",
            "--checkpoint",
            "epoch.pt",
            "--disable-neuco",
            "--disable-ssl4eo",
        ]
    )
    config = config_from_args(args)
    assert config.output_dir == Path("out")
    assert config.checkpoint == Path("epoch.pt")
    assert config.enable_neuco is False
    assert config.enable_ssl4eo is False


def test_unified_cli_rejects_missing_model_source() -> None:
    args = build_parser().parse_args(
        ["--output-dir", "out", "--eurosat-root", "eurosat", "--neuco-root", "neuco"]
    )
    with pytest.raises(ValueError, match="--checkpoint"):
        config_from_args(args)
