"""Tests for the lightweight public package boundary."""

import subprocess
import sys
from pathlib import Path


def test_import_does_not_load_optional_or_model_dependencies() -> None:
    script = """
import sys
import ciip

unexpected = {
    name for name in sys.modules
    if name.split('.', 1)[0] in {'rasterio', 'torch', 'torchgeo', 'torchvision'}
}
if unexpected:
    raise SystemExit(f'import ciip loaded optional dependencies: {sorted(unexpected)}')
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_api_is_discoverable_without_loading_it() -> None:
    script = """
import sys
import ciip

expected = {'S12Dataset', 'Subset', 'available_models', 'generate_splits', 'load'}
if not expected <= set(dir(ciip)):
    raise SystemExit(f'missing public names: {sorted(expected - set(dir(ciip)))}')
if 'torch' in sys.modules or 'rasterio' in sys.modules:
    raise SystemExit('discovering the public API loaded an implementation dependency')
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_dataparallel_training_has_no_second_epoch_loop() -> None:
    path = Path(__file__).resolve().parents[1] / "ciip/open_clip_train/dataparallel/train.py"
    source = path.read_text()
    assert "def train_one_epoch" not in source
    assert "from ciip.open_clip_train.train import train_one_epoch" in source


def test_maintained_evaluation_modules_stay_reviewable() -> None:
    evaluation_dir = Path(__file__).resolve().parents[1] / "ciip/evaluation"
    oversized = {
        path.name: len(path.read_text().splitlines())
        for path in evaluation_dir.glob("*.py")
        if len(path.read_text().splitlines()) > 1000
    }
    assert oversized == {}


def test_first_party_package_paths_are_importable() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "visualizations").exists()
    assert not (root / "intrinsic-dimension").exists()
    assert not (root / "ciip/evaluation/pangaea-results").exists()
