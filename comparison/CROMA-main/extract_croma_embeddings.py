#!/usr/bin/env python3
"""Utility to extract CROMA embeddings for S2L1C tiles."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from use_croma import PretrainedCROMA
from ciip.evaluation.export_neuco_embeddings import (
    E2SChallengeDataset,
    Normalize,
    TemporalMean,
    collate_fn,
    _build_resizer,
)

# Model configuration is kept inside this script as requested.
_MODEL_KWARGS = dict(
    size="base",
    modality="optical",
    image_resolution=120,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/local/ms-data/SSL4EO-S12-downstream/data/data/SSL4EO-S12-v1.1/train"),
        # "/local/ms-data/SSL4EO-S12-downstream/data"),
        help="Path to the root directory containing the NeuCo challenge data.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=Path("/home/juro4948/ciip/comparison/CROMA-main/CROMA_base.pt"),
        help="Path to the pretrained CROMA weights (*.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/juro4948/ciip/comparison/CROMA-main/embeddings/"),
        help="Directory where the extracted embeddings will be stored.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2048,
        help=(
            "Optional upper bound on the number of samples to process. "
            "Defaults to 2048."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use during extraction.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for the DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g. 'cuda', 'cpu'). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def _default_modalities() -> List[str]:
    # Concise helper to mirror export_neuco_embeddings defaults while fixing modality.
    return ['s2l2a'] 


def _build_dataset(data_root: Path) -> E2SChallengeDataset:
    transform = transforms.Compose([
        Normalize(),
        TemporalMean(),
    ])

    

    dataset = E2SChallengeDataset(
        data_path=str(data_root),
        modalities=_default_modalities(),
        transform=transform,
        seasons=4,
        dataset_name="bands",
        randomize_seasons=False,
        concat=True,
        output_file_name=True,
        shift_s2_channels=True,
    )
    return dataset


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    data = batch["data"].to(device=device, non_blocking=True)
    if data.ndim == 5 and data.size(1) == 1:
        data = data.squeeze(1)
    data = data.float()

    # CROMA expects 12 Sentinel-2 channels (B01-B09, B8A, B11, B12). Drop the
    # cirrus band (B10) when present in the NeuCo challenge data.
    if data.size(1) == 13:
        data = torch.cat([data[:, :10], data[:, 11:]], dim=1)

    if data.size(1) != 12:
        raise RuntimeError(
            f"Expected 12 Sentinel-2 channels after preprocessing, but received {data.size(1)}."
        )

    target_resolution = _MODEL_KWARGS["image_resolution"]
    if data.size(-1) != target_resolution or data.size(-2) != target_resolution:
        data = F.adaptive_avg_pool2d(data, output_size=(target_resolution, target_resolution))

    return data


def _trim_batch(
    data: torch.Tensor,
    names: List[str],
    remaining: Optional[int],
) -> tuple[torch.Tensor, List[str]]:
    if remaining is None or remaining >= data.size(0):
        return data, names
    trimmed_data = data[:remaining]
    trimmed_names = names[:remaining]
    return trimmed_data, list(trimmed_names)


def main() -> None:
    args = parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    dataset = _build_dataset(args.data_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )

    # print dataset length
    print(f"Dataset contains {len(dataset)} samples.")

    model = PretrainedCROMA(pretrained_path=str(args.weights_path), **_MODEL_KWARGS)
    model = model.to(device).eval()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings: Dict[str, torch.Tensor] = {}
    processed = 0

    resize_hw = 120
    # Dim sanity (optional)
    # with torch.no_grad():
    #     dummy = torch.zeros(1, dataset[0]['data'].shape[2], *(resize_hw or dataset[0]['data'].shape[-2:]), device=device)  # (B=1,C,H,W)
        # no forward — just ensure C matches expected after transforms

    resizer = _build_resizer(resize_hw).to(device).eval()

    with torch.no_grad():
        # for batch in loader:
        # only loop through first 10 batches
        for i, batch in enumerate(loader):
            if i > 6: 
                break
            file_names = list(batch["file_name"])
            data = _prepare_batch(batch, device)
            data = resizer(data) 

            remaining = None
            if args.max_samples is not None:
                remaining = max(args.max_samples - processed, 0)
                if remaining == 0:
                    break

            data, file_names = _trim_batch(data, file_names, remaining)
            if data.size(0) == 0:
                break
            
            print(data.shape)
            outputs = model(optical_images=data)
            optical_gap = outputs["optical_GAP"].detach().cpu()

            for name, embedding in zip(file_names, optical_gap):
                embeddings[str(name)] = embedding.clone()

            processed += len(file_names)
            if args.max_samples is not None and processed >= args.max_samples:
                break

    if not embeddings:
        raise RuntimeError("No embeddings were generated. Check the dataset path and parameters.")

    ids = list(embeddings.keys())
    stacked = torch.stack(list(embeddings.values()))
    torch.save({"ids": ids, "embeddings": stacked}, output_dir / "s2l2a_embeddings.pt")


if __name__ == "__main__":
    main()
