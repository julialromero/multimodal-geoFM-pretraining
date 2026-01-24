"""Minimal image pair transform utilities used by training scripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
from torchvision import transforms as tv_transforms


@dataclass
class PairGeom:
    """Resize both modalities to a common output size.

    This is a drastically simplified version of the augmentation pipeline used
    in the original training code.  It only performs deterministic resizing so
    that training and evaluation scripts can run without depending on
    repository-relative imports.
    """

    out_size: Tuple[int, int]

    def __call__(self, s1: torch.Tensor, s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        resize = tv_transforms.Resize(self.out_size)
        return resize(s1), resize(s2)


class PairAugmented:
    """Apply optional geometric and photometric augmentations to paired inputs."""

    def __init__(
        self,
        pair_geom: Optional[PairGeom] = None,
        augmentations: Optional[Sequence[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        self.pair_geom = pair_geom
        self.augmentations = list(augmentations or [])

    def __call__(self, s1: torch.Tensor, s2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pair_geom is not None:
            s1, s2 = self.pair_geom(s1, s2)
        for aug in self.augmentations:
            s1 = aug(s1)
            s2 = aug(s2)
        return s1, s2
