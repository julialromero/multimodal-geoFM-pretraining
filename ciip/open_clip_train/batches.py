"""Shared training-batch preparation."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class TrainingBatch:
    """Inputs normalized to the generic two- or three-tower model contract."""

    s1: torch.Tensor
    s2: torch.Tensor
    text: Optional[torch.Tensor] = None

    @property
    def model_inputs(self) -> tuple[torch.Tensor, ...]:
        return (self.s1, self.s2, self.text) if self.text is not None else (self.s1, self.s2)


def prepare_training_batch(batch, *, encoder_pair: str, device, input_dtype) -> TrainingBatch:
    """Move a loader batch to the device and map it to model tower inputs."""
    def vision(name: str) -> torch.Tensor:
        return batch[name].to(device=device, dtype=input_dtype, non_blocking=True)

    def text() -> torch.Tensor:
        return batch["text"].to(device=device, non_blocking=True)

    if encoder_pair == "s2_text":
        return TrainingBatch(vision("s2"), text())
    if encoder_pair == "s1s2_text":
        return TrainingBatch(vision("s1"), vision("s2"), text())
    return TrainingBatch(vision("s1"), vision("s2"))
