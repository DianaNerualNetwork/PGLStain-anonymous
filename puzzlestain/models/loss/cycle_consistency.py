"""Cycle-consistency (L1) loss for the CycleGAN family.

A thin :class:`StainLoss` wrapper around the L1 distance, used for the
cycle-consistency and identity terms of CycleGAN/StegoGAN. The per-term
weights (``lambda_A``/``lambda_B``/``lambda_identity``) stay on the model and
are applied by the strategy, as in the original repo.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


class CycleConsistencyLoss(nn.Module, StainLoss):
    """L1 distance between two images (cycle/identity reconstruction)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the mean absolute difference between ``pred`` and ``target``."""
        return F.l1_loss(pred, target)

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the L1 loss as a :class:`StainLoss` component."""
        if pred is None or target is None:
            raise ValueError(
                "CycleConsistencyLoss requires 'pred' and 'target' kwargs"
            )
        return self.forward(pred, target)

    def name(self) -> str:
        return "cycle_consistency"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("cycle_consistency", CycleConsistencyLoss)

__all__ = ["CycleConsistencyLoss"]
