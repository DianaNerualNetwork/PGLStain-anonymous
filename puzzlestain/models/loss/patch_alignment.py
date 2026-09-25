"""Patch alignment loss from the PPT repository (MICCAI 2024).

Original source: PPT, "High-resolution Medical Image Translation via Patch
Alignment-based Bidirectional Contrastive Learning"
(``models/patch_alignment_loss.py``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


class PatchAlignmentLoss(nn.Module, StainLoss):
    """Patch alignment loss from the PPT repository.

    Unfolds both images into ``height x width`` patches (stride ``height``,
    padding ``height // 2``), sums the absolute per-channel differences and
    returns the mean scaled by ``beta``.

    Args:
        height: Patch height (also used as the vertical stride).
        width: Patch width.
        beta: Global scale applied to the mean distance.
    """

    def __init__(self, height: int = 4, width: int = 4, beta: float = 0.0025) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.beta = beta

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute the patch alignment loss between images ``A`` and ``B``."""
        assert A.size() == B.size(), "A and B must have the same size"
        A_patches = F.unfold(
            A,
            kernel_size=(self.height, self.width),
            padding=(self.height // 2, self.width // 2),
            stride=self.height,
        )
        B_patches = F.unfold(
            B,
            kernel_size=(self.height, self.width),
            padding=(self.height // 2, self.width // 2),
            stride=self.height,
        )
        distances = torch.sum(torch.abs(B_patches - A_patches), dim=1)
        return torch.mean(distances) * self.beta

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the patch alignment loss as a :class:`StainLoss` component."""
        if pred is None or target is None:
            raise ValueError("PatchAlignmentLoss requires 'pred' and 'target' kwargs")
        return self.forward(pred, target)

    def name(self) -> str:
        return "patch_alignment"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("patch_alignment", PatchAlignmentLoss)

__all__ = ["PatchAlignmentLoss"]
