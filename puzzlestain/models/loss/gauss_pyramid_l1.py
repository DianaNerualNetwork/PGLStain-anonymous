"""Gaussian-pyramid L1 loss shared by the CPT family and PPT.

Wraps :class:`~puzzlestain.models.loss.gauss_pyramid.Gauss_Pyramid_Conv`
(ASP/LPTN repository) and computes the mean of the per-level weighted L1
distances between the pyramids of two images. This is the CPT ``loss_GP``
reconstruction term and the PPT frequency loss.

Note: this differs from :class:`~puzzlestain.models.loss.pyramid.PyramidL1Loss`
(``pyramid_l1``), which builds the PyramidPix2pix kornia-based pyramid and
*sums* the per-scale L1 terms.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss
from .gauss_pyramid import Gauss_Pyramid_Conv


class GaussPyramidL1Loss(nn.Module, StainLoss):
    """Mean weighted L1 over a :class:`Gauss_Pyramid_Conv` pyramid.

    Args:
        num_high: Number of high-resolution pyramid levels (the pyramid has
            ``num_high + 1`` entries).
        weights: Per-level weights. ``None`` (default) means uniform weights.
            Can be overridden per call via the ``weights`` kwarg.
    """

    def __init__(self, num_high: int = 5, weights: list[float] | None = None) -> None:
        super().__init__()
        self.pyramid = Gauss_Pyramid_Conv(num_high=num_high)
        if weights is not None and len(weights) != num_high + 1:
            raise ValueError("len(weights) must equal num_high + 1")
        self.weights = weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: list[float] | None = None,
    ) -> torch.Tensor:
        """Return the mean of the per-level weighted L1 distances."""
        p_pred = self.pyramid(pred)
        p_target = self.pyramid(target)
        if weights is None:
            weights = self.weights
        if weights is None:
            weights = [1.0] * len(p_pred)
        loss_pyramid = [
            weight * F.l1_loss(pp, pt)
            for pp, pt, weight in zip(p_pred, p_target, weights)
        ]
        return torch.mean(torch.stack(loss_pyramid))

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the pyramid loss as a :class:`StainLoss` component."""
        if pred is None or target is None:
            raise ValueError("GaussPyramidL1Loss requires 'pred' and 'target' kwargs")
        return self.forward(pred, target, weights)

    def name(self) -> str:
        return "gauss_pyramid_l1"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("gauss_pyramid_l1", GaussPyramidL1Loss)

__all__ = ["GaussPyramidL1Loss"]
