"""MS-SSIM loss wrapper for virtual staining.

This module exposes ``pytorch_msssim.MS_SSIM`` as a :class:`StainLoss` so it
can be composed with other losses in :class:`CompositeLoss`.  It expects images
in the ``[-1, 1]`` range used by the CUT family and internally converts them to
``[0, 1]`` before evaluation.

Original source reference: https://github.com/ccitachi/PSPStain
(MS-SSIM is used as one of the generator losses in PSPStain.)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from pytorch_msssim import MS_SSIM
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise ImportError(
        "MS-SSIM loss requires pytorch_msssim. "
        "Install it with: pip install pytorch-msssim"
    ) from exc

from .base import LossRegistry, StainLoss


class MS_SSIM_Loss(nn.Module, StainLoss):
    """Multi-Scale Structural Similarity loss.

    Args:
        data_range: Pixel value range of the input images.  The default
            ``2.0`` corresponds to the ``[-1, 1]`` range used by PuzzleStain;
            the module rescales internally to ``[0, 1]`` before calling
            ``pytorch_msssim``.
        size_average, win_size, win_sigma, weights, K: Forwarded to
            ``pytorch_msssim.MS_SSIM``.
    """

    def __init__(
        self,
        data_range: float = 2.0,
        size_average: bool = True,
        win_size: int = 11,
        win_sigma: float = 1.5,
        weights: list[float] | None = None,
        K: tuple[float, float] = (0.01, 0.03),
    ) -> None:
        super().__init__()
        self.data_range = data_range
        self.ms_ssim = MS_SSIM(
            data_range=1.0,
            size_average=size_average,
            win_size=win_size,
            win_sigma=win_sigma,
            channel=3,
            weights=weights,
            K=K,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``5 * (1 - MS_SSIM(pred, target))``.

        Mirrors the original PSPStain loss (``util/losses.py``): the raw
        MS-SSIM gap is scaled by 5 and not clamped, so the value can exceed
        5.0 for very dissimilar inputs.

        Args:
            pred: Generated image tensor, shape ``(B, 3, H, W)``, range ``[-1, 1]``.
            target: Target image tensor, same shape and range.

        Returns:
            Scalar MS-SSIM loss.
        """
        # Convert from PuzzleStain's [-1, 1] range to [0, 1].
        pred = (pred + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        return 5.0 * (1.0 - self.ms_ssim(pred, target))

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if pred is None or target is None:
            raise ValueError("MS_SSIM_Loss requires 'pred' and 'target' kwargs")
        return self.forward(pred, target)

    def name(self) -> str:
        return "ms_ssim"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("ms_ssim", MS_SSIM_Loss)

__all__ = ["MS_SSIM_Loss"]
