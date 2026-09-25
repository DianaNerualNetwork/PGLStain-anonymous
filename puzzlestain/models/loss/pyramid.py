"""Pyramid L1 loss used by PyramidPix2pix.

Builds a Gaussian pyramid and computes L1 at each octave. This is the core
supervision signal of PyramidPix2pix (pattern ``L1_L2_L3_L4``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import kornia
except ImportError:  # pragma: no cover - kornia is listed in pyproject.toml
    kornia = None

from .base import LossRegistry, StainLoss


class PyramidL1Loss(nn.Module, StainLoss):
    """Multi-scale L1 loss over a Gaussian pyramid.

    Args:
        num_scales: Number of pyramid levels to supervise (1 to 4).
            ``1`` means only full-resolution L1; ``4`` means L1-L4 as in the
            PyramidPix2pix paper.
        weights: Optional per-scale weights. If None, uniform weights are used.
        blur_kernel_size: Gaussian blur kernel size (must be odd).
        blur_sigma: Gaussian blur sigma.

    Raises:
        ValueError: If ``num_scales`` is not in ``[1, 4]``.
    """

    def __init__(
        self,
        num_scales: int = 4,
        weights: list[float] | None = None,
        blur_kernel_size: int = 3,
        blur_sigma: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        if not 1 <= num_scales <= 4:
            raise ValueError("num_scales must be between 1 and 4")
        self.num_scales = num_scales
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        if weights is None:
            weights = [1.0] * num_scales
        elif len(weights) != num_scales:
            raise ValueError("len(weights) must equal num_scales")
        self.weights = weights

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        if kornia is None:
            raise ImportError("PyramidL1Loss requires kornia. Install it with: pip install kornia")
        return kornia.filters.gaussian_blur2d(
            x, (self.blur_kernel_size, self.blur_kernel_size), self.blur_sigma
        )

    def _build_pyramid(
        self, x: torch.Tensor, num_levels: int
    ) -> list[torch.Tensor]:
        """Build a Gaussian pyramid with ``num_levels`` octaves.

        Each octave applies 4 Gaussian blurs followed by 2x blur-pool downsample.
        This matches the original PyramidPix2pix implementation.
        """
        levels = [x]
        current = x
        for _ in range(num_levels - 1):
            for _ in range(4):
                current = self._blur(current)
            current = kornia.filters.blur_pool2d(current, 1, stride=2)
            levels.append(current)
        return levels

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute multi-scale L1 loss.

        Args:
            pred: Generated image tensor.
            target: Ground-truth image tensor.

        Returns:
            Scalar weighted sum of per-octave L1 losses.
        """
        if pred is None or target is None:
            raise ValueError("PyramidL1Loss requires 'pred' and 'target' kwargs")
        pred_pyramid = self._build_pyramid(pred, self.num_scales)
        target_pyramid = self._build_pyramid(target, self.num_scales)
        total = torch.tensor(0.0, device=pred.device)
        for i, (p, t, w) in enumerate(zip(pred_pyramid, target_pyramid, self.weights)):
            total = total + w * F.l1_loss(p, t)
        return total

    def name(self) -> str:
        return f"pyramid_l1_{self.num_scales}"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("pyramid_l1", PyramidL1Loss)

__all__ = ["PyramidL1Loss"]
