"""LDAM loss from the M2PL-GAN repository.

Original source: https://github.com/Pikachu-one/M2PL-GAN
(``models/m2plgan_model.py`` ``calculate_mmd_loss``). RBF-kernel MMD between
the per-layer patch vectors of two multi-level feature sets, in either
``patchwpatch`` or ``pixelwpiexl`` mode.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


def _extract_patch_vectors(
    feat: torch.Tensor, patch_size: int, stride: int
) -> torch.Tensor:
    """L2-normalized, spatially average-pooled patch vectors of a feature map."""
    patches = feat.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, feat.size(1), patch_size, patch_size)
    pooled = F.adaptive_avg_pool2d(patches, output_size=1)
    return F.normalize(pooled.view(pooled.size(0), -1), dim=1)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """RBF kernel matrix between two sets of vectors."""
    x_size = x.size(0)
    y_size = y.size(0)
    xx = x.unsqueeze(1).expand(x_size, y_size, -1)
    yy = y.unsqueeze(0).expand(x_size, y_size, -1)
    dist = (xx - yy).pow(2).sum(2)
    return torch.exp(-dist / (2 * sigma**2))


def _mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Biased RBF-kernel MMD estimate between two sets of patch vectors."""
    return (
        _rbf_kernel(x, x, sigma).mean()
        + _rbf_kernel(y, y, sigma).mean()
        - 2 * _rbf_kernel(x, y, sigma).mean()
    )


class LDAMLoss(nn.Module, StainLoss):
    """RBF-kernel MMD between patch vectors, summed over layers.

    Two modes, mirroring ``calculate_mmd_loss`` in the original repo:

    - ``patchwpatch``: inputs are multi-level encoder feature maps of shape
      ``(1, C, H, W)``; each map is reduced to L2-normalized patch vectors
      (see :func:`_extract_patch_vectors`) and the per-layer MMD terms are
      summed unweighted.
    - ``pixelwpiexl``: inputs are netF-sampled patch features of shape
      ``(N, C)`` per layer (sampled by the strategy via ``model.netF``); the
      per-layer MMD terms are summed with ``weights``.

    Args:
        patch_size: Side length of the (non-overlapping) square patches
            (``patchwpatch`` only).
        stride: Patch extraction stride (``patchwpatch`` only).
        sigma: RBF kernel bandwidth.
        mode: ``patchwpatch`` or ``pixelwpiexl``.
        weights: Per-layer weights for ``pixelwpiexl`` mode.
    """

    def __init__(
        self,
        patch_size: int = 16,
        stride: int = 16,
        sigma: float = 1.0,
        mode: str = "patchwpatch",
        weights: tuple[float, ...] = (1.0, 0.8, 0.5, 0.3, 0.3),
    ) -> None:
        super().__init__()
        if mode not in ("patchwpatch", "pixelwpiexl"):
            raise NotImplementedError(f"LDAM mode [{mode}] is not recognized")
        self.patch_size = patch_size
        self.stride = stride
        self.sigma = sigma
        self.mode = mode
        self.weights = tuple(weights)

    def forward(
        self,
        feat_fake: list[torch.Tensor],
        feat_real: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the LDAM loss between two multi-level feature sets."""
        total = torch.zeros((), device=feat_fake[0].device)
        if self.mode == "pixelwpiexl":
            if len(feat_fake) > len(self.weights):
                raise ValueError(
                    f"pixelwpiexl mode provides {len(self.weights)} layer "
                    f"weights but received {len(feat_fake)} feature layers"
                )
            for weight, fx, fy in zip(self.weights, feat_fake, feat_real):
                total = total + weight * _mmd_loss(fx, fy, self.sigma)
            return total
        for fx, fy in zip(feat_fake, feat_real):
            x_patches = _extract_patch_vectors(fx, self.patch_size, self.stride)
            y_patches = _extract_patch_vectors(fy, self.patch_size, self.stride)
            total = total + _mmd_loss(x_patches, y_patches, self.sigma)
        return total

    def __call__(
        self,
        feat_fake: list[torch.Tensor] | None = None,
        feat_real: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the LDAM loss as a :class:`StainLoss` component."""
        if feat_fake is None or feat_real is None:
            raise ValueError("LDAMLoss requires 'feat_fake' and 'feat_real' kwargs")
        return self.forward(feat_fake, feat_real)

    def name(self) -> str:
        return "ldam"

    def required_kwargs(self) -> set[str]:
        return {"feat_fake", "feat_real"}


LossRegistry.register("ldam", LDAMLoss)

__all__ = ["LDAMLoss"]
