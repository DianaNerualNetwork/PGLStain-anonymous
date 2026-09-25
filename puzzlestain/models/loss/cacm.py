"""CACM loss from the M2PL-GAN repository.

Original source: https://github.com/Pikachu-one/M2PL-GAN
(``models/PatchGCL.py``). L1 between the per-layer patch cosine-similarity
matrices of two multi-level encoder feature sets.

Note: this differs from
:class:`~puzzlestain.models.loss.cc_matrix.CosineConsistencyLoss`
(``cosine_consistency``), which builds the similarity matrix over *batch
samples* of whole feature maps; CACM builds it over *unfolded patches* within
a single image.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


def _patch_similarity(
    image: torch.Tensor, patch_size: int, stride: int
) -> torch.Tensor:
    """Cosine-similarity matrix between flattened non-overlapping patches.

    Args:
        image: Feature map of shape ``(C, H, W)``.

    Returns:
        Tensor of shape ``(n_patches, n_patches)``.
    """
    C = image.shape[0]
    patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    patches = patches.view(-1, C * patch_size * patch_size)
    patches = F.normalize(patches, dim=1)
    return patches @ patches.t()


class CACMLoss(nn.Module, StainLoss):
    """L1 between per-layer patch cosine-similarity matrices, summed over layers.

    Args:
        patchsize: Side length of the (non-overlapping) square patches.
    """

    def __init__(self, patchsize: int = 64) -> None:
        super().__init__()
        self.patchsize = patchsize

    def forward(
        self,
        feat_real: list[torch.Tensor],
        feat_fake: list[torch.Tensor],
        patchsize: int | None = None,
    ) -> torch.Tensor:
        """Compute the CACM loss between two multi-level feature sets."""
        if patchsize is None:
            patchsize = self.patchsize
        total = torch.zeros((), device=feat_real[0].device)
        for real_feat, fake_feat in zip(feat_real, feat_fake):
            if real_feat.shape[0] != 1:
                raise ValueError("CACM loss requires batch size 1")
            matrix_real = _patch_similarity(
                real_feat.squeeze(0), patchsize, patchsize
            )
            matrix_fake = _patch_similarity(
                fake_feat.squeeze(0), patchsize, patchsize
            )
            total = total + F.l1_loss(matrix_real, matrix_fake)
        return total

    def __call__(
        self,
        feat_real: list[torch.Tensor] | None = None,
        feat_fake: list[torch.Tensor] | None = None,
        patchsize: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the CACM loss as a :class:`StainLoss` component."""
        if feat_real is None or feat_fake is None:
            raise ValueError("CACMLoss requires 'feat_real' and 'feat_fake' kwargs")
        return self.forward(feat_real, feat_fake, patchsize)

    def name(self) -> str:
        return "cacm"

    def required_kwargs(self) -> set[str]:
        return {"feat_real", "feat_fake"}


LossRegistry.register("cacm", CACMLoss)

__all__ = ["CACMLoss"]
