"""Cosine-consistency matrix loss for SIM-GAN / USIGAN.

Computes a per-layer cosine-similarity matrix between two sets of encoder
features and penalizes their L1 difference.  This is the ``CC_loss`` used in
both SIM-GAN and USIGAN.

References:
    https://github.com/MIXAILAB/USIGAN
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


class CosineConsistencyLoss(nn.Module, StainLoss):
    """L1 distance between batch cosine-similarity matrices of two feature sets.

    Args:
        scale: Overall loss scale (the reference uses ``10``).
    """

    def __init__(self, scale: float = 10.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(
        self,
        src_feats: list[torch.Tensor],
        tgt_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute ``scale * L1(S(src), S(tgt))`` averaged over layers.

        Args:
            src_feats: List of feature tensors, each ``(B, C, H, W)``.
            tgt_feats: List of feature tensors with matching shapes.
        """
        total = torch.tensor(0.0, device=src_feats[0].device)
        for src, tgt in zip(src_feats, tgt_feats):
            matrix_src = self._cal_matrix(src).detach()
            matrix_tgt = self._cal_matrix(tgt)
            total = total + F.l1_loss(matrix_src, matrix_tgt)
        return (total / len(src_feats)) * self.scale

    def _cal_matrix(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine similarity of flattened feature maps."""
        batch_size = feat.size(0)
        matrix = torch.zeros(batch_size, batch_size, device=feat.device)
        sub_tensors = torch.split(feat, 1, dim=0)
        for i in range(batch_size):
            for j in range(batch_size):
                vector_i = sub_tensors[i].view(-1)
                vector_j = sub_tensors[j].view(-1)
                matrix[i, j] = F.cosine_similarity(vector_i, vector_j, dim=0)
        return matrix

    def __call__(
        self,
        src_feats: list[torch.Tensor] | None = None,
        tgt_feats: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if src_feats is None or tgt_feats is None:
            raise ValueError(
                "CosineConsistencyLoss requires 'src_feats' and 'tgt_feats' kwargs"
            )
        return self.forward(src_feats, tgt_feats)

    def name(self) -> str:
        return "cosine_consistency"

    def required_kwargs(self) -> set[str]:
        return {"src_feats", "tgt_feats"}


LossRegistry.register("cosine_consistency", CosineConsistencyLoss)

__all__ = ["CosineConsistencyLoss"]
