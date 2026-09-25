"""FocalNCE loss from the PPT repository (MICCAI 2024).

Original source: PPT, "High-resolution Medical Image Translation via Patch
Alignment-based Bidirectional Contrastive Learning" (``models/focalnce.py``).
The logit construction is identical to
:class:`~puzzlestain.models.loss.patchnce.PatchNCELoss`; only the final
criterion differs: a focal loss against the positive class instead of
cross-entropy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


class FocalLoss(nn.Module):
    """Focal loss from the PPT repository (``models/focalnce.py``).

    ``Loss(x, class) = -alpha * (1 - softmax(x)[class])^gamma * log(softmax(x)[class])``,
    averaged over the batch.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs_all = F.softmax(inputs, dim=1)
        probs = probs_all.gather(1, targets.view(-1, 1))
        batch_loss = -self.alpha * torch.pow(1.0 - probs, self.gamma) * probs.log()
        return batch_loss.mean()


class FocalNCELoss(nn.Module, StainLoss):
    """PatchNCE logits with a focal criterion (PPT ``models/focalnce.py``).

    The logit construction is identical to
    :class:`~puzzlestain.models.loss.patchnce.PatchNCELoss` (positive bmm,
    negative bmm with -10.0 diagonal masking, temperature division); only the
    final criterion differs: :class:`FocalLoss` against class 0 instead of
    cross-entropy.

    Args:
        nce_T: Temperature for the InfoNCE softmax.
        batch_size: Minibatch size used to reshape negatives.
    """

    def __init__(self, nce_T: float = 0.07, batch_size: int = 1) -> None:
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.focal_loss = FocalLoss()

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        """Compute the focal NCE loss between sampled patch features."""
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # Positive logit.
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # Negative logits within the current batch (batchSize is 1 in PPT).
        batch_dim_for_bmm = self.batch_size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # Diagonal entries are self-similarities; mask them out.
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=torch.bool)[None]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        return self.focal_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )

    def __call__(
        self,
        feat_q: torch.Tensor | None = None,
        feat_k: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the focal NCE loss as a :class:`StainLoss` component."""
        if feat_q is None or feat_k is None:
            raise ValueError("FocalNCELoss requires 'feat_q' and 'feat_k' kwargs")
        return self.forward(feat_q, feat_k)

    def name(self) -> str:
        return "focal_nce"

    def required_kwargs(self) -> set[str]:
        return {"feat_q", "feat_k"}


LossRegistry.register("focal_nce", FocalNCELoss)

__all__ = ["FocalLoss", "FocalNCELoss"]
