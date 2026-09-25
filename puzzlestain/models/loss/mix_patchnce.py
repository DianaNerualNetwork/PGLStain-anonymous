"""Mix-domain PatchNCE loss from Mix-Domain Contrastive Learning.

The loss augments the standard cross-domain negatives with same-domain
negatives taken from the query features themselves. This widens the negative
set and improves representation learning for image-to-image translation.

Original source:
    https://github.com/ssongwang/mix-domaincontrastivelearning
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import LossRegistry, StainLoss


def _mix_domain_negatives(
    feat_q: torch.Tensor,
    l_neg: torch.Tensor,
    mask_dtype: torch.dtype,
) -> torch.Tensor:
    """Concatenate cross-domain negatives with same-domain query negatives.

    Args:
        feat_q: Query features of shape ``(batch_dim, npatches, dim)``.
        l_neg: Cross-domain negative logits of shape
            ``(batch_dim * npatches, npatches)``.
        mask_dtype: Boolean dtype for the diagonal mask.

    Returns:
        Negative logits of shape ``(batch_dim * npatches, npatches + npatches - 1)``.
    """
    batch_dim, npatches, _ = feat_q.shape

    # Same-domain self-similarity matrix.
    l_neg_aug_batch = torch.bmm(feat_q, feat_q.transpose(2, 1))  # (B, N, N)

    # Remove the diagonal entries (self-similarity) from each batch element.
    diagonal = torch.eye(npatches, device=feat_q.device, dtype=mask_dtype)[
        None, :, :
    ]
    l_neg_aug = l_neg_aug_batch.masked_select(~diagonal.expand_as(l_neg_aug_batch))
    l_neg_aug = l_neg_aug.view(batch_dim * npatches, npatches - 1)

    return torch.cat((l_neg, l_neg_aug), dim=1)


class MixPatchNCELoss(nn.Module, StainLoss):
    """PatchNCE loss with additional same-domain negatives.

    Args:
        nce_T: Temperature for the InfoNCE softmax.
        batch_size: Minibatch size used to reshape negatives.
        nce_includes_all_negatives_from_minibatch: If True, treat every patch in
            the full minibatch as a negative for every query patch.
    """

    def __init__(
        self,
        nce_T: float = 0.07,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
    ) -> None:
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = (
            nce_includes_all_negatives_from_minibatch
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = torch.bool

    def forward(self, feat_q: torch.Tensor, feat_k: torch.Tensor) -> torch.Tensor:
        """Compute the per-patch mix-domain PatchNCE loss."""
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # Positive logit.
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # Negative logits (cross-domain).
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(
            npatches, device=feat_q.device, dtype=self.mask_dtype
        )[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # Mix-domain negatives: add same-domain negatives from the query.
        l_neg = _mix_domain_negatives(feat_q, l_neg, self.mask_dtype)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        loss = self.cross_entropy_loss(
            out,
            torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device),
        )
        return loss

    def __call__(
        self,
        feat_q: torch.Tensor | None = None,
        feat_k: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the mean mix-domain PatchNCE loss."""
        if feat_q is None or feat_k is None:
            raise ValueError(
                "MixPatchNCELoss requires 'feat_q' and 'feat_k' kwargs"
            )
        return self.forward(feat_q, feat_k).mean()

    def name(self) -> str:
        return "mix_patchnce"

    def required_kwargs(self) -> set[str]:
        return {"feat_q", "feat_k"}


LossRegistry.register("mix_patchnce", MixPatchNCELoss)

__all__ = ["MixPatchNCELoss", "_mix_domain_negatives"]
