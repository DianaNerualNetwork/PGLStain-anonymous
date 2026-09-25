"""PatchNCE loss from the CUT repository.

Original source: https://github.com/taesungp/contrastive-unpaired-translation
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import LossRegistry, StainLoss


class PatchNCELoss(nn.Module, StainLoss):
    """Patch-level contrastive loss used by CUT.

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
        """Compute the per-patch PatchNCE loss.

        Returns:
            A tensor of shape ``(num_patches,)`` containing the unreduced
            cross-entropy loss for each query patch.
        """
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # Positive logit: cosine similarity between query and its matching key.
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # Negative logits: similarities against all keys in the batch/image.
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
        """Compute the mean PatchNCE loss.

        This wrapper exposes the loss as a :class:`StainLoss` so it can be used
        inside :class:`CompositeLoss` while keeping ``forward`` compatible with
        the original CUT implementation.
        """
        if feat_q is None or feat_k is None:
            raise ValueError(
                "PatchNCELoss requires 'feat_q' and 'feat_k' kwargs"
            )
        return self.forward(feat_q, feat_k).mean()

    def name(self) -> str:
        return "patchnce"

    def required_kwargs(self) -> set[str]:
        return {"feat_q", "feat_k"}


LossRegistry.register("patchnce", PatchNCELoss)

__all__ = ["PatchNCELoss"]
