"""Mix-domain adaptive supervised PatchNCE loss.

This is the mix-domain extension of :class:`AdaptiveSupervisedPatchNCELoss`:
it appends same-domain query negatives to the cross-domain negative set before
applying the adaptive reweighting schedule.

Original source:
    https://github.com/ssongwang/mix-domaincontrastivelearning
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base import LossRegistry, StainLoss
from .mix_patchnce import _mix_domain_negatives


class MixAdaptiveSupervisedPatchNCELoss(nn.Module, StainLoss):
    """Adaptive supervised PatchNCE loss with same-domain negatives.

    Args:
        nce_T: Temperature for the InfoNCE softmax.
        batch_size: Minibatch size used to reshape negatives.
        nce_includes_all_negatives_from_minibatch: If True, treat every patch in
            the full minibatch as a negative for every query patch.
        asp_loss_mode: Reweighting mode or ``"none"``.
        n_epochs: Number of constant-LR epochs (used to define the schedule).
        n_epochs_decay: Number of decay epochs (used to define the schedule).
    """

    def __init__(
        self,
        nce_T: float = 0.07,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
        asp_loss_mode: str = "none",
        n_epochs: int = 100,
        n_epochs_decay: int = 100,
    ) -> None:
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = (
            nce_includes_all_negatives_from_minibatch
        )
        self.asp_loss_mode = asp_loss_mode
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.total_epochs = n_epochs + n_epochs_decay
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = torch.bool

    def forward(
        self,
        feat_q: torch.Tensor,
        feat_k: torch.Tensor,
        current_epoch: int = -1,
    ) -> torch.Tensor:
        """Compute the unreduced mix-domain adaptive supervised PatchNCE loss."""
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

        # Mix-domain negatives.
        l_neg = _mix_domain_negatives(feat_q, l_neg, self.mask_dtype)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        loss = self.cross_entropy_loss(
            out,
            torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device),
        )

        if self.asp_loss_mode == "none":
            return loss

        scheduler, lookup = self.asp_loss_mode.split("_")[:2]

        t = (current_epoch - 1) / self.total_epochs
        if scheduler == "sigmoid":
            p = 1.0 / (1.0 + np.exp((t - 0.5) * 10))
        elif scheduler == "linear":
            p = 1.0 - t
        elif scheduler == "lambda":
            k = 1.0 - self.n_epochs_decay / self.total_epochs
            m = 1.0 / (1.0 - k)
            p = m - m * t if t >= k else 1.0
        elif scheduler == "zero":
            p = 1.0
        else:
            raise ValueError(f"Unrecognized scheduler: {scheduler}")

        w0 = 1.0
        x = l_pos.squeeze().detach()
        if lookup == "top":
            x = torch.where(x > 0.0, x, torch.zeros_like(x))
            w1 = torch.sqrt(1.0 - (x - 1.0) ** 2)
        elif lookup == "linear":
            w1 = torch.relu(x)
        elif lookup == "bell":
            sigma, mu, sc = 1.0, 0.0, 4.0
            w1 = (
                1.0
                / (sigma * np.sqrt(2.0 * torch.pi))
                * torch.exp(-(((x - 0.5) * sc - mu) ** 2) / (2.0 * sigma**2))
            )
        elif lookup == "uniform":
            w1 = torch.ones_like(x)
        else:
            raise ValueError(f"Unrecognized lookup: {lookup}")

        w = p * w0 + (1.0 - p) * w1
        w = w / w.sum() * len(w)
        loss = loss * w
        return loss

    def __call__(
        self,
        feat_q: torch.Tensor | None = None,
        feat_k: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the mean mix-domain adaptive supervised PatchNCE loss."""
        if feat_q is None or feat_k is None:
            raise ValueError(
                "MixAdaptiveSupervisedPatchNCELoss requires 'feat_q' and 'feat_k' kwargs"
            )
        current_epoch = kwargs.get("current_epoch", -1)
        return self.forward(feat_q, feat_k, current_epoch=current_epoch).mean()

    def name(self) -> str:
        return "mix_asp"

    def required_kwargs(self) -> set[str]:
        return {"feat_q", "feat_k"}


LossRegistry.register("mix_asp", MixAdaptiveSupervisedPatchNCELoss)

__all__ = ["MixAdaptiveSupervisedPatchNCELoss"]
