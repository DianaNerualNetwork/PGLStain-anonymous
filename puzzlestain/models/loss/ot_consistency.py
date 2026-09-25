"""Optimal-transport consistency losses for SIM-GAN and USIGAN.

These losses use entropy-regularized optimal transport to enforce cycle-like
consistency between source, target, and generated patch features.

References:
    https://github.com/MIXAILAB/USIGAN
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss
from .sinkhorn import optimal_transport, unbalanced_optimal_transport


class MCLoss(nn.Module, StainLoss):
    """Marginal/OT consistency loss used by SIM-GAN.

    Computes transport plans ``T(src, tgt)`` and ``T(tgt, gen)`` and penalizes
    their L1 difference.  Source/target features are detached so gradients flow
    only through the generated features.

    Args:
        eps: Entropic regularization for Sinkhorn.
        max_iter: Number of Sinkhorn iterations.
        cost_type: ``"easy"`` or ``"hard"`` cost construction.
        scale: Overall loss scale (the reference uses ``10000``).
        batch_size: Minibatch size used to reshape features.
        nce_includes_all_negatives_from_minibatch: If True, treat the whole
            minibatch as one batch dimension for OT.
    """

    def __init__(
        self,
        eps: float = 1.0,
        max_iter: int = 50,
        cost_type: str = "easy",
        scale: float = 10000.0,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.cost_type = cost_type
        self.scale = scale
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = (
            nce_includes_all_negatives_from_minibatch
        )

    def forward(
        self,
        feat_src: torch.Tensor,
        feat_tgt: torch.Tensor,
        feat_gen: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``scale * L1(T(src,tgt), T(tgt,gen))``.

        Args:
            feat_src: Source features ``(B*P, D)`` or ``(P, D)``.
            feat_tgt: Target features, same shape.
            feat_gen: Generated features, same shape (gradients enabled).
        """
        dim = feat_src.shape[1]
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim = 1
        else:
            batch_dim = self.batch_size

        ot_src = feat_src.view(batch_dim, -1, dim).detach()
        ot_tgt = feat_tgt.view(batch_dim, -1, dim).detach()
        ot_gen = feat_gen.view(batch_dim, -1, dim)

        f1 = optimal_transport(
            ot_src, ot_tgt, eps=self.eps, max_iter=self.max_iter, cost_type=self.cost_type
        )
        f2 = optimal_transport(
            ot_tgt, ot_gen, eps=self.eps, max_iter=self.max_iter, cost_type=self.cost_type
        )
        return F.l1_loss(f1, f2) * self.scale

    def __call__(
        self,
        feat_src: torch.Tensor | None = None,
        feat_tgt: torch.Tensor | None = None,
        feat_gen: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if feat_src is None or feat_tgt is None or feat_gen is None:
            raise ValueError("MCLoss requires 'feat_src', 'feat_tgt' and 'feat_gen' kwargs")
        return self.forward(feat_src, feat_tgt, feat_gen)

    def name(self) -> str:
        return "mc_consistency"

    def required_kwargs(self) -> set[str]:
        return {"feat_src", "feat_tgt", "feat_gen"}


class UMCLoss(nn.Module, StainLoss):
    """Unbalanced OT cycle consistency loss used by USIGAN.

    Computes direct transport ``T(src, gen)`` and indirect transport
    ``T(src, tgt) @ T(tgt, gen)`` and penalizes their L1 difference.

    References:
        https://github.com/MIXAILAB/USIGAN (models/MC_loss.py)
    """

    def __init__(
        self,
        eps: float = 1.0,
        tau: float = 0.001,
        max_iter: int = 50,
        cost_type: str = "easy",
        scale: float = 10000.0,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.tau = tau
        self.max_iter = max_iter
        self.cost_type = cost_type
        self.scale = scale
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = (
            nce_includes_all_negatives_from_minibatch
        )

    def forward(
        self,
        feat_src: torch.Tensor,
        feat_tgt: torch.Tensor,
        feat_gen: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``scale * L1(T(src,gen), T(src,tgt) @ T(tgt,gen))``."""
        dim = feat_src.shape[1]
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim = 1
        else:
            batch_dim = self.batch_size

        ot_src = feat_src.view(batch_dim, -1, dim).detach()
        ot_tgt = feat_tgt.view(batch_dim, -1, dim).detach()
        ot_gen = feat_gen.view(batch_dim, -1, dim)

        f1 = unbalanced_optimal_transport(
            ot_src,
            ot_tgt,
            eps=self.eps,
            tau=self.tau,
            max_iter=self.max_iter,
            cost_type=self.cost_type,
        )
        f2 = unbalanced_optimal_transport(
            ot_tgt,
            ot_gen,
            eps=self.eps,
            tau=self.tau,
            max_iter=self.max_iter,
            cost_type=self.cost_type,
        )
        f_indirect = torch.matmul(f1, f2)
        f_direct = unbalanced_optimal_transport(
            ot_src,
            ot_gen,
            eps=self.eps,
            tau=self.tau,
            max_iter=self.max_iter,
            cost_type=self.cost_type,
        )
        return F.l1_loss(f_direct, f_indirect) * self.scale

    def __call__(
        self,
        feat_src: torch.Tensor | None = None,
        feat_tgt: torch.Tensor | None = None,
        feat_gen: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if feat_src is None or feat_tgt is None or feat_gen is None:
            raise ValueError("UMCLoss requires 'feat_src', 'feat_tgt' and 'feat_gen' kwargs")
        return self.forward(feat_src, feat_tgt, feat_gen)

    def name(self) -> str:
        return "umc_consistency"

    def required_kwargs(self) -> set[str]:
        return {"feat_src", "feat_tgt", "feat_gen"}


LossRegistry.register("mc_consistency", MCLoss)
LossRegistry.register("umc_consistency", UMCLoss)

__all__ = ["MCLoss", "UMCLoss"]
