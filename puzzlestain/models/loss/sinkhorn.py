"""Sinkhorn optimal transport utilities for contrastive learning.

Adapted from the MoNCE / USIGAN / SIM-GAN repositories, which use a lightweight
entropy-regularized Sinkhorn solver to compute transport plans between patch
feature sets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sinkhorn(
    dot: torch.Tensor,
    max_iter: int = 100,
    reg: float | None = None,
) -> torch.Tensor:
    """Balanced Sinkhorn iteration for a batch of similarity matrices.

    Args:
        dot: Similarity matrix of shape ``(n, in_size, out_size)``. Expected
            to be strictly positive (e.g. ``exp(cost / eps)``).
        max_iter: Number of Sinkhorn iterations.
        reg: Deprecated, kept for compatibility with older call sites.

    Returns:
        Transport plan of shape ``(n, in_size, out_size)``.
    """
    del reg  # unused; kept for backward-compatible signatures
    n, in_size, out_size = dot.shape
    K = dot
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    a = float(out_size) / float(in_size)
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        v = 1.0 / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
    return u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))


def optimal_transport(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1.0,
    max_iter: int = 100,
    cost_type: str | None = None,
) -> torch.Tensor:
    """Compute an optimal-transport plan between two patch-feature sets.

    Args:
        q: Query features of shape ``(n, in_size, dim)``.
        k: Key features of shape ``(m, out_size, dim)``.
        eps: Entropic regularization parameter.
        max_iter: Sinkhorn iterations.
        cost_type: How to turn cosine similarities into costs. ``"easy"`` uses
            ``1 - sim``; ``"hard"`` uses ``sim`` directly. ``None`` defaults to
            ``"easy"``.

    Returns:
        Transport plan of shape ``(n * m, out_size, in_size)``.
    """
    n, in_size, _ = q.shape
    m, out_size, _ = k.shape

    # Cosine similarity between every query patch and every key patch.
    cost = torch.einsum("bid,bod->bio", q, k)  # (n, m, in_size, out_size)

    if cost_type == "hard":
        K = cost.clone()
    else:
        # "easy" and default: cost = 1 - similarity.
        K = 1.0 - cost.clone()

    # Mask the diagonal to prevent self-matching within the same image.
    diagonal = torch.eye(in_size, device=q.device, dtype=torch.bool)[None, :, :]
    K.masked_fill_(diagonal, -10.0)

    K = K.reshape(-1, in_size, out_size)
    K = torch.exp(K / eps)
    K = sinkhorn(K, max_iter=max_iter)
    return K.permute(0, 2, 1).contiguous()


def unbalanced_sinkhorn(
    dot: torch.Tensor,
    tau: float = 0.1,
    max_iter: int = 100,
) -> torch.Tensor:
    """Unbalanced Sinkhorn iteration for a batch of similarity matrices.

    This is the UOT variant used by USIGAN, which relaxes the hard marginal
    constraints with a KL penalty controlled by ``tau``.

    Args:
        dot: Positive similarity matrix of shape ``(n, in_size, out_size)``.
        tau: Relaxation parameter for mass conservation.
        max_iter: Number of Sinkhorn iterations.

    Returns:
        Transport plan of shape ``(n, in_size, out_size)``.
    """
    n, in_size, out_size = dot.shape
    K = dot
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    for _ in range(max_iter):
        u = torch.exp(-tau * u) * (
            1.0 / (torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size) + 1e-8)
        )
        v = torch.exp(-tau * v) * (
            1.0 / (torch.bmm(u.view(n, 1, in_size), K).view(n, out_size) + 1e-8)
        )
    return u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))


def unbalanced_optimal_transport(
    q: torch.Tensor,
    k: torch.Tensor,
    eps: float = 1.0,
    tau: float = 0.1,
    max_iter: int = 100,
    cost_type: str | None = None,
) -> torch.Tensor:
    """Compute an unbalanced optimal-transport plan between patch features.

    Args:
        q: Query features of shape ``(n, in_size, dim)``.
        k: Key features of shape ``(m, out_size, dim)``.
        eps: Entropic regularization parameter.
        tau: Relaxation parameter for mass conservation.
        max_iter: Sinkhorn iterations.
        cost_type: ``"easy"`` uses ``1 - sim``; ``"hard"`` uses ``sim``.

    Returns:
        Transport plan of shape ``(n * m, out_size, in_size)``.
    """
    n, in_size, _ = q.shape
    m, out_size, _ = k.shape

    cost = torch.einsum("bid,bod->bio", q, k)  # (n, m, in_size, out_size)
    if cost_type == "hard":
        K = cost.clone()
    else:
        K = 1.0 - cost.clone()

    diagonal = torch.eye(in_size, device=q.device, dtype=torch.bool)[None, :, :]
    K.masked_fill_(diagonal, -10.0)

    K = K.reshape(-1, in_size, out_size)
    K = torch.exp(K / eps)
    K = unbalanced_sinkhorn(K, tau=tau, max_iter=max_iter)
    return K.permute(0, 2, 1).contiguous()


__all__ = [
    "sinkhorn",
    "optimal_transport",
    "unbalanced_sinkhorn",
    "unbalanced_optimal_transport",
]
