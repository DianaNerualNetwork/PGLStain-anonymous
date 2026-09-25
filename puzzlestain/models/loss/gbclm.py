"""GBCLM loss (graph-based contrastive loss) from the M2PL-GAN repository.

Original source: https://github.com/Pikachu-one/M2PL-GAN
(``models/PatchGCL.py``). Bidirectional InfoNCE over graph encodings of
netF-sampled patch features, with the original's ``conv_type`` variants
(``tag`` / ``gcn`` / ``gat`` / ``none``). The original builds DGL graphs;
this port is pure PyTorch and numerically mirrors DGL's ``TAGConv``
(``D^-1/2 (A+I) D^-1/2`` propagation, k=4 hops), ``GraphConv`` and
``GATConv``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows; mirrors ``Normalize(2)`` in the original PatchGCL.py."""
    norm = x.pow(2).sum(1, keepdim=True).pow(1.0 / 2.0)
    return x.div(norm)


class Embed(nn.Module):
    """Linear projection followed by L2 normalization (PatchGCL ``Embed``)."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return _l2_normalize(self.linear(x))


class TAGConvEncoder(nn.Module):
    """Pure-PyTorch port of DGL ``TAGConv`` + L2 norm (PatchGCL ``Encoder``).

    Computes ``Linear(concat[h, Ph, P^2h, ..., P^kh])`` where
    ``P = D^-1/2 (A+I) D^-1/2`` is the symmetrically normalized adjacency,
    matching DGL's message-passing implementation for a symmetric binary
    adjacency with self-loops.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_hop: int = 4) -> None:
        super().__init__()
        self.num_hop = num_hop
        self.lin = nn.Linear(in_dim * (num_hop + 1), hidden_dim)

    def forward(self, feat: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        fstack = [feat]
        for _ in range(self.num_hop):
            fstack.append(adj_norm @ fstack[-1])
        return _l2_normalize(self.lin(torch.cat(fstack, dim=-1)))


class GCNEncoder(nn.Module):
    """Pure-PyTorch port of DGL ``GraphConv`` (``norm='both'``) + L2 norm.

    Computes ``Linear(P @ h)`` where ``P = D^-1/2 (A+I) D^-1/2`` is the
    symmetrically normalized adjacency, matching DGL's message-passing
    implementation for a symmetric binary adjacency with self-loops.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)

    def forward(self, feat: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return _l2_normalize(self.linear(adj_norm @ feat))


class GATEncoder(nn.Module):
    """Pure-PyTorch port of DGL ``GATConv`` (multi-head) + L2 norm.

    Mirrors DGL's ``GATConv(in_dim, out_dim, num_heads)`` defaults
    (``negative_slope=0.2``, no feature/attention dropout, no residual):
    the score of edge ``u -> v`` is ``LeakyReLU(attn_l . Wh_u + attn_r .
    Wh_v)``, softmax-normalized per destination node over its neighbors
    (binary adjacency with self-loops), and head outputs are concatenated.
    Initialization mirrors DGL's ``reset_parameters`` (``attn_l``/``attn_r``
    Xavier with the ReLU gain, zero bias); ``fc.weight`` is re-initialized by
    ``init_weights`` in :meth:`GNNLoss.create_mlp`, exactly like the original
    repo's ``init_net``.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, out_dim))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(1, num_heads, out_dim))
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        mask = (adj > 0).unsqueeze(-1)  # [V, U, 1]: edge u -> v, self-loops in
        ft = self.fc(feat).view(-1, self.num_heads, self.out_dim)
        el = (ft * self.attn_l).sum(dim=-1)  # source scores [U, H]
        er = (ft * self.attn_r).sum(dim=-1)  # destination scores [V, H]
        scores = F.leaky_relu(er.unsqueeze(1) + el.unsqueeze(0), negative_slope=0.2)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)  # per destination, over sources
        out = torch.einsum("vuh,uhf->vhf", attn, ft) + self.bias
        return _l2_normalize(out.flatten(1))


class GNNLoss(nn.Module, StainLoss):
    """Graph-based contrastive loss (GBCLM), ported from PatchGCL.GNNLoss.

    Per nce layer, lazily creates an ``Embed`` projection and a graph
    encoder selected by ``conv_type`` (``tag`` / ``gcn`` / ``gat`` /
    ``none``). The adjacency is binary (cosine similarity > ``nonzero_th``)
    built from the detached target embeddings and reused for the source
    graph, exactly like the original ``nonzero_graph`` with
    ``exist_adj=adj_t``. The loss is the sum of both InfoNCE directions
    (temperature 0.07, positive on the diagonal), averaged over layers.

    Note: mirroring the original repo, these parameters receive gradients
    through the G loss but are *not* registered in any optimizer, so they are
    never stepped. The M2PL-GAN strategy exposes this module via
    ``get_networks`` so it is checkpointed.

    Args:
        nc: Embedding dimension (matches ``netF_nc``).
        num_hop: Number of TAGConv propagation hops (``tag`` only).
        nonzero_th: Cosine-similarity threshold for the binary adjacency.
        nce_T: InfoNCE temperature.
        conv_type: Graph convolution type: ``tag`` | ``gcn`` | ``gat`` |
            ``none``.
    """

    def __init__(
        self,
        nc: int = 256,
        num_hop: int = 4,
        nonzero_th: float = 0.6,
        nce_T: float = 0.07,
        conv_type: str = "tag",
    ) -> None:
        super().__init__()
        if conv_type not in ("tag", "gcn", "gat", "none"):
            raise NotImplementedError(
                f"conv_type [{conv_type}] is not recognized"
            )
        self.nc = nc
        self.num_hop = num_hop
        self.nonzero_th = nonzero_th
        self.nce_T = nce_T
        self.conv_type = conv_type
        self.mlp_init = False

    def create_mlp(self, feats: list[torch.Tensor]) -> None:
        """Lazily create per-layer modules once real feature shapes are known."""
        # Imported here (not at module top) to avoid a circular import between
        # the loss package and the CUT family at registration time.
        from ..cut_family.networks.cut_networks import init_weights

        device = feats[0].device
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            setattr(self, f"embed_{mlp_id}", Embed(input_nc, self.nc))
            if self.conv_type == "tag":
                encoder: nn.Module | None = TAGConvEncoder(
                    self.nc, self.nc, self.num_hop
                )
            elif self.conv_type == "gcn":
                encoder = GCNEncoder(self.nc, self.nc)
            elif self.conv_type == "gat":
                encoder = GATEncoder(self.nc, self.nc // 4, num_heads=4)
            else:  # "none": no graph convolution, mirrors ``conv1 = None``
                encoder = None
            setattr(self, f"gnn_{mlp_id}", encoder)
        init_weights(self, init_type="normal", init_gain=0.02)
        self.to(device)
        self.mlp_init = True

    def _build_adjacency(self, feat: torch.Tensor) -> torch.Tensor:
        """Return ``D^-1/2 (A+I) D^-1/2`` for the binary thresholded graph."""
        sim = feat @ feat.t()  # feat is L2-normalized -> cosine similarity
        adj = (sim > self.nonzero_th).float()
        adj.fill_diagonal_(1.0)  # mirrors dgl.add_self_loop
        deg = adj.sum(1).clamp(min=1)
        dinv = deg.pow(-0.5)
        return adj * dinv.unsqueeze(1) * dinv.unsqueeze(0)

    def _layer_loss(
        self,
        f_es: torch.Tensor,
        f_et: torch.Tensor,
        gnn: nn.Module | None,
        adj_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Bidirectional node-wise InfoNCE between source and target graphs."""
        if gnn is None:  # conv_type "none": L2-normalized embeddings only
            f_gt = _l2_normalize(f_et).detach()
            f_gs = _l2_normalize(f_es)
        else:
            f_gt = gnn(f_et, adj_norm).detach()
            f_gs = gnn(f_es, adj_norm)
        num_patches = f_gs.shape[0]
        diagonal = torch.eye(num_patches, device=f_es.device, dtype=torch.bool)[None]
        label = torch.zeros(num_patches, dtype=torch.long, device=f_es.device)
        f_gt_reshape = f_gt.view(1, -1, self.nc)
        f_gs_reshape = f_gs.view(1, -1, self.nc)

        gs_pos = torch.einsum("nc,nc->n", [f_gt, f_gs]).unsqueeze(-1)
        gs_neg = torch.bmm(f_gt_reshape, f_gs_reshape.transpose(2, 1))
        gs_neg.masked_fill_(diagonal, -10.0)
        out_gs = torch.cat([gs_pos, gs_neg.view(-1, num_patches)], dim=1) / self.nce_T
        loss_gs = F.cross_entropy(out_gs.contiguous(), label)

        gt_pos = torch.einsum("nc,nc->n", [f_gs, f_gt]).unsqueeze(-1)
        gt_neg = torch.bmm(f_gs_reshape, f_gt_reshape.transpose(2, 1))
        gt_neg.masked_fill_(diagonal, -10.0)
        out_gt = torch.cat([gt_pos, gt_neg.view(-1, num_patches)], dim=1) / self.nce_T
        loss_gt = F.cross_entropy(out_gt.contiguous(), label)

        return loss_gs + loss_gt

    def forward(
        self,
        feat_s: list[torch.Tensor],
        feat_t: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the GBCLM loss between sampled source/target patch features."""
        if not self.mlp_init:
            self.create_mlp(feat_s)
        total = torch.zeros((), device=feat_s[0].device)
        for mlp_id, (f_es, f_et) in enumerate(zip(feat_s, feat_t)):
            embed = getattr(self, f"embed_{mlp_id}")
            gnn = getattr(self, f"gnn_{mlp_id}")
            f_es = embed(f_es)
            f_et = embed(f_et)
            adj_norm = self._build_adjacency(f_et.detach())
            total = total + self._layer_loss(f_es, f_et, gnn, adj_norm)
        return total / len(feat_s)

    def __call__(
        self,
        feat_s: list[torch.Tensor] | None = None,
        feat_t: list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the GBCLM loss as a :class:`StainLoss` component."""
        if feat_s is None or feat_t is None:
            raise ValueError("GNNLoss requires 'feat_s' and 'feat_t' kwargs")
        return self.forward(feat_s, feat_t)

    def name(self) -> str:
        return "gbclm"

    def required_kwargs(self) -> set[str]:
        return {"feat_s", "feat_t"}


LossRegistry.register("gbclm", GNNLoss)

__all__ = ["Embed", "GATEncoder", "GCNEncoder", "GNNLoss", "TAGConvEncoder"]
