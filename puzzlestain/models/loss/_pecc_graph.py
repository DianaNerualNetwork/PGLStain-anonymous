"""Graph-encoding primitives used internally by PECC.

The implementations are pure PyTorch equivalents of the graph projection
operators used by the correspondence-calibration objective: TAG convolution,
graph convolution, and graph attention over dense patch graphs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows."""
    norm = x.pow(2).sum(1, keepdim=True).pow(1.0 / 2.0)
    return x.div(norm)


class Embed(nn.Module):
    """Linear projection followed by L2 normalization."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return l2_normalize(self.linear(x))


class TAGConvEncoder(nn.Module):
    """Pure-PyTorch TAG convolution followed by L2 normalization."""

    def __init__(self, in_dim: int, hidden_dim: int, num_hop: int = 4) -> None:
        super().__init__()
        self.num_hop = num_hop
        self.lin = nn.Linear(in_dim * (num_hop + 1), hidden_dim)

    def forward(self, feat: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        fstack = [feat]
        for _ in range(self.num_hop):
            fstack.append(adj_norm @ fstack[-1])
        return l2_normalize(self.lin(torch.cat(fstack, dim=-1)))


class GCNEncoder(nn.Module):
    """Pure-PyTorch graph convolution followed by L2 normalization."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)

    def forward(self, feat: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.linear(adj_norm @ feat))


class GATEncoder(nn.Module):
    """Pure-PyTorch multi-head graph attention followed by L2 normalization."""

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
        mask = (adj > 0).unsqueeze(-1)
        ft = self.fc(feat).view(-1, self.num_heads, self.out_dim)
        el = (ft * self.attn_l).sum(dim=-1)
        er = (ft * self.attn_r).sum(dim=-1)
        scores = F.leaky_relu(er.unsqueeze(1) + el.unsqueeze(0), negative_slope=0.2)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        out = torch.einsum("vuh,uhf->vhf", attn, ft) + self.bias
        return l2_normalize(out.flatten(1))


__all__ = ["Embed", "GATEncoder", "GCNEncoder", "TAGConvEncoder", "l2_normalize"]
