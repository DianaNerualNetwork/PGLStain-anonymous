"""Protein Expression Aware Correspondence Calibration (PECC).

OD-guided correspondence calibration for weakly-paired virtual staining. Three
design problems are addressed:

1. Misaligned positives (P0). v2 pairs patch ``i`` of ``fake_B`` with patch
   ``i`` of ``real_B``, but weakly-paired H&E/IHC data is spatially
   misaligned, so index-aligned positives routinely link different tissue
   classes (measured: ~16% of pairs differ by more than ``od_sigma`` in OD).
   v3 pairs each fake patch with the real patch of *nearest OD value*
   (hard nearest neighbour within the sampled patch set), which is
   misalignment-robust by construction.
2. One-sided graphs (P0). v2 builds the OD graph from ``real_B`` only and
   never looks at ``fake_B`` (``image_s`` was accepted but unused). v3 builds
   bilateral graphs: each side's features are aggregated over *its own* OD
   graph, and the two sides are linked only through the OD-matched positives.
3. Saturated OD signal (P3). v2 inherits the PCSM ``[-1, 1]``-clamped OD
   estimator, whose output is effectively binary (~64% pixels near 0, ~36%
   pinned at ~2.26). For patch-pair graph gating a graded signal is needed,
   so v3 uses a ``[0, 1]``-range focal OD (``_proper_od_map``). The PCSM
   parity path in ``od_graph.py``/``pcsm.py`` is left untouched.

Additionally the structure-matching loss :class:`PECCStructureLoss`
implements the "graph vs graph" idea directly: the OD graph of ``fake_B`` is
compared to the OD graph of ``real_B`` through permutation-invariant
statistics (sorted degree sequence, normalized-Laplacian spectrum, weighted
clustering coefficient), so the constraint is immune to spatial
misalignment while still pushing the generated image's OD structure toward
the real one. Gradients flow into ``fake_B`` through its OD map.

Layer coverage (P1, shallow-layer embedding collapse measured in v2: pairwise
cosine std 0.02 at layer 0) is handled by the strategy, which passes only the
selected deep layers (default nce layers 8/12/16) to this loss.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._pecc_graph import (
    Embed,
    GATEncoder,
    GCNEncoder,
    TAGConvEncoder,
    l2_normalize,
)
from .base import LossRegistry, StainLoss


def _proper_od_map(image: torch.Tensor) -> torch.Tensor:
    """Compute focal OD (DAB) map from an image in ``[-1, 1]``.

    Unlike ``od_graph.compute_od_map`` (PCSM parity, clamps the ``[-1, 1]``
    input directly), this first rescales to ``[0, 1]`` so the Beer-Lambert
    logarithm sees physical transmitted intensities, yielding a graded
    protein-expression signal instead of a saturated two-level mask.

    Args:
        image: RGB image of shape (B, 3, H, W) in ``[-1, 1]``.

    Returns:
        OD map of shape (B, H, W).
    """
    image = ((image + 1.0) / 2.0).permute(0, 2, 3, 1)  # (B, H, W, 3) in [0, 1]

    # HED color deconvolution matrices (Ruifrok-Johnston)
    rgb_from_hed = torch.tensor(
        [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
        device=image.device,
        dtype=image.dtype,
    )
    coeffs = torch.tensor(
        [0.2125, 0.7154, 0.0721], device=image.device, dtype=image.dtype
    ).view(3, 1)
    hed_from_rgb = torch.linalg.inv(rgb_from_hed)

    # Separate stains
    rgb_clamped = torch.clamp(image, min=1e-6)
    log_adjust = torch.log(torch.tensor(1e-6, device=image.device, dtype=image.dtype))
    stains = torch.matmul(torch.log(rgb_clamped) / log_adjust, hed_from_rgb)
    stains = torch.maximum(stains, torch.zeros_like(stains))

    # Extract DAB channel (index 2)
    null = torch.zeros_like(stains[:, :, :, 0])
    ihc_d = torch.stack((null, null, stains[:, :, :, 2]), dim=-1)

    # Combine back to RGB
    log_rgb = -torch.matmul((ihc_d * -log_adjust), rgb_from_hed)
    rgb_d = torch.exp(log_rgb)
    rgb_d = torch.clamp(rgb_d, min=0.0, max=1.0)

    # Convert to grayscale
    grey_d = torch.matmul(rgb_d, coeffs)
    grey_d = torch.clamp(grey_d, 0.0, 1.0)

    # Compute focal OD
    alpha = 1.8
    adjust_calibration = torch.tensor(
        10 ** (-(math.e) ** (1 / alpha)), device=image.device, dtype=image.dtype
    )
    FOD = torch.log10(1.0 / (grey_d + adjust_calibration))
    FOD = torch.clamp(FOD, min=0.0)
    FOD = FOD**alpha  # (B, H, W, 1)

    return FOD.squeeze(-1)  # (B, H, W)


def _od_nearest(src_od: torch.Tensor, tgt_od: torch.Tensor) -> torch.Tensor:
    """Hard OD nearest-neighbour matching between two sampled patch sets.

    Args:
        src_od: OD values of the query patches, shape (N,).
        tgt_od: OD values of the candidate patches, shape (N,).

    Returns:
        Long tensor (N,): for each query patch, the index of the candidate
        patch with the closest OD value.
    """
    dist = torch.abs(src_od.unsqueeze(1) - tgt_od.unsqueeze(0))  # (N, N)
    return dist.argmin(dim=1)


class PECCCorrespondenceLoss(nn.Module, StainLoss):
    """OD-guided hierarchical graph contrastive loss with OD-matched positives.

    Per layer, both ``fake_B`` and ``real_B`` patches get their own OD graph
    (feature cosine x OD kernel) and global graph (feature cosine only);
    features are aggregated over the own-side graph by the shared GNN
    encoders and concatenated (hierarchical). The bidirectional InfoNCE uses
    OD nearest-neighbour positives across the two images instead of
    index-aligned ones.

    Args:
        nc: Embedding dimension (matches ``netF_nc``).
        num_hop: Number of TAGConv propagation hops (``tag`` only).
        nonzero_th: Cosine-similarity threshold for graph binarization.
        od_sigma: Bandwidth of the OD similarity kernel, on the ``[0, 1]``-
            range proper OD scale (much smaller than v2's saturated scale).
        nce_T: InfoNCE temperature.
        conv_type: Graph convolution type: ``tag`` | ``gcn`` | ``gat`` | ``none``.
        graph_mode: Graph-topology ablation mode. ``dual`` uses one
            feature-similarity graph and one OD-calibrated feature graph.
            ``feature_only`` feeds the feature graph to both independent
            encoders, while ``od_only`` feeds the OD-calibrated graph to both.
            Reusing both encoder slots preserves parameter count, output
            dimensionality, and contrastive-logit scale across the ablation.
        pairing: How InfoNCE positives are chosen. ``od_nearest`` (default,
            the v3 design) pairs each patch with the cross-image patch of
            nearest OD value. ``index_softgate`` keeps index-aligned
            candidates but down-weights each query by the OD agreement
            ``exp(-|dOD| / od_sigma)``. ``index_uniform`` is the controlled
            no-gate ablation: it keeps the same index-aligned candidates and
            dual-graph encoders while weighting every query equally.
    """

    def __init__(
        self,
        nc: int = 256,
        num_hop: int = 4,
        nonzero_th: float = 0.6,
        od_sigma: float = 0.1,
        nce_T: float = 0.07,
        conv_type: str = "tag",
        graph_mode: str = "dual",
        pairing: str = "od_nearest",
    ) -> None:
        super().__init__()
        if conv_type not in ("tag", "gcn", "gat", "none"):
            raise NotImplementedError(
                f"conv_type [{conv_type}] is not recognized"
            )
        if pairing not in ("od_nearest", "index_softgate", "index_uniform"):
            raise NotImplementedError(
                f"pairing [{pairing}] is not recognized"
            )
        if graph_mode not in ("dual", "feature_only", "od_only"):
            raise NotImplementedError(
                f"graph_mode [{graph_mode}] is not recognized"
            )
        self.nc = nc
        self.num_hop = num_hop
        self.nonzero_th = nonzero_th
        self.od_sigma = od_sigma
        self.nce_T = nce_T
        self.conv_type = conv_type
        self.graph_mode = graph_mode
        self.pairing = pairing
        self.mlp_init = False

    def _select_branch_adjacencies(
        self,
        adj_od_s: torch.Tensor,
        adj_od_t: torch.Tensor,
        adj_feature_s: torch.Tensor,
        adj_feature_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select topology inputs while retaining both encoder branches."""
        if self.graph_mode == "feature_only":
            return adj_feature_s, adj_feature_t, adj_feature_s, adj_feature_t
        if self.graph_mode == "od_only":
            return adj_od_s, adj_od_t, adj_od_s, adj_od_t
        return adj_od_s, adj_od_t, adj_feature_s, adj_feature_t

    def create_mlp(self, feats: list[torch.Tensor]) -> None:
        """Lazily create per-layer modules once real feature shapes are known."""
        from ..cut_family.networks.cut_networks import init_weights

        device = feats[0].device
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            setattr(self, f"embed_{mlp_id}", Embed(input_nc, self.nc))

            # Create two GNN encoders: OD and global (shared by both sides,
            # mirroring v2 where one encoder pair processes both f_es/f_et).
            if self.conv_type == "tag":
                od_encoder = TAGConvEncoder(self.nc, self.nc, self.num_hop)
                global_encoder = TAGConvEncoder(self.nc, self.nc, self.num_hop)
            elif self.conv_type == "gcn":
                od_encoder = GCNEncoder(self.nc, self.nc)
                global_encoder = GCNEncoder(self.nc, self.nc)
            elif self.conv_type == "gat":
                od_encoder = GATEncoder(self.nc, self.nc // 4, num_heads=4)
                global_encoder = GATEncoder(self.nc, self.nc // 4, num_heads=4)
            else:  # "none"
                od_encoder = None
                global_encoder = None

            setattr(self, f"gnn_od_{mlp_id}", od_encoder)
            setattr(self, f"gnn_global_{mlp_id}", global_encoder)

        init_weights(self, init_type="normal", init_gain=0.02)
        self.to(device)
        self.mlp_init = True

    def _build_adjacency(self, feat: torch.Tensor) -> torch.Tensor:
        """Return ``D^-1/2 (A+I) D^-1/2`` for the binary thresholded graph."""
        sim = feat @ feat.t()  # feat is L2-normalized -> cosine similarity
        adj = (sim > self.nonzero_th).float()
        adj.fill_diagonal_(1.0)
        deg = adj.sum(1).clamp(min=1)
        dinv = deg.pow(-0.5)
        return adj * dinv.unsqueeze(1) * dinv.unsqueeze(0)

    def _build_od_adjacency(
        self, feat: torch.Tensor, od_values: torch.Tensor
    ) -> torch.Tensor:
        """Build adjacency matrix weighted by feature x OD similarity."""
        sim_feat = feat @ feat.t()  # (N, N)
        od_diff = torch.abs(od_values.unsqueeze(0) - od_values.unsqueeze(1))
        sim_od = torch.exp(-od_diff / self.od_sigma)  # (N, N)
        sim = sim_feat * sim_od
        adj = (sim > self.nonzero_th).float()
        adj.fill_diagonal_(1.0)
        deg = adj.sum(1).clamp(min=1)
        dinv = deg.pow(-0.5)
        return adj * dinv.unsqueeze(1) * dinv.unsqueeze(0)

    def _layer_loss(
        self,
        f_es: torch.Tensor,
        f_et: torch.Tensor,
        gnn_od: nn.Module | None,
        gnn_global: nn.Module | None,
        adj_od_s: torch.Tensor,
        adj_od_t: torch.Tensor,
        adj_global_s: torch.Tensor,
        adj_global_t: torch.Tensor,
        s_to_t: torch.Tensor,
        t_to_s: torch.Tensor,
        w_s: torch.Tensor | None = None,
        w_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bidirectional InfoNCE with OD-matched or soft-gated positives.

        Args:
            s_to_t: (N,) fake patch index -> matched real patch index.
            t_to_s: (N,) real patch index -> matched fake patch index.
            w_s: Optional (N,) per-query weights for the source->target CE
                (``index_softgate`` pairing); ``None`` = uniform.
            w_t: Optional (N,) per-query weights for the target->source CE.
        """
        # Aggregate each side over its own graphs (target side detached).
        if gnn_od is None:
            f_gs_od = l2_normalize(f_es)
            f_gt_od = l2_normalize(f_et).detach()
        else:
            f_gs_od = gnn_od(f_es, adj_od_s)
            f_gt_od = gnn_od(f_et, adj_od_t).detach()
        if gnn_global is None:
            f_gs_global = l2_normalize(f_es)
            f_gt_global = l2_normalize(f_et).detach()
        else:
            f_gs_global = gnn_global(f_es, adj_global_s)
            f_gt_global = gnn_global(f_et, adj_global_t).detach()

        f_gs = torch.cat([f_gs_od, f_gs_global], dim=-1)  # (N, 2*nc)
        f_gt = torch.cat([f_gt_od, f_gt_global], dim=-1)  # (N, 2*nc)

        num_patches = f_gs.shape[0]
        arange = torch.arange(num_patches, device=f_gs.device)
        label = torch.zeros(num_patches, dtype=torch.long, device=f_gs.device)

        # Source -> target: query fake i, positive = matched real s_to_t[i].
        pos_gs = (f_gs * f_gt[s_to_t]).sum(dim=-1, keepdim=True)  # (N, 1)
        neg_gs = f_gs @ f_gt.t()  # (N, N)
        neg_gs[arange, s_to_t] = -10.0
        out_gs = torch.cat([pos_gs, neg_gs], dim=1) / self.nce_T
        loss_gs = self._weighted_ce(out_gs.contiguous(), label, w_s)

        # Target -> source: query real j, positive = matched fake t_to_s[j].
        pos_gt = (f_gt * f_gs[t_to_s]).sum(dim=-1, keepdim=True)  # (N, 1)
        neg_gt = f_gt @ f_gs.t()  # (N, N)
        neg_gt[arange, t_to_s] = -10.0
        out_gt = torch.cat([pos_gt, neg_gt], dim=1) / self.nce_T
        loss_gt = self._weighted_ce(out_gt.contiguous(), label, w_t)

        return loss_gs + loss_gt

    @staticmethod
    def _weighted_ce(
        out: torch.Tensor, label: torch.Tensor, w: torch.Tensor | None
    ) -> torch.Tensor:
        """Cross-entropy with optional per-query weights (normalized mean)."""
        ce = F.cross_entropy(out, label, reduction="none")
        if w is None:
            return ce.mean()
        return (ce * w).sum() / w.sum().clamp(min=1e-8)

    def forward(
        self,
        feat_s: list[torch.Tensor],
        feat_t: list[torch.Tensor],
        image_s: torch.Tensor,
        image_t: torch.Tensor,
        patch_ids: list[torch.Tensor],
        feat_map_sizes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Compute OD-guided hierarchical PECC loss with matched positives.

        Args:
            feat_s: Sampled fake_B patch features (one tensor per layer).
            feat_t: Sampled real_B patch features (same sampled positions).
            image_s: Generated image ``fake_B`` (B, 3, H, W) in [-1, 1].
            image_t: Real image ``real_B`` (B, 3, H, W) in [-1, 1].
            patch_ids: Sampled patch indices for each layer.
            feat_map_sizes: (H_feat, W_feat) per layer for OD downsampling.

        Returns:
            Scalar loss.
        """
        if not self.mlp_init:
            self.create_mlp(feat_s)

        # Proper OD maps, computed once per image at full resolution (P3).
        od_s_map = _proper_od_map(image_s)  # (B, H, W)
        od_t_map = _proper_od_map(image_t)

        total = torch.zeros((), device=feat_s[0].device)
        for mlp_id, (f_es, f_et) in enumerate(zip(feat_s, feat_t)):
            embed = getattr(self, f"embed_{mlp_id}")
            gnn_od = getattr(self, f"gnn_od_{mlp_id}")
            gnn_global = getattr(self, f"gnn_global_{mlp_id}")

            f_es = embed(f_es)
            f_et = embed(f_et)

            # OD values at this layer's sampled patch positions.
            H_feat, W_feat = feat_map_sizes[mlp_id]
            ids = patch_ids[mlp_id].clamp(0, H_feat * W_feat - 1)
            od_s = F.interpolate(
                od_s_map.unsqueeze(1), size=(H_feat, W_feat),
                mode="bilinear", align_corners=True,
            ).squeeze(1).flatten(1)[:, ids].squeeze(0)  # (N,)
            od_t = F.interpolate(
                od_t_map.unsqueeze(1), size=(H_feat, W_feat),
                mode="bilinear", align_corners=True,
            ).squeeze(1).flatten(1)[:, ids].squeeze(0)

            # Positive pairing (P0): OD nearest neighbour by default;
            # ``index_softgate`` keeps index-aligned positives and
            # down-weights each pair by its OD agreement instead.
            if self.pairing in ("index_softgate", "index_uniform"):
                arange = torch.arange(od_s.shape[0], device=od_s.device)
                s_to_t = arange
                t_to_s = arange
                if self.pairing == "index_softgate":
                    # Detached: the gate must not become a gradient path,
                    # otherwise the model could lower the loss by
                    # *increasing* the OD mismatch of hard pairs.
                    w = torch.exp(
                        -torch.abs(od_s - od_t) / self.od_sigma
                    ).detach()
                    w_s = w
                    w_t = w
                else:
                    w_s = None
                    w_t = None
            else:
                s_to_t = _od_nearest(od_s, od_t)
                t_to_s = _od_nearest(od_t, od_s)
                w_s = None
                w_t = None

            # Bilateral graphs: each side aggregates over its own graph (P0).
            adj_od_s = self._build_od_adjacency(f_es, od_s)
            adj_od_t = self._build_od_adjacency(f_et.detach(), od_t)
            adj_global_s = self._build_adjacency(f_es)
            adj_global_t = self._build_adjacency(f_et.detach())

            (
                adj_branch_od_s,
                adj_branch_od_t,
                adj_branch_global_s,
                adj_branch_global_t,
            ) = self._select_branch_adjacencies(
                adj_od_s,
                adj_od_t,
                adj_global_s,
                adj_global_t,
            )

            total = total + self._layer_loss(
                f_es, f_et, gnn_od, gnn_global,
                adj_branch_od_s, adj_branch_od_t,
                adj_branch_global_s, adj_branch_global_t,
                s_to_t, t_to_s, w_s, w_t,
            )

        return total / len(feat_s)

    def __call__(
        self,
        feat_s: list[torch.Tensor] | None = None,
        feat_t: list[torch.Tensor] | None = None,
        image_s: torch.Tensor | None = None,
        image_t: torch.Tensor | None = None,
        patch_ids: list[torch.Tensor] | None = None,
        feat_map_sizes: list[tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the loss as a :class:`StainLoss` component."""
        if feat_s is None or feat_t is None:
            raise ValueError("PECCCorrespondenceLoss requires 'feat_s' and 'feat_t'")
        if image_s is None or image_t is None:
            raise ValueError("PECCCorrespondenceLoss requires 'image_s' and 'image_t'")
        if patch_ids is None or feat_map_sizes is None:
            raise ValueError(
                "PECCCorrespondenceLoss requires 'patch_ids' and 'feat_map_sizes'"
            )
        return self.forward(
            feat_s, feat_t, image_s, image_t, patch_ids, feat_map_sizes
        )

    def name(self) -> str:
        return "pecc_correspondence"

    def required_kwargs(self) -> set[str]:
        return {"feat_s", "feat_t", "image_s", "image_t", "patch_ids", "feat_map_sizes"}


class PECCStructureLoss(nn.Module, StainLoss):
    """Permutation-invariant OD graph structure matching loss.

    Builds soft OD graphs (``A = exp(-|dOD| / sigma)``, self-loops included)
    of ``fake_B`` and ``real_B`` over the same randomly sampled pixel
    positions, and matches their structure through statistics that do not
    depend on node ordering:

    1. Sorted degree sequence, normalized by node count (L1).
    2. Sorted normalized-Laplacian spectrum (L1).
    3. Weighted clustering coefficient ``trace(P^3) / N`` with ``P = D^-1 A``
       (L1).

    Because no node index ever enters the comparison, the constraint is
    immune to spatial misalignment; gradients reach ``fake_B`` through its
    OD map, pushing the generated stain structure toward the real one.

    Args:
        od_sigma: Bandwidth of the OD similarity kernel (``[0, 1]``-range
            proper OD scale).
        num_patches: Number of pixel positions sampled per graph.
        lambda_deg: Weight of the degree-sequence term.
        lambda_spec: Weight of the Laplacian-spectrum term.
        lambda_clust: Weight of the clustering-coefficient term.
        scale: Overall loss scale.
    """

    def __init__(
        self,
        od_sigma: float = 0.1,
        num_patches: int = 256,
        lambda_deg: float = 1.0,
        lambda_spec: float = 0.5,
        lambda_clust: float = 0.01,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.od_sigma = od_sigma
        self.num_patches = num_patches
        self.lambda_deg = lambda_deg
        self.lambda_spec = lambda_spec
        self.lambda_clust = lambda_clust
        self.scale = scale

    def _soft_adjacency(self, od_values: torch.Tensor) -> torch.Tensor:
        """Return the soft OD adjacency ``exp(-|dOD| / sigma)`` (B, N, N)."""
        od_diff = torch.abs(od_values.unsqueeze(-1) - od_values.unsqueeze(-2))
        return torch.exp(-od_diff / self.od_sigma)

    def _graph_stats(
        self, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (sorted degrees, sorted Laplacian spectrum, clustering).

        The degree-sequence statistic is normalized by the node count (i.e.
        mean edge weight per node) so the term is resolution-independent and
        does not dominate the other terms by a factor of ``N``. The raw
        (unnormalized) degree is kept for the Laplacian/transition
        normalizations, which require the true row sums.
        """
        num_nodes = adj.shape[1]
        deg_raw = adj.sum(dim=-1)  # (B, N)
        deg_sorted = torch.sort(deg_raw / num_nodes, dim=-1).values

        dinv = deg_raw.clamp(min=1e-8).pow(-0.5)
        norm = adj * dinv.unsqueeze(-1) * dinv.unsqueeze(-2)
        eye = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
        laplacian = eye.unsqueeze(0) - norm
        spectrum = torch.linalg.eigvalsh(laplacian)  # (B, N), ascending

        transition = adj / deg_raw.clamp(min=1e-8).unsqueeze(-1)  # (B, N, N)
        clustering = torch.diagonal(
            transition @ transition @ transition, dim1=-2, dim2=-1
        ).sum(dim=-1) / num_nodes  # (B,)

        return deg_sorted, spectrum, clustering

    def forward(
        self,
        image_s: torch.Tensor,
        image_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the structure matching loss between fake_B and real_B.

        Args:
            image_s: Generated image ``fake_B`` (B, 3, H, W) in [-1, 1].
            image_t: Real image ``real_B`` (B, 3, H, W) in [-1, 1].

        Returns:
            Scalar loss.
        """
        od_s = _proper_od_map(image_s).flatten(1)  # (B, H*W)
        od_t = _proper_od_map(image_t).flatten(1)

        num_pixels = od_s.shape[1]
        num_patches = min(self.num_patches, num_pixels)
        ids = torch.randperm(num_pixels, device=od_s.device)[:num_patches]
        od_s = od_s[:, ids]  # (B, N)
        od_t = od_t[:, ids]

        adj_s = self._soft_adjacency(od_s)
        adj_t = self._soft_adjacency(od_t)

        deg_s, spec_s, clust_s = self._graph_stats(adj_s)
        deg_t, spec_t, clust_t = self._graph_stats(adj_t)

        loss_deg = F.l1_loss(deg_s, deg_t)
        loss_spec = F.l1_loss(spec_s, spec_t)
        loss_clust = F.l1_loss(clust_s, clust_t)

        total = (
            self.lambda_deg * loss_deg
            + self.lambda_spec * loss_spec
            + self.lambda_clust * loss_clust
        )
        return total * self.scale

    def __call__(
        self,
        image_s: torch.Tensor | None = None,
        image_t: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the loss as a :class:`StainLoss` component."""
        if image_s is None or image_t is None:
            raise ValueError("PECCStructureLoss requires 'image_s' and 'image_t'")
        return self.forward(image_s, image_t)

    def name(self) -> str:
        return "pecc_structure"

    def required_kwargs(self) -> set[str]:
        return {"image_s", "image_t"}


LossRegistry.register("pecc_correspondence", PECCCorrespondenceLoss)
LossRegistry.register("pecc_structure", PECCStructureLoss)

LossRegistry.register("od_struct_match", PECCStructureLoss)

__all__ = ["PECCCorrespondenceLoss", "PECCStructureLoss"]
