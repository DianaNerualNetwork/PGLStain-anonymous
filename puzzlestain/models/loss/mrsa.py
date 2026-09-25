"""Marginal Relational Stain Alignment (MRSA).

This module is intentionally separate from the PECC correspondence and
structure losses. It treats tissue-patch optical-density (OD) values as an
unordered set and matches the generated and target empirical distributions by
their one-dimensional Wasserstein-1 distance. No spatial or sample-index
correspondence is assumed.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .base import LossRegistry, StainLoss
from .pecc import _proper_od_map


class MRSAMarginalLoss(nn.Module, StainLoss):
    """Match unordered tissue-patch OD distributions with empirical W1.

    OD is first averaged within non-overlapping patches, using an independently
    estimated tissue mask on each side. Fixed empirical quantiles turn the two
    variable-size patch sets into equally sized, sorted samples. The mean L1
    distance between those quantiles is the one-dimensional Wasserstein-1
    distance approximation used for training.

    Args:
        patch_size: Side length of each non-overlapping OD pooling patch.
        num_quantiles: Number of empirical quantiles used to approximate W1.
        tissue_threshold: A pixel is tissue when its mean RGB intensity in
            ``[0, 1]`` is below this threshold.
        min_tissue_fraction: Minimum tissue fraction for a patch to be used.
        min_tissue_patches: If too few patches pass the threshold, use this
            many patches with the highest tissue fractions as a finite fallback.
    """

    def __init__(
        self,
        patch_size: int = 16,
        num_quantiles: int = 128,
        tissue_threshold: float = 0.95,
        min_tissue_fraction: float = 0.25,
        min_tissue_patches: int = 8,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if num_quantiles <= 0:
            raise ValueError("num_quantiles must be positive")
        if not 0.0 <= tissue_threshold <= 1.0:
            raise ValueError("tissue_threshold must be in [0, 1]")
        if not 0.0 <= min_tissue_fraction <= 1.0:
            raise ValueError("min_tissue_fraction must be in [0, 1]")
        if min_tissue_patches <= 0:
            raise ValueError("min_tissue_patches must be positive")

        self.patch_size = patch_size
        self.num_quantiles = num_quantiles
        self.tissue_threshold = tissue_threshold
        self.min_tissue_fraction = min_tissue_fraction
        self.min_tissue_patches = min_tissue_patches

    def _patch_od_sets(
        self, image: torch.Tensor, od_map: torch.Tensor
    ) -> list[torch.Tensor]:
        """Return one tissue-filtered, unordered patch-OD set per image."""
        height, width = image.shape[-2:]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"image size {(height, width)} must be at least patch_size "
                f"{self.patch_size}"
            )

        # The hard mask is deliberately detached: it selects tissue but does
        # not provide a shortcut gradient through the brightness threshold.
        image_unit = ((image.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
        tissue = (image_unit.mean(dim=1, keepdim=True) < self.tissue_threshold).to(
            dtype=od_map.dtype
        )

        tissue_fraction = F.avg_pool2d(
            tissue, kernel_size=self.patch_size, stride=self.patch_size
        )
        tissue_weighted_od = F.avg_pool2d(
            od_map.unsqueeze(1) * tissue,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patch_od = tissue_weighted_od / tissue_fraction.clamp_min(1e-6)

        sets: list[torch.Tensor] = []
        for batch_idx in range(image.shape[0]):
            fractions = tissue_fraction[batch_idx, 0].flatten()
            values = patch_od[batch_idx, 0].flatten()
            keep = fractions >= self.min_tissue_fraction
            if int(keep.sum()) < self.min_tissue_patches:
                fallback_count = min(self.min_tissue_patches, fractions.numel())
                keep_ids = torch.topk(fractions, fallback_count).indices
                selected = values[keep_ids]
            else:
                selected = values[keep]
            sets.append(selected)
        return sets

    def _wasserstein_1d(
        self, source_values: torch.Tensor, target_values: torch.Tensor
    ) -> torch.Tensor:
        """Approximate empirical one-dimensional W1 with fixed quantiles."""
        quantiles = (
            torch.arange(
                self.num_quantiles,
                device=source_values.device,
                dtype=torch.float32,
            )
            + 0.5
        ) / self.num_quantiles
        source_quantiles = torch.quantile(source_values.float(), quantiles)
        target_quantiles = torch.quantile(target_values.float(), quantiles)
        return F.l1_loss(source_quantiles, target_quantiles)

    def forward(self, image_s: torch.Tensor, image_t: torch.Tensor) -> torch.Tensor:
        """Return mean OD-set W1; gradients flow only through ``image_s``."""
        if image_s.ndim != 4 or image_t.ndim != 4:
            raise ValueError("image_s and image_t must have shape (B, C, H, W)")
        if image_s.shape[0] != image_t.shape[0]:
            raise ValueError("image_s and image_t must have the same batch size")
        if image_s.shape[1] != 3 or image_t.shape[1] != 3:
            raise ValueError("OD-set matching expects three-channel RGB images")

        source_od = _proper_od_map(image_s)
        # The same-case real target is an anchor, never an optimization target.
        target_image = image_t.detach()
        target_od = _proper_od_map(target_image)
        source_sets = self._patch_od_sets(image_s, source_od)
        target_sets = self._patch_od_sets(target_image, target_od)
        losses = [
            self._wasserstein_1d(source, target)
            for source, target in zip(source_sets, target_sets)
        ]
        return torch.stack(losses).mean()

    def name(self) -> str:
        """Return the registry/logging key."""
        return "mrsa_marginal"

    def required_kwargs(self) -> set[str]:
        """Declare image inputs consumed by this component."""
        return {"image_s", "image_t"}

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        """Support the common keyword-only :class:`StainLoss` interface."""
        return super().__call__(**kwargs)


class MRSARelationalLoss(nn.Module, StainLoss):
    """Match unordered soft OD graphs with detached-coupling entropic UGW.

    Each tissue patch becomes a node with two continuous expression features:
    mean focal DAB OD and soft positive-pixel fraction. Edges are soft RBF
    affinities in that feature space. The source and target graphs are built
    independently and may contain different numbers of nodes.

    Args:
        patch_size: Non-overlapping patch size for node statistics.
        max_nodes: Maximum tissue nodes retained per graph.
        min_nodes: Fallback number of highest-tissue patches.
        tissue_threshold: Mean RGB threshold in [0, 1] for tissue pixels.
        min_tissue_fraction: Minimum tissue fraction for a valid patch.
        positive_od_threshold: Center of the soft DAB-positive indicator.
        positive_temperature: Temperature of the soft positive indicator.
        od_sigma: RBF scale for patch mean OD.
        positive_sigma: RBF scale for positive fraction.
        sinkhorn_epsilon: Entropic regularization in the UGW coupling solver.
        mass_penalty: Marginal-relaxation strength; larger approaches balanced.
        balanced: If ``True``, enforce the prescribed node-mass marginals with
            balanced Sinkhorn scaling. The default ``False`` preserves UGW.
        outer_iterations: Number of GW linearization/coupling updates.
        sinkhorn_iterations: Scaling iterations per UGW update.
    """

    def __init__(
        self,
        patch_size: int = 32,
        max_nodes: int = 64,
        min_nodes: int = 8,
        tissue_threshold: float = 0.95,
        min_tissue_fraction: float = 0.25,
        positive_od_threshold: float = 0.02,
        positive_temperature: float = 0.01,
        od_sigma: float = 0.1,
        positive_sigma: float = 0.25,
        sinkhorn_epsilon: float = 0.05,
        mass_penalty: float = 1.0,
        balanced: bool = False,
        outer_iterations: int = 6,
        sinkhorn_iterations: int = 30,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if max_nodes <= 0 or min_nodes <= 0 or min_nodes > max_nodes:
            raise ValueError("require 0 < min_nodes <= max_nodes")
        if not 0.0 <= tissue_threshold <= 1.0:
            raise ValueError("tissue_threshold must be in [0, 1]")
        if not 0.0 <= min_tissue_fraction <= 1.0:
            raise ValueError("min_tissue_fraction must be in [0, 1]")
        for name, value in {
            "positive_temperature": positive_temperature,
            "od_sigma": od_sigma,
            "positive_sigma": positive_sigma,
            "sinkhorn_epsilon": sinkhorn_epsilon,
            "mass_penalty": mass_penalty,
        }.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if outer_iterations <= 0 or sinkhorn_iterations <= 0:
            raise ValueError("solver iteration counts must be positive")

        self.patch_size = patch_size
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.tissue_threshold = tissue_threshold
        self.min_tissue_fraction = min_tissue_fraction
        self.positive_od_threshold = positive_od_threshold
        self.positive_temperature = positive_temperature
        self.od_sigma = od_sigma
        self.positive_sigma = positive_sigma
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.mass_penalty = mass_penalty
        self.balanced = balanced
        self.outer_iterations = outer_iterations
        self.sinkhorn_iterations = sinkhorn_iterations

    def _extract_nodes(
        self, image: torch.Tensor, od_map: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return per-image node features and normalized node masses."""
        height, width = image.shape[-2:]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"image size {(height, width)} must be at least patch_size "
                f"{self.patch_size}"
            )

        image_unit = ((image.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
        tissue = (image_unit.mean(dim=1, keepdim=True) < self.tissue_threshold).to(
            od_map.dtype
        )
        positive_probability = torch.sigmoid(
            (od_map - self.positive_od_threshold) / self.positive_temperature
        ).unsqueeze(1)

        tissue_fraction = F.avg_pool2d(
            tissue, kernel_size=self.patch_size, stride=self.patch_size
        )
        patch_mean_od = F.avg_pool2d(
            od_map.unsqueeze(1) * tissue,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ) / tissue_fraction.clamp_min(1e-6)
        patch_positive_fraction = F.avg_pool2d(
            positive_probability * tissue,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ) / tissue_fraction.clamp_min(1e-6)

        graphs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_idx in range(image.shape[0]):
            fractions = tissue_fraction[batch_idx, 0].flatten()
            mean_od = patch_mean_od[batch_idx, 0].flatten()
            positive_fraction = patch_positive_fraction[batch_idx, 0].flatten()
            valid_ids = torch.nonzero(
                fractions >= self.min_tissue_fraction, as_tuple=False
            ).flatten()

            if valid_ids.numel() < self.min_nodes:
                count = min(self.min_nodes, fractions.numel())
                node_ids = torch.topk(fractions, count).indices
            elif valid_ids.numel() > self.max_nodes:
                local_ids = torch.topk(fractions[valid_ids], self.max_nodes).indices
                node_ids = valid_ids[local_ids]
            else:
                node_ids = valid_ids

            features = torch.stack(
                (mean_od[node_ids], positive_fraction[node_ids]), dim=1
            )
            masses = fractions[node_ids].clamp_min(1e-6)
            masses = masses / masses.sum()
            graphs.append((features, masses))
        return graphs

    def _soft_adjacency(self, features: torch.Tensor) -> torch.Tensor:
        """Build the fully differentiable soft OD-relation adjacency."""
        scales = features.new_tensor([self.od_sigma, self.positive_sigma])
        differences = (features[:, None, :] - features[None, :, :]) / scales
        squared_distance = differences.square().sum(dim=-1)
        return torch.exp(-0.5 * squared_distance)

    @staticmethod
    def _gw_cost_matrix(
        source_graph: torch.Tensor,
        target_graph: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        """Linearized squared-loss GW cost for a fixed coupling."""
        source_mass = coupling.sum(dim=1)
        target_mass = coupling.sum(dim=0)
        source_term = source_graph.square() @ source_mass
        target_term = target_graph.square() @ target_mass
        cross_term = source_graph @ coupling @ target_graph.transpose(0, 1)
        return (
            source_term[:, None] + target_term[None, :] - 2.0 * cross_term
        ).clamp_min(0.0)

    def _unbalanced_sinkhorn(
        self,
        cost: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Solve one entropic unbalanced OT subproblem by scaling."""
        reference = source_mass[:, None] * target_mass[None, :]
        kernel = reference * torch.exp(-cost / self.sinkhorn_epsilon).clamp_min(1e-30)
        exponent = (
            1.0
            if self.balanced
            else self.mass_penalty / (self.mass_penalty + self.sinkhorn_epsilon)
        )
        source_scale = torch.ones_like(source_mass)
        target_scale = torch.ones_like(target_mass)
        for _ in range(self.sinkhorn_iterations):
            source_ratio = source_mass / (kernel @ target_scale).clamp_min(1e-12)
            source_scale = source_ratio.clamp(1e-8, 1e8).pow(exponent)
            target_ratio = target_mass / (
                kernel.transpose(0, 1) @ source_scale
            ).clamp_min(1e-12)
            target_scale = target_ratio.clamp(1e-8, 1e8).pow(exponent)
        return (source_scale[:, None] * kernel * target_scale[None, :]).clamp_min(1e-30)

    def _solve_coupling(
        self,
        source_graph: torch.Tensor,
        target_graph: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate a detached entropic UGW coupling."""
        with torch.no_grad():
            source_graph = source_graph.detach()
            target_graph = target_graph.detach()
            source_mass = source_mass.detach()
            target_mass = target_mass.detach()
            coupling = source_mass[:, None] * target_mass[None, :]
            for _ in range(self.outer_iterations):
                cost = self._gw_cost_matrix(source_graph, target_graph, coupling)
                coupling = self._unbalanced_sinkhorn(cost, source_mass, target_mass)
            return coupling.detach()

    def forward(self, image_s: torch.Tensor, image_t: torch.Tensor) -> torch.Tensor:
        """Return fixed-coupling UGW structural loss for a batch."""
        if image_s.ndim != 4 or image_t.ndim != 4:
            raise ValueError("image_s and image_t must have shape (B, C, H, W)")
        if image_s.shape[0] != image_t.shape[0]:
            raise ValueError("image_s and image_t must have the same batch size")
        if image_s.shape[1] != 3 or image_t.shape[1] != 3:
            raise ValueError("OD graph matching expects three-channel RGB images")

        source_od = _proper_od_map(image_s)
        target_image = image_t.detach()
        target_od = _proper_od_map(target_image)
        source_graphs = self._extract_nodes(image_s, source_od)
        target_graphs = self._extract_nodes(target_image, target_od)

        batch_losses = []
        for (source_features, source_mass), (
            target_features,
            target_mass,
        ) in zip(source_graphs, target_graphs):
            source_adjacency = self._soft_adjacency(source_features)
            target_adjacency = self._soft_adjacency(target_features).detach()
            coupling = self._solve_coupling(
                source_adjacency,
                target_adjacency,
                source_mass,
                target_mass,
            )
            structural_cost = self._gw_cost_matrix(
                source_adjacency, target_adjacency, coupling
            )
            transported_mass = coupling.sum().clamp_min(1e-8)
            batch_losses.append(
                (structural_cost * coupling).sum() / transported_mass.square()
            )
        return torch.stack(batch_losses).mean()

    def name(self) -> str:
        """Return the registry/logging key."""
        return "mrsa_relational"

    def required_kwargs(self) -> set[str]:
        """Declare image inputs consumed by this component."""
        return {"image_s", "image_t"}

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        """Support the common keyword-only StainLoss interface."""
        return super().__call__(**kwargs)



LossRegistry.register("mrsa_marginal", MRSAMarginalLoss)
LossRegistry.register("mrsa_relational", MRSARelationalLoss)

# Compatibility aliases for checkpoints using development loss identifiers.
LossRegistry.register("od_set_wasserstein", MRSAMarginalLoss)
LossRegistry.register("od_graph_ugw", MRSARelationalLoss)

__all__ = ["MRSAMarginalLoss", "MRSARelationalLoss"]
