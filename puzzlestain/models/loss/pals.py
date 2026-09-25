"""PALS (Protein-Aware Learning Strategy) loss for PSPStain.

The loss operates in the optical-density (OD) space of DAB-stained images and
produces a multi-level pathology-aware reconstruction term plus pseudo masks
that PCLS uses for cross-image prototype consistency.

Original source: https://github.com/ccitachi/PSPStain
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Any

import math

import torch
import torch.nn as nn

from .base import LossRegistry, StainLoss


class MLPALoss(nn.Module, StainLoss):
    """Multi-Level Protein Awareness loss.

    Expects images in the ``[-1, 1]`` range (the PuzzleStain/CUT default) and
    consumes them directly, mirroring the original PALS.  Negative values are
    clamped to ``1e-6`` inside :meth:`_separate_stains` before the log.

    Args:
        alpha: Focal optical density exponent.
        thresh_FOD: Threshold below which FOD values are zeroed.
        thresh_mask: Threshold used to build the binary pseudo mask.
        num_bins: Number of histogram bins for the histo-level term.
        num_blocks: Number of spatial blocks for the block-level term.
    """

    def __init__(
        self,
        alpha: float = 1.8,
        thresh_FOD: float = 0.15,
        thresh_mask: float = 0.68,
        num_bins: int = 20,
        num_blocks: int = 16,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.thresh_FOD = thresh_FOD
        self.thresh_mask = thresh_mask
        self.num_bins = num_bins
        self.num_blocks = num_blocks

        # H&E / DAB stain separation matrices (Ruifrok-Johnston).
        self.register_buffer(
            "rgb_from_hed",
            torch.tensor(
                [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]]
            ),
        )
        self.register_buffer(
            "coeffs",
            torch.tensor([0.2125, 0.7154, 0.0721]).view(3, 1),
        )
        self.register_buffer(
            "hed_from_rgb", torch.linalg.inv(self.rgb_from_hed)
        )
        self.register_buffer(
            "adjust_Calibration",
            torch.tensor(10 ** (-(math.e) ** (1 / alpha))),
        )
        self.register_buffer("log_adjust", torch.log(torch.tensor(1e-6)))

        self.mse_loss = nn.MSELoss()
        self.mse_loss_2 = nn.MSELoss(reduction="none")

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute MLPA loss and pseudo masks.

        Args:
            inputs: Generated image tensor, shape ``(B, 3, H, W)``, range ``[-1, 1]``.
            targets: Target image tensor, same shape and range.

        Returns:
            ``(loss_MLPA, mask_inputs, mask_targets)``.  Masks have shape
            ``(B, H, W)`` and values ``{0, 1}``.
        """
        # The original PALS feeds the raw [-1, 1] tensors directly into the
        # optical-density math; _separate_stains clamps negatives to 1e-6.
        inputs_reshape = inputs.permute(0, 2, 3, 1)
        targets_reshape = targets.permute(0, 2, 3, 1)

        inputs_OD, input_block, input_histo, input_mask = self._compute_OD(
            inputs_reshape
        )
        targets_OD, target_block, target_histo, target_mask = self._compute_OD(
            targets_reshape
        )

        area = inputs.shape[2] * inputs.shape[3]

        MLPA_avg = self.mse_loss_2(inputs_OD, targets_OD) / (area**2)
        MLPA_histo = (
            (
                (input_histo / area - target_histo / area) ** 2
            ).sum(1)
        ) / inputs.shape[0]
        MLPA_block = self.mse_loss(
            input_block / (area / self.num_blocks),
            target_block / (area / self.num_blocks),
        )

        cond = (inputs_OD - targets_OD >= targets_OD * -0.4) & (
            inputs_OD - targets_OD <= targets_OD * 0.4
        )
        loss_MLPA = torch.sum(
            torch.where(cond, MLPA_histo, MLPA_avg + MLPA_histo)
        )
        loss_MLPA = loss_MLPA + MLPA_block

        return loss_MLPA, input_mask, target_mask

    def _compute_OD(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute OD features, histogram, block sums and pseudo mask."""
        assert image.shape[-1] == 3

        ihc_hed = self._separate_stains(image, self.hed_from_rgb)
        null = torch.zeros_like(ihc_hed[:, :, :, 0])
        ihc_d = self._combine_stains(
            torch.stack((null, null, ihc_hed[:, :, :, 2]), dim=-1),
            self.rgb_from_hed,
        )
        grey_d = self._rgb2gray(ihc_d)
        grey_d = torch.clamp(grey_d, 0.0, 1.0)

        FOD = torch.log10(1.0 / (grey_d + self.adjust_Calibration))
        FOD = torch.clamp(FOD, min=0.0)
        FOD = FOD**self.alpha
        FOD_relu = torch.where(
            FOD < self.thresh_FOD,
            torch.zeros_like(FOD),
            FOD,
        )

        mask_OD = torch.where(
            FOD < self.thresh_mask,
            torch.zeros_like(FOD),
            FOD,
        )
        mask_OD = mask_OD.squeeze(-1).detach()
        mask_OD = (mask_OD > 0).float()

        flattened_img_2 = FOD.squeeze(-1).flatten(1, 2)

        avg = torch.sum(FOD_relu, dim=(1, 2, 3))

        block_size = int(math.sqrt(self.num_blocks))
        tensor_blocks = FOD_relu.squeeze(-1).unfold(
            1, image.shape[1] // block_size, image.shape[1] // block_size
        ).unfold(
            2, image.shape[2] // block_size, image.shape[2] // block_size
        )
        block = tensor_blocks.sum(dim=(3, 4))

        histo = self._calculate_histo_sums(
            flattened_img_2, self.num_bins, 0.0, math.e
        )

        return avg, block, histo, mask_OD

    def _separate_stains(
        self, rgb: torch.Tensor, conv_matrix: torch.Tensor, *, channel_axis: int = -1
    ) -> torch.Tensor:
        rgb = torch.clamp(rgb, min=1e-6)
        stains = torch.matmul(torch.log(rgb) / self.log_adjust, conv_matrix)
        stains = torch.maximum(stains, torch.zeros_like(stains))
        return stains

    def _combine_stains(
        self, stains: torch.Tensor, conv_matrix: torch.Tensor, *, channel_axis: int = -1
    ) -> torch.Tensor:
        log_rgb = -torch.matmul((stains * -self.log_adjust), conv_matrix)
        rgb = torch.exp(log_rgb)
        return torch.clamp(rgb, min=0.0, max=1.0)

    def _rgb2gray(self, rgb: torch.Tensor, *, channel_axis: int = -1) -> torch.Tensor:
        return torch.matmul(rgb, self.coeffs)

    def _calculate_histo_sums(
        self, features: torch.Tensor, num_histos: int, min_val: float, max_val: float
    ) -> torch.Tensor:
        bucket_width = (max_val - min_val) / num_histos
        normalized = (features - min_val) / bucket_width
        histo_indices = normalized.clamp(0, num_histos - 1).long()

        batch_sums = torch.zeros(
            features.shape[0], num_histos, device=features.device
        )
        for i in range(features.shape[0]):
            for j in range(num_histos):
                indices_in_histo = (histo_indices[i] == j).nonzero(as_tuple=True)[0]
                if indices_in_histo.numel() > 0:
                    batch_sums[i, j] = features[i, indices_in_histo].sum()
        return batch_sums

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the scalar MLPA loss.

        For mask access use :meth:`forward` directly.
        """
        if pred is None or target is None:
            raise ValueError("MLPALoss requires 'pred' and 'target' kwargs")
        loss, _, _ = self.forward(pred, target)
        return loss

    def name(self) -> str:
        return "mlpa"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("mlpa", MLPALoss)

__all__ = ["MLPALoss"]
