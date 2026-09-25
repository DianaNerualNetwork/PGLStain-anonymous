"""PCSM (Pathological Correspondence Self-Mining) loss for USIGAN.

This loss compares focal optical-density (OD) maps of the generated and target
images.  It is adapted from the USIGAN pathological consistency module.

Reference:
    https://github.com/MIXAILAB/USIGAN
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


class PCSMLoss(nn.Module, StainLoss):
    """Pathological correspondence self-mining loss.

    Expects images in the ``[-1, 1]`` range. For parity with the reference
    implementation (USIGAN ``PCM_loss.py``), the tanh-range tensors are fed
    directly into the OD computation — negative channels are clamped to
    ``1e-6`` inside :meth:`_separate_stains`, which the reference results were
    trained with. Do NOT rescale to ``[0, 1]`` first: that neutralizes the
    pathology prior (FOD maps collapse below ``thresh_FOD``).

    Args:
        alpha: Focal OD exponent.
        thresh_FOD: Threshold below which FOD values are zeroed.
        thresh_mask: Threshold for the pseudo mask (unused in the loss term but
            kept for parity with the reference).
        scale: Overall loss scale.
    """

    def __init__(
        self,
        alpha: float = 1.8,
        thresh_FOD: float = 0.15,
        thresh_mask: float = 0.68,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.thresh_FOD = thresh_FOD
        self.thresh_mask = thresh_mask
        self.scale = scale

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
        self.register_buffer("hed_from_rgb", torch.linalg.inv(self.rgb_from_hed))
        self.register_buffer(
            "adjust_Calibration",
            torch.tensor(10 ** (-(math.e) ** (1 / alpha))),
        )
        self.register_buffer("log_adjust", torch.log(torch.tensor(1e-6)))
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute OD-map L1 + average-OD MSE loss.

        Args:
            pred: Generated image ``(B, 3, H, W)`` in ``[-1, 1]``.
            target: Target image, same shape.
        """
        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        pred_od, pred_avg, _ = self._compute_od(pred)
        target_od, target_avg, _ = self._compute_od(target)

        matrix_pred = self._cal_matrix(pred_od)
        matrix_target = self._cal_matrix(target_od)

        loss = F.l1_loss(matrix_pred, matrix_target)
        area = pred.shape[1] * pred.shape[2]
        loss = loss + self.mse_loss(pred_avg, target_avg) / (area**2)
        return loss * self.scale

    def _compute_od(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(flattened_OD, avg_OD, mask)``."""
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
        mask = torch.where(
            FOD < self.thresh_mask,
            torch.zeros_like(FOD),
            FOD,
        )
        mask = mask.squeeze(-1).detach()
        mask = (mask > 0).float()

        flattened = FOD.flatten(1, 2)
        avg = torch.sum(FOD_relu, dim=(1, 2, 3))
        return flattened, avg, mask

    def _cal_matrix(self, feats: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine-similarity matrix across the batch."""
        batch_size = feats.size(0)
        matrix = torch.zeros(batch_size, batch_size, device=feats.device)
        for i in range(batch_size):
            for j in range(batch_size):
                matrix[i, j] = F.cosine_similarity(
                    feats[i].view(-1), feats[j].view(-1), dim=0
                )
        return matrix

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

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if pred is None or target is None:
            raise ValueError("PCSMLoss requires 'pred' and 'target' kwargs")
        return self.forward(pred, target)

    def name(self) -> str:
        return "pcsm"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("pcsm", PCSMLoss)

__all__ = ["PCSMLoss"]
