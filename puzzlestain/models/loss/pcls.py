"""Combined pathology loss for PSPStain: MLPA + CTPC.

This module wraps the pretrained PCLS segmentation network and exposes a single
StainLoss that combines the Multi-Level Protein Awareness (MLPA) loss with the
Cross-image Tumor Prototype Consistency (CTPC) loss.  It is intended to be used
inside :class:`CompositeLoss` under the registered name ``"psp_pathology"``.

The pretrained ``UNet_pro`` checkpoint(s) are third-party assets that are not
shipped with PuzzleStain; their existence is validated at construction and a
missing checkpoint raises :class:`FileNotFoundError` pointing to the official
download source (PSPStain: single net, PGVMS: one net per stain).

Original sources:
    https://github.com/ccitachi/PSPStain
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..cut_family.networks.pcls import UNetPro
from .base import LossRegistry, StainLoss
from .pals import MLPALoss

#: Official download source of the pretrained UNet_pro checkpoint (PSPStain).
_PSPSTAIN_PRETRAIN_URL = "https://github.com/ccitachi/PSPStain/tree/main/pretrain"
#: Official download source of the stain-specific UNet_pro checkpoints (PGVMS).
_PGVMS_PRETRAIN_URL = (
    "https://drive.google.com/drive/folders/1ekuPcvVLlX0D0IQ-OIHdh5L3zwnU-s4l"
)


def _ctpc_loss(
    input_logits: torch.Tensor,
    target_logits: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Cross-prototype consistency loss used by CTPC."""
    target_logits = target_logits.clone()
    target_logits = target_logits.view(
        target_logits.size(0), target_logits.size(1), -1
    )
    target_logits = target_logits.transpose(1, 2)  # [N, HW, C]
    return criterion(
        input_logits.transpose(1, 2),
        target_logits.transpose(1, 2).squeeze(1),
    )


class PSPStainPathologyLoss(nn.Module, StainLoss):
    """Combined MLPA + CTPC loss for PSPStain.

    Supports two segmentation-network configurations:

    - Single net (PSPStain): one shared ``UNet_pro`` for every sample.
    - Multi net (PGVMS): one stain-specific ``UNet_pro`` per stain, routed by
      the ``label`` argument of :meth:`forward`.

    Args:
        seg_pretrained_path: Path to the pretrained ``UNet_pro`` checkpoint.
            Mutually exclusive with ``seg_pretrained_paths``.
        lambda_CTPC: Weight for the CTPC term.
        alpha, thresh_FOD, thresh_mask, num_bins, num_blocks: MLPA hyperparameters.
        seg_pretrained_paths: Optional ``{stain_name: checkpoint_path}`` map
            loading one segmentation net per stain (PGVMS). Mutually exclusive
            with ``seg_pretrained_path``.
    """

    #: Key under which the single shared segmentation net is stored.
    _DEFAULT_SEG_KEY = "default"

    def __init__(
        self,
        seg_pretrained_path: str | None = None,
        lambda_CTPC: float = 2.5,
        alpha: float = 1.8,
        thresh_FOD: float = 0.15,
        thresh_mask: float = 0.68,
        num_bins: int = 20,
        num_blocks: int = 16,
        seg_pretrained_paths: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if (seg_pretrained_path is None) == (seg_pretrained_paths is None):
            raise ValueError(
                "Exactly one of 'seg_pretrained_path' or 'seg_pretrained_paths' "
                "must be provided"
            )
        self.lambda_CTPC = lambda_CTPC
        self.mlpa = MLPALoss(
            alpha=alpha,
            thresh_FOD=thresh_FOD,
            thresh_mask=thresh_mask,
            num_bins=num_bins,
            num_blocks=num_blocks,
        )
        paths = (
            {self._DEFAULT_SEG_KEY: seg_pretrained_path}
            if seg_pretrained_path is not None
            else dict(seg_pretrained_paths)
        )
        missing = {key: path for key, path in paths.items() if not os.path.isfile(path)}
        if missing:
            if seg_pretrained_path is not None:
                hint = (
                    "Download it from the PSPStain authors: "
                    f"{_PSPSTAIN_PRETRAIN_URL} (e.g. MIST_unet_seg.pth), then "
                    "point loss.components.psp_pathology.seg_pretrained_path at it."
                )
            else:
                hint = (
                    "Download them from the PGVMS authors: "
                    f"{_PGVMS_PRETRAIN_URL}, then set "
                    "loss.components.psp_pathology.seg_pretrained_paths.<STAIN> "
                    "accordingly."
                )
            raise FileNotFoundError(
                "Pretrained UNet_pro segmentation checkpoint(s) required by the "
                f"'psp_pathology' loss were not found: {missing}. {hint}"
            )
        self.netSegs = nn.ModuleDict()
        for key, path in paths.items():
            net = UNetPro(in_chns=3, class_num=2)
            state = torch.load(path, map_location="cpu")
            net.load_state_dict(state)
            # Mirroring the original PCLS: netSeg stays in train mode (batch-norm
            # statistics and dropout remain active); only its parameters are frozen.
            for param in net.parameters():
                param.requires_grad = False
            self.netSegs[key] = net
        self.ce_loss = nn.CrossEntropyLoss()

    @property
    def netSeg(self) -> UNetPro:
        """The shared segmentation net (single-net configuration only)."""
        return self.netSegs[self._DEFAULT_SEG_KEY]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        label: str | None = None,
    ) -> torch.Tensor:
        """Compute ``MLPA + lambda_CTPC * CTPC``.

        Args:
            pred: Generated image tensor, shape ``(B, 3, H, W)``, range ``[-1, 1]``.
            target: Target image tensor, same shape and range.
            label: Stain name selecting the stain-specific segmentation net.
                Required when built with ``seg_pretrained_paths``; must be
                ``None`` (or omitted) for the single-net configuration.

        Returns:
            Scalar combined loss.
        """
        seg_key = self._resolve_seg_key(label)
        loss_MLPA, mask_A, mask_B = self.mlpa.forward(pred, target)
        loss_CTPC = self._compute_ctpc(pred, target, mask_A, mask_B, seg_key)
        return loss_MLPA + self.lambda_CTPC * loss_CTPC

    def _resolve_seg_key(self, label: str | None) -> str:
        """Map ``label`` to the segmentation-net key, validating the combination."""
        if self._DEFAULT_SEG_KEY in self.netSegs:
            if label is not None:
                raise ValueError(
                    "Single-net PSPStainPathologyLoss does not take a 'label'"
                )
            return self._DEFAULT_SEG_KEY
        if label is None:
            raise ValueError(
                "Multi-net PSPStainPathologyLoss requires a stain 'label'; "
                f"available: {list(self.netSegs.keys())}"
            )
        if label not in self.netSegs:
            raise ValueError(
                f"Unknown stain label {label!r}; available: {list(self.netSegs.keys())}"
            )
        return label

    def _compute_ctpc(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_A: torch.Tensor,
        mask_B: torch.Tensor,
        seg_key: str,
    ) -> torch.Tensor:
        """Compute cross-image tumor prototype consistency loss.

        Mirroring the original PCLS, this runs with autograd enabled so the
        CTPC gradients flow back to ``pred`` (``fake_B``); the seg net itself
        contributes no gradients because its parameters are frozen.
        """
        net_seg = self.netSegs[seg_key]
        batch_size = pred.shape[0]
        loss_CTPC = torch.tensor(0.0, device=pred.device)
        for i in range(batch_size):
            image_dual = torch.cat(
                (pred[i].unsqueeze(0), target[i].unsqueeze(0)), dim=0
            )
            m_a = mask_A[i].unsqueeze(0).unsqueeze(1).long()
            m_b = mask_B[i].unsqueeze(0).unsqueeze(1).long()

            _, crossproout = net_seg(image_dual)
            ctpc = _ctpc_loss(
                crossproout,
                torch.cat((m_a, m_b), dim=0),
                self.ce_loss,
            )
            loss_CTPC = loss_CTPC + torch.mean(ctpc)
        return loss_CTPC / batch_size

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if pred is None or target is None:
            raise ValueError(
                "PSPStainPathologyLoss requires 'pred' and 'target' kwargs"
            )
        return self.forward(pred, target, label=kwargs.get("label"))

    def name(self) -> str:
        return "psp_pathology"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("psp_pathology", PSPStainPathologyLoss)

__all__ = ["PSPStainPathologyLoss", "_ctpc_loss"]
