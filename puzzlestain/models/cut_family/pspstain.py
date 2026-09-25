"""PSPStain model and strategy for the CUT family.

PSPStain extends CPT (CUT + Gaussian-pyramid reconstruction) with two pathology-
aware generator losses:

- Multi-Level Protein Awareness (MLPA) loss plus Cross-image Tumor Prototype
  Consistency (CTPC) loss, combined in :class:`PSPStainPathologyLoss`.
- Multi-Scale SSIM (MS-SSIM) loss.

Reference implementations:
    https://github.com/ccitachi/PSPStain/models/PALS.py
    https://github.com/ccitachi/PSPStain/models/PCLS.py
    https://github.com/ccitachi/PSPStain/models/networks.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..loss.base import CompositeLoss, StainLoss
from ..registry import ModelRegistry
from .base import BaseCUTProcessor
from .cpt import CPTModel, CPTStrategy

if TYPE_CHECKING:
    from ...configs.schema import PuzzleStainConfig


class PSPStainModel(CPTModel):
    """PSPStain: CPT + pathology-aware losses.

    Args:
        lambda_ssim: Weight for the MS-SSIM term.
        lambda_pathology: Weight for the combined MLPA + CTPC pathology term.
        lambda_CTPC: Inner weight for CTPC inside the pathology loss.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 32,
        netG: str = "resnet_6blocks",
        netD: str = "n_layers",
        netF: str = "mlp_sample",
        n_layers_D: int = 5,
        normG: str = "instance",
        normD: str = "instance",
        init_type: str = "xavier",
        init_gain: float = 0.02,
        no_dropout: bool = True,
        no_antialias: bool = False,
        no_antialias_up: bool = False,
        nce_idt: bool = True,
        lambda_GAN: float = 1.0,
        lambda_NCE: float = 1.0,
        lambda_gp: float = 10.0,
        gp_weights: list[float] | str = "uniform",
        lambda_ssim: float = 0.05,
        lambda_pathology: float = 1.0,
        lambda_CTPC: float = 2.5,
        nce_layers: str = "0,4,8,12,16",
        nce_T: float = 0.07,
        num_patches: int = 256,
        netF_nc: int = 256,
        flip_equivariance: bool = False,
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "spectral",
        d_lr_factor: float = 20.0,
    ) -> None:
        self.lambda_ssim = lambda_ssim
        self.lambda_pathology = lambda_pathology
        self.lambda_CTPC = lambda_CTPC
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            ndf=ndf,
            netG=netG,
            netD=netD,
            netF=netF,
            n_layers_D=n_layers_D,
            normG=normG,
            normD=normD,
            init_type=init_type,
            init_gain=init_gain,
            no_dropout=no_dropout,
            no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
            nce_idt=nce_idt,
            lambda_GAN=lambda_GAN,
            lambda_NCE=lambda_NCE,
            nce_layers=nce_layers,
            nce_T=nce_T,
            num_patches=num_patches,
            netF_nc=netF_nc,
            flip_equivariance=flip_equivariance,
            lambda_gp=lambda_gp,
            gp_weights=gp_weights,
            crop_size=crop_size,
            gpu_ids=gpu_ids,
            no_d=no_d,
            weight_norm=weight_norm,
            d_lr_factor=d_lr_factor,
        )
        # Original PSPStain trains the discriminator with lr/20
        # (PSPStain_model.py: optimizer_D = Adam(..., lr=opt.lr/20)).


class PSPStainStrategy(CPTStrategy):
    """Training strategy for PSPStain.

    Extends the CPT strategy by adding MS-SSIM and the combined MLPA+CTPC
    pathology loss to the generator loss.  Both new losses expect ``pred`` and
    ``target`` images, so they are looked up from the configured composite loss.
    """

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return CPT losses plus ``loss_SSIM`` and ``loss_Pathology``."""
        loss_G, losses = super()._compute_generator_loss(
            real_A,
            real_B,
            fake_B,
            idt_B,
            flipped_for_equivariance,
        )

        loss_SSIM = self._compute_ms_ssim_loss(fake_B, real_B)
        if loss_SSIM is not None:
            loss_G = loss_G + loss_SSIM
            losses["loss_SSIM"] = loss_SSIM

        loss_Pathology = self._compute_pathology_loss(fake_B, real_B)
        if loss_Pathology is not None:
            loss_G = loss_G + loss_Pathology
            losses["loss_Pathology"] = loss_Pathology

        return loss_G, losses

    def _compute_ms_ssim_loss(
        self, fake_B: torch.Tensor, real_B: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute the MS-SSIM generator loss."""
        if getattr(self.model, "lambda_ssim", 0.0) <= 0.0:
            return None
        loss = self._find_ms_ssim_loss()(pred=fake_B, target=real_B)
        return loss * self.model.lambda_ssim

    def _compute_pathology_loss(
        self, fake_B: torch.Tensor, real_B: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute the combined MLPA + CTPC pathology loss."""
        if getattr(self.model, "lambda_pathology", 0.0) <= 0.0:
            return None
        loss = self._find_pathology_loss()(pred=fake_B, target=real_B)
        return loss * self.model.lambda_pathology

    def _find_ms_ssim_loss(self) -> StainLoss:
        """Extract the MS_SSIM_Loss component from the configured composite loss."""
        name = "ms_ssim"
        loss = self._find_loss_by_name(name)
        if loss is None:
            raise RuntimeError(f"No MS_SSIM_Loss found in the configured composite loss")
        return loss

    def _find_pathology_loss(self) -> StainLoss:
        """Extract the PSPStainPathologyLoss component from the composite loss."""
        name = "psp_pathology"
        loss = self._find_loss_by_name(name)
        if loss is None:
            raise RuntimeError(
                f"No PSPStainPathologyLoss found in the configured composite loss"
            )
        return loss

    def _find_loss_by_name(self, name: str) -> StainLoss | None:
        """Return the first component whose :meth:`name` matches ``name``."""
        if isinstance(self.loss_fn, StainLoss) and self.loss_fn.name() == name:
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if loss_fn.name() == name:
                    return loss_fn
        return None


ModelRegistry.register(
    "pspstain",
    model_cls=PSPStainModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=PSPStainStrategy,
)

__all__ = ["PSPStainModel", "PSPStainStrategy"]
