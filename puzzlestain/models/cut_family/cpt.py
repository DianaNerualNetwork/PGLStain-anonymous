"""CPT (Contrastive Paired Translation) model."""

from __future__ import annotations

import torch

from ..loss.base import CompositeLoss, StainLoss
from ..registry import ModelRegistry
from .base import BaseCUTProcessor
from .cut import CUTModel, CUTStrategy


class CPTModel(CUTModel):
    """CPT: CUT + Gaussian-pyramid reconstruction loss.

    This model extends :class:`CUTModel` with an additional L1 loss between
    Gaussian pyramids of the generated and real target images. It mirrors the
    ``CPTModel`` from the ASP repository with ``lambda_asp=0``.
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
        nce_layers: str = "0,4,8,12,16",
        nce_T: float = 0.07,
        num_patches: int = 256,
        netF_nc: int = 256,
        flip_equivariance: bool = False,
        lambda_gp: float = 10.0,
        gp_weights: list[float] | str = "uniform",
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "none",
        d_lr_factor: float = 1.0,
    ) -> None:
        if isinstance(gp_weights, str):
            if gp_weights == "uniform":
                gp_weights = [1.0] * 6
            else:
                gp_weights = list(eval(gp_weights))
        self.lambda_gp = lambda_gp
        self.gp_weights = gp_weights
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
            crop_size=crop_size,
            gpu_ids=gpu_ids,
            no_d=no_d,
            weight_norm=weight_norm,
            d_lr_factor=d_lr_factor,
        )


class CPTStrategy(CUTStrategy):
    """Training strategy for CPT.

    Extends the plain CUT step with the Gaussian-pyramid reconstruction
    (``loss_GP``) and adaptive supervised PatchNCE (``loss_ASP``) terms,
    mirroring ``CPTModel`` in the ASP repository. Strategies whose models
    define ``lambda_gp``/``lambda_asp`` pick up these terms by inheriting
    from this class. The pyramid reconstruction term uses the
    ``gauss_pyramid_l1`` component of the configured composite loss.
    """

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return CUT losses plus ``loss_GP`` and ``loss_ASP``."""
        loss_G, losses = super()._compute_generator_loss(
            real_A, real_B, fake_B, idt_B, flipped_for_equivariance
        )

        loss_GP = self._compute_gp_loss(fake_B, real_B)
        if loss_GP is not None:
            loss_G = loss_G + loss_GP
            losses["loss_GP"] = loss_GP

        loss_ASP = self._compute_asp_loss(real_B, fake_B, flipped_for_equivariance)
        if loss_ASP is not None:
            loss_G = loss_G + loss_ASP
            losses["loss_ASP"] = loss_ASP

        return loss_G, losses

    def _compute_gp_loss(
        self,
        fake_B: torch.Tensor,
        real_B: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute the Gaussian-pyramid reconstruction loss (CPT family)."""
        if getattr(self.model, "lambda_gp", 0.0) <= 0.0:
            return None
        gp_loss = self._find_loss_by_name("gauss_pyramid_l1")
        if gp_loss is None:
            raise RuntimeError(
                "No gauss_pyramid_l1 loss found in the configured composite loss"
            )
        loss = gp_loss(pred=fake_B, target=real_B, weights=self.model.gp_weights)
        return loss * self.model.lambda_gp

    def _compute_asp_loss(
        self,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> torch.Tensor | None:
        """Compute the adaptive supervised PatchNCE loss (ASP family)."""
        if getattr(self.model, "lambda_asp", 0.0) <= 0.0:
            return None
        return self._calculate_ASP_loss(
            real_B, fake_B, flipped_for_equivariance
        )

    def _calculate_ASP_loss(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> torch.Tensor:
        """Compute the adaptive supervised PatchNCE loss between ``src`` and ``tgt``."""
        feat_q = self.model.netG(tgt, self.model.nce_layers, encode_only=True)
        if self.model.flip_equivariance and flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = self.model.netG(src, self.model.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.model.netF(feat_k, self.model.num_patches, None)
        feat_q_pool, _ = self.model.netF(feat_q, self.model.num_patches, sample_ids)

        total = 0.0
        n_layers = len(self.model.nce_layers)
        asp_loss = self._find_asp_loss()
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            total = total + asp_loss(feat_q=f_q, feat_k=f_k, current_epoch=self.current_epoch)
        return (total / n_layers) * self.model.lambda_asp

    def _find_asp_loss(self) -> StainLoss:
        """Extract the AdaptiveSupervisedPatchNCELoss component from the composite loss."""
        if isinstance(self.loss_fn, StainLoss) and self.loss_fn.name() == "asp":
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if loss_fn.name() == "asp":
                    return loss_fn
        raise RuntimeError("No AdaptiveSupervisedPatchNCELoss found in the configured loss function")

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
    "cpt",
    model_cls=CPTModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=CPTStrategy,
)


__all__ = ["CPTModel", "CPTStrategy"]
