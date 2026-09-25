"""SIM-GAN model and strategy for the CUT family.

SIM-GAN extends CPT (CUT + Gaussian-pyramid reconstruction) with two extra
losses computed on patch features:

- ``MC_Loss``: optimal-transport consistency between source/target/generated
  patch features.
- ``CC_loss``: L1 distance between batch cosine-similarity matrices of target
  and generated encoder features.

Reference:
    https://github.com/xianchaoguan/SIM-GAN/SIM-GAN/models/cut_model.py
    https://github.com/xianchaoguan/SIM-GAN/SIM-GAN/models/MC_loss.py
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


class SIMGANModel(CPTModel):
    """SIM-GAN: CPT + MC consistency + cosine-consistency matrix.

    Args:
        lambda_mc: Weight for the MC consistency term.
        lambda_cc: Weight for the cosine-consistency matrix term.
        mc_eps, mc_max_iter, mc_cost_type: MC loss hyperparameters.
        cc_scale, mc_scale: Internal scales carried by the loss modules.
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
        lambda_mc: float = 1.0,
        lambda_cc: float = 1.0,
        mc_eps: float = 1.0,
        mc_max_iter: int = 50,
        mc_cost_type: str = "easy",
        mc_scale: float = 10000.0,
        cc_scale: float = 10.0,
        nce_layers: str = "0,4,8,12,16",
        nce_T: float = 0.07,
        num_patches: int = 256,
        netF_nc: int = 256,
        flip_equivariance: bool = False,
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "none",
        d_lr_factor: float = 1.0,
    ) -> None:
        self.lambda_mc = lambda_mc
        self.lambda_cc = lambda_cc
        self.mc_eps = mc_eps
        self.mc_max_iter = mc_max_iter
        self.mc_cost_type = mc_cost_type
        self.mc_scale = mc_scale
        self.cc_scale = cc_scale
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


class SIMGANStrategy(CPTStrategy):
    """Training strategy for SIM-GAN."""

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return CPT losses plus ``loss_MC`` and ``loss_CC``."""
        loss_G, losses = super()._compute_generator_loss(
            real_A,
            real_B,
            fake_B,
            idt_B,
            flipped_for_equivariance,
        )

        loss_MC = self._compute_mc_loss(real_A, real_B, fake_B, flipped_for_equivariance)
        if loss_MC is not None:
            loss_G = loss_G + loss_MC
            losses["loss_MC"] = loss_MC

        loss_CC = self._compute_cc_loss(real_B, fake_B)
        if loss_CC is not None:
            loss_G = loss_G + loss_CC
            losses["loss_CC"] = loss_CC

        return loss_G, losses

    def _compute_mc_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> torch.Tensor | None:
        """Compute the MC consistency loss."""
        if getattr(self.model, "lambda_mc", 0.0) <= 0.0:
            return None
        feats = self._extract_ot_features(real_A, real_B, fake_B, flipped_for_equivariance)
        mc_loss = self._find_mc_loss()
        total = torch.tensor(0.0, device=fake_B.device)
        for f_src, f_tgt, f_gen in zip(
            feats["feat_src"], feats["feat_tgt"], feats["feat_gen"]
        ):
            total = total + mc_loss(feat_src=f_src, feat_tgt=f_tgt, feat_gen=f_gen)
        loss = total / len(self.model.nce_layers)
        return loss * self.model.lambda_mc

    def _compute_cc_loss(
        self, real_B: torch.Tensor, fake_B: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute the cosine-consistency matrix loss."""
        if getattr(self.model, "lambda_cc", 0.0) <= 0.0:
            return None
        feat_tgt = self.model.netG(fake_B, self.model.nce_layers, encode_only=True)
        feat_src = self.model.netG(real_B, self.model.nce_layers, encode_only=True)
        loss = self._find_cc_loss()(src_feats=feat_src, tgt_feats=feat_tgt)
        return loss * self.model.lambda_cc

    def _extract_ot_features(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> dict[str, torch.Tensor]:
        """Return pooled src/tgt/gen features for the OT losses."""
        feat_src = self.model.netG(real_A, self.model.nce_layers, encode_only=True)
        feat_tgt = self.model.netG(real_B, self.model.nce_layers, encode_only=True)
        feat_gen = self.model.netG(fake_B, self.model.nce_layers, encode_only=True)

        if self.model.flip_equivariance and flipped_for_equivariance:
            feat_src = [torch.flip(f, [3]) for f in feat_src]
            feat_tgt = [torch.flip(f, [3]) for f in feat_tgt]
            feat_gen = [torch.flip(f, [3]) for f in feat_gen]

        feat_src_pool, sample_ids = self.model.netF(
            feat_src, self.model.num_patches, None
        )
        feat_tgt_pool, _ = self.model.netF(feat_tgt, self.model.num_patches, sample_ids)
        feat_gen_pool, _ = self.model.netF(feat_gen, self.model.num_patches, sample_ids)

        return {
            "feat_src": feat_src_pool,
            "feat_tgt": feat_tgt_pool,
            "feat_gen": feat_gen_pool,
        }

    def _find_mc_loss(self) -> StainLoss:
        """Extract MCLoss from the configured composite loss."""
        loss = self._find_loss_by_name("mc_consistency")
        if loss is None:
            raise RuntimeError("No MCLoss found in the configured composite loss")
        return loss

    def _find_cc_loss(self) -> StainLoss:
        """Extract CosineConsistencyLoss from the configured composite loss."""
        loss = self._find_loss_by_name("cosine_consistency")
        if loss is None:
            raise RuntimeError(
                "No CosineConsistencyLoss found in the configured composite loss"
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
    "simgan",
    model_cls=SIMGANModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=SIMGANStrategy,
)

__all__ = ["SIMGANModel", "SIMGANStrategy"]
