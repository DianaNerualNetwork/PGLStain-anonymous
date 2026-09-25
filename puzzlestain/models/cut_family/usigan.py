"""USIGAN model and strategy for the CUT family.

USIGAN extends CPT (CUT + Gaussian-pyramid reconstruction) with three extra
losses:

- ``UMC_Loss``: unbalanced OT cycle consistency between source/target/generated
  patch features.
- ``PCSM_Loss``: pathological correspondence self-mining based on DAB optical
  density maps.
- ``CC_loss``: L1 distance between batch cosine-similarity matrices of target
  and generated encoder features.

References:
    https://github.com/MIXAILAB/USIGAN/models/cut_model.py
    https://github.com/MIXAILAB/USIGAN/models/MC_loss.py
    https://github.com/MIXAILAB/USIGAN/models/PCM_loss.py
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


class USIGANModel(CPTModel):
    """USIGAN: CPT + UMC + PCSM + cosine-consistency matrix.

    Args:
        lambda_umc: Weight for the UMC consistency term.
        lambda_pcsm: Weight for the PCSM term.
        lambda_cc: Weight for the cosine-consistency matrix term.
        umc_eps, umc_tau, umc_max_iter, umc_cost_type: UMC loss hyperparameters.
        umc_scale, cc_scale, pcsm_scale: Internal loss scales.
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
        lambda_umc: float = 1.0,
        lambda_pcsm: float = 1.0,
        lambda_cc: float = 1.0,
        umc_eps: float = 1.0,
        umc_tau: float = 0.001,
        umc_max_iter: int = 50,
        umc_cost_type: str = "easy",
        umc_scale: float = 10000.0,
        cc_scale: float = 10.0,
        pcsm_scale: float = 1.0,
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
        self.lambda_umc = lambda_umc
        self.lambda_pcsm = lambda_pcsm
        self.lambda_cc = lambda_cc
        self.umc_eps = umc_eps
        self.umc_tau = umc_tau
        self.umc_max_iter = umc_max_iter
        self.umc_cost_type = umc_cost_type
        self.umc_scale = umc_scale
        self.cc_scale = cc_scale
        self.pcsm_scale = pcsm_scale
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


class USIGANStrategy(CPTStrategy):
    """Training strategy for USIGAN."""

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return CPT losses plus ``loss_UMC``, ``loss_PCSM`` and ``loss_CC``."""
        loss_G, losses = super()._compute_generator_loss(
            real_A,
            real_B,
            fake_B,
            idt_B,
            flipped_for_equivariance,
        )

        loss_UMC = self._compute_umc_loss(
            real_A, real_B, fake_B, flipped_for_equivariance
        )
        if loss_UMC is not None:
            loss_G = loss_G + loss_UMC
            losses["loss_UMC"] = loss_UMC

        loss_PCSM = self._compute_pcsm_loss(real_B, fake_B)
        if loss_PCSM is not None:
            loss_G = loss_G + loss_PCSM
            losses["loss_PCSM"] = loss_PCSM

        loss_CC = self._compute_cc_loss(real_B, fake_B)
        if loss_CC is not None:
            loss_G = loss_G + loss_CC
            losses["loss_CC"] = loss_CC

        return loss_G, losses

    def _compute_umc_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> torch.Tensor | None:
        """Compute the UMC consistency loss."""
        if getattr(self.model, "lambda_umc", 0.0) <= 0.0:
            return None
        feats = self._extract_ot_features(real_A, real_B, fake_B, flipped_for_equivariance)
        umc_loss = self._find_umc_loss()
        total = torch.tensor(0.0, device=fake_B.device)
        for f_src, f_tgt, f_gen in zip(
            feats["feat_src"], feats["feat_tgt"], feats["feat_gen"]
        ):
            total = total + umc_loss(feat_src=f_src, feat_tgt=f_tgt, feat_gen=f_gen)
        loss = total / len(self.model.nce_layers)
        return loss * self.model.lambda_umc

    def _compute_pcsm_loss(
        self, real_B: torch.Tensor, fake_B: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute the PCSM loss."""
        if getattr(self.model, "lambda_pcsm", 0.0) <= 0.0:
            return None
        loss = self._find_pcsm_loss()(pred=fake_B, target=real_B)
        return loss * self.model.lambda_pcsm

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

    def _find_umc_loss(self) -> StainLoss:
        """Extract UMCLoss from the configured composite loss."""
        loss = self._find_loss_by_name("umc_consistency")
        if loss is None:
            raise RuntimeError("No UMCLoss found in the configured composite loss")
        return loss

    def _find_pcsm_loss(self) -> StainLoss:
        """Extract PCSMLoss from the configured composite loss."""
        loss = self._find_loss_by_name("pcsm")
        if loss is None:
            raise RuntimeError("No PCSMLoss found in the configured composite loss")
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
    "usigan",
    model_cls=USIGANModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=USIGANStrategy,
)

__all__ = ["USIGANModel", "USIGANStrategy"]
