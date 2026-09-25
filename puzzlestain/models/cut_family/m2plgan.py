"""M2PL-GAN model: FastCUT + CACM + LDAM + GBCLM.

Ported from https://github.com/Pikachu-one/M2PL-GAN (``models/m2plgan_model.py``
and ``models/PatchGCL.py``). On top of the FastCUT step (LSGAN + PatchNCE with
``nce_idt=False`` and ``flip_equivariance=False``) it adds three losses between
the multi-level encoder features of ``real_B`` and ``fake_B``, all consumed
from the configured composite loss by their registered names:

- CACM (``cacm``): L1 between per-layer patch cosine-similarity matrices.
- LDAM (``ldam``): RBF-kernel MMD between per-layer patch vectors
  (``patchwpatch`` mode) or between netF-sampled patch features with
  per-layer weights (``pixelwpiexl`` mode).
- GBCLM (``gbclm``): bidirectional InfoNCE over graph encodings of the
  netF-sampled patch features (``conv_type`` ``tag``/``gcn``/``gat``/
  ``none``). The original builds DGL graphs; this port is pure PyTorch and
  numerically mirrors DGL's ``TAGConv`` (``D^-1/2 (A+I) D^-1/2``
  propagation, k=4 hops), ``GraphConv`` and ``GATConv``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..loss.base import CompositeLoss, StainLoss
from ..loss.gbclm import GNNLoss
from ..registry import ModelRegistry
from .base import BaseCUTProcessor
from .cut import CUTModel, CUTStrategy


class M2PLGANModel(CUTModel):
    """M2PL-GAN: FastCUT generator/projection/discriminator.

    Mirrors ``M2PLGANModel`` from the original repository with the FastCUT
    configuration used in the paper (``nce_idt=False``,
    ``flip_equivariance=False``, ``lambda_NCE=1.0``). The GBCLM module lives
    in the configured composite loss (registered name ``gbclm``); the
    strategy exposes it via ``get_networks`` so it is checkpointed.
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
        nce_idt: bool = False,
        lambda_GAN: float = 1.0,
        lambda_NCE: float = 1.0,
        nce_layers: str = "0,4,8,12,16",
        nce_T: float = 0.07,
        num_patches: int = 256,
        netF_nc: int = 256,
        flip_equivariance: bool = False,
        lambda_CACM: float = 1.0,
        lambda_LDAM: float = 10.0,
        lambda_GBCLM: float = 0.1,
        patchsize: int = 64,
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "spectral",
        d_lr_factor: float = 1.0,
    ) -> None:
        self.lambda_CACM = lambda_CACM
        self.lambda_LDAM = lambda_LDAM
        self.lambda_GBCLM = lambda_GBCLM
        self.patchsize = patchsize
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


class M2PLGANStrategy(CUTStrategy):
    """Training strategy for M2PL-GAN.

    Reuses the FastCUT D-then-G update from the CUT family and adds the CACM,
    LDAM and GBCLM terms to the generator loss, mirroring
    ``M2PLGANModel.compute_G_loss`` in the original repo. The three terms are
    looked up from the configured composite loss by their registered names
    (``cacm`` / ``ldam`` / ``gbclm``).
    """

    def get_networks(self) -> dict[str, nn.Module]:
        """Return the CUT networks plus the GBCLM module for checkpointing."""
        networks = super().get_networks()
        gnn = self._find_loss_by_name("gbclm")
        if not isinstance(gnn, nn.Module):
            raise RuntimeError("The gbclm loss component must be an nn.Module")
        networks["GNN"] = gnn
        return networks

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the FastCUT losses plus ``loss_CACM``/``loss_LDAM``/``loss_GBCLM``."""
        loss_G, losses = super()._compute_generator_loss(
            real_A, real_B, fake_B, idt_B, flipped_for_equivariance
        )
        model = self.model

        feat_fake_B = model.netG(fake_B, model.nce_layers, encode_only=True)
        feat_real_B = model.netG(real_B, model.nce_layers, encode_only=True)

        if model.lambda_CACM > 0.0:
            loss_CACM = (
                self._find_loss_by_name("cacm")(
                    feat_real=feat_real_B,
                    feat_fake=feat_fake_B,
                    patchsize=model.patchsize,
                )
                * model.lambda_CACM
            )
            loss_G = loss_G + loss_CACM
            losses["loss_CACM"] = loss_CACM

        if model.lambda_LDAM > 0.0:
            ldam_loss = self._find_loss_by_name("ldam")
            if getattr(ldam_loss, "mode", "patchwpatch") == "pixelwpiexl":
                # pixelwpiexl mode: MMD over netF-sampled features (shared ids).
                feats_fake_ld, sample_ids = model.netF(
                    feat_fake_B, model.num_patches, None
                )
                feats_real_ld, _ = model.netF(
                    feat_real_B, model.num_patches, sample_ids
                )
                loss_LDAM = (
                    ldam_loss(feat_fake=feats_fake_ld, feat_real=feats_real_ld)
                    * model.lambda_LDAM
                )
            else:
                loss_LDAM = (
                    ldam_loss(feat_fake=feat_fake_B, feat_real=feat_real_B)
                    * model.lambda_LDAM
                )
            loss_G = loss_G + loss_LDAM
            losses["loss_LDAM"] = loss_LDAM

        if model.lambda_GBCLM > 0.0:
            # GNN contrastive loss over netF-sampled features (shared sample ids).
            feats_fake, sample_ids = model.netF(feat_fake_B, model.num_patches, None)
            feats_real, _ = model.netF(feat_real_B, model.num_patches, sample_ids)
            loss_GBCLM = (
                self._find_loss_by_name("gbclm")(feat_s=feats_fake, feat_t=feats_real)
                * model.lambda_GBCLM
            )
            loss_G = loss_G + loss_GBCLM
            losses["loss_GBCLM"] = loss_GBCLM

        return loss_G, losses

    def _find_loss_by_name(self, name: str) -> StainLoss:
        """Return the composite component whose :meth:`name` matches ``name``."""
        if isinstance(self.loss_fn, StainLoss) and self.loss_fn.name() == name:
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if loss_fn.name() == name:
                    return loss_fn
        raise RuntimeError(
            f"No {name} loss found in the configured composite loss"
        )


ModelRegistry.register(
    "m2plgan",
    model_cls=M2PLGANModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=M2PLGANStrategy,
)


__all__ = ["GNNLoss", "M2PLGANModel", "M2PLGANStrategy"]
