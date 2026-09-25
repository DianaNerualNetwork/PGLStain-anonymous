"""ASP (Adaptive Supervised PatchNCE) model."""

from __future__ import annotations

from ..registry import ModelRegistry
from .base import BaseCUTProcessor
from .cpt import CPTModel, CPTStrategy


class ASPModel(CPTModel):
    """ASP: CPT + adaptive supervised PatchNCE loss.

    This model extends :class:`CPTModel` with an additional adaptive
    supervised PatchNCE term between the generated target and the real
    target. It mirrors the ASP experiment 0 configuration from the ASP
    repository: ``CUT_mode=FastCUT`` (``nce_idt=False``), ``lambda_NCE=10.0``,
    ``lambda_gp=10.0``, ``lambda_asp=10.0``.
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
        lambda_NCE: float = 10.0,
        nce_layers: str = "0,4,8,12,16",
        nce_T: float = 0.07,
        num_patches: int = 256,
        netF_nc: int = 256,
        flip_equivariance: bool = False,
        lambda_gp: float = 10.0,
        gp_weights: list[float] | str = "[0.015625,0.03125,0.0625,0.125,0.25,1.0]",
        lambda_asp: float = 10.0,
        asp_loss_mode: str = "lambda_linear",
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "none",
        d_lr_factor: float = 1.0,
    ) -> None:
        self.lambda_asp = lambda_asp
        self.asp_loss_mode = asp_loss_mode
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
        # The ASP reference trains G/D/F with the same learning rate; the
        # CPT-specific ``d_lr_factor=10`` does not apply here.


class ASPStrategy(CPTStrategy):
    """Training strategy for ASP."""

    pass


ModelRegistry.register(
    "asp",
    model_cls=ASPModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=ASPStrategy,
)


__all__ = ["ASPModel", "ASPStrategy"]
