"""CUT and FastCUT models for unpaired/paired image-to-image translation."""

from __future__ import annotations

from typing import Any

from ..registry import ModelRegistry
from .base import BaseCUTModel, BaseCUTProcessor
from .networks import define_D, define_F, define_G
from .strategy_base import BaseCUTStrategy


class CUTModel(BaseCUTModel):
    """CUT / FastCUT generator + projection + discriminator.

    Mirrors ``CUTModel`` from the original CUT repository. The ``cut_mode``
    flag selects between the two variants:

    - ``CUT`` (default): ``nce_idt=True``, ``lambda_NCE=1.0``.
    - ``FastCUT``: ``nce_idt=False``, ``lambda_NCE=10.0``,
      ``flip_equivariance=True``.
    """

    def _build_networks(self, no_d: bool) -> None:
        gpu_ids = self.gpu_ids if self.gpu_ids else []
        self.netG = define_G(
            self.input_nc,
            self.output_nc,
            self.ngf,
            self.netG_name,
            norm=self.normG,
            use_dropout=not self.no_dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
            no_antialias=self.no_antialias,
            no_antialias_up=self.no_antialias_up,
            gpu_ids=gpu_ids,
            opt=self,
            weight_norm=self.weight_norm,
        )
        self.netF = define_F(
            self.input_nc,
            self.netF_name,
            norm=self.normG,
            use_dropout=not self.no_dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
            no_antialias=self.no_antialias,
            gpu_ids=gpu_ids,
            opt=self,
        )
        if not no_d:
            self.netD = define_D(
                self.output_nc,
                self.ndf,
                self.netD_name,
                n_layers_D=self.n_layers_D,
                norm=self.normD,
                init_type=self.init_type,
                init_gain=self.init_gain,
                no_antialias=self.no_antialias,
                gpu_ids=gpu_ids,
                opt=self,
                weight_norm=self.weight_norm,
            )

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        netG: str = "resnet_9blocks",
        netD: str = "n_layers",
        netF: str = "mlp_sample",
        n_layers_D: int = 3,
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
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "none",
        d_lr_factor: float = 1.0,
    ) -> None:
        self.ngf = ngf
        self.ndf = ndf
        self.netG_name = netG
        self.netD_name = netD
        self.netF_name = netF
        self.n_layers_D = n_layers_D
        self.normG = normG
        self.normD = normD
        self.init_type = init_type
        self.init_gain = init_gain
        self.no_dropout = no_dropout
        self.no_antialias = no_antialias
        self.no_antialias_up = no_antialias_up
        self.nce_idt = nce_idt
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.nce_layers = [int(x) for x in nce_layers.split(",")]
        self.nce_T = nce_T
        self.num_patches = num_patches
        self.netF_nc = netF_nc
        self.flip_equivariance = flip_equivariance
        self.crop_size = crop_size
        self.d_lr_factor = d_lr_factor
        self.gpu_ids = gpu_ids
        self.weight_norm = weight_norm
        super().__init__(input_nc=input_nc, output_nc=output_nc, no_d=no_d)


class CUTStrategy(BaseCUTStrategy):
    """Training strategy for CUT / FastCUT."""

    pass


ModelRegistry.register(
    "cut",
    model_cls=CUTModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=CUTStrategy,
)


__all__ = ["CUTModel", "CUTStrategy"]
