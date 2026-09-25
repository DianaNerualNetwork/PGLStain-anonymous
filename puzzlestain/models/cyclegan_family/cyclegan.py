"""CycleGAN model for unpaired image-to-image translation."""

from __future__ import annotations

import logging

from ..networks import define_D, define_G
from ..image_pool import ImagePool
from ..registry import ModelRegistry
from .base import BaseCycleGANModel, BaseCycleGANProcessor
from .strategy_base import BaseCycleGANStrategy

logger = logging.getLogger(__name__)


class CycleGANModel(BaseCycleGANModel):
    """CycleGAN generators + discriminators with image pools.

    Mirrors ``CycleGANModel`` from the original CycleGAN/pix2pix repository:
    ``netG_A`` maps A->B, ``netG_B`` maps B->A, ``netD_A`` discriminates
    domain-B images and ``netD_B`` domain-A images. The default configuration
    uses ``resnet_9blocks`` generators, ``basic`` (70x70 PatchGAN)
    discriminators, instance normalization and no dropout.

    The ``gan_mode`` parameter is stored for parity with the original repo's
    options; the actual GAN objective is built from the loss config.
    """

    def _build_networks(self, no_d: bool) -> None:
        self.netG_A = define_G(
            self.input_nc,
            self.output_nc,
            self.ngf,
            self.netG_name,
            norm=self.norm,
            use_dropout=self.use_dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
        )
        self.netG_B = define_G(
            self.output_nc,
            self.input_nc,
            self.ngf,
            self.netG_name,
            norm=self.norm,
            use_dropout=self.use_dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
        )
        if not no_d:
            self.netD_A = define_D(
                self.output_nc,
                self.ndf,
                self.netD_name,
                n_layers_D=self.n_layers_D,
                norm=self.norm,
                init_type=self.init_type,
                init_gain=self.init_gain,
            )
            self.netD_B = define_D(
                self.input_nc,
                self.ndf,
                self.netD_name,
                n_layers_D=self.n_layers_D,
                norm=self.norm,
                init_type=self.init_type,
                init_gain=self.init_gain,
            )
            self.fake_A_pool = ImagePool(self.pool_size)
            self.fake_B_pool = ImagePool(self.pool_size)

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        netG: str = "resnet_9blocks",
        netD: str = "basic",
        n_layers_D: int = 3,
        norm: str = "instance",
        init_type: str = "normal",
        init_gain: float = 0.02,
        use_dropout: bool = False,
        pool_size: int = 50,
        lambda_A: float = 10.0,
        lambda_B: float = 10.0,
        lambda_identity: float = 0.5,
        gan_mode: str = "lsgan",
        no_d: bool = False,
        d_lr_factor: float = 1.0,
    ) -> None:
        self.ngf = ngf
        self.ndf = ndf
        self.netG_name = netG
        self.netD_name = netD
        self.n_layers_D = n_layers_D
        self.norm = norm
        self.init_type = init_type
        self.init_gain = init_gain
        self.use_dropout = use_dropout
        self.pool_size = pool_size
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_identity
        self.gan_mode = gan_mode
        self.d_lr_factor = d_lr_factor
        super().__init__(input_nc=input_nc, output_nc=output_nc, no_d=no_d)


class CycleGANStrategy(BaseCycleGANStrategy):
    """Training strategy for CycleGAN."""

    pass


ModelRegistry.register(
    "cyclegan",
    model_cls=CycleGANModel,
    processor_cls=BaseCycleGANProcessor,
    strategy_cls=CycleGANStrategy,
)


__all__ = ["CycleGANModel", "CycleGANStrategy"]
