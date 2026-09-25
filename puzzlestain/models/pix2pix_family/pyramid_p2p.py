"""PyramidPix2Pix variant within the Pix2Pix family."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..loss.base import CompositeLoss
from ..networks import define_D, define_G
from ..registry import ModelRegistry
from .base import BasePix2PixModel, BasePix2PixProcessor
from .strategy_base import BasePix2PixStrategy

if TYPE_CHECKING:
    from ...configs.schema import PuzzleStainConfig


class PyramidP2PModel(BasePix2PixModel):
    """PyramidPix2pix generator + discriminator pair."""

    def _build_networks(self, no_d: bool) -> None:
        self.netG = define_G(
            self.input_nc,
            self.output_nc,
            self.ngf,
            self.netG_name,
            norm=self.norm,
            use_dropout=self.use_dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
        )
        if not no_d:
            self.netD = define_D(
                self.input_nc + self.output_nc,
                self.ndf,
                self.netD_name,
                n_layers_D=self.n_layers_D,
                norm=self.norm,
                init_type=self.init_type,
                init_gain=self.init_gain,
            )

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        netG: str = "resnet_9blocks",
        netD: str = "basic",
        n_layers_D: int = 3,
        norm: str = "batch",
        use_dropout: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
        no_d: bool = False,
        d_lr_factor: float = 1.0,
    ) -> None:
        self.ngf = ngf
        self.ndf = ndf
        self.netG_name = netG
        self.netD_name = netD
        self.n_layers_D = n_layers_D
        self.norm = norm
        self.use_dropout = use_dropout
        self.init_type = init_type
        self.init_gain = init_gain
        self.d_lr_factor = d_lr_factor
        super().__init__(input_nc=input_nc, output_nc=output_nc, no_d=no_d)


class PyramidP2PStrategy(BasePix2PixStrategy):
    """Training strategy for PyramidPix2pix: D-first, then G."""

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        source = inputs["source_image"]
        target = inputs["target_image"]

        output = self.model(source_image=source)
        fake = output.pred_image

        losses: dict[str, float] = {}

        # ---- Update D ----
        if "D" in optimizers and self.model.netD is not None:
            opt_D = optimizers["D"]
            opt_D.zero_grad()
            for param in self.model.netD.parameters():
                param.requires_grad = True

            fake_AB = torch.cat([source, fake.detach()], dim=1)
            pred_fake = self.model.netD(fake_AB)
            loss_D_fake = self._find_gan_loss()(prediction=pred_fake, target_is_real=False)

            real_AB = torch.cat([source, target], dim=1)
            pred_real = self.model.netD(real_AB)
            loss_D_real = self._find_gan_loss()(prediction=pred_real, target_is_real=True)

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            opt_D.step()

            losses["loss_D_fake"] = loss_D_fake.item()
            losses["loss_D_real"] = loss_D_real.item()
            losses["loss_D"] = loss_D.item()

        # ---- Update G ----
        opt_G = optimizers["G"]
        opt_G.zero_grad()
        if self.model.netD is not None:
            for param in self.model.netD.parameters():
                param.requires_grad = False

        fake_AB = torch.cat([source, fake], dim=1)
        pred_fake = self.model.netD(fake_AB)

        loss_G = self.loss_fn(prediction=pred_fake, target_is_real=True, pred=fake, target=target)
        loss_G.backward()
        opt_G.step()

        losses["loss_G"] = loss_G.item()
        if isinstance(self.loss_fn, CompositeLoss):
            self._log_generator_breakdown(losses, self.loss_fn.get_last_breakdown())

        return losses


ModelRegistry.register(
    "pix2pix_pyramid",
    model_cls=PyramidP2PModel,
    processor_cls=BasePix2PixProcessor,
    strategy_cls=PyramidP2PStrategy,
)

__all__ = ["PyramidP2PModel", "PyramidP2PStrategy"]
