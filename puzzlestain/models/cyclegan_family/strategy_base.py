"""Shared training strategy machinery for the CycleGAN family."""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ..loss.base import CompositeLoss, StainLoss
from ..loss.gan import GANLoss
from ..strategy.base import TrainingStrategy
from .base import BaseCycleGANModel, BaseCycleGANProcessor

if TYPE_CHECKING:
    from ...configs.schema import PuzzleStainConfig

logger = logging.getLogger(__name__)


class BaseCycleGANStrategy(TrainingStrategy):
    """Common training strategy for CycleGAN-family models.

    The update order follows the original CycleGAN repo (generators first,
    then discriminators — the reverse of the CUT family):

      1. Forward: ``fake_B = G_A(real_A)``, ``rec_A = G_B(fake_B)``,
         ``fake_A = G_B(real_B)``, ``rec_B = G_A(fake_A)``.
      2. Freeze D_A/D_B, backprop the generator loss, step ``optimizer_G``.
      3. Unfreeze D_A/D_B, backprop D_A and D_B separately, step
         ``optimizer_D``.

    The GAN loss is taken from the configured composite loss; the cycle and
    identity L1 terms use the ``cycle_consistency`` component of the composite
    loss, with the per-term weights (``lambda_A``/``lambda_B``/
    ``lambda_identity``) applied here, as in the original repo.

    Args:
        model: A :class:`BaseCycleGANModel` subclass with ``netG_A``,
            ``netG_B`` and optional ``netD_A``/``netD_B``.
        processor: A :class:`BaseCycleGANProcessor`.
        loss_fn: Composite loss containing a :class:`GANLoss` component.
        config: Resolved experiment config.
    """

    def __init__(
        self,
        model: BaseCycleGANModel,
        processor: BaseCycleGANProcessor,
        loss_fn: StainLoss,
        config: "PuzzleStainConfig",
    ) -> None:
        self.model = model
        self.processor = processor
        self.loss_fn = loss_fn
        self.config = config

    def get_networks(self) -> dict[str, nn.Module]:
        networks: dict[str, nn.Module] = {
            "G": self.model.netG_A,
            "G_B": self.model.netG_B,
        }
        if self.model.netD_A is not None:
            networks["D_A"] = self.model.netD_A
        if self.model.netD_B is not None:
            networks["D_B"] = self.model.netD_B
        return networks

    def get_optimizers(
        self, config: "PuzzleStainConfig"
    ) -> dict[str, torch.optim.Optimizer]:
        opt_cfg = config.training
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        optimizers: dict[str, torch.optim.Optimizer] = {
            "G": torch.optim.Adam(
                itertools.chain(
                    self.model.netG_A.parameters(), self.model.netG_B.parameters()
                ),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=weight_decay,
            ),
        }
        if self.model.netD_A is not None and self.model.netD_B is not None:
            optimizers["D"] = torch.optim.Adam(
                itertools.chain(
                    self.model.netD_A.parameters(), self.model.netD_B.parameters()
                ),
                lr=opt_cfg.lr / self.model.d_lr_factor,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=weight_decay,
            )
        return optimizers

    def get_schedulers(
        self,
        optimizers: dict[str, torch.optim.Optimizer],
        config: "PuzzleStainConfig",
    ) -> dict[str, Any]:
        """Create LR schedulers matching the original linear policy."""
        opt_cfg = config.training
        policy = getattr(opt_cfg, "lr_policy", "linear")
        if policy != "linear":
            return {name: None for name in optimizers}

        max_epochs = opt_cfg.max_epochs
        n_epochs = getattr(opt_cfg, "n_epochs", max_epochs // 2)
        n_epochs_decay = getattr(opt_cfg, "n_epochs_decay", max_epochs - n_epochs)

        def lambda_rule(epoch: int) -> float:
            return 1.0 - max(0, epoch + 1 - n_epochs) / float(n_epochs_decay + 1)

        schedulers: dict[str, Any] = {}
        for name, opt in optimizers.items():
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda_rule
            )
        return schedulers

    def _forward(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the generator forward pass, mirroring ``CycleGANModel.forward``."""
        fake_B = self.model.netG_A(real_A)  # G_A(A)
        rec_A = self.model.netG_B(fake_B)  # G_B(G_A(A))
        fake_A = self.model.netG_B(real_B)  # G_B(B)
        rec_B = self.model.netG_A(fake_A)  # G_A(G_B(B))
        return fake_B, rec_A, fake_A, rec_B

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        rec_A: torch.Tensor,
        fake_A: torch.Tensor,
        rec_B: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(loss_G, losses)``, mirroring ``CycleGANModel.backward_G``."""
        lambda_A = self.model.lambda_A
        lambda_B = self.model.lambda_B
        lambda_idt = self.model.lambda_identity
        gan_loss = self._find_gan_loss()
        cycle_loss = self._find_cycle_loss()
        losses: dict[str, torch.Tensor] = {}

        # GAN loss D_A(G_A(A)) and D_B(G_B(B)).
        loss_G_A = gan_loss(prediction=self.model.netD_A(fake_B), target_is_real=True)
        loss_G_B = gan_loss(prediction=self.model.netD_B(fake_A), target_is_real=True)
        # Forward cycle loss ||G_B(G_A(A)) - A|| and backward ||G_A(G_B(B)) - B||.
        loss_cycle_A = cycle_loss(pred=rec_A, target=real_A) * lambda_A
        loss_cycle_B = cycle_loss(pred=rec_B, target=real_B) * lambda_B
        losses["loss_G_A"] = loss_G_A
        losses["loss_G_B"] = loss_G_B
        losses["loss_cycle_A"] = loss_cycle_A
        losses["loss_cycle_B"] = loss_cycle_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B

        # Identity loss (only when lambda_identity > 0):
        # ||G_A(B) - B|| * lambda_B and ||G_B(A) - A|| * lambda_A.
        if lambda_idt > 0:
            idt_A = self.model.netG_A(real_B)
            loss_idt_A = cycle_loss(pred=idt_A, target=real_B) * lambda_B * lambda_idt
            idt_B = self.model.netG_B(real_A)
            loss_idt_B = cycle_loss(pred=idt_B, target=real_A) * lambda_A * lambda_idt
            losses["loss_idt_A"] = loss_idt_A
            losses["loss_idt_B"] = loss_idt_B
            loss_G = loss_G + loss_idt_A + loss_idt_B

        return loss_G, losses

    def _backward_D(
        self,
        netD: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backprop one discriminator, mirroring ``backward_D_basic``.

        ``fake`` comes from the image pool (which stores the generated,
        non-detached tensors); it is detached here, as in the original repo.
        Returns ``(loss_D, loss_D_real, loss_D_fake)`` after calling
        ``loss_D.backward()``.
        """
        gan_loss = self._find_gan_loss()
        # Real
        pred_real = netD(real)
        loss_D_real = gan_loss(prediction=pred_real, target_is_real=True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = gan_loss(prediction=pred_fake, target_is_real=False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D, loss_D_real, loss_D_fake

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        real_A = inputs["source_image"]
        real_B = inputs["target_image"]

        # Forward: compute fake images and reconstruction images.
        fake_B, rec_A, fake_A, rec_B = self._forward(real_A, real_B)
        losses: dict[str, float] = {}

        # ---- Update G_A and G_B ----
        opt_G = optimizers["G"]
        self._set_requires_grad(
            [self.model.netD_A, self.model.netD_B], requires_grad=False
        )
        opt_G.zero_grad()
        loss_G, g_losses = self._compute_generator_loss(
            real_A, real_B, fake_B, rec_A, fake_A, rec_B
        )
        loss_G.backward()
        opt_G.step()
        self._set_requires_grad(
            [self.model.netD_A, self.model.netD_B], requires_grad=True
        )

        for name, value in g_losses.items():
            losses[name] = value.item()
        losses["loss_G"] = loss_G.item()

        # ---- Update D_A and D_B ----
        if self.model.netD_A is not None and "D" in optimizers:
            opt_D = optimizers["D"]
            opt_D.zero_grad()
            fake_B_pool = self.model.fake_B_pool.query(fake_B)
            loss_D_A, _, _ = self._backward_D(self.model.netD_A, real_B, fake_B_pool)
            fake_A_pool = self.model.fake_A_pool.query(fake_A)
            loss_D_B, _, _ = self._backward_D(self.model.netD_B, real_A, fake_A_pool)
            opt_D.step()

            losses["loss_D_A"] = loss_D_A.item()
            losses["loss_D_B"] = loss_D_B.item()
            losses["loss_D"] = loss_D_A.item() + loss_D_B.item()

        return losses

    def sample_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
    ) -> dict[str, torch.Tensor] | None:
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        real_A = inputs["source_image"]
        real_B = inputs["target_image"]
        with torch.no_grad():
            fake_B = self.model(source_image=real_A).pred_image
        return {
            "source": real_A,
            "pred": fake_B,
            "target": real_B,
        }

    def replace_networks(self, networks: dict[str, nn.Module]) -> None:
        """Swap prepared networks back into the model wrapper."""
        if "model" in networks:
            self.model = networks["model"]
        else:
            if "G" in networks:
                self.model.netG_A = networks["G"]
            if "G_B" in networks:
                self.model.netG_B = networks["G_B"]
            if "D_A" in networks and self.model.netD_A is not None:
                self.model.netD_A = networks["D_A"]
            if "D_B" in networks and self.model.netD_B is not None:
                self.model.netD_B = networks["D_B"]

    @staticmethod
    def _set_requires_grad(
        nets: list[nn.Module | None], requires_grad: bool = False
    ) -> None:
        """Toggle ``requires_grad`` on the given networks (D freeze for G update)."""
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def _find_gan_loss(self) -> GANLoss:
        """Extract the GANLoss component from the configured composite loss."""
        if isinstance(self.loss_fn, GANLoss):
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if isinstance(loss_fn, GANLoss):
                    return loss_fn
        raise RuntimeError("No GANLoss found in the configured loss function")

    def _find_cycle_loss(self) -> StainLoss:
        """Extract the ``cycle_consistency`` component from the composite loss."""
        name = "cycle_consistency"
        if isinstance(self.loss_fn, StainLoss) and self.loss_fn.name() == name:
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if loss_fn.name() == name:
                    return loss_fn
        raise RuntimeError(
            f"No {name} loss found in the configured composite loss"
        )


__all__ = ["BaseCycleGANStrategy"]
