"""Shared training strategy machinery for the Pix2Pix family."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from ..base import StainOutput
from ..loss.base import CompositeLoss, StainLoss
from ..loss.gan import GANLoss
from ..strategy.base import TrainingStrategy
from .base import BasePix2PixModel, BasePix2PixProcessor

if TYPE_CHECKING:
    from ...configs.schema import PuzzleStainConfig


class BasePix2PixStrategy(TrainingStrategy):
    """Common training strategy for Pix2Pix-family models.

    Subclasses must implement :meth:`training_step` to define the update order
    (e.g. D-first or G-first) and how the composite loss is decomposed.

    Args:
        model: A :class:`BasePix2PixModel` with ``netG`` and optional ``netD``.
        processor: A :class:`BasePix2PixProcessor`.
        loss_fn: Composite loss combining GAN and reconstruction terms.
        config: Resolved experiment config.
    """

    def __init__(
        self,
        model: BasePix2PixModel,
        processor: BasePix2PixProcessor,
        loss_fn: StainLoss,
        config: "PuzzleStainConfig",
    ) -> None:
        self.model = model
        self.processor = processor
        self.loss_fn = loss_fn
        self.config = config

    def get_networks(self) -> dict[str, nn.Module]:
        networks: dict[str, nn.Module] = {"G": self.model.netG}
        if self.model.netD is not None:
            networks["D"] = self.model.netD
        return networks

    def get_optimizers(
        self, config: "PuzzleStainConfig"
    ) -> dict[str, torch.optim.Optimizer]:
        opt_cfg = config.training
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        optimizers: dict[str, torch.optim.Optimizer] = {
            "G": torch.optim.Adam(
                self.model.netG.parameters(),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=weight_decay,
            ),
        }
        if self.model.netD is not None:
            optimizers["D"] = torch.optim.Adam(
                self.model.netD.parameters(),
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
        """Create LR schedulers matching the original pix2pix linear policy."""
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
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)
        return schedulers

    def sample_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
    ) -> dict[str, torch.Tensor] | None:
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        source = inputs["source_image"]
        target = inputs["target_image"]
        with torch.no_grad():
            fake = self.model(source_image=source).pred_image
        return {
            "source": source,
            "pred": fake,
            "target": target,
        }

    def replace_networks(self, networks: dict[str, nn.Module]) -> None:
        """Swap prepared networks back into the model wrapper."""
        if "model" in networks:
            self.model = networks["model"]
        else:
            if "G" in networks:
                self.model.netG = networks["G"]
            if "D" in networks and self.model.netD is not None:
                self.model.netD = networks["D"]

    def _find_gan_loss(self) -> GANLoss:
        """Extract the GANLoss component from the configured composite loss."""
        if isinstance(self.loss_fn, GANLoss):
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if isinstance(loss_fn, GANLoss):
                    return loss_fn
        raise RuntimeError("No GANLoss found in the configured loss function")

    def _log_generator_breakdown(
        self, losses: dict[str, float], breakdown: dict[str, float]
    ) -> None:
        """Map composite loss breakdown keys to strategy-specific log names."""
        mapping: dict[str, str] = {
            "gan_feat": "loss_G_GAN_Feat",
            "gan": "loss_G_GAN",
            "vgg": "loss_G_VGG",
            "pyramid_l1": "loss_recon",
        }
        for key, value in breakdown.items():
            if key == "loss/total":
                continue
            name = key.removeprefix("loss/")
            for prefix, log_name in mapping.items():
                if name.startswith(prefix):
                    losses[log_name] = value
                    break


__all__ = ["BasePix2PixStrategy"]
