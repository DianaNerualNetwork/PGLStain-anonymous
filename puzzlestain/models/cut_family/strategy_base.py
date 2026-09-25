"""Shared training strategy machinery for the CUT family."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import random
import torch
import torch.nn as nn

from ..base import StainOutput
from ..loss.base import CompositeLoss, StainLoss
from ..loss.gan import GANLoss
from ..strategy.base import TrainingStrategy
from .base import BaseCUTModel, BaseCUTProcessor

if TYPE_CHECKING:
    from ...configs.schema import PuzzleStainConfig


class BaseCUTStrategy(TrainingStrategy):
    """Common training strategy for CUT-family models.

    The update order follows the original CUT repo: discriminator first, then
    generator + projection network. The strategy supports the optional CUT
    losses (NCE identity) via flags on the model instance. The CPT/ASP-only
    Gaussian-pyramid and adaptive supervised PatchNCE terms live in
    ``CPTStrategy`` (see ``cpt.py``).

    Args:
        model: A :class:`BaseCUTModel` subclass with ``netG``, ``netF`` and
            optional ``netD``.
        processor: A :class:`BaseCUTProcessor`.
        loss_fn: Composite loss combining GAN and PatchNCE terms.
        config: Resolved experiment config.
    """

    #: Weight applied to the summed ``(loss_NCE + loss_NCE_Y)`` identity terms.
    #: The CUT reference averages them (0.5); the CPT/MDCL reference
    #: (``cpt_model.py``) sums them without averaging (1.0).
    nce_idt_weight = 0.5

    def __init__(
        self,
        model: BaseCUTModel,
        processor: BaseCUTProcessor,
        loss_fn: StainLoss,
        config: "PuzzleStainConfig",
    ) -> None:
        self.model = model
        self.processor = processor
        self.loss_fn = loss_fn
        self.config = config
        self.current_epoch = 1

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for epoch-dependent losses (ASP, SRC)."""
        self.current_epoch = epoch

    def _build_optimizer_F(
        self, optimizers: dict[str, torch.optim.Optimizer]
    ) -> torch.optim.Optimizer:
        """Create the Adam optimizer for netF once its lazy MLPs exist.

        The projection network ``netF`` cannot be instantiated until the first
        real forward pass has created its MLPs. This helper mirrors the ASP
        reference, where ``optimizer_F`` is built inside
        ``data_dependent_initialize`` immediately after the first G-loss
        backward.
        """
        opt_cfg = self.config.training
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        return torch.optim.Adam(
            self.model.netF.parameters(),
            lr=opt_cfg.lr,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            weight_decay=weight_decay,
        )

    def data_dependent_initialize(
        self,
        batch: dict[str, torch.Tensor],
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> None:
        """Reference-style warmup forward/backward used to create netF MLPs.

        Mirrors ``CUTModel.data_dependent_initialize``: one forward pass, one
        D-loss backward and one G-loss backward with *no* optimizer step.  This
        ensures the RNG consumption (patch sampling, etc.) matches the original
        repo before the real training loop starts.
        """
        device = next(self.model.parameters()).device
        inputs = self.processor.preprocess(batch, device=device)
        real_A = inputs["source_image"]
        real_B = inputs["target_image"]

        fake_B, idt_B, flipped = self._forward_generator(real_A, real_B)

        # D backward (fake is detached at the call site).
        if self.model.netD is not None and "D" in optimizers:
            loss_D, _, _ = self._compute_discriminator_loss(fake_B, real_B)
            optimizers["D"].zero_grad()
            loss_D.backward()
            optimizers["D"].zero_grad()

        # G backward.  This is also the first real forward of netF, so its
        # lazy MLPs are materialized here using the same RNG state as the ASP
        # reference (after the first augmented batch has been drawn).
        loss_G, *_ = self._compute_generator_loss(
            real_A, real_B, fake_B, idt_B, flipped
        )
        optimizers["G"].zero_grad()
        loss_G.backward()
        optimizers["G"].zero_grad()

        # Create the F optimizer now that netF's MLPs exist, matching the ASP
        # reference timing.
        if "F" not in optimizers:
            optimizers["F"] = self._build_optimizer_F(optimizers)

    def _forward_generator(
        self, real_A: torch.Tensor, real_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
        """Run the generator forward pass and return the generated outputs."""
        batch_size = real_A.size(0)
        flipped_for_equivariance = False
        if self.model.nce_idt:
            real = torch.cat((real_A, real_B), dim=0)
            if self.model.flip_equivariance:
                flipped_for_equivariance = self.model.training and (np.random.random() < 0.5)
                if flipped_for_equivariance:
                    real = torch.flip(real, [3])
            fake = self.model.netG(real)
            fake_B = fake[:batch_size]
            idt_B = fake[batch_size:]
        else:
            real = real_A
            if self.model.flip_equivariance:
                flipped_for_equivariance = self.model.training and (np.random.random() < 0.5)
                if flipped_for_equivariance:
                    real = torch.flip(real, [3])
            fake_B = self.model.netG(real)
            idt_B = None
        return fake_B, idt_B, flipped_for_equivariance

    def _compute_discriminator_loss(
        self, fake_B: torch.Tensor, real_B: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (loss_D, loss_D_fake, loss_D_real)."""
        pred_fake = self.model.netD(fake_B.detach())
        loss_D_fake = self._find_gan_loss()(prediction=pred_fake, target_is_real=False)
        pred_real = self.model.netD(real_B)
        loss_D_real = self._find_gan_loss()(prediction=pred_real, target_is_real=True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, loss_D_fake, loss_D_real

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(loss_G, losses)`` where ``losses`` maps names to tensors.

        Only terms that were actually computed are included. Subclasses extend
        the returned dict with their own terms (``CPTStrategy`` adds
        ``loss_GP``/``loss_ASP``, USIGAN adds ``loss_UMC``/``loss_PCSM``/
        ``loss_CC``), and ``training_step`` logs every entry.
        """
        losses: dict[str, torch.Tensor] = {}

        loss_G_GAN = self._compute_gan_loss(fake_B)
        loss_G = loss_G_GAN * self.model.lambda_GAN
        losses["loss_G_GAN"] = loss_G_GAN

        if getattr(self.model, "lambda_NCE", 0.0) > 0.0:
            loss_NCE = self._compute_nce_loss(real_A, fake_B, flipped_for_equivariance)
            loss_G = loss_G + loss_NCE
            losses["loss_NCE"] = loss_NCE

            if self.model.nce_idt and idt_B is not None:
                loss_NCE_Y = self._compute_nce_y_loss(
                    real_B, idt_B, flipped_for_equivariance
                )
                # CUT reference averages the two NCE terms (0.5); CPT/MDCL
                # subclasses set ``nce_idt_weight = 1.0`` to sum them.
                loss_G = (
                    loss_G - loss_NCE + (loss_NCE + loss_NCE_Y) * self.nce_idt_weight
                )
                losses["loss_NCE_Y"] = loss_NCE_Y

        return loss_G, losses

    def _compute_gan_loss(self, fake_B: torch.Tensor) -> torch.Tensor:
        """Compute the generator adversarial loss."""
        pred_fake = self.model.netD(fake_B)
        return self._find_gan_loss()(prediction=pred_fake, target_is_real=True)

    def _compute_nce_loss(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> torch.Tensor:
        """Compute the source-to-fake PatchNCE loss."""
        return self._calculate_NCE_loss(real_A, fake_B, flipped_for_equivariance)

    def _compute_nce_y_loss(
        self,
        real_B: torch.Tensor,
        idt_B: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> torch.Tensor:
        """Compute the target-to-identity PatchNCE loss (CUT mode)."""
        return self._calculate_NCE_loss(real_B, idt_B, flipped_for_equivariance)

    def get_networks(self) -> dict[str, nn.Module]:
        networks: dict[str, nn.Module] = {
            "G": self.model.netG,
            "F": self.model.netF,
        }
        if self.model.netD is not None:
            networks["D"] = self.model.netD
        return networks

    def get_optimizers(
        self, config: "PuzzleStainConfig"
    ) -> dict[str, torch.optim.Optimizer]:
        opt_cfg = config.training
        weight_decay = getattr(opt_cfg, "weight_decay", 0.0)
        # netF's optimizer is created lazily in data_dependent_initialize once
        # its MLPs exist, mirroring the ASP reference.
        optimizers: dict[str, torch.optim.Optimizer] = {
            "G": torch.optim.Adam(
                self.model.netG.parameters(),
                lr=opt_cfg.lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                weight_decay=weight_decay,
            ),
        }
        if self.model.netD is not None:
            d_lr = opt_cfg.lr / self.model.d_lr_factor
            optimizers["D"] = torch.optim.Adam(
                self.model.netD.parameters(),
                lr=d_lr,
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
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda_rule)
        return schedulers

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

        fake_B, idt_B, flipped = self._forward_generator(real_A, real_B)
        losses: dict[str, float] = {}

        # ---- Update D ----
        if self.model.netD is not None and "D" in optimizers:
            opt_D = optimizers["D"]
            opt_D.zero_grad()
            loss_D, loss_D_fake, loss_D_real = self._compute_discriminator_loss(fake_B, real_B)
            loss_D.backward()
            opt_D.step()

            losses["loss_D_fake"] = loss_D_fake.item()
            losses["loss_D_real"] = loss_D_real.item()
            losses["loss_D"] = loss_D.item()

        # ---- Update G and F ----
        opt_G = optimizers["G"]
        opt_F = optimizers["F"]
        opt_G.zero_grad()
        opt_F.zero_grad()

        loss_G, g_losses = self._compute_generator_loss(
            real_A, real_B, fake_B, idt_B, flipped
        )
        loss_G.backward()
        opt_G.step()
        opt_F.step()

        for name, value in g_losses.items():
            losses[name] = value.item()
        losses["loss_G"] = loss_G.item()
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
                self.model.netG = networks["G"]
            if "F" in networks:
                self.model.netF = networks["F"]
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

    def _calculate_NCE_loss(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        flipped_for_equivariance: bool,
    ) -> torch.Tensor:
        """Compute the PatchNCE loss between ``src`` and ``tgt``."""
        feat_q = self.model.netG(tgt, self.model.nce_layers, encode_only=True)
        if self.model.flip_equivariance and flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = self.model.netG(src, self.model.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.model.netF(feat_k, self.model.num_patches, None)
        feat_q_pool, _ = self.model.netF(feat_q, self.model.num_patches, sample_ids)

        total = 0.0
        n_layers = len(self.model.nce_layers)
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            total = total + self._find_patchnce_loss()(feat_q=f_q, feat_k=f_k)
        return (total / n_layers) * self.model.lambda_NCE

    def _find_patchnce_loss(self) -> StainLoss:
        """Extract the PatchNCELoss component from the composite loss."""
        if isinstance(self.loss_fn, StainLoss) and self.loss_fn.name() == "patchnce":
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if loss_fn.name() == "patchnce":
                    return loss_fn
        raise RuntimeError("No PatchNCELoss found in the configured loss function")


__all__ = ["BaseCUTStrategy"]
