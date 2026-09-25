"""GAN loss adapted from PyramidPix2pix / pix2pix.

Original source: https://github.com/bupt-ai-cz/BCI (PyramidPix2pix)
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LossRegistry, StainLoss


class GANLoss(nn.Module, StainLoss):
    """GAN objective supporting vanilla, LSGAN, WGAN-GP, hinge, nonsaturating."""

    def __init__(
        self,
        gan_mode: str,
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ("wgangp", "nonsaturating", "hinge"):
            self.loss = None
        else:
            raise NotImplementedError(f"gan mode {gan_mode} not implemented")

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Create a label tensor with the same size and device as the prediction."""
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction).to(prediction.device)

    def __call__(
        self,
        prediction: torch.Tensor | list[list[torch.Tensor]] | None = None,
        target_is_real: bool | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Calculate GAN loss.

        Supports two discriminator output formats:
          - Single tensor (standard PatchGAN output).
          - List of lists of tensors (Pix2PixHD multi-scale discriminator,
            where the last tensor of each inner list is the scale's final
            prediction).

        Args:
            prediction: Discriminator output map or multi-scale outputs.
            target_is_real: Whether the target label is real (True) or fake (False).

        Returns:
            Scalar GAN loss.
        """
        if prediction is None or target_is_real is None:
            raise ValueError("GANLoss requires 'prediction' and 'target_is_real' kwargs")

        # Pix2PixHD multi-scale format: list of lists, take the last prediction
        # of each scale.
        if isinstance(prediction, list) and len(prediction) > 0 and isinstance(prediction[0], list):
            loss = torch.tensor(0.0, device=prediction[0][0].device)
            for input_i in prediction:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)  # type: ignore[union-attr]
            return loss

        if self.gan_mode in ("lsgan", "vanilla"):
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            return self.loss(prediction, target_tensor)  # type: ignore[union-attr]
        if self.gan_mode == "wgangp":
            return (-prediction.mean() if target_is_real else prediction.mean())
        if self.gan_mode == "hinge":
            # Discriminator-side hinge terms: relu(1 - real) / relu(1 + fake).
            # The generator's hinge objective is ``-prediction.mean()``; since
            # ``target_is_real=True`` here produces the D-real term, strategies
            # compute the G-side term directly.
            if target_is_real:
                return F.relu(1.0 - prediction).mean()
            return F.relu(1.0 + prediction).mean()
        if self.gan_mode == "nonsaturating":
            # Mirrors the CUT repo: per-sample softplus reduced to a scalar.
            if target_is_real:
                return F.softplus(-prediction).mean()
            return F.softplus(prediction).mean()
        raise NotImplementedError(f"gan mode {self.gan_mode} not implemented")

    def name(self) -> str:
        return f"gan_{self.gan_mode}"

    def required_kwargs(self) -> set[str]:
        return {"prediction", "target_is_real"}


LossRegistry.register("gan", GANLoss)

__all__ = ["GANLoss"]
