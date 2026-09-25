"""VGG feature matching loss for virtual staining.

Mirrors the original Pix2PixHD VGGLoss but exposed as a reusable StainLoss
component so it can be composed with GAN and pixel losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models

from .base import LossRegistry, StainLoss


class Vgg19(torch.nn.Module):
    """VGG19 feature extractor used by VGGLoss."""

    def __init__(self, requires_grad: bool = False) -> None:
        super().__init__()
        try:
            vgg_pretrained_features = models.vgg19(weights="DEFAULT").features
        except TypeError:
            vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor) -> list[torch.Tensor]:
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class VGGLoss(nn.Module, StainLoss):
    """VGG feature matching loss.

    Computes a weighted L1 distance between VGG19 feature maps of the
    prediction and target. The weights follow the original Pix2PixHD repo.

    Args:
        lambda_feat: Overall multiplier for the loss (default 10.0), matching
            the original Pix2PixHD ``lambda_feat`` option.
        move_to_device_on_first_call: If True (default), the frozen VGG network
            is moved to the same device as the input tensors on the first call.
            This avoids assuming a specific device at construction time.
    """

    def __init__(
        self,
        lambda_feat: float = 10.0,
        move_to_device_on_first_call: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_feat = lambda_feat
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self._device_moved = not move_to_device_on_first_call

    def __call__(
        self,
        pred: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if pred is None or target is None:
            raise ValueError("VGGLoss requires 'pred' and 'target' kwargs")

        if not self._device_moved:
            self.vgg = self.vgg.to(pred.device)
            self._device_moved = True

        x_vgg, y_vgg = self.vgg(pred), self.vgg(target)
        loss = 0.0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return torch.tensor(loss * self.lambda_feat, device=pred.device) if not isinstance(loss, torch.Tensor) else loss * self.lambda_feat

    def name(self) -> str:
        return "vgg"

    def required_kwargs(self) -> set[str]:
        return {"pred", "target"}


LossRegistry.register("vgg", VGGLoss)

__all__ = ["VGGLoss", "Vgg19"]
