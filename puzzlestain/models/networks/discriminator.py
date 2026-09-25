"""Discriminator architectures adapted from PyramidPix2pix / pix2pix.

Original source: https://github.com/bupt-ai-cz/BCI (PyramidPix2pix)
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn as nn


class ConvDiscriminator(nn.Module):
    """Single-convolution discriminator used for the ``conv`` pattern."""

    def __init__(self, input_nc: int) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(input_nc, 64, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.conv1_1(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (``basic`` / ``n_layers``)."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func is nn.InstanceNorm2d
        else:
            use_bias = norm_layer is nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(x)


class PixelDiscriminator(nn.Module):
    """1x1 PixelGAN discriminator."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func is nn.InstanceNorm2d
        else:
            use_bias = norm_layer is nn.InstanceNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)
