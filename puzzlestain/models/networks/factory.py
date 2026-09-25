"""Factory functions for building pix2pix-compatible generators and discriminators.

Original source: https://github.com/bupt-ai-cz/BCI (PyramidPix2pix)
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import torch.nn as nn

from .discriminator import ConvDiscriminator, NLayerDiscriminator, PixelDiscriminator
from .generator import (
    AttentionUnetGenerator,
    ResnetGenerator,
    UnetGenerator,
)
from .utils import get_norm_layer, init_net


def define_G(
    input_nc: int,
    output_nc: int,
    ngf: int,
    netG: str,
    norm: str = "batch",
    use_dropout: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
    n_downsampling: int = 2,
) -> nn.Module:
    """Create a generator.

    Args:
        input_nc: Number of channels in input images.
        output_nc: Number of channels in output images.
        ngf: Number of filters in the last conv layer.
        netG: Architecture name: ``resnet_9blocks`` | ``resnet_6blocks`` |
            ``unet_256`` | ``unet_128`` | ``attention_unet_32``.
        norm: Normalization layer: ``batch`` | ``instance`` | ``none``.
        use_dropout: Whether to use dropout layers.
        init_type: Weight initialization method.
        init_gain: Scaling factor.
        n_downsampling: Number of downsampling blocks (ResNet generators only;
            TDKStain uses 3, the pix2pix default is 2).

    Returns:
        Initialized generator network.
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
            n_blocks=9, n_downsampling=n_downsampling,
        )
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(
            input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
            n_blocks=6, n_downsampling=n_downsampling,
        )
    elif netG == "unet_128":
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "unet_256":
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "attention_unet_32":
        net = AttentionUnetGenerator(
            input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout
        )
    else:
        raise NotImplementedError(f"Generator model name [{netG}] is not recognized")
    return init_net(net, init_type, init_gain)


def define_D(
    input_nc: int,
    ndf: int,
    netD: str,
    n_layers_D: int = 3,
    norm: str = "batch",
    init_type: str = "normal",
    init_gain: float = 0.02,
) -> nn.Module:
    """Create a discriminator.

    Args:
        input_nc: Number of channels in input images.
        ndf: Number of filters in the first conv layer.
        netD: Architecture name: ``basic`` | ``n_layers`` | ``pixel`` | ``conv``.
        n_layers_D: Number of conv layers when ``netD == "n_layers"``.
        norm: Normalization layer.
        init_type: Weight initialization method.
        init_gain: Scaling factor.

    Returns:
        Initialized discriminator network.
    """
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "pixel":
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == "conv":
        net = ConvDiscriminator(input_nc)
    else:
        raise NotImplementedError(f"Discriminator model name [{netD}] is not recognized")
    return init_net(net, init_type, init_gain)
