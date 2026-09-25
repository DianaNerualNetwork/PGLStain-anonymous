"""Generator architectures adapted from PyramidPix2pix / pix2pix.

Original source: https://github.com/bupt-ai-cz/BCI (PyramidPix2pix)
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn as nn


class ResnetGenerator(nn.Module):
    """Resnet-based generator with downsampling/upsampling blocks."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = "reflect",
        n_downsampling: int = 2,
    ) -> None:
        super().__init__()
        if n_blocks < 0:
            raise ValueError("n_blocks must be non-negative")

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func is nn.InstanceNorm2d
        else:
            use_bias = norm_layer is nn.InstanceNorm2d

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    ngf * mult // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult // 2),
                nn.ReLU(True),
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(
        self, input: torch.Tensor, layers: list[int] = [], encode_only: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | list[torch.Tensor]:
        """Forward pass, optionally tapping intermediate encoder features.

        With ``layers`` empty (the default) this is the plain image-to-image
        forward. With layer indices given, returns ``(output, feats)``, or
        just ``feats`` when ``encode_only`` stops the pass at the last tapped
        layer (mirrors the CUT family's feature-extraction convention).
        """
        # Avoid mutating a caller-provided list across calls.
        layers = list(layers)
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats: list[torch.Tensor] = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return feat, feats
        return self.model(input)


class ResnetBlock(nn.Module):
    """ResNet block with skip connection."""

    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer: Callable[..., nn.Module],
        use_dropout: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.conv_block = self._build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    @staticmethod
    def _build_conv_block(
        dim: int,
        padding_type: str,
        norm_layer: Callable[..., nn.Module],
        use_dropout: bool,
        use_bias: bool,
    ) -> nn.Sequential:
        conv_block: list[nn.Module] = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x + self.conv_block(x)


class UnetGenerator(nn.Module):
    """U-Net generator with skip connections."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int = 64,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    """U-Net submodule with skip connection.

    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func is nn.InstanceNorm2d
        else:
            use_bias = norm_layer is nn.InstanceNorm2d

        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down: list[nn.Module] = [downconv]
            up: list[nn.Module] = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class AttentionUnetGenerator(nn.Module):
    """Attention U-Net generator used in PyramidPix2pix."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int = 64,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )
        for _ in range(num_downs - 5):
            unet_block = AttentionUnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        unet_block = AttentionUnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = AttentionUnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = AttentionUnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.model(x)


class AttentionUnetSkipConnectionBlock(nn.Module):
    """Attention-gated U-Net submodule."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        self.innermost = innermost
        if input_nc is None:
            input_nc = outer_nc

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func is nn.InstanceNorm2d
        else:
            use_bias = norm_layer is nn.InstanceNorm2d

        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)

        self.W = nn.Sequential(
            nn.Conv2d(outer_nc, outer_nc, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outer_nc),
        )
        self.theta = nn.Conv2d(
            outer_nc, outer_nc, kernel_size=2, stride=2, padding=0, bias=False
        )
        self.phi = nn.Conv2d(
            inner_nc * 2, outer_nc, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.psi = nn.Conv2d(
            outer_nc, 1, kernel_size=1, stride=1, padding=0, bias=True
        )

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            gating = down
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
            gating = down
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model += [nn.Dropout(0.5)]
            gating = down + [submodule]

        self.model = nn.Sequential(*model)
        self.gating = nn.Sequential(*gating)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self.outermost:
            return self.model(x)
        if self.innermost:
            return torch.cat([x, self.model(x)], 1)

        input_size = x.size()
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        g = self.gating(x)
        phi_g = nn.functional.interpolate(
            self.phi(g), size=theta_x_size[2:], mode="bilinear", align_corners=False
        )
        frelu = nn.functional.relu(theta_x + phi_g, inplace=False)
        sigm_psif = torch.sigmoid(self.psi(frelu))
        sigm_psi_f = nn.functional.interpolate(
            sigm_psif, size=input_size[2:], mode="bilinear", align_corners=False
        )
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return torch.cat([W_y, self.model(x)], 1)
