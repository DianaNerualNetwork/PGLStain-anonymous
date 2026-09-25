"""Schrödinger-Bridge (SB) network architectures for UNSB.

Ported from the original UNSB repository (ICLR 2024).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    """Per-feature L2 normalization used by the z-mapping network."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embedding (transformer frequencies)."""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding followed by a 2-layer MLP."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        act: nn.Module = nn.LeakyReLU(0.2),
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, temp: torch.Tensor) -> torch.Tensor:  # noqa: D102
        temb = get_timestep_embedding(temp, self.embedding_dim)
        return self.main(temb)


class AdaptiveLayer(nn.Module):
    """FiLM conditioning layer: ``gamma(z) * x + beta(z)``."""

    def __init__(self, in_channel: int, style_dim: int) -> None:
        super().__init__()
        self.style_net = nn.Linear(style_dim, in_channel * 2)
        # NOTE: these bias tweaks are overwritten by the global weight init
        # (init_net sets all Linear biases to 0), exactly as in the original
        # repo, where define_G applies init_net after construction.
        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, input: torch.Tensor, style: torch.Tensor) -> torch.Tensor:  # noqa: D102
        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        return gamma * input + beta


class ResnetBlock_cond(nn.Module):
    """ResNet block with timestep and latent-code conditioning."""

    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer: Callable[..., nn.Module],
        use_dropout: bool,
        use_bias: bool,
        temb_dim: int,
        z_dim: int,
    ) -> None:
        super().__init__()
        self.conv_block, self.adaptive, self.conv_fin = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, temb_dim, z_dim
        )

    def build_conv_block(
        self,
        dim: int,
        padding_type: str,
        norm_layer: Callable[..., nn.Module],
        use_dropout: bool,
        use_bias: bool,
        temb_dim: int,
        z_dim: int,
    ) -> tuple[nn.ModuleList, AdaptiveLayer, nn.ModuleList]:
        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()
        p = 0
        if padding_type == "reflect":
            self.conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            self.conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")

        self.conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]
        self.adaptive = AdaptiveLayer(dim, z_dim)
        self.conv_fin += [nn.ReLU(True)]
        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            self.conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            self.conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(f"padding [{padding_type}] is not implemented")
        self.conv_fin += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        self.Dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.Dense_time.bias)

        self.style = nn.Linear(z_dim, dim * 2)
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

        return self.conv_block, self.adaptive, self.conv_fin

    def forward(
        self, x: torch.Tensor, time_cond: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        time_input = self.Dense_time(time_cond)
        out = x
        for n, layer in enumerate(self.conv_block):
            out = layer(out)
            if n == 0:
                # Time vector added after the first ReflectionPad, before conv1
                # (unusual placement ported exactly from the original repo).
                out = out + time_input[:, :, None, None]
        out = self.adaptive(out, z)
        for layer in self.conv_fin:
            out = layer(out)
        return x + out


class ResnetGenerator_ncsn(nn.Module):
    """Timestep/latent-conditioned ResNet generator (SB transition kernel).

    ``forward(x, time_cond, z)`` runs the full translation; the CUT-style
    ``layers``/``encode_only`` path extracts encoder features for PatchNCE.
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 9,
        padding_type: str = "reflect",
        n_mlp: int = 3,
    ) -> None:
        assert n_blocks >= 0
        super().__init__()
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
        self.ngf = ngf
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
                Downsample(ngf * mult * 2),
            ]
        self.model_res = nn.ModuleList()
        mult = 2**n_downsampling
        for _ in range(n_blocks):
            self.model_res += [
                ResnetBlock_cond(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                    temb_dim=4 * ngf,
                    z_dim=4 * ngf,
                )
            ]

        model_upsample: list[nn.Module] = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_upsample += [
                Upsample(ngf * mult),
                nn.Conv2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]
        model_upsample += [nn.ReflectionPad2d(3)]
        model_upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_upsample = nn.Sequential(*model_upsample)

        mapping_layers: list[nn.Module] = [
            PixelNorm(),
            nn.Linear(self.ngf * 4, self.ngf * 4),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(n_mlp):
            mapping_layers.append(nn.Linear(self.ngf * 4, self.ngf * 4))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)

        modules_emb: list[nn.Module] = [nn.Linear(self.ngf, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        modules_emb += [nn.Linear(self.ngf * 4, self.ngf * 4)]
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)

    def forward(
        self,
        x: torch.Tensor,
        time_cond: torch.Tensor,
        z: torch.Tensor,
        layers: list[int] | None = None,
        encode_only: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        layers = layers or []
        z_embed = self.z_transform(z)
        temb = get_timestep_embedding(time_cond, self.ngf)
        time_embed = self.time_embed(temb)
        if len(layers) > 0:
            feat = x
            feats: list[torch.Tensor] = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
            for layer_id, layer in enumerate(self.model_res):
                feat = layer(feat, time_embed, z_embed)
                if layer_id + len(self.model) in layers:
                    feats.append(feat)
                if layer_id + len(self.model) == layers[-1] and encode_only:
                    return feats
            return feat, feats

        out = self.model(x)
        for layer in self.model_res:
            out = layer(out, time_embed, z_embed)
        return self.model_upsample(out)


class ConvBlock_cond(nn.Module):
    """Time-conditioned conv block for the PatchGAN discriminator."""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        t_emb_dim: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
        downsample: bool = True,
        use_bias: bool | None = None,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        if norm_layer is not None:
            self.use_norm = True
            self.norm = norm_layer(out_channel)
        else:
            self.use_norm = False
        self.act = nn.LeakyReLU(0.2, True)
        self.down = Downsample(out_channel)
        self.dense = nn.Linear(t_emb_dim, out_channel)

    def forward(self, input: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:  # noqa: D102
        out = self.conv1(input)
        # Time embedding added after conv, before norm (as in the original).
        out += self.dense(t_emb)[..., None, None]
        if self.use_norm:
            out = self.norm(out)
        out = self.act(out)
        if self.downsample:
            out = self.down(out)
        return out


class NLayerDiscriminator_ncsn(nn.Module):
    """Time-conditioned 70x70 PatchGAN discriminator / SB energy net.

    ``forward(input, t_emb, input2=None)`` channel-concatenates ``input2``
    when given; this is how the energy net consumes paired (x_t, x_{t+1})
    tensors (12 input channels).
    """

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
        self.model_main = nn.ModuleList()
        kw = 4
        padw = 1
        self.model_main.append(
            ConvBlock_cond(
                input_nc,
                ndf,
                4 * ndf,
                kernel_size=kw,
                stride=1,
                padding=padw,
                use_bias=use_bias,
            )
        )

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.model_main.append(
                ConvBlock_cond(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    4 * ndf,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    use_bias=use_bias,
                    norm_layer=norm_layer,
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                4 * ndf,
                kernel_size=kw,
                stride=1,
                padding=padw,
                use_bias=use_bias,
                norm_layer=norm_layer,
                downsample=False,
            )
        )
        self.final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.t_embed = TimestepEmbedding(
            embedding_dim=4 * ndf,
            hidden_dim=4 * ndf,
            output_dim=4 * ndf,
            act=nn.LeakyReLU(0.2),
        )

    def forward(
        self,
        input: torch.Tensor,
        t_emb: torch.Tensor,
        input2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_emb = self.t_embed(t_emb)
        if input2 is not None:
            out = torch.cat([input, input2], dim=1)
        else:
            out = input
        for layer in self.model_main:
            out = layer(out, t_emb)
        return self.final_conv(out)


def get_filter(filt_size: int = 3) -> torch.Tensor:
    """Return a normalized binomial filter kernel."""
    if filt_size == 1:
        a = np.array([1.0])
    elif filt_size == 2:
        a = np.array([1.0, 1.0])
    elif filt_size == 3:
        a = np.array([1.0, 2.0, 1.0])
    elif filt_size == 4:
        a = np.array([1.0, 3.0, 3.0, 1.0])
    elif filt_size == 5:
        a = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
    elif filt_size == 6:
        a = np.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])
    elif filt_size == 7:
        a = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])
    else:
        raise ValueError(f"Unsupported filter size: {filt_size}")

    filt = torch.Tensor(a[:, None] * a[None, :])
    return filt / torch.sum(filt)


def get_pad_layer(pad_type: str) -> type[nn.Module]:
    """Return the padding layer class for the given padding type."""
    if pad_type in ("refl", "reflect"):
        PadLayer: type[nn.Module] = nn.ReflectionPad2d
    elif pad_type in ("repl", "replicate"):
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        raise NotImplementedError(f"Pad type [{pad_type}] not recognized")
    return PadLayer


class Downsample(nn.Module):
    """Antialiased stride-2 downsampling (FIR filter)."""

    def __init__(
        self,
        channels: int,
        pad_type: str = "reflect",
        filt_size: int = 3,
        stride: int = 2,
        pad_off: int = 0,
    ) -> None:
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(np.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample(nn.Module):
    """Antialiased stride-2 transposed-conv upsampling (FIR filter)."""

    def __init__(self, channels: int, pad_type: str = "repl", filt_size: int = 4, stride: int = 2) -> None:
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer("filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:  # noqa: D102
        ret_val = F.conv_transpose2d(
            self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1]
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        return ret_val[:, :, :-1, :-1]
