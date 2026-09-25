"""CUT network building blocks ported from the CUT repository.

Original source: https://github.com/taesungp/contrastive-unpaired-translation
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def get_filter(filt_size: int = 3) -> torch.Tensor:
    """Return a binomial filter kernel of size ``filt_size``.

    The kernel is normalized so that its elements sum to one.
    """
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
    filt = filt / torch.sum(filt)
    return filt


class Downsample(nn.Module):
    """Anti-aliased downsampling by convolving with a binomial filter."""

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
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, :: self.stride, :: self.stride]
            return self.pad(inp)[:, :, :: self.stride, :: self.stride]
        return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    """Nearest-neighbor or bilinear upsampling wrapper."""

    def __init__(self, scale_factor: int, mode: str = "nearest") -> None:
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    """Anti-aliased upsampling using a transposed convolution with a binomial filter."""

    def __init__(
        self,
        channels: int,
        pad_type: str = "repl",
        filt_size: int = 4,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        ret_val = F.conv_transpose2d(
            self.pad(inp),
            self.filt,
            stride=self.stride,
            padding=1 + self.pad_size,
            groups=inp.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type: str) -> type[nn.Module]:
    """Return a 2D padding layer class by name."""
    if pad_type in ("refl", "reflect"):
        return nn.ReflectionPad2d
    if pad_type in ("repl", "replicate"):
        return nn.ReplicationPad2d
    if pad_type == "zero":
        return nn.ZeroPad2d
    raise ValueError(f"Pad type [{pad_type}] not recognized")


class Identity(nn.Module):
    """Identity mapping module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_norm_layer(norm_type: str = "instance") -> Callable[[int], nn.Module]:
    """Return a 2D normalization layer constructor.

    Args:
        norm_type: One of ``batch``, ``instance``, or ``none``.
    """
    if norm_type == "batch":
        norm_layer: Callable[[int], nn.Module] = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":

        def norm_layer(x: int) -> nn.Module:  # noqa: ARG001
            return Identity()
    else:
        raise NotImplementedError(
            f"normalization layer [{norm_type}] is not found"
        )
    return norm_layer


def init_weights(
    net: nn.Module, init_type: str = "normal", init_gain: float = 0.02, debug: bool = False
) -> None:
    """Initialize network weights.

    Args:
        net: Network to initialize.
        init_type: One of ``normal``, ``xavier``, ``kaiming``, ``orthogonal``.
        init_gain: Scaling factor for normal, xavier and orthogonal.
        debug: Print layer names while initializing.
    """

    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if debug:
                print(classname)
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(
    net: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
    gpu_ids: list[int] | None = None,
    debug: bool = False,
    initialize_weights: bool = True,
) -> nn.Module:
    """Initialize a network's weights.

    ``gpu_ids`` is accepted for backward compatibility with older configs but
    is inert: device placement is handled by accelerate (``prepare``), and
    lazily-created submodules follow their input feature's device instead.
    """
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


def _apply_weight_norm(net: nn.Module, weight_norm: str = "none") -> None:
    """Wrap every conv layer of ``net`` per the ASP repo's ``weight_norm`` flag.

    ``"spectral"`` wraps all ``Conv2d``/``ConvTranspose2d`` layers with
    :func:`torch.nn.utils.spectral_norm` (mirroring the ASP repository);
    ``"none"`` leaves the network untouched. Applied before ``init_net`` so
    weight initialization still reaches the underlying conv weights.
    """
    if weight_norm == "none":
        return
    if weight_norm != "spectral":
        raise NotImplementedError(
            f"weight_norm [{weight_norm}] is not recognized"
        )
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.utils.spectral_norm(module)


def define_G(
    input_nc: int,
    output_nc: int,
    ngf: int,
    netG: str,
    norm: str = "batch",
    use_dropout: bool = False,
    init_type: str = "normal",
    init_gain: float = 0.02,
    no_antialias: bool = False,
    no_antialias_up: bool = False,
    gpu_ids: list[int] | None = None,
    opt: Any | None = None,
    weight_norm: str = "none",
) -> nn.Module:
    """Create a CUT/ContraAST generator.

    Only ResNet-based generators are supported here.
    """
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == "resnet_9blocks":
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
            n_blocks=9,
            opt=opt,
        )
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
            n_blocks=6,
            opt=opt,
        )
    elif netG == "resnet_4blocks":
        net = ResnetGenerator(
            input_nc,
            output_nc,
            ngf,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
            n_blocks=4,
            opt=opt,
        )
    else:
        raise NotImplementedError(
            f"Generator model name [{netG}] is not recognized"
        )
    _apply_weight_norm(net, weight_norm)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_F(
    input_nc: int,
    netF: str,
    norm: str = "batch",  # noqa: ARG001
    use_dropout: bool = False,  # noqa: ARG001
    init_type: str = "normal",
    init_gain: float = 0.02,
    no_antialias: bool = False,  # noqa: ARG001
    gpu_ids: list[int] | None = None,
    opt: Any | None = None,
) -> nn.Module:
    """Create the patch feature projection network ``F``."""
    nc = opt.netF_nc if opt is not None else 256
    if netF == "sample":
        net = PatchSampleF(
            use_mlp=False,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
            nc=nc,
        )
    elif netF == "mlp_sample":
        net = PatchSampleF(
            use_mlp=True,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
            nc=nc,
        )
    else:
        raise NotImplementedError(
            f"projection model name [{netF}] is not recognized"
        )
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(
    input_nc: int,
    ndf: int,
    netD: str,
    n_layers_D: int = 3,
    norm: str = "batch",
    init_type: str = "normal",
    init_gain: float = 0.02,
    no_antialias: bool = False,
    gpu_ids: list[int] | None = None,
    opt: Any | None = None,
    weight_norm: str = "none",
) -> nn.Module:
    """Create a PatchGAN discriminator."""
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == "basic":
        net = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias
        )
    elif netD == "n_layers":
        net = NLayerDiscriminator(
            input_nc,
            ndf,
            n_layers_D,
            norm_layer=norm_layer,
            no_antialias=no_antialias,
        )
    else:
        raise NotImplementedError(
            f"Discriminator model name [{netD}] is not recognized"
        )
    _apply_weight_norm(net, weight_norm)
    return init_net(net, init_type, init_gain, gpu_ids)


class Normalize(nn.Module):
    """L2 normalize along the channel dimension."""

    def __init__(self, power: int = 2) -> None:
        super().__init__()
        self.power = power

    def forward(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        norm = (x + 1e-7).pow(self.power).sum(dim, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out


class ResnetGenerator(nn.Module):
    """ResNet-based generator with optional anti-aliased down/upsampling."""

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
        n_blocks: int = 6,
        padding_type: str = "reflect",
        no_antialias: bool = False,
        no_antialias_up: bool = False,
        opt: Any | None = None,
    ) -> None:
        assert n_blocks >= 0
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            if no_antialias:
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
            else:
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
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            else:
                model += [
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

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(
        self, input: torch.Tensor, layers: list[int] = [], encode_only: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | list[torch.Tensor]:
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
    """Residual block used by the CUT generator."""

    def __init__(
        self,
        dim: int,
        padding_type: str,
        norm_layer: Callable[[int], nn.Module],
        use_dropout: bool,
        use_bias: bool,
    ) -> None:
        super().__init__()
        self.conv_block = self._build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def _build_conv_block(
        self,
        dim: int,
        padding_type: str,
        norm_layer: Callable[[int], nn.Module],
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
            raise NotImplementedError(
                f"padding [{padding_type}] is not implemented"
            )

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
            raise NotImplementedError(
                f"padding [{padding_type}] is not implemented"
            )
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class PatchSampleF(nn.Module):
    """Sample and project patch features for PatchNCE."""

    def __init__(
        self,
        use_mlp: bool = False,
        init_type: str = "normal",
        init_gain: float = 0.02,
        nc: int = 256,
        gpu_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids or []

    def create_mlp(self, feats: list[torch.Tensor]) -> None:
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(
                nn.Linear(input_nc, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc),
            )
            # The MLPs are created lazily at first forward, after accelerate
            # has placed the model, so follow the feature's device rather than
            # relying on gpu_ids (empty under accelerate-managed placement).
            mlp.to(feat.device)
            setattr(self, f"mlp_{mlp_id}", mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(
        self,
        feats: list[torch.Tensor],
        num_patches: int = 64,
        patch_ids: list[torch.Tensor] | None = None,
        return_all: bool = False,
    ) -> tuple[list[torch.Tensor], list[Any]] | tuple[
        list[torch.Tensor], list[torch.Tensor], list[Any]
    ]:
        """Sample and project patch features.

        Args:
            feats: List of encoder feature maps.
            num_patches: Number of patches to sample per feature map. ``0``
                returns all patches reshaped to spatial form.
            patch_ids: Optional pre-computed patch indices; if None, random
                indices are drawn for ``num_patches > 0``.
            return_all: If True, additionally return all patches (after MLP and
                L2 norm) reshaped to spatial form. Used by NEGCUT's
                ``neg_gen_al`` variant.

        Returns:
            By default ``(return_feats, return_ids)``. When ``return_all`` is
            True, ``(return_feats, return_feats_all, return_ids)``.
        """
        return_ids: list[Any] = []
        return_feats: list[torch.Tensor] = []
        return_feats_all: list[torch.Tensor] = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[: int(min(num_patches, patch_id.shape[0]))]
                if isinstance(patch_id, np.ndarray):
                    patch_id = torch.from_numpy(patch_id).long().to(feat.device)
                else:
                    patch_id = patch_id.long().to(feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            else:
                x_sample = feat_reshape
                patch_id = []

            if self.use_mlp:
                mlp = getattr(self, f"mlp_{feat_id}")
                x_sample = mlp(x_sample)

            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape(
                    [B, x_sample.shape[-1], H, W]
                )

            if return_all:
                # x_sample currently contains sampled patches when num_patches > 0;
                # rebuild the full spatial feature map from all patches.
                if num_patches > 0:
                    x_sample_all = feat_reshape.flatten(0, 1)
                    if self.use_mlp:
                        x_sample_all = mlp(x_sample_all)
                    x_sample_all = self.l2norm(x_sample_all)
                    x_sample_all = x_sample_all.view(B, -1, self.nc)
                    x_sample_all = x_sample_all.permute(0, 2, 1).reshape(
                        [B, self.nc, H, W]
                    )
                else:
                    x_sample_all = x_sample
                return_feats.append(x_sample)
                return_feats_all.append(x_sample_all)
            else:
                return_feats.append(x_sample)

        if return_all:
            return return_feats, return_feats_all, return_ids
        return return_feats, return_ids


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with optional anti-aliased downsampling."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        no_antialias: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            sequence: list[nn.Module] = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        else:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf),
            ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if no_antialias:
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
            else:
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
                    Downsample(ndf * nf_mult),
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


class NLayerBackbone(nn.Module):
    """PatchGAN backbone that outputs intermediate features instead of logits."""

    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: Callable[[int], nn.Module] = nn.InstanceNorm2d,
    ) -> None:
        super().__init__()
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
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        self.feat_dim = ndf * nf_mult
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = [
    "Downsample",
    "Upsample",
    "Upsample2",
    "get_filter",
    "get_pad_layer",
    "get_norm_layer",
    "Identity",
    "init_weights",
    "init_net",
    "define_G",
    "define_F",
    "define_D",
    "Normalize",
    "ResnetGenerator",
    "ResnetBlock",
    "PatchSampleF",
    "NLayerDiscriminator",
    "NLayerBackbone",
]
