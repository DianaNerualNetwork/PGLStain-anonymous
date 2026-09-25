"""PPT model: Patch alignment-based bidirectional contrastive learning.

Ported from the PPT repository (MICCAI 2024, "High-resolution Medical Image
Translation via Patch Alignment-based Bidirectional Contrastive Learning",
``models/ppt.py`` and friends). On top of a CUT-style step (LSGAN + patch
contrastive loss with a lazily-built PatchSampleF projection head) it uses:

- FocalNCELoss (registered as ``focal_nce``): InfoNCE logits identical to
  PatchNCE, but the final criterion is a FocalLoss (alpha=0.25, gamma=2)
  against the positive class instead of cross-entropy. The contrastive term is
  bidirectional on the identity pair:
  ``NCE(A, fake_B) + NCE(B, idt_B) + NCE(idt_B, B)``.
- PatchAlignmentLoss (registered as ``patch_alignment``): mean L1 between 4x4
  unfolded patches (stride 4, padding 2) of ``real_B`` and ``fake_B``, scaled
  by beta=0.0025.
- VGGLoss (registered as ``vgg``): pix2pixHD-style VGG19 feature matching
  (``lambda_feat=1.0``; no ImageNet normalization, target detached).
- Frequency loss (registered as ``gauss_pyramid_l1``): L1 over a 6-level
  Gaussian pyramid, all weights 1.0.

All four are consumed from the configured composite loss, like the other
CUT-family methods.

Architecture deviation from the CUT family: the original PPT generator reads
``n_downsampling`` via ``getattr(opt, 'n_downsampling', 4)`` and the option is
never defined, so it uses *four* down/upsampling stages (bottleneck
``ngf * 16``) instead of CUT's two. ``PPTResnetGenerator`` reproduces this.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn as nn

from ..loss.base import CompositeLoss, StainLoss
from ..registry import ModelRegistry
from .base import BaseCUTProcessor
from .cut import CUTModel, CUTStrategy
from .networks.cut_networks import (
    Downsample,
    ResnetBlock,
    Upsample,
    get_norm_layer,
    init_net,
)
from .networks import define_D, define_F


class PPTResnetGenerator(nn.Module):
    """ResNet generator of the PPT repository (``models/networks.py``).

    Identical to the CUT-family ResnetGenerator except that it uses four
    anti-aliased down/upsampling stages (the original reads
    ``getattr(opt, 'n_downsampling', 4)`` and the option is never defined).
    Composed from the shared CUT building blocks, which were verified
    line-by-line identical to PPT's ``Downsample``/``Upsample``/``ResnetBlock``.
    Only the configuration PPT uses is implemented: ``padding_type='reflect'``,
    no dropout, ``no_antialias=False`` and ``no_antialias_up=False``.
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        norm_layer: Callable[[int], nn.Module] | None = None,
        n_blocks: int = 9,
        n_downsampling: int = 4,
    ) -> None:
        assert n_blocks >= 0
        super().__init__()
        if norm_layer is None:
            norm_layer = get_norm_layer("instance")
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

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type="reflect",
                    norm_layer=norm_layer,
                    use_dropout=False,
                    use_bias=use_bias,
                )
            ]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
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
        """Standard forward, or intermediate features for the NCE loss."""
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


class PPTModel(CUTModel):
    """PPT generator/projection/discriminator trio.

    Mirrors ``PPT_model`` from the original repository with its fixed
    hyperparameters: ResNet generator with 9 blocks and 4 downsampling stages,
    PatchSampleF projection (use_mlp=True, nc=256), NLayerDiscriminator with
    n_layers=3, instance norm, ``init_type='normal'`` and LSGAN. The forward
    pass always concatenates ``real_A`` and ``real_B`` (``nce_idt=True``,
    ``flip_equivariance=False``).
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 64,
        netG: str = "resnet_9blocks",  # noqa: ARG002
        netD: str = "n_layers",
        netF: str = "mlp_sample",
        n_layers_D: int = 3,
        normG: str = "instance",
        normD: str = "instance",
        init_type: str = "normal",
        init_gain: float = 0.02,
        no_dropout: bool = True,
        no_antialias: bool = False,
        no_antialias_up: bool = False,
        nce_idt: bool = True,
        lambda_GAN: float = 1.0,
        lambda_NCE: float = 1.0,
        nce_layers: str = "0,4,8,12,16",
        nce_T: float = 0.07,
        num_patches: int = 256,
        netF_nc: int = 256,
        flip_equivariance: bool = False,
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "none",
        d_lr_factor: float = 1.0,
    ) -> None:
        super().__init__(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            ndf=ndf,
            netG=netG,
            netD=netD,
            netF=netF,
            n_layers_D=n_layers_D,
            normG=normG,
            normD=normD,
            init_type=init_type,
            init_gain=init_gain,
            no_dropout=no_dropout,
            no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
            nce_idt=nce_idt,
            lambda_GAN=lambda_GAN,
            lambda_NCE=lambda_NCE,
            nce_layers=nce_layers,
            nce_T=nce_T,
            num_patches=num_patches,
            netF_nc=netF_nc,
            flip_equivariance=flip_equivariance,
            crop_size=crop_size,
            gpu_ids=gpu_ids,
            no_d=no_d,
            weight_norm=weight_norm,
            d_lr_factor=d_lr_factor,
        )

    def _build_networks(self, no_d: bool) -> None:
        """Build the PPT networks (4-stage ResnetGenerator + shared F and D)."""
        gpu_ids = self.gpu_ids if self.gpu_ids else []
        norm_layer = get_norm_layer(self.normG)
        netG = PPTResnetGenerator(
            self.input_nc,
            self.output_nc,
            self.ngf,
            norm_layer=norm_layer,
            n_blocks=9,
            n_downsampling=4,
        )
        self.netG = init_net(netG, self.init_type, self.init_gain, gpu_ids)
        self.netF = define_F(
            self.input_nc,
            self.netF_name,
            norm=self.normG,
            use_dropout=not self.no_dropout,
            init_type=self.init_type,
            init_gain=self.init_gain,
            no_antialias=self.no_antialias,
            gpu_ids=gpu_ids,
            opt=self,
        )
        if not no_d:
            self.netD = define_D(
                self.output_nc,
                self.ndf,
                self.netD_name,
                n_layers_D=self.n_layers_D,
                norm=self.normD,
                init_type=self.init_type,
                init_gain=self.init_gain,
                no_antialias=self.no_antialias,
                gpu_ids=gpu_ids,
                opt=self,
            )


class PPTStrategy(CUTStrategy):
    """Training strategy for PPT.

    Reuses the CUT-family D-then-G update and replaces the generator loss with
    the PPT objective (all coefficients 1.0), mirroring
    ``PPT_model.backward_G``::

        loss_G = loss_G_GAN + loss_freq + loss_contrast
                 + loss_patch_alignment + loss_content

    where ``loss_contrast = NCE(A, fake_B) + NCE(B, idt_B) + NCE(idt_B, B)``
    with the focal NCE criterion. All four PPT-specific terms are looked up
    from the configured composite loss by their registered names
    (``focal_nce`` / ``patch_alignment`` / ``vgg`` / ``gauss_pyramid_l1``).
    """

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,  # noqa: ARG002
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the PPT generator loss and its named components."""
        if idt_B is None:
            raise ValueError("PPT requires nce_idt=True so that idt_B exists")
        losses: dict[str, torch.Tensor] = {}

        # Adversarial loss (LSGAN on the fake prediction).
        loss_G_GAN = self._compute_gan_loss(fake_B)
        losses["loss_G_GAN"] = loss_G_GAN

        # Bidirectional contrastive loss with the focal criterion.
        loss_contrast = (
            self._calculate_focal_nce_loss(real_A, fake_B)
            + self._calculate_focal_nce_loss(real_B, idt_B)
            + self._calculate_focal_nce_loss(idt_B, real_B)
        )
        losses["loss_contrast"] = loss_contrast

        # Patch alignment loss between real_B and fake_B.
        loss_patch_alignment = self._find_loss_by_name("patch_alignment")(
            pred=real_B, target=fake_B
        )
        losses["loss_patch_alignment"] = loss_patch_alignment

        # VGG content loss (target detached inside VGGLoss).
        loss_content = self._find_loss_by_name("vgg")(pred=fake_B, target=real_B)
        losses["loss_content"] = loss_content

        # Frequency loss over the Gaussian pyramid (uniform level weights).
        loss_freq = self._find_loss_by_name("gauss_pyramid_l1")(
            pred=fake_B, target=real_B
        )
        losses["loss_freq"] = loss_freq

        loss_G = (
            loss_G_GAN + loss_freq + loss_contrast + loss_patch_alignment + loss_content
        )
        return loss_G, losses

    def _calculate_focal_nce_loss(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        """Focal NCE between ``src`` and ``tgt``, mirroring ``calculate_NCE_loss``."""
        feat_q = self.model.netG(tgt, self.model.nce_layers, encode_only=True)
        feat_k = self.model.netG(src, self.model.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.model.netF(feat_k, self.model.num_patches, None)
        feat_q_pool, _ = self.model.netF(feat_q, self.model.num_patches, sample_ids)

        focal_nce = self._find_loss_by_name("focal_nce")
        total: torch.Tensor | float = 0.0
        n_layers = len(self.model.nce_layers)
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            total = total + focal_nce(feat_q=f_q, feat_k=f_k)
        return total / n_layers

    def _find_loss_by_name(self, name: str) -> StainLoss:
        """Return the composite component whose :meth:`name` matches ``name``."""
        if isinstance(self.loss_fn, StainLoss) and self.loss_fn.name() == name:
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if loss_fn.name() == name:
                    return loss_fn
        raise RuntimeError(
            f"No {name} loss found in the configured composite loss"
        )


ModelRegistry.register(
    "ppt",
    model_cls=PPTModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=PPTStrategy,
)


__all__ = [
    "PPTModel",
    "PPTResnetGenerator",
    "PPTStrategy",
]
