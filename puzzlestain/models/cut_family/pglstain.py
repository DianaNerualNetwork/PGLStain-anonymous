"""PGLStain model and training strategy.

PGLStain extends CPT with two independently configurable paper modules:
Protein Expression Aware Correspondence Calibration (PECC) and Marginal
Relational Stain Alignment (MRSA). The strategy keeps the established loss
order while exposing one clean model entry point.
"""

from __future__ import annotations

import torch

from ..registry import ModelRegistry
from .base import BaseCUTProcessor
from .cpt import CPTModel, CPTStrategy


class PGLStainModel(CPTModel):
    """PGLStain: CPT with PECC and MRSA supervision.

    Args:
        lambda_pecc: Weight for the hierarchical graph contrastive loss.
        lambda_od_struct: Weight for PECC OD-structure matching.
        lambda_od_set: Weight for MRSA marginal alignment.
        lambda_od_graph: Weight for MRSA relational alignment.
        pecc_layers: Comma-separated subset of ``nce_layers`` on which the
            graph contrastive loss runs (e.g. ``"8,12,16"``).
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        ngf: int = 64,
        ndf: int = 32,
        netG: str = "resnet_6blocks",
        netD: str = "n_layers",
        netF: str = "mlp_sample",
        n_layers_D: int = 5,
        normG: str = "instance",
        normD: str = "instance",
        init_type: str = "xavier",
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
        lambda_gp: float = 10.0,
        gp_weights: list[float] | str = "uniform",
        lambda_pecc: float = 0.1,
        lambda_od_struct: float = 0.1,
        lambda_od_set: float = 5.0,
        lambda_od_graph: float = 1.0,
        pecc_layers: str = "8,12,16",
        crop_size: int = 512,
        gpu_ids: list[int] | None = None,
        no_d: bool = False,
        weight_norm: str = "spectral",
        d_lr_factor: float = 1.0,
    ) -> None:
        self.lambda_pecc = lambda_pecc
        self.lambda_od_struct = lambda_od_struct
        self.lambda_od_set = lambda_od_set
        self.lambda_od_graph = lambda_od_graph
        for name, value in {
            "lambda_pecc": lambda_pecc,
            "lambda_od_struct": lambda_od_struct,
            "lambda_od_set": lambda_od_set,
            "lambda_od_graph": lambda_od_graph,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.pecc_layers = [int(x) for x in pecc_layers.split(",")]
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
            lambda_gp=lambda_gp,
            gp_weights=gp_weights,
            crop_size=crop_size,
            gpu_ids=gpu_ids,
            no_d=no_d,
            weight_norm=weight_norm,
            d_lr_factor=d_lr_factor,
        )


class PGLStainStrategy(CPTStrategy):
    """Training strategy for PGLStain."""

    def get_networks(self) -> dict[str, torch.nn.Module]:
        """Return the CUT networks plus the PECC module for checkpointing."""
        networks = super().get_networks()
        if self.model.lambda_pecc > 0.0:
            pecc = self._find_loss_by_name("pecc_correspondence")
            if not isinstance(pecc, torch.nn.Module):
                raise RuntimeError(
                    "The pecc_correspondence loss component must be an nn.Module"
                )
            # Store the correspondence module with its paper-facing name.
            networks["PECC"] = pecc
        return networks

    def _pecc_layer_indices(self) -> list[int]:
        """Map ``model.pecc_layers`` to indices into ``model.nce_layers``."""
        selected = set(self.model.pecc_layers)
        indices = [
            i for i, layer in enumerate(self.model.nce_layers) if layer in selected
        ]
        if len(indices) != len(selected):
            raise ValueError(
                f"pecc_layers {self.model.pecc_layers} must be a subset of "
                f"nce_layers {self.model.nce_layers}"
            )
        return indices

    def _compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_B: torch.Tensor,
        idt_B: torch.Tensor | None,
        flipped_for_equivariance: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return CPT losses plus enabled PECC and MRSA terms."""
        loss_G, losses = super()._compute_generator_loss(
            real_A, real_B, fake_B, idt_B, flipped_for_equivariance
        )
        model = self.model

        if model.lambda_pecc > 0.0:
            # Extract features at all NCE layers, then keep the PECC subset.
            feat_fake_B = model.netG(fake_B, model.nce_layers, encode_only=True)
            feat_real_B = model.netG(real_B, model.nce_layers, encode_only=True)
            feats_fake, sample_ids = model.netF(feat_fake_B, model.num_patches, None)
            feats_real, _ = model.netF(feat_real_B, model.num_patches, sample_ids)

            indices = self._pecc_layer_indices()
            sub_fake = [feats_fake[i] for i in indices]
            sub_real = [feats_real[i] for i in indices]
            sub_ids = [sample_ids[i] for i in indices]
            sub_sizes = [
                (feat_fake_B[i].shape[2], feat_fake_B[i].shape[3]) for i in indices
            ]

            pecc_loss = self._find_loss_by_name("pecc_correspondence")
            if pecc_loss is None:
                raise RuntimeError(
                    "No pecc_correspondence loss found in the configured composite loss"
                )
            loss_PECC = (
                pecc_loss(
                    feat_s=sub_fake,
                    feat_t=sub_real,
                    image_s=fake_B,
                    image_t=real_B,
                    patch_ids=sub_ids,
                    feat_map_sizes=sub_sizes,
                )
                * model.lambda_pecc
            )
            loss_G = loss_G + loss_PECC
            losses["loss_PECC"] = loss_PECC

        if model.lambda_od_struct > 0.0:
            struct_loss = self._find_loss_by_name("pecc_structure")
            if struct_loss is None:
                raise RuntimeError(
                    "No pecc_structure loss found in the configured composite loss"
                )
            loss_ODStruct = (
                struct_loss(image_s=fake_B, image_t=real_B) * model.lambda_od_struct
            )
            loss_G = loss_G + loss_ODStruct
            losses["loss_ODStruct"] = loss_ODStruct

        if model.lambda_od_set > 0.0:
            marginal_loss = self._find_loss_by_name("mrsa_marginal")
            if marginal_loss is None:
                raise RuntimeError(
                    "No mrsa_marginal loss found in the configured composite loss"
                )
            loss_ODSet = (
                marginal_loss(image_s=fake_B, image_t=real_B)
                * model.lambda_od_set
            )
            loss_G = loss_G + loss_ODSet
            losses["loss_ODSet"] = loss_ODSet

        if model.lambda_od_graph > 0.0:
            relational_loss = self._find_loss_by_name("mrsa_relational")
            if relational_loss is None:
                raise RuntimeError(
                    "No mrsa_relational loss found in the configured composite loss"
                )
            loss_ODGraph = (
                relational_loss(image_s=fake_B, image_t=real_B)
                * model.lambda_od_graph
            )
            loss_G = loss_G + loss_ODGraph
            losses["loss_ODGraph"] = loss_ODGraph

        return loss_G, losses


ModelRegistry.register(
    "pglstain",
    model_cls=PGLStainModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=PGLStainStrategy,
)

# Compatibility aliases for checkpoints using development model identifiers.
for legacy_name in (
    "od_guided_higraph_v3",
    "od_guided_higraph_v3_softgate_odset_odugw",
):
    ModelRegistry.register(
        legacy_name,
        model_cls=PGLStainModel,
        processor_cls=BaseCUTProcessor,
        strategy_cls=PGLStainStrategy,
    )


__all__ = ["PGLStainModel", "PGLStainStrategy"]
