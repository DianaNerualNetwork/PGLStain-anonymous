"""Loss components for PGLStain and the included comparison methods."""

from __future__ import annotations

from .base import CompositeLoss, LossRegistry, StainLoss
from . import (
    asp,
    cacm,
    cc_matrix,
    cycle_consistency,
    focal_nce,
    gan,
    gauss_pyramid,
    gauss_pyramid_l1,
    gbclm,
    ldam,
    mix_asp,
    mix_patchnce,
    mrsa,
    ot_consistency,
    pals,
    patch_alignment,
    patchnce,
    pcls,
    pcsm,
    pecc,
    pyramid,
    ssim,
    vgg,
)
from .gan import GANLoss
from .gauss_pyramid import Gauss_Pyramid_Conv
from .gauss_pyramid_l1 import GaussPyramidL1Loss
from .gbclm import GNNLoss
from .mrsa import MRSAMarginalLoss, MRSARelationalLoss
from .patchnce import PatchNCELoss
from .pecc import PECCCorrespondenceLoss, PECCStructureLoss
from .pyramid import PyramidL1Loss

__all__ = [
    "CompositeLoss",
    "GANLoss",
    "GNNLoss",
    "GaussPyramidL1Loss",
    "Gauss_Pyramid_Conv",
    "LossRegistry",
    "MRSAMarginalLoss",
    "MRSARelationalLoss",
    "PECCCorrespondenceLoss",
    "PECCStructureLoss",
    "PatchNCELoss",
    "PyramidL1Loss",
    "StainLoss",
]
