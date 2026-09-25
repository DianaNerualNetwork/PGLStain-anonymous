"""Inference package for PuzzleStain.

Exports the public :class:`StainPredictor` used to generate virtual-stain images
from trained checkpoints.
"""

from __future__ import annotations

from .predictor import StainPredictor
from .spatial import PhysicalInferenceSpec

__all__ = ["PhysicalInferenceSpec", "StainPredictor"]
