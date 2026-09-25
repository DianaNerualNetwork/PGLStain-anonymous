"""UNSB (Unpaired Neural Schrödinger Bridge, ICLR 2024).

Faithful port of the original UNSB repository's ``SBModel``:
https://github.com/chenxy99/UNSB

Reference training command::

    python train.py --dataroot /path/to/MIST/ER/TrainValAB \
        --name mist_er --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0

The model/strategy logic lives in :mod:`.sb_base`; this module only provides
the registered variant. Differences from the original repo are documented in
:mod:`.sb_base` (second-batch source, single ``lambda_NCE`` weighting).
"""

from __future__ import annotations

from ..registry import ModelRegistry
from .base import BaseDiffusionProcessor
from .sb_base import BaseSBModel, BaseSBStrategy


class UNSBModel(BaseSBModel):
    """UNSB model: SB bridge with PatchNCE + identity NCE (``nce_idt=True``)."""

    pass


class UNSBStrategy(BaseSBStrategy):
    """Training strategy for UNSB."""

    pass


ModelRegistry.register(
    "unsb",
    model_cls=UNSBModel,
    processor_cls=BaseDiffusionProcessor,
    strategy_cls=UNSBStrategy,
)


__all__ = ["UNSBModel", "UNSBStrategy"]
