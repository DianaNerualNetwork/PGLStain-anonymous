"""MDCL (Mix-Domain Contrastive Learning) for the CUT family.

MDCL augments every contrastive term with same-domain query negatives: the
PatchNCE branches use ``mix_patchnce`` and the adaptive supervised branch uses
``mix_asp``. The model is the full CPT architecture (CUT + Gaussian-pyramid
reconstruction + adaptive supervised PatchNCE), matching the original repo's
``python -m experiments mist train 0`` (``model=cpt``, ``CUT_mode=CUT``).

Reference:
    https://github.com/ssongwang/mix-domaincontrastivelearning/models/cpt_model.py
"""

from __future__ import annotations

from ..loss.base import CompositeLoss, StainLoss
from ..registry import ModelRegistry
from .asp import ASPModel
from .base import BaseCUTProcessor
from .cpt import CPTStrategy


class MixASPStrategy(CPTStrategy):
    """Training strategy with the mix-domain adaptive supervised loss.

    Inherits the CPT-layer ASP machinery and looks up the adaptive supervised
    loss by the registered name ``"mix_asp"``.
    """

    def _find_asp_loss(self) -> StainLoss:
        """Extract the MixAdaptiveSupervisedPatchNCELoss component."""
        name = "mix_asp"
        if isinstance(self.loss_fn, StainLoss) and self.loss_fn.name() == name:
            return self.loss_fn
        if isinstance(self.loss_fn, CompositeLoss):
            for loss_fn, _ in self.loss_fn.losses:
                if loss_fn.name() == name:
                    return loss_fn
        raise RuntimeError(
            f"No {name} loss found in the configured composite loss"
        )


class MDCLStrategy(MixASPStrategy):
    """Training strategy for MDCL.

    Full CPT step whose contrastive terms are looked up by their mix-domain
    registered names (``mix_patchnce`` for the NCE branches, ``mix_asp`` for
    the adaptive supervised branch), and whose NCE identity terms are summed
    rather than averaged, mirroring the MDCL repo's ``cpt_model.py``.
    """

    nce_idt_weight = 1.0

    def _find_patchnce_loss(self) -> StainLoss:
        """Extract the MixPatchNCELoss component from the configured composite loss."""
        name = "mix_patchnce"
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
    "mdcl",
    model_cls=ASPModel,
    processor_cls=BaseCUTProcessor,
    strategy_cls=MDCLStrategy,
)

__all__ = ["MDCLStrategy", "MixASPStrategy"]
