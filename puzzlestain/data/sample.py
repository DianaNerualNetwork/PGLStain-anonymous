"""Unified, flat data container for a single virtual-staining sample.

A sample is indexed at the **stain level**: one source image paired with exactly
one target stain image. Multi-stain panels are unfolded into multiple stain-level
samples by the Dataset, so the collator and sampler never need special handling
for "some stains present / some missing".

``StainSample`` uses dynamic fields so that auxiliary images (DAB masks, nuclei
maps, etc.) can be attached without modifying this module.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np


class StainSample:
    """One (source, target_stain) pair or one labeled source image.

    Fields are stored dynamically via ``__dict__`` so callers can attach any
    number of auxiliary images (``dab_mask``, ``nuclei_map``, ...) without
    changing the class definition. The well-known fields below are documented
    here and remain accessible as attributes.

    Attributes:
        source_image: Input source patch, channels-last ``(H, W, C)``.
        target_image: Target stain patch, channels-last ``(H, W, C)``.
        target_stain: Stain name, e.g. ``"ER"``, ``"CD3"``.
        target_label: Integer class index, e.g. ``3`` for ER in the IHC4BC
            ordering. Used by condition-based models.
        target_onehot: One-hot class vector; computed from ``target_label`` when
            requested.
        input_keys: Field names consumed as model inputs.
        target_keys: Field names produced as supervision targets.
        metadata: Free-form per-sample provenance (paths, dataset uid, etc.).
    """

    source_image: Optional[np.ndarray] = None
    target_image: Optional[np.ndarray] = None
    target_stain: Optional[str] = None
    target_label: Optional[int] = None
    target_onehot: Optional[np.ndarray] = None
    input_keys: list[str]
    target_keys: list[str]
    metadata: dict[str, Any]

    def __init__(self, **fields: Any) -> None:
        """Construct a sample from keyword fields.

        Well-known defaults are provided when omitted; unknown fields are stored
        as-is so auxiliary signals can be attached by name.
        """
        fields.setdefault("input_keys", ["source_image"])
        fields.setdefault("target_keys", ["target_image"])
        fields.setdefault("metadata", {})
        self.__dict__.update(fields)

    def replace(self, **changes: Any) -> "StainSample":
        """Return a shallow copy with the given fields replaced.

        Mirrors ``dataclasses.replace`` for the old fixed-field dataclass.
        """
        new = copy.copy(self)
        new.__dict__.update(changes)
        return new

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{k}={type(v).__name__}" for k, v in self.__dict__.items()
        )
        return f"StainSample({fields})"
