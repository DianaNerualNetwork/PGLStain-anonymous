"""Kind-driven collation for virtual-staining samples.

Stacks source/target/auxiliary images and gathers stain names / labels /
metadata into per-sample lists. Fields
are routed by ``FieldKind`` rather than by hardcoded names.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .fields import FIELD_KINDS, FieldKind
from .sample import StainSample


def stain_collate_fn(samples: list[StainSample]) -> dict[str, Any]:
    """Collate a list of ``StainSample`` into a numpy batch dict.

    Source, target, and auxiliary images are stacked to ``(B, H, W, C)`` numpy
    arrays (or ``(B, C, H, W)`` tensors if the transform already converted
    them). ``target_stain``, ``target_label`` and ``metadata`` are gathered as
    arrays or lists.
    """
    if not samples:
        return {}

    batch: dict[str, Any] = {}

    for name, kind in FIELD_KINDS.items():
        items = [getattr(sample, name, None) for sample in samples]
        present = [item is not None for item in items]
        if not any(present):
            continue
        if not all(present):
            missing = [
                index for index, is_present in enumerate(present) if not is_present
            ]
            raise ValueError(
                f"Inconsistent field {name!r} in batch: missing from sample "
                f"indices {missing}"
            )

        if kind in (
            FieldKind.SOURCE_IMAGE,
            FieldKind.TARGET_IMAGE,
            FieldKind.AUX_IMAGE,
        ):
            if isinstance(items[0], torch.Tensor):
                batch[name] = torch.stack(items)
            else:
                batch[name] = np.stack(items)
        elif kind is FieldKind.TARGET_LABEL:
            batch[name] = np.array(items, dtype=np.int64)
        elif kind is FieldKind.TARGET_ONEHOT:
            batch[name] = np.stack(items)
        elif kind in (FieldKind.STAIN_NAME, FieldKind.TEXT):
            batch[name] = items
        elif kind is FieldKind.SCALAR:
            batch[name] = np.array(items)
        elif kind is FieldKind.METADATA:
            batch[name] = items

    return batch
