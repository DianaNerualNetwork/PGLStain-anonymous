"""Field-kind vocabulary and the per-task field-role contract.

Every Sample flat field is mapped
to a semantic ``FieldKind``. Collation routes by kind rather than by field name,
so adding a new stain/task/auxiliary image does not require editing the collate
function.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator


class FieldKind(str, Enum):
    """Semantic kind of a flat ``StainSample`` field."""

    SOURCE_IMAGE = "source_image"
    TARGET_IMAGE = "target_image"
    AUX_IMAGE = "aux_image"
    STAIN_NAME = "stain_name"
    TARGET_LABEL = "target_label"
    TARGET_ONEHOT = "target_onehot"
    TEXT = "text"
    SCALAR = "scalar"
    METADATA = "metadata"


#: Global map from ``StainSample`` field name to its ``FieldKind``.
FIELD_KINDS: dict[str, FieldKind] = {
    "source_image": FieldKind.SOURCE_IMAGE,
    "target_image": FieldKind.TARGET_IMAGE,
    "target_stain": FieldKind.STAIN_NAME,
    "target_label": FieldKind.TARGET_LABEL,
    "target_onehot": FieldKind.TARGET_ONEHOT,
    "cond_index": FieldKind.SCALAR,
    "sample_uid": FieldKind.TEXT,
    "dataset_uid": FieldKind.TEXT,
    "panel_id": FieldKind.TEXT,
    "metadata": FieldKind.METADATA,
}


def register_field_kind(name: str, kind: FieldKind) -> None:
    """Register a field name with its semantic kind.

    This lets auxiliary images and task-specific fields participate in
    kind-driven collation without hard-coding every possible name.
    """
    FIELD_KINDS[name] = kind


def ensure_aux_field(name: str) -> None:
    """Ensure ``name`` is registered as an auxiliary image field."""
    register_field_kind(name, FieldKind.AUX_IMAGE)


def iter_image_field_names(sample: Any) -> list[str]:
    """Return registered image field names present on ``sample``."""
    image_kinds = {
        FieldKind.SOURCE_IMAGE,
        FieldKind.TARGET_IMAGE,
        FieldKind.AUX_IMAGE,
    }
    return [
        name
        for name in sample.__dict__
        if name in FIELD_KINDS and FIELD_KINDS[name] in image_kinds
    ]


assert all(
    isinstance(kind, FieldKind) for kind in FIELD_KINDS.values()
), "FIELD_KINDS values must all be FieldKind members"


class TaskContract(BaseModel):
    """Per-task declaration of input vs. target field roles."""

    input_keys: list[str]
    target_keys: list[str]

    @model_validator(mode="after")
    def check_roles(self) -> "TaskContract":
        overlap = set(self.input_keys) & set(self.target_keys)
        if overlap:
            raise ValueError(
                f"input_keys and target_keys must be disjoint; overlap: {sorted(overlap)}"
            )
        unknown = [
            k for k in (*self.input_keys, *self.target_keys) if k not in FIELD_KINDS
        ]
        if unknown:
            raise ValueError(f"unknown field key(s): {sorted(set(unknown))}")
        if len(self.target_keys) < 1:
            raise ValueError("at least one target key is required")
        return self
