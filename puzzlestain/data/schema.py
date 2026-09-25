"""Pydantic validation models for virtual-staining dataset metadata.

All enum values are strings for Parquet/JSON compatibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PanelPolicy(str, Enum):
    """How panel annotations are organized in a dataset shard."""

    COMPLETE = "complete"  # every HE has all panel members
    PARTIAL = "partial"  # every HE has a subset of members from the same WSI
    MIXED = (
        "mixed"  # different HEs from different WSIs, each paired with one/more stains
    )


class StainMetaRecord(BaseModel):
    """One row in the virtual-stain sample meta table.

    This is the runtime schema; converters may carry extra columns as long as
    the required columns are present.
    """

    sample_uid: str
    he_path: str
    target_path: str
    target_stain: str
    dataset_split: DatasetSplit
    dataset_uid: str
    panel_id: str


class DatasetInfo(BaseModel):
    """Sidecar ``dataset_info.json`` schema."""

    dataset_uid: str
    panel_id: str
    panel_members: list[str]
    panel_policy: PanelPolicy = PanelPolicy.MIXED
    source_domain: str = "HE"
    image_shape: Optional[list[int]] = None
    dtype: str = "uint8"
    directory_format: Optional[str] = None
