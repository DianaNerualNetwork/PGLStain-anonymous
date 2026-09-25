"""Shared Dataset utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..backends import BackendFactory, ImageBackend
from ..fields import TaskContract

logger = logging.getLogger(__name__)
_SHARD_COL = "__shard_idx"


def load_sidecar(meta_path: str | Path) -> dict:
    """Read ``dataset_info.json`` next to a parquet file."""
    p = Path(meta_path)
    for candidate in (
        p.parent / "dataset_info.json",
        p.parent.parent / "dataset_info.json",
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except (OSError, ValueError):
                return {}
    return {}


def select_rows(
    df: pd.DataFrame,
    splits: Optional[list[str]],
    row_filter: Optional[dict[str, Any]],
) -> pd.DataFrame:
    """Filter a meta table by split and column-value equality."""
    n0 = len(df)
    if splits:
        df = df[df["dataset_split"].isin(splits)]
    for col, val in (row_filter or {}).items():
        allowed = val if isinstance(val, (list, tuple, set)) else [val]
        df = df[df[col].isin(list(allowed))]
    if len(df) == 0:
        raise ValueError(
            f"Row selection removed all {n0} rows "
            f"(splits={splits}, row_filter={row_filter})"
        )
    if len(df) != n0:
        logger.info("Row selection kept %d/%d rows", len(df), n0)
    return df.reset_index(drop=True)


def check_unique_sample_uid(df: pd.DataFrame) -> None:
    """Fail loud if sample_uids collide across shards."""
    dup = df[df["sample_uid"].duplicated(keep=False)]
    if not dup.empty:
        offenders = sorted(dup["sample_uid"].unique().tolist())
        raise ValueError(f"Duplicate sample_uid across shards: {offenders}")


class VirtualStainDatasetBase:
    """Mixin with common Dataset construction helpers."""

    def _resolve_backend(self, backend: Optional[str | ImageBackend]) -> ImageBackend:
        if backend is None:
            backend = BackendFactory.create("default")
        if isinstance(backend, str):
            backend = BackendFactory.create(backend)
        return backend

    def _resolve_contract(self) -> TaskContract:
        return TaskContract(input_keys=["source_image"], target_keys=["target_image"])

    def _assert_panel_consistency(
        self,
        shard_infos: list[dict],
        panel_id: str,
    ) -> None:
        """Ensure all shards share the same panel_id and compatible settings."""
        panel_ids = {
            info.get("panel_id") for info in shard_infos if info.get("panel_id")
        }
        if panel_ids and panel_ids != {panel_id}:
            raise ValueError(
                f"panel_id mismatch: expected {panel_id!r}, found {panel_ids}"
            )

    def _resolve_panel_members(
        self,
        shard_infos: list[dict],
        columns: list[str],
    ) -> list[str]:
        """Resolve panel members from sidecars and column names.

        Members are intersected across shards (order follows the first
        sidecar). A warning listing the dropped members and each shard's
        declaration is logged when the intersection is lossy.
        """
        declared = [
            list(info.get("panel_members", []))
            for info in shard_infos
            if info.get("panel_members")
        ]
        inferred = [
            col.replace("target_stain_", "").replace("stain_", "").replace("_path", "")
            for col in columns
            if col.startswith("stain_") and col.endswith("_path")
        ]
        if inferred:
            declared.append(inferred)

        # Labeled-image-folder mode: panel members are optional; derive from
        # unique target_label values so that selected_stains filtering still works.
        if not declared and "target_label" in columns:
            return []

        if not declared:
            raise ValueError("Cannot resolve panel_members from sidecars or columns")
        common = set(declared[0]).intersection(*declared[1:])
        # Preserve the order declared by the first sidecar rather than sorting.
        members = [m for m in declared[0] if m in common]
        dropped = sorted(set().union(*declared) - set(members))
        if dropped:
            logger.warning(
                "panel_members resolved to %s by intersection; dropped %s. "
                "Shard declarations: %s",
                members,
                dropped,
                {
                    str(info.get("dataset_uid", "?")): info.get("panel_members")
                    for info in shard_infos
                    if info.get("panel_members")
                },
            )
        return members
