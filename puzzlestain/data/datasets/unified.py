"""Unified stain-level virtual-stain Dataset.

Supports both single-shard and multi-shard (multi-dataset) training. Data can be
provided as:

1. A list of pre-built ``sample_meta.parquet`` files.
2. A single parquet file.
3. In-memory ``DataFrame``s produced by directory scanners.

Every sample is one ``(source, target_stain)`` pair, so complete panels, partial
panels, and mixed datasets all collapse to the same index space.

When the source table only carries ``target_label`` (no ``target_path``), the
sample is treated as a labeled source image for conditioning/classification tasks.

Auxiliary images (DAB masks, nuclei maps, etc.) can be attached by listing their
path columns in ``aux_columns``. The columns are expected to be named
``{name}_path`` and will be exposed as sample fields named ``{name}``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..backends import ImageBackend
from ..fields import TaskContract, ensure_aux_field
from ..sample import StainSample
from ..transforms.base import Compose
from .base import (
    _SHARD_COL,
    VirtualStainDatasetBase,
    check_unique_sample_uid,
    load_sidecar,
    select_rows,
)

logger = logging.getLogger(__name__)

_MetaSource = str | Path | pd.DataFrame


def _discover_shard_parquets(directory: Path) -> list[Path]:
    """Discover per-shard ``sample_meta.parquet`` files under a shard root.

    A shard root is a directory that contains a top-level ``dataset_info.json``
    with a ``shards`` list, where each entry names a subdirectory holding its
    own ``sample_meta.parquet`` and ``dataset_info.json``. Returns an empty
    list when the directory does not match this layout.
    """
    info_path = directory / "dataset_info.json"
    if not info_path.is_file():
        return []
    try:
        info = json.loads(info_path.read_text())
    except (OSError, ValueError):
        return []
    shards = info.get("shards")
    if not isinstance(shards, list):
        return []
    parquet_paths: list[Path] = []
    for shard in shards:
        parquet_path = directory / shard / "sample_meta.parquet"
        if parquet_path.is_file():
            parquet_paths.append(parquet_path)
    return parquet_paths


def _resolve_aux_paths(row: pd.Series, aux_columns: list[str]) -> dict[str, str]:
    """Build ``field_name -> relative path`` for auxiliary images."""
    paths: dict[str, str] = {}
    for col in aux_columns:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val) or val == "":
            continue
        # Column ``dab_mask_path`` becomes field ``dab_mask``.
        field_name = col[: -len("_path")] if col.endswith("_path") else col
        paths[field_name] = str(val)
    return paths


class VirtualStainDataset(Dataset, VirtualStainDatasetBase):
    """Stain-level Dataset for paired virtual staining.

    Args:
        sample_meta: One parquet path, one DataFrame, or a list of either
            (multi-shard).
        patch_root: One root path or a list aligned with ``sample_meta``. Not
            needed when ``sample_meta`` is a DataFrame that already contains a
            ``__root_dir`` column.
        panel_id: Panel identifier.
        selected_stains: Stains to include; ``None`` means all panel members.
        panel_policy: Optional filter; ``"complete"``/``"partial"``/``"mixed"``.
        num_classes: Number of stain classes; used to build one-hot vectors.
            Inferred from sidecar ``panel_members`` when ``None``.
        return_onehot: Whether to populate ``target_onehot`` on each sample.
        transforms: Augmentation pipeline operating on ``StainSample``.
        splits: Filter by ``dataset_split``.
        row_filter: Column -> value(s) equality filter.
        backend: ``"pil"`` or an ``ImageBackend`` instance.
        aux_columns: Column names in the parquet that hold auxiliary image
            paths. Each column ``foo_path`` is exposed as sample field ``foo``.
        input_keys: Field names treated as model inputs. Defaults to
            ``["source_image"]``.
        target_keys: Field names treated as supervision targets. Defaults to
            ``["target_image"]``.
    """

    def __init__(
        self,
        sample_meta: _MetaSource | list[_MetaSource],
        patch_root: Optional[str | Path | list[str | Path]] = None,
        panel_id: str = "default",
        selected_stains: Optional[list[str]] = None,
        panel_policy: Optional[str] = None,
        num_classes: Optional[int] = None,
        return_onehot: bool = False,
        transforms: Optional[Compose] = None,
        splits: Optional[list[str]] = None,
        row_filter: Optional[dict[str, Any]] = None,
        backend: Optional[str | ImageBackend] = None,
        max_dataset_size: int = 0,
        aux_columns: Optional[list[str]] = None,
        input_keys: Optional[list[str]] = None,
        target_keys: Optional[list[str]] = None,
    ):
        super().__init__()
        self.panel_id = panel_id
        self.transforms = transforms
        self.backend = self._resolve_backend(backend)
        self.return_onehot = return_onehot
        self.aux_columns = list(aux_columns or [])
        for col in self.aux_columns:
            field_name = col[: -len("_path")] if col.endswith("_path") else col
            ensure_aux_field(field_name)
        self._contract = TaskContract(
            input_keys=list(input_keys or ["source_image"]),
            target_keys=list(target_keys or ["target_image"]),
        )

        frames, infos = self._normalize_meta_sources(sample_meta, patch_root)
        self._init_from_frames(
            frames,
            infos,
            selected_stains,
            panel_policy,
            splits,
            row_filter,
            max_dataset_size=max_dataset_size,
        )
        self._num_classes = num_classes or len(self.panel_members) or 1

    @classmethod
    def _from_frames(
        cls,
        frames: list[pd.DataFrame],
        infos: list[dict],
        *,
        panel_id: str,
        selected_stains: Optional[list[str]] = None,
        panel_policy: Optional[str] = None,
        num_classes: Optional[int] = None,
        return_onehot: bool = False,
        transforms: Optional[Compose] = None,
        backend: Optional[str | ImageBackend] = None,
        max_dataset_size: int = 0,
        aux_columns: Optional[list[str]] = None,
        input_keys: Optional[list[str]] = None,
        target_keys: Optional[list[str]] = None,
    ) -> "VirtualStainDataset":
        """Construct directly from scanner outputs (bypasses parquet loading)."""
        obj = cls.__new__(cls)
        Dataset.__init__(obj)
        obj.panel_id = panel_id
        obj.transforms = transforms
        obj.backend = obj._resolve_backend(backend)
        obj.return_onehot = return_onehot
        obj.aux_columns = list(aux_columns or [])
        for col in obj.aux_columns:
            field_name = col[: -len("_path")] if col.endswith("_path") else col
            ensure_aux_field(field_name)
        obj._contract = TaskContract(
            input_keys=list(input_keys or ["source_image"]),
            target_keys=list(target_keys or ["target_image"]),
        )

        for shard_idx, df in enumerate(frames):
            df[_SHARD_COL] = shard_idx
        obj._init_from_frames(
            frames,
            infos,
            selected_stains,
            panel_policy,
            splits=None,
            row_filter=None,
            max_dataset_size=max_dataset_size,
        )
        obj._num_classes = num_classes or len(obj.panel_members) or 1
        return obj

    def _normalize_meta_sources(
        self,
        sample_meta: _MetaSource | list[_MetaSource],
        patch_root: Optional[str | Path | list[str | Path]],
    ) -> tuple[list[pd.DataFrame], list[dict]]:
        """Turn heterogeneous inputs into a list of DataFrames + sidecars."""
        if isinstance(sample_meta, (str, Path, pd.DataFrame)):
            sample_meta = [sample_meta]
        else:
            sample_meta = list(sample_meta)

        # Expand shard-root directories (e.g. MIST_parquet/ with dataset_info.json
        # listing shards) into the individual sample_meta.parquet paths.
        expanded_sources: list[_MetaSource] = []
        for src in sample_meta:
            if isinstance(src, pd.DataFrame):
                expanded_sources.append(src)
                continue
            p = Path(src)
            if p.is_dir():
                shard_paths = _discover_shard_parquets(p)
                if shard_paths:
                    expanded_sources.extend(shard_paths)
                    continue
            expanded_sources.append(src)
        sample_meta = expanded_sources

        if patch_root is None:
            root_dirs = [None] * len(sample_meta)
        elif isinstance(patch_root, (str, Path)):
            root_dirs = [patch_root] * len(sample_meta)
        else:
            root_dirs = list(patch_root)
            if len(root_dirs) != len(sample_meta):
                raise ValueError("sample_meta and patch_root must have the same length")

        frames: list[pd.DataFrame] = []
        infos: list[dict] = []

        for shard_idx, (src, root) in enumerate(zip(sample_meta, root_dirs)):
            if isinstance(src, pd.DataFrame):
                df = src.copy()
                info = {}
                if "__root_dir" not in df.columns and root is not None:
                    df["__root_dir"] = str(root)
            else:
                p = Path(src)
                df = pd.read_parquet(src)
                info = load_sidecar(src)
                if "__root_dir" not in df.columns:
                    if root is not None:
                        df["__root_dir"] = str(root)
                    elif p.is_dir():
                        # Directory of parquet files: resolve relative paths against
                        # the directory itself.
                        df["__root_dir"] = str(p)
                    else:
                        # Single parquet file without an explicit root: resolve
                        # relative paths against the directory that contains it.
                        df["__root_dir"] = str(p.parent)

            df[_SHARD_COL] = shard_idx
            frames.append(df)
            infos.append(info)

        return frames, infos

    def _init_from_frames(
        self,
        frames: list[pd.DataFrame],
        infos: list[dict],
        selected_stains: Optional[list[str]],
        panel_policy: Optional[str],
        splits: Optional[list[str]],
        row_filter: Optional[dict[str, Any]],
        max_dataset_size: int = 0,
    ) -> None:
        self.shard_infos = infos
        self._assert_panel_consistency(self.shard_infos, self.panel_id)
        self._assert_policy_consistency(panel_policy)

        self.meta_table = pd.concat(frames, ignore_index=True)
        check_unique_sample_uid(self.meta_table)
        self.meta_table = select_rows(self.meta_table, splits, row_filter)

        # Normalize to strings so integer labels compare correctly.
        self.panel_members = [
            str(m)
            for m in self._resolve_panel_members(
                self.shard_infos, list(self.meta_table.columns)
            )
        ]
        self.selected_stains = [str(s) for s in (selected_stains or self.panel_members)]
        invalid = set(self.selected_stains) - set(self.panel_members)
        if invalid:
            raise ValueError(
                f"selected_stains {sorted(invalid)} not in panel_members "
                f"{self.panel_members}"
            )

        self.entries = self._build_stain_level_entries()
        if max_dataset_size > 0:
            self.entries = self.entries[:max_dataset_size]

    def _assert_policy_consistency(self, panel_policy: Optional[str]) -> None:
        if panel_policy is None:
            return
        for info in self.shard_infos:
            if info.get("panel_policy") and info.get("panel_policy") != panel_policy:
                raise ValueError(
                    f"panel_policy mismatch: required {panel_policy!r}, "
                    f"got {info.get('panel_policy')!r} in {info.get('dataset_uid')}"
                )

    def _build_stain_level_entries(self) -> list[dict[str, Any]]:
        """Unfold the meta table into one entry per (source, target_stain) pair."""
        entries: list[dict[str, Any]] = []
        long_form = "target_stain" in self.meta_table.columns
        has_target_path = "target_path" in self.meta_table.columns
        has_target_label = "target_label" in self.meta_table.columns

        for _, row in self.meta_table.iterrows():
            shard_idx = int(row[_SHARD_COL])
            dataset_uid = str(
                row.get("dataset_uid")
                or self.shard_infos[shard_idx].get("dataset_uid", "unknown")
            )
            sample_uid = str(row["sample_uid"])
            source_path = str(row.get("source_path") or row.get("he_path"))
            root = str(row["__root_dir"])

            if long_form:
                stain = str(row["target_stain"])
                if stain not in self.selected_stains:
                    continue
                target_path = str(row["target_path"]) if has_target_path else ""
                target_label = self._resolve_label(row, stain)
                entries.append(
                    self._make_entry(
                        sample_uid,
                        dataset_uid,
                        shard_idx,
                        root,
                        source_path,
                        target_path,
                        stain,
                        target_label,
                        row=row,
                    )
                )
            elif has_target_label:
                # Labeled-image-folder mode: one row = one source image + label.
                label = int(row["target_label"])
                stain = self._label_to_stain(label)
                if stain is not None and stain not in self.selected_stains:
                    continue
                entries.append(
                    self._make_entry(
                        sample_uid,
                        dataset_uid,
                        shard_idx,
                        root,
                        source_path,
                        "",
                        stain or "",
                        label,
                        row=row,
                    )
                )
            else:
                # Wide-form panel parquet.
                for stain in self.selected_stains:
                    col = f"stain_{stain}_path"
                    if col in row.index and pd.notna(row[col]):
                        entries.append(
                            self._make_entry(
                                sample_uid,
                                dataset_uid,
                                shard_idx,
                                root,
                                source_path,
                                str(row[col]),
                                stain,
                                self._resolve_label(row, stain),
                                row=row,
                            )
                        )

        if not entries:
            raise ValueError("No stain-level entries built; check selected_stains")

        return entries

    def _resolve_label(self, row: pd.Series, stain: str) -> Optional[int]:
        """Resolve integer label from row or sidecar label_mapping."""
        if "target_label" in row.index and pd.notna(row["target_label"]):
            return int(row["target_label"])
        for info in self.shard_infos:
            mapping = info.get("label_mapping")
            if isinstance(mapping, dict) and stain in mapping:
                return int(mapping[stain])
        if stain in self.panel_members:
            return self.panel_members.index(stain)
        return None

    def _label_to_stain(self, label: int) -> Optional[str]:
        """Map integer label back to stain name when possible."""
        for info in self.shard_infos:
            mapping = info.get("label_mapping")
            if isinstance(mapping, dict):
                for stain, idx in mapping.items():
                    if int(idx) == label:
                        return stain
        if 0 <= label < len(self.panel_members):
            return self.panel_members[label]
        return None

    def _make_entry(
        self,
        sample_uid: str,
        dataset_uid: str,
        shard_idx: int,
        root: str,
        source_path: str,
        target_path: str,
        stain: str,
        target_label: Optional[int] = None,
        row: Optional[pd.Series] = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "entry_uid": (
                f"{dataset_uid}_{sample_uid}_{stain}"
                if stain
                else f"{dataset_uid}_{sample_uid}"
            ),
            "sample_uid": sample_uid,
            "dataset_uid": dataset_uid,
            "shard_idx": shard_idx,
            "root": root,
            "source_path": source_path,
            "target_path": target_path,
            "target_stain": stain,
            "target_label": target_label,
        }
        if row is not None and self.aux_columns:
            entry["aux_paths"] = _resolve_aux_paths(row, self.aux_columns)
        return entry

    def _onehot(self, label: int) -> np.ndarray:
        vec = np.zeros(self._num_classes, dtype=np.float32)
        vec[label] = 1.0
        return vec

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> StainSample:
        entry = self.entries[idx]
        root = Path(entry["root"])

        source_image = self.backend.read(root / entry["source_path"])

        target_image = None
        if entry.get("target_path"):
            target_image = self.backend.read(root / entry["target_path"])

        target_label = entry.get("target_label")
        target_onehot = None
        if self.return_onehot and target_label is not None:
            target_onehot = self._onehot(target_label)

        sample_kwargs: dict[str, Any] = {
            "source_image": source_image,
            "target_image": target_image,
            "target_stain": entry.get("target_stain") or None,
            "target_label": target_label,
            "target_onehot": target_onehot,
            "input_keys": list(self._contract.input_keys),
            "target_keys": list(self._contract.target_keys),
            "metadata": {
                "entry_uid": entry["entry_uid"],
                "sample_uid": entry["sample_uid"],
                "dataset_uid": entry["dataset_uid"],
                "panel_id": self.panel_id,
                "shard_idx": entry["shard_idx"],
                "source_path": str(entry["source_path"]),
                "target_path": (
                    str(entry["target_path"]) if entry.get("target_path") else ""
                ),
            },
        }

        for name, aux_path in entry.get("aux_paths", {}).items():
            sample_kwargs[name] = self.backend.read(root / aux_path)

        sample = StainSample(**sample_kwargs)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
