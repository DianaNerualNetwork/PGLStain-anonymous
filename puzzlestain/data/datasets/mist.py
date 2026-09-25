"""MIST/IHC4BC virtual-stain dataset loader and converter.

The MIST layout places each target stain in its own subdirectory with paired
``trainA``/``trainB`` folders (and optional ``testA``/``testB``). The source
images live in the ``*A`` folders and the target stains in the ``*B`` folders.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..backends import ImageBackend
from ..scanners import scan_mist_style_directory
from ..transforms.base import Compose
from .unified import VirtualStainDataset

logger = logging.getLogger(__name__)


class MISTVirtualStainDataset(VirtualStainDataset):
    """Virtual-stain dataset for the MIST/IHC4BC directory layout.

    Args:
        dataroot: Root directory containing one subdirectory per stain.
        panel_id: Panel identifier. Defaults to ``breast_mist_4plex_20x``.
        panel_members: Full panel member list. Defaults to the standard IHC4BC
            panel ``["ER", "PR", "Ki67", "HER2"]``.
        selected_stains: Stains to include; ``None`` includes all panel members.
        splits: Split prefixes to load (default ``("train",)``).
        source_suffix: Source-domain folder suffix (default ``"A"``).
        target_suffix: Target-domain folder suffix (default ``"B"``).
        panel_policy: Panel policy metadata.
        return_onehot: Populate ``target_onehot`` on each sample.
        transforms: Augmentation pipeline.
        backend: Image backend or registered name.
        max_dataset_size: Cap dataset entries at this value when positive.
    """

    PANEL_MEMBERS = ["ER", "PR", "Ki67", "HER2"]
    DEFAULT_PANEL_ID = "breast_mist_4plex_20x"

    def __new__(
        cls,
        dataroot: str | Path,
        panel_id: str = DEFAULT_PANEL_ID,
        panel_members: Optional[list[str]] = None,
        selected_stains: Optional[list[str]] = None,
        splits: tuple[str, ...] = ("train",),
        source_suffix: str = "A",
        target_suffix: str = "B",
        panel_policy: str = "mixed",
        return_onehot: bool = False,
        transforms: Optional[Compose] = None,
        backend: Optional[str | ImageBackend] = None,
        max_dataset_size: int = 0,
        aux_columns: Optional[list[str]] = None,
        input_keys: Optional[list[str]] = None,
        target_keys: Optional[list[str]] = None,
    ) -> VirtualStainDataset:
        results = scan_mist_style_directory(
            panel_root=dataroot,
            panel_id=panel_id,
            panel_members=panel_members or cls.PANEL_MEMBERS,
            splits=splits,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
            panel_policy=panel_policy,
        )
        if not results:
            raise ValueError(f"No valid MIST shards found under {dataroot}")

        n_classes = len(results[0][1].get("label_mapping") or panel_members or cls.PANEL_MEMBERS)

        return VirtualStainDataset._from_frames(
            frames=[df for df, _ in results],
            infos=[info for _, info in results],
            panel_id=panel_id,
            selected_stains=selected_stains,
            panel_policy=panel_policy,
            num_classes=n_classes,
            return_onehot=return_onehot,
            transforms=transforms,
            backend=backend,
            max_dataset_size=max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )

    @classmethod
    def from_parquet(
        cls,
        parquet_root: str | Path,
        raw_root: str | Path,
        panel_id: str = DEFAULT_PANEL_ID,
        panel_members: Optional[list[str]] = None,
        selected_stains: Optional[list[str]] = None,
        splits: tuple[str, ...] = ("train",),
        split_root_name: str = "TrainValAB",
        panel_policy: str = "mixed",
        return_onehot: bool = False,
        transforms: Optional[Compose] = None,
        backend: Optional[str | ImageBackend] = None,
        max_dataset_size: int = 0,
        aux_columns: Optional[list[str]] = None,
        input_keys: Optional[list[str]] = None,
        target_keys: Optional[list[str]] = None,
    ) -> VirtualStainDataset:
        """Load a pre-converted MIST parquet tree using the raw images.

        This is useful when metadata has been pre-scanned into
        ``sample_meta.parquet`` + ``dataset_info.json`` shards but the actual
        patch images still live in the original MIST/IHC4BC directory tree.

        Args:
            parquet_root: Directory containing one shard subdirectory per stain
                (as produced by ``convert``).
            raw_root: Original MIST/IHC4BC root directory with one subdirectory
                per target stain.
            split_root_name: Intermediate folder that holds the split folders
                (default ``"TrainValAB"``).
            aux_columns: Auxiliary image path columns to load alongside the
                source/target images.
            input_keys: Field names treated as model inputs.
            target_keys: Field names treated as supervision targets.
        """
        parquet_root = Path(parquet_root)
        raw_root = Path(raw_root)

        top_info_path = parquet_root / "dataset_info.json"
        if not top_info_path.is_file():
            raise ValueError(f"Missing top-level dataset_info.json in {parquet_root}")
        top_info = json.loads(top_info_path.read_text())
        shards = top_info.get("shards") or []
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"No shards declared in {top_info_path}")

        sample_meta_paths: list[Path] = []
        patch_roots: list[Path] = []

        for shard in shards:
            shard_dir = parquet_root / shard
            parquet_path = shard_dir / "sample_meta.parquet"
            if not parquet_path.is_file():
                logger.warning("Skipping missing shard parquet: %s", parquet_path)
                continue

            sidecar_path = shard_dir / "dataset_info.json"
            target_domain = shard
            if sidecar_path.is_file():
                sidecar = json.loads(sidecar_path.read_text())
                target_domain = sidecar.get("target_domain") or shard

            sample_meta_paths.append(parquet_path)
            patch_roots.append(raw_root / target_domain / split_root_name)

        if not sample_meta_paths:
            raise ValueError(f"No valid MIST shards found under {parquet_root}")

        return VirtualStainDataset(
            sample_meta=sample_meta_paths,
            patch_root=patch_roots,
            panel_id=panel_id,
            selected_stains=selected_stains,
            panel_policy=panel_policy,
            return_onehot=return_onehot,
            transforms=transforms,
            splits=list(splits) if splits else None,
            backend=backend,
            max_dataset_size=max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )

    @classmethod
    def convert(
        cls,
        panel_root: str | Path,
        out_root: str | Path,
        panel_id: str = DEFAULT_PANEL_ID,
        panel_members: Optional[list[str]] = None,
        splits: tuple[str, ...] = ("train", "val", "test"),
        source_suffix: str = "A",
        target_suffix: str = "B",
        panel_policy: str = "mixed",
    ) -> list[dict[str, Any]]:
        """Convert a MIST/IHC4BC directory tree to Parquet shards.

        Produces one shard directory per stain under ``out_root``, each
        containing ``sample_meta.parquet`` and ``dataset_info.json``.
        Also writes a top-level ``dataset_info.json`` indexing all shards
        for downstream consumers (``from_parquet``, ``convert_domain_pool``).

        Returns:
            A list of the emitted sidecar dictionaries.
        """
        results = scan_mist_style_directory(
            panel_root=panel_root,
            panel_id=panel_id,
            panel_members=panel_members or cls.PANEL_MEMBERS,
            splits=splits,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
            panel_policy=panel_policy,
        )
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        infos: list[dict[str, Any]] = []
        for df, info in results:
            dataset_uid = info["dataset_uid"]
            shard_dir = out_root / dataset_uid
            shard_dir.mkdir(parents=True, exist_ok=True)

            df_out = df.drop(columns=["__root_dir"], errors="ignore")
            df_out.to_parquet(shard_dir / "sample_meta.parquet", index=False)
            (shard_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
            logger.info("Wrote %s: %d rows", dataset_uid, len(df_out))
            infos.append(info)

        # Write top-level dataset_info.json indexing all shards.
        shard_names = [info["dataset_uid"] for info in infos]
        top_info = {
            "panel_id": panel_id,
            "panel_members": panel_members or cls.PANEL_MEMBERS,
            "panel_policy": panel_policy,
            "directory_format": "MIST",
            "shards": shard_names,
        }
        (out_root / "dataset_info.json").write_text(json.dumps(top_info, indent=2))
        logger.info("Wrote top-level dataset_info.json with %d shards", len(shard_names))

        return infos

    @classmethod
    def from_raw(
        cls,
        dataroot: str | Path,
        panel_id: str = DEFAULT_PANEL_ID,
        panel_members: Optional[list[str]] = None,
        selected_stains: Optional[list[str]] = None,
        splits: tuple[str, ...] = ("train",),
        return_onehot: bool = False,
        transforms: Optional[Compose] = None,
        backend: Optional[str | ImageBackend] = None,
        max_dataset_size: int = 0,
        aux_columns: Optional[list[str]] = None,
        input_keys: Optional[list[str]] = None,
        target_keys: Optional[list[str]] = None,
    ) -> VirtualStainDataset:
        """Convenience alias for ``MISTVirtualStainDataset(...)``."""
        return cls(
            dataroot=dataroot,
            panel_id=panel_id,
            panel_members=panel_members,
            selected_stains=selected_stains,
            splits=splits,
            return_onehot=return_onehot,
            transforms=transforms,
            backend=backend,
            max_dataset_size=max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )

    @classmethod
    def info(cls, panel_id: str = DEFAULT_PANEL_ID) -> dict[str, Any]:
        """Return panel metadata for the default MIST/IHC4BC panel."""
        return {
            "panel_id": panel_id,
            "panel_members": cls.PANEL_MEMBERS,
            "panel_policy": "mixed",
            "source_domain": "HE",
            "target_domain": "IHC",
        }

    @classmethod
    def discover_samples(
        cls,
        dataroot: str | Path,
        panel_members: Optional[list[str]] = None,
        splits: tuple[str, ...] = ("train", "val", "test"),
    ) -> list[pd.DataFrame]:
        """Scan ``dataroot`` and return the raw shard DataFrames.

        This is useful for inspecting a MIST layout before building a full
        ``VirtualStainDataset``.
        """
        results = scan_mist_style_directory(
            panel_root=dataroot,
            panel_id=cls.DEFAULT_PANEL_ID,
            panel_members=panel_members or cls.PANEL_MEMBERS,
            splits=splits,
        )
        return [df for df, _ in results]


__all__ = ["MISTVirtualStainDataset"]
