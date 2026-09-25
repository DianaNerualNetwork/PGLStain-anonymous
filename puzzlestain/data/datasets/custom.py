"""Custom virtual-stain dataset loader for arbitrary paired layouts.

This wraps ``scan_paired_directory`` so users can point ``train.py`` at a raw
``dataroot`` using one of the supported layouts (subdirectory-per-stain like
MIST, flat paired with filename suffixes, or labeled image folders).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..backends import ImageBackend
from ..scanners import scan_paired_directory
from ..transforms.base import Compose
from .unified import VirtualStainDataset


class CustomVirtualStainDataset(VirtualStainDataset):
    """Direct loader for arbitrary paired-image directory layouts.

    The underlying scanner supports:

    * Layout A: one subdirectory per stain with ``{split}A``/``{split}B``
      folders (MIST/IHC4BC style).
    * Layout B: flat paired folders where the stain is encoded in the filename
      suffix, e.g. ``case_001_HE.png`` and ``case_001_ER.png``.

    Args:
        dataroot: Root directory containing paired images.
        panel_id: Panel identifier.
        panel_members: Stain names. In Layout A inferred from subdirectories
            when omitted; required for Layout B.
        selected_stains: Stains to include; ``None`` means all panel members.
        splits: Split prefixes to scan.
        source_suffix: Source-domain folder suffix (default ``"A"``).
        target_suffix: Target-domain folder suffix (default ``"B"``).
        panel_policy: Panel policy metadata.
        label_mapping: Optional map from stain name to integer class index.
        return_onehot: Populate ``target_onehot`` on each sample.
        stain_from_subdir: ``True`` for Layout A, ``False`` for Layout B.
        source_stain: Suffix identifying source images in Layout B.
        transforms: Augmentation pipeline.
        backend: Image backend or registered name.
        max_dataset_size: Cap dataset entries at this value when positive.
    """

    def __new__(
        cls,
        dataroot: str | Path,
        panel_id: str,
        panel_members: Optional[list[str]] = None,
        selected_stains: Optional[list[str]] = None,
        splits: tuple[str, ...] = ("train",),
        source_suffix: str = "A",
        target_suffix: str = "B",
        panel_policy: str = "mixed",
        label_mapping: Optional[dict[str, int]] = None,
        return_onehot: bool = False,
        stain_from_subdir: bool = True,
        source_stain: str = "HE",
        transforms: Optional[Compose] = None,
        backend: Optional[str | ImageBackend] = None,
        max_dataset_size: int = 0,
    ) -> VirtualStainDataset:
        results = scan_paired_directory(
            panel_root=dataroot,
            panel_id=panel_id,
            panel_members=panel_members,
            splits=splits,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
            panel_policy=panel_policy,
            label_mapping=label_mapping,
            stain_from_subdir=stain_from_subdir,
            source_stain=source_stain,
        )
        if not results:
            raise ValueError(f"No valid paired shards found under {dataroot}")

        n_classes = (
            len(
                results[0][1].get("label_mapping")
                or panel_members
                or results[0][1].get("panel_members")
                or []
            )
            or 1
        )

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
        )


__all__ = ["CustomVirtualStainDataset"]
