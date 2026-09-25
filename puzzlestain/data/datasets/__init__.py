"""Convenience loaders for common raw project layouts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..backends import ImageBackend
from ..scanners import (
    scan_aligned_directory,
    scan_labeled_image_folder,
    scan_paired_directory,
)
from ..transforms.base import Compose
from .custom import CustomVirtualStainDataset
from .mist import MISTVirtualStainDataset
from .panel import StainPanelDataset
from .unified import VirtualStainDataset


class AlignedVirtualStainDataset(VirtualStainDataset):
    """Direct loader for the Pix2PixHD aligned folder layout.

    Layout::

        dataroot/
          trainA/  trainB/
          valA/    valB/
          testA/   testB/

    This matches Pix2PixHD's ``AlignedDataset`` when ``label_nc == 0``.

    Args:
        dataroot: Root directory containing ``trainA``/``trainB`` etc.
        panel_id: Panel identifier.
        target_stain: Target stain/domain name (e.g. ``"ER"``).
        splits: Split prefixes to scan.
        source_suffix: Source-domain folder suffix (default ``"A"``).
        target_suffix: Target-domain folder suffix (default ``"B"``).
        panel_policy: Panel policy metadata.
        return_onehot: Populate ``target_onehot`` on each sample.
        transforms: Augmentation pipeline.
        backend: Image backend or registered name.
        max_dataset_size: Cap dataset entries at this value when positive.
        aux_columns: Auxiliary image path columns to load alongside the
            source/target images.
        input_keys: Field names treated as model inputs.
        target_keys: Field names treated as supervision targets.
    """

    def __new__(
        cls,
        dataroot: str | Path,
        panel_id: str,
        target_stain: str,
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
        results = scan_aligned_directory(
            panel_root=dataroot,
            panel_id=panel_id,
            target_stain=target_stain,
            splits=splits,
            source_suffix=source_suffix,
            target_suffix=target_suffix,
            panel_policy=panel_policy,
        )
        df, info = results[0]

        panel_members = [target_stain]
        num_classes = 1

        return VirtualStainDataset._from_frames(
            frames=[df],
            infos=[info],
            panel_id=panel_id,
            selected_stains=panel_members,
            panel_policy=panel_policy,
            num_classes=num_classes,
            return_onehot=return_onehot,
            transforms=transforms,
            backend=backend,
            max_dataset_size=max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )


class PairedVirtualStainDataset(VirtualStainDataset):
    """Direct loader for paired-image directories.

    Supports two layouts selected by ``stain_from_subdir``:

    **Layout A** (MIST/IHC4BC style)::

        panel_root/
          CD3/
            trainA/  trainB/
          CD8/
            trainA/  trainB/

    **Layout B** (flat paired + filename suffix)::

        dataroot/
          trainA/
            case_001_HE.png
            case_002_HE.png
          trainB/
            case_001_ER.png
            case_001_HER2.png
            case_002_ER.png

    In Layout B the stain is extracted from the filename suffix
    (``case_001_ER.png`` -> ``ER``). Cases are paired by the stem prefix
    and ``source_stain`` selects the input domain (default ``HE``).

    Args:
        dataroot: Root directory containing paired source/target folders.
        panel_id: Panel identifier.
        panel_members: List of stain names. Required for Layout B; inferred
            from subdirectories in Layout A when omitted.
        selected_stains: Stains to include; ``None`` means all panel members.
        splits: Split prefixes to scan.
        source_suffix: Source-domain folder suffix (default ``A``).
        target_suffix: Target-domain folder suffix (default ``B``).
        panel_policy: Panel policy metadata.
        label_mapping: Optional map from stain name to integer class index.
            Emits ``target_label`` when provided.
        return_onehot: Populate ``target_onehot`` on each sample.
        stain_from_subdir: ``True`` for Layout A, ``False`` for Layout B.
        source_stain: Suffix that identifies source images in Layout B
            (default ``HE``). Change this when the input domain is not H&E.
        transforms: Augmentation pipeline.
        backend: Image backend or registered name.
        max_dataset_size: Cap dataset entries at this value when positive.
        aux_columns: Auxiliary image path columns to load alongside the
            source/target images.
        input_keys: Field names treated as model inputs.
        target_keys: Field names treated as supervision targets.
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
        aux_columns: Optional[list[str]] = None,
        input_keys: Optional[list[str]] = None,
        target_keys: Optional[list[str]] = None,
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
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )


class LabeledImageFolderVirtualStainDataset(VirtualStainDataset):
    """Direct loader for single-image folders with labels in filenames.

    Layout::

        root/
          xxx_0.png
          yyy_1.png

    The integer suffix is the class label. This matches the classic
    ``DatasetFolder`` / ``ImageFolder`` pattern used in conditional generation
    and classification baselines.
    """

    def __new__(
        cls,
        root: str | Path,
        panel_id: str,
        panel_members: Optional[list[str]] = None,
        label_mapping: Optional[dict[str, int]] = None,
        split: Optional[str] = None,
        return_onehot: bool = False,
        transforms: Optional[Compose] = None,
        backend: Optional[str | ImageBackend] = None,
        max_dataset_size: int = 0,
        aux_columns: Optional[list[str]] = None,
        input_keys: Optional[list[str]] = None,
        target_keys: Optional[list[str]] = None,
    ) -> VirtualStainDataset:
        results = scan_labeled_image_folder(
            root=root,
            panel_id=panel_id,
            label_mapping=label_mapping,
            split=split,
        )
        df, info = results[0]

        if panel_members is None and label_mapping is not None:
            panel_members = sorted(label_mapping.keys())
        elif panel_members is None:
            panel_members = sorted(df["target_label"].astype(int).unique().tolist())

        # Update sidecar so downstream panel_member resolution sees integers.
        info["panel_members"] = [str(m) for m in panel_members]

        num_classes = len(info.get("label_mapping") or panel_members or []) or 1

        return VirtualStainDataset._from_frames(
            frames=[df],
            infos=[info],
            panel_id=panel_id,
            selected_stains=panel_members,
            num_classes=num_classes,
            return_onehot=return_onehot,
            transforms=transforms,
            backend=backend,
            max_dataset_size=max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )


__all__ = [
    "AlignedVirtualStainDataset",
    "CustomVirtualStainDataset",
    "LabeledImageFolderVirtualStainDataset",
    "MISTVirtualStainDataset",
    "PairedVirtualStainDataset",
    "StainPanelDataset",
    "VirtualStainDataset",
]
