"""Directory scanners that turn raw project layouts into Parquet-ready tables.

Each scanner returns a ``(DataFrame, sidecar_dict)`` pair per dataset shard,
making it easy to either feed a Dataset directly or persist to the standard
``sample_meta.parquet`` + ``dataset_info.json`` layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _common_image_suffixes() -> tuple[str, ...]:
    return (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _list_image_files(directory: Path) -> dict[str, Path]:
    suffixes = _common_image_suffixes()
    return {
        p.name: p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    }


def _parse_suffix_files(
    files: dict[str, Path], stain_candidates: set[str]
) -> dict[str, dict[str, str]]:
    """Group image files by case id and stain suffix.

    ``case_id_stain.png`` is parsed into ``cases[case_id][stain] = filename``.
    Files whose last ``_`` token is not a recognized stain are ignored.
    """
    cases: dict[str, dict[str, str]] = {}
    for filename in files:
        stem = Path(filename).stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        case_id, stain = parts
        if stain not in stain_candidates:
            continue
        cases.setdefault(case_id, {})[stain] = filename
    return cases


def scan_mist_style_directory(
    panel_root: str | Path,
    panel_id: str,
    panel_members: Optional[list[str]] = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    source_suffix: str = "A",
    target_suffix: str = "B",
    panel_policy: str = "mixed",
    label_mapping: Optional[dict[str, int]] = None,
    split_root_name: str = "TrainValAB",
) -> list[tuple[pd.DataFrame, dict]]:
    """Scan a MIST/IHC4BC-style directory tree into shards.

    Layout::

        panel_root/
          ER/
            TrainValAB/
              trainA/  trainB/
              valA/    valB/
              testA/   testB/
          PR/
            TrainValAB/
              trainA/  trainB/
              ...

    The emitted ``dataset_uid`` for each shard is ``source_to_{stain_lower}``.

    Args:
        panel_root: Root directory with one subdirectory per stain.
        panel_id: Panel identifier shared across stains.
        panel_members: List of stain names. Inferred from subdirectories when
            omitted.
        splits: Split prefixes to scan.
        source_suffix: Source-domain suffix (default ``"A"``).
        target_suffix: Target-domain suffix (default ``"B"``).
        panel_policy: ``"complete"``, ``"partial"`` or ``"mixed"``.
        label_mapping: Optional map from stain name to integer class index.
        split_root_name: Name of the intermediate folder that holds the split
            folders (default ``"TrainValAB"``).

    Returns:
        One ``(DataFrame, sidecar)`` tuple per stain shard.
    """
    panel_root = Path(panel_root)
    stain_dirs = sorted([d for d in panel_root.iterdir() if d.is_dir()])
    panel_members = panel_members or [d.name for d in stain_dirs]
    label_mapping = label_mapping or {name: i for i, name in enumerate(panel_members)}
    results: list[tuple[pd.DataFrame, dict]] = []

    for stain_dir in stain_dirs:
        if stain_dir.name not in panel_members:
            continue
        split_root = stain_dir / split_root_name
        if not split_root.is_dir():
            continue

        rows: list[dict] = []
        for split in splits:
            src_dir = split_root / f"{split}{source_suffix}"
            tgt_dir = split_root / f"{split}{target_suffix}"
            if not src_dir.exists() or not tgt_dir.exists():
                continue

            src_files = _list_image_files(src_dir)
            tgt_files = _list_image_files(tgt_dir)

            for filename in sorted(set(src_files.keys()) & set(tgt_files.keys())):
                stem = Path(filename).stem
                stain_name = stain_dir.name
                dataset_uid = f"source_to_{stain_name.lower()}"
                rows.append(
                    {
                        "sample_uid": f"{dataset_uid}_{split}_{stem}",
                        "source_path": str(
                            Path(f"{split}{source_suffix}") / filename
                        ),
                        "target_path": str(
                            Path(f"{split}{target_suffix}") / filename
                        ),
                        "target_stain": stain_name,
                        "target_label": label_mapping[stain_name],
                        "dataset_split": split,
                        "dataset_uid": dataset_uid,
                        "panel_id": panel_id,
                    }
                )

        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["__root_dir"] = str(split_root)
        sidecar = {
            "dataset_uid": f"he_to_{stain_dir.name.lower()}",
            "panel_id": panel_id,
            "panel_members": panel_members,
            "panel_policy": panel_policy,
            "source_domain": "source",
            "target_domain": stain_dir.name,
            "directory_format": "MIST",
            "label_mapping": label_mapping,
        }
        results.append((df, sidecar))

    return results


def scan_paired_directory(
    panel_root: str | Path,
    panel_id: str,
    panel_members: Optional[list[str]] = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    source_suffix: str = "A",
    target_suffix: str = "B",
    panel_policy: str = "mixed",
    label_mapping: Optional[dict[str, int]] = None,
    stain_from_subdir: bool = True,
    source_stain: str = "HE",
) -> list[tuple[pd.DataFrame, dict]]:
    """Scan a paired-image directory into shards.

    Supports two common layouts controlled by ``stain_from_subdir``:

    **Layout A** (``stain_from_subdir=True``, MIST/IHC4BC style)::

        panel_root/
          CD3/
            trainA/  trainB/
            valA/    valB/
          CD8/
            trainA/  trainB/
            ...

    **Layout B** (``stain_from_subdir=False``, flat + filename suffix)::

        panel_root/
          trainA/
            case_001_HE.png
            case_002_HE.png
          trainB/
            case_001_ER.png
            case_001_HER2.png
            case_002_ER.png

    In Layout B the stain name is extracted from the filename suffix
    (e.g. ``case_001_ER.png`` -> ``ER``). Files are grouped by the
    case id (the stem before the last ``_``). A sample is emitted for
    every case that has ``source_stain`` in the source folder and a
    target stain from ``panel_members`` in the target folder.

    When ``source_suffix`` and ``target_suffix`` are the same, source
    and target images are looked up in the same folder; this supports
    single-folder panel datasets where each case provides all stains.

    Args:
        panel_root: Root directory.
        panel_id: Panel identifier shared across stains.
        panel_members: List of stain names. In Layout A inferred from
            subdirectories; in Layout B required and used to filter
            filename suffixes and assign label ids.
        splits: Split prefixes to scan.
        source_suffix: Source-domain suffix (default ``A``).
        target_suffix: Target-domain suffix (default ``B``).
        panel_policy: ``"complete"``, ``"partial"`` or ``"mixed"``.
        label_mapping: Optional map from stain name to integer class index.
            Emits a ``target_label`` column when provided; otherwise the
            index in ``panel_members`` is used.
        stain_from_subdir: ``True`` for Layout A, ``False`` for Layout B.
        source_stain: Stain suffix that identifies source images in Layout B
            (default ``"HE"``). Set to another panel member when the input
            domain is not H&E.

    Returns:
        One ``(DataFrame, sidecar)`` tuple per shard.
    """
    panel_root = Path(panel_root)

    if stain_from_subdir:
        stain_dirs = sorted([d for d in panel_root.iterdir() if d.is_dir()])
        panel_members = panel_members or [d.name for d in stain_dirs]
    else:
        if panel_members is None:
            raise ValueError("panel_members is required when stain_from_subdir=False")
        stain_dirs = [panel_root]

    label_mapping = label_mapping or {name: i for i, name in enumerate(panel_members)}
    results: list[tuple[pd.DataFrame, dict]] = []

    for stain_dir in stain_dirs:
        if stain_from_subdir:
            stain_candidates = {stain_dir.name}
        else:
            stain_candidates = set(panel_members)

        rows: list[dict] = []

        for split in splits:
            src_dir = stain_dir / f"{split}{source_suffix}"
            tgt_dir = stain_dir / f"{split}{target_suffix}"
            if not src_dir.exists() or not tgt_dir.exists():
                continue

            src_files = _list_image_files(src_dir)
            tgt_files = _list_image_files(tgt_dir)

            if stain_from_subdir:
                # Layout A: paired folders share the same filename within a
                # stain-specific subdirectory.
                for filename in sorted(set(src_files.keys()) & set(tgt_files.keys())):
                    stem = Path(filename).stem
                    stain_name = stain_dir.name
                    dataset_uid = f"source_to_{stain_name.lower()}"
                    rows.append(
                        {
                            "sample_uid": f"{dataset_uid}_{split}_{stem}",
                            "source_path": str(
                                Path(f"{split}{source_suffix}") / filename
                            ),
                            "target_path": str(
                                Path(f"{split}{target_suffix}") / filename
                            ),
                            "target_stain": stain_name,
                            "target_label": label_mapping[stain_name],
                            "dataset_split": split,
                            "dataset_uid": dataset_uid,
                            "panel_id": panel_id,
                        }
                    )
            else:
                # Layout B: stain is encoded in the filename suffix. Group by
                # case id and pair source_stain in the source folder with each
                # available target stain in the target folder. The source stain
                # is parsed even when it is not listed as a target panel member.
                parse_candidates = stain_candidates | {source_stain}
                src_cases = _parse_suffix_files(src_files, parse_candidates)
                tgt_cases = _parse_suffix_files(tgt_files, parse_candidates)
                dataset_uid = f"{panel_id}_v1"

                for case_id in sorted(src_cases.keys()):
                    if source_stain not in src_cases[case_id]:
                        continue
                    src_filename = src_cases[case_id][source_stain]
                    for stain_name in panel_members:
                        if stain_name == source_stain:
                            continue
                        if stain_name not in tgt_cases.get(case_id, {}):
                            continue
                        tgt_filename = tgt_cases[case_id][stain_name]
                        stem = Path(src_filename).stem
                        rows.append(
                            {
                                "sample_uid": f"{dataset_uid}_{split}_{case_id}_{stain_name}",
                                "source_path": str(
                                    Path(f"{split}{source_suffix}") / src_filename
                                ),
                                "target_path": str(
                                    Path(f"{split}{target_suffix}") / tgt_filename
                                ),
                                "target_stain": stain_name,
                                "target_label": label_mapping[stain_name],
                                "dataset_split": split,
                                "dataset_uid": dataset_uid,
                                "panel_id": panel_id,
                            }
                        )

        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["__root_dir"] = str(stain_dir)
        sidecar = {
            "dataset_uid": (
                f"he_to_{stain_dir.name.lower()}"
                if stain_from_subdir
                else f"{panel_id}_v1"
            ),
            "panel_id": panel_id,
            "panel_members": panel_members,
            "panel_policy": panel_policy,
            "source_domain": "source",
            "target_domain": stain_dir.name if stain_from_subdir else panel_id,
            "directory_format": "PAIRED_SUBDIR" if stain_from_subdir else "PAIRED_FLAT",
            "label_mapping": label_mapping,
        }
        results.append((df, sidecar))

    return results


def scan_aligned_directory(
    panel_root: str | Path,
    panel_id: str,
    target_stain: str,
    splits: tuple[str, ...] = ("train", "val", "test"),
    source_suffix: str = "A",
    target_suffix: str = "B",
    panel_policy: str = "mixed",
) -> list[tuple[pd.DataFrame, dict]]:
    """Scan the Pix2PixHD-aligned directory layout into one shard.

    Layout::

        dataroot/
          trainA/  trainB/
          valA/    valB/
          testA/   testB/

    Source images live in ``{split}{source_suffix}`` and target images in
    ``{split}{target_suffix}``. This matches the Pix2PixHD ``AlignedDataset``
    when ``label_nc == 0``.

    Args:
        panel_root: Root directory containing ``trainA``/``trainB`` etc.
        panel_id: Panel identifier.
        target_stain: Name of the target stain/domain (e.g. ``"ER"``).
        splits: Split prefixes to scan.
        source_suffix: Source-domain folder suffix (default ``"A"``).
        target_suffix: Target-domain folder suffix (default ``"B"``).
        panel_policy: ``"complete"``, ``"partial"`` or ``"mixed"``.

    Returns:
        A single-element list with the shard ``(DataFrame, sidecar)``.
    """
    panel_root = Path(panel_root)
    label_mapping = {target_stain: 0}
    dataset_uid = f"he_to_{target_stain.lower()}"
    rows: list[dict] = []

    for split in splits:
        src_dir = panel_root / f"{split}{source_suffix}"
        tgt_dir = panel_root / f"{split}{target_suffix}"
        if not src_dir.exists() or not tgt_dir.exists():
            continue

        src_files = _list_image_files(src_dir)
        tgt_files = _list_image_files(tgt_dir)

        for filename in sorted(set(src_files.keys()) & set(tgt_files.keys())):
            stem = Path(filename).stem
            rows.append(
                {
                    "sample_uid": f"{dataset_uid}_{split}_{stem}",
                    "source_path": str(Path(f"{split}{source_suffix}") / filename),
                    "target_path": str(Path(f"{split}{target_suffix}") / filename),
                    "target_stain": target_stain,
                    "target_label": label_mapping[target_stain],
                    "dataset_split": split,
                    "dataset_uid": dataset_uid,
                    "panel_id": panel_id,
                }
            )

    if not rows:
        raise ValueError(f"No aligned {source_suffix}/{target_suffix} pairs found under {panel_root}")

    df = pd.DataFrame(rows)
    df["__root_dir"] = str(panel_root)
    sidecar = {
        "dataset_uid": dataset_uid,
        "panel_id": panel_id,
        "panel_members": [target_stain],
        "panel_policy": panel_policy,
        "source_domain": "source",
        "target_domain": target_stain,
        "directory_format": "ALIGNED",
        "label_mapping": label_mapping,
    }
    return [(df, sidecar)]


def scan_labeled_image_folder(
    root: str | Path,
    panel_id: str,
    label_mapping: Optional[dict[str, int]] = None,
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    label_from_filename: bool = True,
    split: Optional[str] = None,
) -> list[tuple[pd.DataFrame, dict]]:
    """Scan a single-image folder with labels encoded in filenames.

    Layout::

        root/
          xxx_0.png
          yyy_1.png
          ...

    This matches the classic PyTorch ``ImageFolder`` / ``DatasetFolder`` pattern
    used in many conditional-generation baselines: the filename suffix ``_N`` is
    the integer class label. It is also useful for source-only classification or
    conditioning datasets.

    Args:
        root: Directory containing images.
        panel_id: Panel identifier.
        label_mapping: Optional map from string label token to integer index.
            If ``label_from_filename`` is True and the suffix is a string stain
            name (e.g. ``xxx_ER.png``), this mapping is used; otherwise the
            suffix is parsed as an integer directly.
        extensions: Allowed image extensions.
        label_from_filename: If True, parse the last ``_token`` of the filename
            stem as the label; otherwise every image gets label ``0``.
        split: Optional split name to write into ``dataset_split``.

    Returns:
        A single-element list of ``(DataFrame, sidecar)``.
    """
    root = Path(root)
    rows: list[dict] = []

    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        stem = path.stem
        if label_from_filename:
            token = stem.split("_")[-1]
            if label_mapping is not None:
                if token not in label_mapping:
                    continue
                label = label_mapping[token]
            else:
                try:
                    label = int(token)
                except ValueError:
                    continue
        else:
            label = 0

        rows.append(
            {
                "sample_uid": f"{panel_id}_{stem}",
                "source_path": str(path.name),
                "target_label": label,
                "dataset_split": split if split is not None else "train",
                "dataset_uid": f"{panel_id}_v1",
                "panel_id": panel_id,
            }
        )

    if not rows:
        raise ValueError(f"No valid images found under {root}")

    df = pd.DataFrame(rows)
    df["__root_dir"] = str(root)
    sidecar = {
        "dataset_uid": f"{panel_id}_v1",
        "panel_id": panel_id,
        "panel_members": sorted(set(label_mapping)) if label_mapping else [],
        "panel_policy": "mixed",
        "source_domain": "source",
        "target_domain": panel_id,
        "directory_format": "LabeledImageFolder",
        "label_mapping": label_mapping,
    }
    return [(df, sidecar)]
