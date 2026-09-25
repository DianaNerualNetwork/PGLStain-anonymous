"""Per-class panel dataset for multi-target stain translation.

A self-contained dataset pipeline for methods whose training step consumes
one image from *every* panel class at once (e.g. StainExpert, where class 0
is the HE source domain and each remaining class is a target stain). It
reads the standard MIST parquet shard tree (``sample_meta.parquet`` +
``dataset_info.json`` per stain) but intentionally lives outside
:class:`~puzzlestain.data.datasets.unified.VirtualStainDataset` so the
stain-level (source, target) pair semantics of the existing data layer are
untouched.

Each sample carries a single ``panel_images`` field — one optionally resized,
randomly cropped, [-1, 1]-normalized tensor per class, shape
``(num_classes, C, H, W)`` — which the kind-driven collator stacks to
``(B, num_classes, C, H, W)``.
Classes are index-based and unpaired, mirroring the original
``UnpairedDataset``: class ``i``'s image is taken at the requested index
when in range, otherwise drawn at random from the class pool.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..fields import FieldKind, register_field_kind
from ..sample import StainSample

logger = logging.getLogger(__name__)

#: Field carrying the per-class image stack; registered as an image kind so
#: ``stain_collate_fn`` stacks it without any collator changes.
PANEL_IMAGES_FIELD = "panel_images"

register_field_kind(PANEL_IMAGES_FIELD, FieldKind.AUX_IMAGE)


class StainPanelDataset(torch.utils.data.Dataset):
    """One-image-per-class panel dataset built from MIST parquet shards.

    Args:
        class_pools: Resolved image paths per class; ``class_pools[0]`` is
            the source-domain (HE) pool.
        class_names: Class names aligned with ``class_pools`` (index 0 is
            the source domain).
        resize_size: Optional square whole-image resize applied before the
            crop. ``None`` preserves the original random-crop-only behavior.
            When set, it must be greater than or equal to ``crop_size``.
        crop_size: Square random-crop size applied independently to every
            class image (the original's ``RandomCrop``). Images smaller than
            the crop are upscaled to it first.
        max_dataset_size: Cap the dataset length when positive; ``0`` means
            the total number of images across all class pools (the
            original's ``__len__`` semantics).

    Attributes:
        panel_members: Target stain names (classes 1..K), exposed for the
            training entry point's logging.
        selected_stains: Same as ``panel_members``.
    """

    def __init__(
        self,
        class_pools: list[list[Path]],
        class_names: list[str],
        crop_size: int = 512,
        max_dataset_size: int = 0,
        resize_size: int | None = None,
    ) -> None:
        if not class_pools or any(len(pool) == 0 for pool in class_pools):
            raise ValueError("Every class pool must contain at least one image")
        self.class_pools = class_pools
        if resize_size is not None and resize_size < crop_size:
            raise ValueError("resize_size must be greater than or equal to crop_size")
        self.class_names = list(class_names)
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.panel_members = self.class_names[1:]
        self.selected_stains = self.panel_members
        total = sum(len(pool) for pool in class_pools)
        self._length = min(total, max_dataset_size) if max_dataset_size > 0 else total

    @classmethod
    def from_parquet(
        cls,
        parquet_root: str | Path,
        raw_root: str | Path,
        selected_stains: list[str] | None = None,
        splits: tuple[str, ...] = ("train",),
        split_root_name: str = "TrainValAB",
        source_name: str = "HE",
        crop_size: int = 512,
        max_dataset_size: int = 0,
        resize_size: int | None = None,
    ) -> StainPanelDataset:
        """Build class pools from a MIST parquet tree and its raw images.

        Args:
            parquet_root: Directory containing one shard subdirectory per
                stain (as produced by ``MISTVirtualStainDataset.convert``).
            raw_root: Original MIST/IHC4BC root with one subdirectory per
                target stain.
            selected_stains: Target stains to include, defining the order of
                classes 1..K; ``None`` uses the top-level panel order.
            splits: ``dataset_split`` values to keep (default train only).
            split_root_name: Intermediate folder holding the split folders.
            source_name: Name of the source-domain class (class 0).
            resize_size: Optional square whole-image resize before cropping.
                ``None`` preserves the original random-crop-only behavior.
                When set, it must be at least ``crop_size``.
            crop_size: Square random-crop size.
            max_dataset_size: Cap the dataset length when positive.
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

        # Map each shard to its target stain and per-shard image root.
        shard_frames: dict[str, pd.DataFrame] = {}
        shard_roots: dict[str, Path] = {}
        for shard in shards:
            parquet_path = parquet_root / shard / "sample_meta.parquet"
            if not parquet_path.is_file():
                logger.warning("Skipping missing shard parquet: %s", parquet_path)
                continue
            sidecar_path = parquet_root / shard / "dataset_info.json"
            target_domain = shard
            if sidecar_path.is_file():
                sidecar = json.loads(sidecar_path.read_text())
                target_domain = sidecar.get("target_domain") or shard
            df = pd.read_parquet(parquet_path)
            if splits:
                df = df[df["dataset_split"].isin(list(splits))]
            shard_frames[target_domain] = df.reset_index(drop=True)
            shard_roots[target_domain] = raw_root / target_domain / split_root_name

        stains = (
            list(selected_stains)
            if selected_stains
            else [m for m in top_info.get("panel_members", []) if m in shard_frames]
        )
        missing = [s for s in stains if s not in shard_frames]
        if missing:
            raise ValueError(
                f"No parquet shard found for selected_stains={missing}; "
                f"available: {sorted(shard_frames)}"
            )
        if not shard_frames:
            raise ValueError(f"No valid MIST shards found under {parquet_root}")

        def resolve(stain: str, rel_paths: pd.Series) -> list[Path]:
            return [shard_roots[stain] / rel for rel in rel_paths.tolist()]

        # Class 0 is the source domain; the HE trainA set is replicated
        # across MIST stain folders, so the first shard's pool is
        # representative (deduplicated by filename for safety).
        he_stain = stains[0]
        he_rows = shard_frames[he_stain]
        seen: set[str] = set()
        he_pool: list[Path] = []
        for rel in he_rows["source_path"].tolist():
            if Path(rel).name in seen:
                continue
            seen.add(Path(rel).name)
            he_pool.append(shard_roots[he_stain] / rel)

        class_pools = [he_pool]
        for stain in stains:
            class_pools.append(resolve(stain, shard_frames[stain]["target_path"]))

        class_names = [source_name, *stains]
        for name, pool in zip(class_names, class_pools):
            logger.info("Panel class %-6s -> %d images", name, len(pool))

        return cls(
            class_pools=class_pools,
            class_names=class_names,
            resize_size=resize_size,
            crop_size=crop_size,
            max_dataset_size=max_dataset_size,
        )

    def __len__(self) -> int:
        """Total number of images across all class pools (original semantics)."""
        return self._length

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load one RGB image as a resized/cropped normalized CHW tensor."""
        img = Image.open(path).convert("RGB")
        if self.resize_size is not None and img.size != (
            self.resize_size,
            self.resize_size,
        ):
            img = img.resize((self.resize_size, self.resize_size), Image.BICUBIC)
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            img = img.resize(
                (max(w, self.crop_size), max(h, self.crop_size)),
                Image.LANCZOS,
            )
            w, h = img.size
        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, C) in [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor * 2.0 - 1.0

    def __getitem__(self, index: int) -> StainSample:
        """Return one image per class as a stacked ``panel_images`` field.

        Class ``i``'s image is taken at ``index`` when in range, otherwise
        drawn at random from the class pool (unpaired classes, mirroring the
        original ``UnpairedDataset``).
        """
        images = []
        for pool in self.class_pools:
            path = pool[index] if index < len(pool) else random.choice(pool)
            images.append(self._load_image(path))
        return StainSample(
            **{PANEL_IMAGES_FIELD: torch.stack(images)},
            metadata={"class_names": self.class_names},
        )


__all__ = ["PANEL_IMAGES_FIELD", "StainPanelDataset"]
