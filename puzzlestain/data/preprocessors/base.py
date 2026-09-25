"""Offline preprocessors that derive auxiliary images from raw patches.

Each preprocessor consumes one input image per row (typically the target/IHC
patch) and writes one or more derived images to an output directory. A new
parquet is produced with additional ``{name}_path`` columns referencing the
derived images, ready for ``VirtualStainDataset(aux_columns=...)``. Derived
images that already exist under the output directory are reused instead of
recomputed (idempotent reruns / link-only reuse of a previous tree).
"""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..backends import BackendFactory

logger = logging.getLogger(__name__)


class Preprocessor(abc.ABC):
    """Abstract base for an auxiliary-image preprocessor.

    A preprocessor is a pure function ``image -> dict[str, image]``. The runner
    handles I/O: reading the input column, saving outputs, and updating the
    parquet.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short preprocessor name used in CLI and directory names."""
        ...

    @property
    @abc.abstractmethod
    def output_columns(self) -> list[str]:
        """Names of the derived images produced by this preprocessor."""
        ...

    @abc.abstractmethod
    def process(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Process one image and return a mapping ``output_name -> image``.

        All returned arrays should have the same spatial size as the input so
        that spatial augmentation stays aligned downstream.
        """
        ...

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return self.process(image)


class PreprocessorRegistry:
    """Registry of built-in and user-provided preprocessors."""

    _registry: dict[str, type[Preprocessor]] = {}

    @classmethod
    def register(cls, name: str, preprocessor_cls: type[Preprocessor]) -> None:
        """Register a preprocessor class under ``name``."""
        cls._registry[name] = preprocessor_cls

    @classmethod
    def get(cls, name: str) -> type[Preprocessor]:
        """Return the preprocessor class registered under ``name``."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown preprocessor: {name}. "
                f"Registered: {sorted(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered preprocessor names."""
        return sorted(cls._registry.keys())


def _auto_register_builtins() -> None:
    """Lazily register built-in preprocessors if their dependencies are present."""
    # Import inside the function to avoid hard dependencies at package import.
    try:
        from .dab import DABPreprocessor

        PreprocessorRegistry.register(DABPreprocessor().name, DABPreprocessor)
    except ImportError as exc:  # pragma: no cover
        logger.debug("DAB preprocessor not available: %s", exc)

    try:
        from .nuclei import NucleiPreprocessor

        PreprocessorRegistry.register(NucleiPreprocessor().name, NucleiPreprocessor)
    except ImportError as exc:  # pragma: no cover
        logger.debug("Nuclei preprocessor not available: %s", exc)


_auto_register_builtins()


def run_preprocessors(
    sample_meta: str | Path,
    dataroot: str | Path,
    preprocessors: list[str | Preprocessor],
    out_meta: str | Path,
    out_images: str | Path,
    input_column: str = "target_path",
    splits: Optional[list[str]] = None,
    backend: Optional[str | Any] = None,
) -> pd.DataFrame:
    """Run preprocessors on a parquet and emit a new parquet + derived images.

    Args:
        sample_meta: Input parquet path (or shard-root directory).
        dataroot: Root directory containing the raw images referenced by the
            parquet. Used when the parquet lacks a ``__root_dir`` column.
        preprocessors: List of preprocessor names or ``Preprocessor`` instances.
        out_meta: Output parquet path.
        out_images: Output directory for derived images.
        input_column: Parquet column to use as input image (default
            ``target_path``).
        splits: Optional split filter applied before processing.
        backend: Image backend name or instance.

    Returns:
        The updated DataFrame.
    """
    from ..datasets.unified import VirtualStainDataset

    out_meta = Path(out_meta)
    out_images = Path(out_images)
    out_images.mkdir(parents=True, exist_ok=True)

    # Load the table without building full stain-level entries; we only need
    # the raw rows and the resolved root directory.
    df = pd.read_parquet(sample_meta)
    if "__root_dir" not in df.columns:
        df["__root_dir"] = str(dataroot)

    if splits:
        df = df[df["dataset_split"].isin(splits)].reset_index(drop=True)

    backend_reader = BackendFactory.create(backend or "default")

    # Instantiate named preprocessors.
    instances: list[Preprocessor] = []
    for p in preprocessors:
        if isinstance(p, str):
            instances.append(PreprocessorRegistry.get(p)())
        else:
            instances.append(p)

    # Create one output subdirectory per output column.
    output_dirs: dict[str, Path] = {}
    for proc in instances:
        for col in proc.output_columns:
            d = out_images / col
            d.mkdir(parents=True, exist_ok=True)
            output_dirs[col] = d

    path_col_cache: dict[str, list[str]] = {col: [] for col in output_dirs}

    n_rows = len(df)
    n_reused = 0
    for idx, row in df.iterrows():
        root = Path(str(row["__root_dir"]))
        in_path = root / str(row[input_column])

        # Expected output per column (deterministic name). Derived images
        # that already exist are reused instead of recomputed, so reruns are
        # idempotent and a previously generated tree can simply be linked by
        # pointing --out-images at it.
        in_name = Path(str(row[input_column])).name
        expected = {
            col: output_dirs[col] / _replace_suffix(in_name, f"_{col}.png")
            for col in output_dirs
        }

        image = None
        for proc in instances:
            missing_cols = [c for c in proc.output_columns if not expected[c].is_file()]
            if not missing_cols:
                continue
            if image is None:
                image = backend_reader.read(in_path)
            outputs = proc(image)
            for col in missing_cols:
                out_arr = outputs.get(col)
                if out_arr is None:
                    continue
                _save_image(out_arr, expected[col])

        if image is None:
            n_reused += 1
        for col in output_dirs:
            path_col_cache[col].append(
                str(expected[col].resolve()) if expected[col].is_file() else ""
            )

        if (idx + 1) % 100 == 0 or idx == n_rows - 1:
            logger.info("Preprocessed %d/%d rows", idx + 1, n_rows)

    if n_reused:
        logger.info("Reused existing derived images for %d/%d rows", n_reused, n_rows)

    for col, paths in path_col_cache.items():
        df[f"{col}_path"] = paths

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_meta, index=False)
    logger.info("Wrote %s with columns %s", out_meta, df.columns.tolist())

    # Propagate panel metadata next to the output parquet; without a sidecar,
    # VirtualStainDataset cannot validate selected_stains (panel_members = []).
    # Prefer the input sidecar; otherwise rebuild it from the table itself.
    from ..datasets.base import load_sidecar

    sidecar = load_sidecar(sample_meta)
    if not sidecar:
        sidecar = {}
        for key in ("dataset_uid", "panel_id", "panel_policy"):
            if key in df.columns and df[key].nunique() == 1:
                sidecar[key] = str(df[key].iloc[0])
        if "target_stain" in df.columns:
            sidecar["panel_members"] = sorted(df["target_stain"].astype(str).unique())
    if sidecar.get("panel_members"):
        (out_meta.parent / "dataset_info.json").write_text(
            json.dumps(sidecar, indent=2)
        )
        logger.info("Wrote %s", out_meta.parent / "dataset_info.json")
    return df


def _replace_suffix(filename: str, suffix: str) -> str:
    """Append a suffix before the extension, e.g. ``a.png`` -> ``a_dab_mask.png``."""
    p = Path(filename)
    return f"{p.stem}{suffix}"


def _save_image(arr: np.ndarray, path: Path) -> None:
    """Save a channels-last numpy image as PNG."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    elif arr.shape[2] == 1:
        Image.fromarray(arr.squeeze(2), mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
