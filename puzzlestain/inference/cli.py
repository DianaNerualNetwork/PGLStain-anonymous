"""Command-line entry point for virtual-stain inference.

Usage::

    puzzlestain-predict <checkpoint_dir> <dataset> <stain> <exp_name> <source_dir>
        [--gt-dir <gt_dir>] -v

The default ``resize`` mode preserves the original direct-resize behavior.
Strict ``mpp-full`` and ``mpp-sliding`` modes require an explicit input
MPP and write a reproducibility manifest. BatchNorm-based Pix2Pix checkpoints
trained with batch size one may require ``--train-mode`` to match their
original repository's inference convention.

Predicted images are written under ``PATHOLOGY_CACHE_ROOT`` using the same
layout that ``puzzlestain-eval`` consumes::

    {PATHOLOGY_CACHE_ROOT}/
      └── {dataset}/
            └── {stain}/
                  └── {exp_name}/
                        ├── source/   <- spatially prepared HE images
                        ├── fake/     <- generated virtual-stain images
                        └── gt/       <- spatially prepared real IHC images
                                       (when --gt-dir is given)

If ``PATHOLOGY_CACHE_ROOT`` is not set, the current directory is used and a
warning is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL import __version__ as pillow_version

from ..logging_utils import configure_logging
from .predictor import StainPredictor
from .spatial import PhysicalInferenceSpec, prepare_physical_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puzzlestain-predict",
        description=(
            "Generate virtual-stain images from a trained checkpoint and save "
            "them in the evaluation directory layout."
        ),
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Checkpoint directory containing config.yaml and trainer_state.pt.",
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (top-level directory under PATHOLOGY_CACHE_ROOT).",
    )
    parser.add_argument(
        "stain",
        help="Stain type, e.g. ER, PR, HER2, Ki67.",
    )
    parser.add_argument(
        "exp_name",
        help="Experiment/model name used for the output subdirectory.",
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing source HE images.",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Directory containing ground-truth IHC images (matched by filename stem).",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help=(
            "Root directory for virtual-stain results. "
            "Defaults to PATHOLOGY_CACHE_ROOT env var, then current directory."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda / cpu). Default: auto.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=None,
        help=(
            "Override the legacy direct-resize size. Valid only with "
            "--inference-mode=resize."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        choices=("resize", "mpp-full", "mpp-sliding"),
        default="resize",
        help=(
            "Spatial inference protocol. 'resize' preserves the original behavior; "
            "strict MPP modes resample once to a declared physical resolution."
        ),
    )
    parser.add_argument(
        "--input-mpp",
        type=float,
        default=None,
        help="Required patch resolution in microns per pixel for strict MPP modes.",
    )
    parser.add_argument(
        "--target-mpp",
        type=float,
        default=None,
        help=(
            "Override the checkpoint's data.target_mpp for strict MPP modes. "
            "Required when an older checkpoint does not record it."
        ),
    )
    parser.add_argument(
        "--gt-mpp",
        type=float,
        default=None,
        help="Ground-truth patch MPP; defaults to --input-mpp.",
    )
    parser.add_argument(
        "--full-size",
        type=int,
        default=None,
        help=(
            "Exact post-MPP square size for mpp-full. Defaults to the training "
            "resize canvas recorded in the checkpoint."
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help=(
            "Tile size for mpp-sliding. Defaults to the training crop size, "
            "then 512 for older checkpoints."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=128,
        help="Tile overlap for mpp-sliding. Default: 128.",
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        help=(
            "Save float32 .npy arrays instead of uint8 PNGs. The output range "
            "depends on the model and inference mode; CUT/PGLStain resize "
            "inference returns [-1, 1]."
        ),
    )
    parser.add_argument(
        "--train-mode",
        action="store_true",
        help=(
            "Run inference with the model in training mode. This makes BatchNorm "
            "use the input image's own statistics instead of the accumulated running "
            "statistics, matching the default behavior of the original PyramidP2P "
            "test script. Useful when batch_size=1 + BatchNorm was used during "
            "training and eval-mode results look washed out."
        ),
    )
    parser.add_argument(
        "--seed-per-image",
        action="store_true",
        help=(
            "Fix the random seed per image (derived from image content) so "
            "repeated runs produce identical outputs even with dropout active "
            "(i.e. with --train-mode)."
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print INFO-level logs.",
    )
    return parser


def _resolve_cache_root(args_root: str | None) -> Path:
    """Return the cache root from CLI arg, env var, or current directory."""
    if args_root:
        return Path(args_root)
    env = os.environ.get("PATHOLOGY_CACHE_ROOT")
    if env:
        return Path(env)
    logger.warning(
        "PATHOLOGY_CACHE_ROOT not set and --cache-root not given; using current directory"
    )
    return Path(".")


def _collect_image_paths(
    directory: Path,
    *,
    strict: bool = False,
) -> dict[str, Path]:
    """Map filename stem to path for image files in deterministic order."""

    paths: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in paths:
            message = (
                f"Duplicate image stem {path.stem!r} in {directory}: "
                f"{paths[path.stem].name!r} and {path.name!r}"
            )
            if strict:
                raise ValueError(message)
            logger.warning("%s; keeping the first file", message)
            continue
        paths[path.stem] = path
    return paths


def _prepare_eval_layout(
    cache_root: Path,
    dataset: str,
    stain: str,
    exp_name: str,
    source_dir: Path,
    gt_dir: Path | None,
    resize_size: int,
    interpolation: int = Image.BICUBIC,
) -> tuple[Path, Path, Path]:
    """Create source/fake/gt directories and copy source/gt images into them.

    Source and ground-truth images are resized to ``resize_size`` with the
    same filter the predictor applies to its inputs, so fake, source,
    and gt share one resolution and metrics compare like with like.

    Returns:
        (exp_root, fake_dir, source_dir_out) so the caller can write fake images.
    """
    exp_root = cache_root / dataset / stain / exp_name
    source_out = exp_root / "source"
    fake_out = exp_root / "fake"
    gt_out = exp_root / "gt"

    source_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)
    if gt_dir is not None:
        gt_out.mkdir(parents=True, exist_ok=True)

    source_paths = _collect_image_paths(source_dir)
    if not source_paths:
        raise ValueError(f"No images found in source directory: {source_dir}")

    gt_paths: dict[str, Path] = {}
    if gt_dir is not None:
        gt_paths = _collect_image_paths(gt_dir)
        missing_gt = set(source_paths) - set(gt_paths)
        if missing_gt:
            logger.warning(
                "No ground-truth match for %d source image(s): %s",
                len(missing_gt),
                sorted(missing_gt)[:10],
            )

    for stem, src_path in source_paths.items():
        dst_name = f"{stem}.png"
        src_img = Image.open(src_path).convert("RGB")
        if src_img.size != (resize_size, resize_size):
            src_img = src_img.resize((resize_size, resize_size), interpolation)
        src_img.save(source_out / dst_name)
        if stem in gt_paths:
            gt_img = Image.open(gt_paths[stem]).convert("RGB")
            if gt_img.size != (resize_size, resize_size):
                gt_img = gt_img.resize((resize_size, resize_size), interpolation)
            gt_img.save(gt_out / dst_name)

    return exp_root, fake_out, source_out


def _resolve_physical_spec(
    args: argparse.Namespace,
    predictor: StainPredictor,
) -> PhysicalInferenceSpec:
    """Resolve a strict MPP protocol without guessing missing physical scale."""

    if args.crop_size is not None:
        raise ValueError("--crop-size is valid only with --inference-mode=resize")
    if args.input_mpp is None:
        raise ValueError(f"--input-mpp is required for {args.inference_mode}")

    target_mpp = (
        args.target_mpp if args.target_mpp is not None else predictor.target_mpp
    )
    if target_mpp is None:
        raise ValueError(
            "This checkpoint does not record data.target_mpp; provide "
            "--target-mpp explicitly"
        )

    if args.inference_mode == "mpp-full":
        if args.tile_size is not None:
            raise ValueError("--tile-size is valid only with mpp-sliding")
        full_size = (
            args.full_size
            if args.full_size is not None
            else predictor.training_resize_size
        )
        if full_size is None:
            raise ValueError(
                "The checkpoint does not record a square training resize canvas; "
                "provide --full-size explicitly"
            )
        return PhysicalInferenceSpec(
            mode="mpp-full",
            input_mpp=args.input_mpp,
            target_mpp=target_mpp,
            full_size=full_size,
        )

    if args.full_size is not None:
        raise ValueError("--full-size is valid only with mpp-full")
    tile_size = (
        args.tile_size
        if args.tile_size is not None
        else predictor.training_crop_size or 512
    )
    return PhysicalInferenceSpec(
        mode="mpp-sliding",
        input_mpp=args.input_mpp,
        target_mpp=target_mpp,
        tile_size=tile_size,
        overlap=args.tile_overlap,
    )


def _prepare_mpp_eval_layout(
    cache_root: Path,
    dataset: str,
    stain: str,
    exp_name: str,
    source_dir: Path,
    gt_dir: Path | None,
    spec: PhysicalInferenceSpec,
    *,
    gt_mpp: float,
    interpolation: int,
) -> tuple[Path, Path, Path]:
    """Validate, resample once to target MPP, and create evaluation artifacts."""

    exp_root = cache_root / dataset / stain / exp_name
    if exp_root.exists() and (not exp_root.is_dir() or any(exp_root.iterdir())):
        raise FileExistsError(
            f"Strict inference refuses the non-empty experiment path: {exp_root}"
        )

    source_paths = _collect_image_paths(source_dir, strict=True)
    if not source_paths:
        raise ValueError(f"No images found in source directory: {source_dir}")

    gt_paths: dict[str, Path] = {}
    if gt_dir is not None:
        gt_paths = _collect_image_paths(gt_dir, strict=True)
        missing_gt = sorted(set(source_paths) - set(gt_paths))
        if missing_gt:
            raise ValueError(
                "Strict inference requires one ground truth for every source; "
                f"missing {len(missing_gt)} stem(s): {missing_gt[:10]}"
            )

    gt_spec = replace(spec, input_mpp=gt_mpp)

    def prepared(path: Path, image_spec: PhysicalInferenceSpec) -> np.ndarray:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        return prepare_physical_image(rgb, image_spec, interpolation=interpolation)

    # Validate the complete set before creating output directories, so a bad
    # sample cannot leave a partially populated benchmark experiment.
    for stem, source_path in source_paths.items():
        source = prepared(source_path, spec)
        if stem in gt_paths:
            target = prepared(gt_paths[stem], gt_spec)
            if source.shape != target.shape:
                raise ValueError(
                    f"MPP-prepared source/ground-truth shape mismatch for {stem!r}: "
                    f"{source.shape} versus {target.shape}"
                )

    source_out = exp_root / "source"
    fake_out = exp_root / "fake"
    gt_out = exp_root / "gt"
    source_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)
    if gt_dir is not None:
        gt_out.mkdir(parents=True, exist_ok=True)

    for stem, source_path in source_paths.items():
        Image.fromarray(prepared(source_path, spec)).save(source_out / f"{stem}.png")
        if stem in gt_paths:
            Image.fromarray(prepared(gt_paths[stem], gt_spec)).save(
                gt_out / f"{stem}.png"
            )

    return exp_root, fake_out, source_out


def _write_prediction_manifest(
    exp_root: Path,
    checkpoint_dir: Path,
    source_dir: Path,
    gt_dir: Path | None,
    predictor: StainPredictor,
    spec: PhysicalInferenceSpec,
    *,
    gt_mpp: float,
    normalized: bool,
) -> Path:
    """Atomically write machine-readable provenance for strict inference."""

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def dimensions(directory: Path) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(path) as image:
                    width, height = image.size
            elif path.suffix.lower() == ".npy":
                array = np.load(path, mmap_mode="r")
                height, width = array.shape[:2]
            else:
                continue
            counts[f"{width}x{height}"] += 1
        return dict(sorted(counts.items()))

    try:
        package_version = metadata.version("puzzlestain")
    except metadata.PackageNotFoundError:
        package_version = "unknown"

    interpolation_name = {
        Image.NEAREST: "nearest",
        Image.BILINEAR: "bilinear",
        Image.BICUBIC: "bicubic",
        Image.LANCZOS: "lanczos",
    }.get(predictor.interpolation, str(predictor.interpolation))
    checkpoint_dir = checkpoint_dir.resolve()
    gt_out = exp_root / "gt"
    protocol: dict[str, Any] = {
        "mode": spec.mode,
        "input_mpp": spec.input_mpp,
        "target_mpp": spec.target_mpp,
        "scale_factor": spec.input_mpp / spec.target_mpp,
        "dimension_rounding": "round_half_up",
        "interpolation": interpolation_name,
        "model_mode": "train" if predictor.train_mode else "eval",
        "seed_strategy": (
            "first 64 bits of SHA-256(image bytes), reduced to signed 63-bit"
        ),
    }
    if spec.mode == "mpp-full":
        protocol["full_size"] = spec.full_size
    else:
        protocol.update(
            {
                "tile_size": spec.tile_size,
                "overlap": spec.overlap,
                "stride": spec.tile_size - spec.overlap,
                "blend": "raised_cosine",
                "padding": "reflect; edge for singleton axes",
                "traversal": "row_major",
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "state_file": "trainer_state.pt",
            "state_sha256": sha256(checkpoint_dir / "trainer_state.pt"),
            "config_file": "config.yaml",
            "config_sha256": sha256(checkpoint_dir / "config.yaml"),
        },
        "protocol": protocol,
        "training_spatial_config": {
            "resize_size": predictor.training_resize_size,
            "crop_size": predictor.training_crop_size,
            "target_mpp": predictor.target_mpp,
        },
        "inputs": {
            "source_directory": str(source_dir.resolve()),
            "ground_truth_directory": (
                str(gt_dir.resolve()) if gt_dir is not None else None
            ),
            "ground_truth_mpp": gt_mpp if gt_dir is not None else None,
            "source_dimensions": dimensions(source_dir),
        },
        "outputs": {
            "experiment_root": str(exp_root.resolve()),
            "artifact_format": "npy-float32" if normalized else "png-uint8",
            "source_dimensions": dimensions(exp_root / "source"),
            "prediction_dimensions": dimensions(exp_root / "fake"),
            "ground_truth_dimensions": (dimensions(gt_out) if gt_out.is_dir() else {}),
        },
        "software": {
            "python": platform.python_version(),
            "puzzlestain": package_version,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "command": list(sys.argv),
    }
    manifest_path = exp_root / "prediction_manifest.json"
    temporary_path = exp_root / ".prediction_manifest.json.tmp"
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    configure_logging("INFO" if args.verbose else "WARNING")

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(2)

    gt_dir: Path | None = None
    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        if not gt_dir.is_dir():
            print(f"Error: gt directory not found: {gt_dir}", file=sys.stderr)
            sys.exit(2)

    cache_root = _resolve_cache_root(args.cache_root)

    logger.info("Loading checkpoint from %s", args.checkpoint_dir)
    predictor = StainPredictor.from_checkpoint(
        args.checkpoint_dir,
        device=args.device,
        train_mode=args.train_mode,
        seed_per_image=args.seed_per_image,
    )

    stain_names = getattr(predictor.model, "stain_names", None)
    if stain_names is not None:
        # Multi-stain checkpoint: map the requested stain to its domain label.
        lookup = {name.lower(): idx for idx, name in enumerate(stain_names)}
        if args.stain.lower() not in lookup:
            print(
                f"Error: stain {args.stain!r} is not in this checkpoint's "
                f"stain table; available: {list(stain_names)}",
                file=sys.stderr,
            )
            sys.exit(2)
        predictor.target_label = lookup[args.stain.lower()]
        logger.info(
            "Multi-stain checkpoint: target stain %s -> label %d",
            args.stain,
            predictor.target_label,
        )

    mpp_spec: PhysicalInferenceSpec | None = None
    gt_mpp: float | None = None
    if args.inference_mode == "resize":
        if args.crop_size is not None:
            predictor.crop_size = args.crop_size
        exp_root, fake_dir, _ = _prepare_eval_layout(
            cache_root=cache_root,
            dataset=args.dataset,
            stain=args.stain,
            exp_name=args.exp_name,
            source_dir=source_dir,
            gt_dir=gt_dir,
            resize_size=predictor.crop_size,
            interpolation=predictor.interpolation,
        )
    else:
        try:
            mpp_spec = _resolve_physical_spec(args, predictor)
            gt_mpp = args.gt_mpp if args.gt_mpp is not None else mpp_spec.input_mpp
            exp_root, fake_dir, _ = _prepare_mpp_eval_layout(
                cache_root=cache_root,
                dataset=args.dataset,
                stain=args.stain,
                exp_name=args.exp_name,
                source_dir=source_dir,
                gt_dir=gt_dir,
                spec=mpp_spec,
                gt_mpp=gt_mpp,
                interpolation=predictor.interpolation,
            )
        except (FileExistsError, ValueError) as error:
            parser.error(str(error))

    written = predictor.predict_dir(
        source_dir,
        fake_dir,
        return_normalized=args.normalized,
        suffix=".png" if not args.normalized else ".npy",
        mpp_spec=mpp_spec,
    )

    print(f"Wrote {len(written)} predictions to {fake_dir}")
    print(f"Evaluation layout ready at {exp_root}")
    if mpp_spec is not None:
        assert gt_mpp is not None
        manifest_path = _write_prediction_manifest(
            exp_root,
            Path(args.checkpoint_dir),
            source_dir,
            gt_dir,
            predictor,
            mpp_spec,
            gt_mpp=gt_mpp,
            normalized=args.normalized,
        )
        print(f"Reproducibility manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
