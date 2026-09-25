"""CLI entry point for offline auxiliary-image preprocessing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .base import PreprocessorRegistry, run_preprocessors

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute auxiliary images and extend a parquet."
    )
    parser.add_argument(
        "--sample-meta",
        required=True,
        help="Input sample_meta.parquet or shard-root directory.",
    )
    parser.add_argument(
        "--dataroot",
        required=True,
        help="Root directory containing the raw patch images.",
    )
    parser.add_argument(
        "--preprocessors",
        nargs="+",
        required=True,
        choices=PreprocessorRegistry.list(),
        help="One or more preprocessors to run.",
    )
    parser.add_argument(
        "--out-meta",
        required=True,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--out-images",
        required=True,
        help="Output directory for derived images.",
    )
    parser.add_argument(
        "--input-column",
        default="target_path",
        help="Parquet column used as preprocessor input (default: target_path).",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated splits to process (default: all).",
    )
    parser.add_argument(
        "--backend",
        default="default",
        help="Image backend name (default: default).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",")]

    run_preprocessors(
        sample_meta=args.sample_meta,
        dataroot=args.dataroot,
        preprocessors=args.preprocessors,
        out_meta=args.out_meta,
        out_images=args.out_images,
        input_column=args.input_column,
        splits=splits,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
