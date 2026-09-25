"""CLI tool to convert MIST/IHC4BC-style directories to Parquet shards.

Usage::

    python -m puzzlestain.data.tools.convert_mist \
        --panel-root /data/IHC4BC \
        --panel-id breast_mist_4plex_20x \
        --out-root /data/IHC4BC_parquet

Produces::

    out-root/
      source_to_er/
        sample_meta.parquet
        dataset_info.json
      source_to_pr/
        sample_meta.parquet
        dataset_info.json
      ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from puzzlestain.data.datasets.mist import MISTVirtualStainDataset

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MIST/IHC4BC layout to Parquet shards"
    )
    parser.add_argument(
        "--panel-root", required=True, help="Root with one subdirectory per stain"
    )
    parser.add_argument(
        "--panel-id",
        default=MISTVirtualStainDataset.DEFAULT_PANEL_ID,
        help="Panel identifier",
    )
    parser.add_argument("--out-root", required=True, help="Output directory")
    parser.add_argument(
        "--splits", default="train,val,test", help="Comma-separated split prefixes"
    )
    parser.add_argument("--a-suffix", default="A")
    parser.add_argument("--b-suffix", default="B")
    parser.add_argument("--panel-policy", default="mixed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    MISTVirtualStainDataset.convert(
        panel_root=args.panel_root,
        out_root=args.out_root,
        panel_id=args.panel_id,
        splits=tuple(args.splits.split(",")),
        source_suffix=args.a_suffix,
        target_suffix=args.b_suffix,
        panel_policy=args.panel_policy,
    )


if __name__ == "__main__":
    main()
