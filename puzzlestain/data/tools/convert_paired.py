"""CLI tool to convert paired directories to Parquet shards.

Usage::

    # Layout A (MIST/IHC4BC style)
    python -m puzzlestain.data.tools.convert_paired \
        --dataroot /data/IHC4BC \
        --panel-id breast_ihc_4plex \
        --out-root /data/IHC4BC_parquet \
        --stain-from-subdir

    # Layout B (flat paired + filename suffix)
    python -m puzzlestain.data.tools.convert_paired \
        --dataroot /data/PGVMS \
        --panel-id breast_ihc_4plex \
        --panel-members ER,PR,Ki67,HER2 \
        --out-root /data/PGVMS_parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from puzzlestain.data.scanners import scan_paired_directory

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert paired layout to Parquet shards"
    )
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--panel-members", default=None, help="Comma-separated stain names"
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--source-suffix", default="A")
    parser.add_argument("--target-suffix", default="B")
    parser.add_argument("--panel-policy", default="mixed")
    parser.add_argument("--stain-from-subdir", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    panel_members = None
    if args.panel_members:
        panel_members = [s.strip() for s in args.panel_members.split(",")]

    results = scan_paired_directory(
        panel_root=args.dataroot,
        panel_id=args.panel_id,
        panel_members=panel_members,
        splits=tuple(args.splits.split(",")),
        source_suffix=args.source_suffix,
        target_suffix=args.target_suffix,
        panel_policy=args.panel_policy,
        stain_from_subdir=args.stain_from_subdir,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for df, info in results:
        dataset_uid = info["dataset_uid"]
        shard_dir = out_root / dataset_uid
        shard_dir.mkdir(parents=True, exist_ok=True)

        df_out = df.drop(columns=["__root_dir"], errors="ignore")
        df_out.to_parquet(shard_dir / "sample_meta.parquet", index=False)
        (shard_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
        logger.info("Wrote %s: %d rows", dataset_uid, len(df_out))


if __name__ == "__main__":
    main()
