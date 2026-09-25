"""CLI tool to build a single domain-pool shard from paired stain shards.

Any-to-any multi-stain methods (e.g. UMDST, GramGAN, PD-UniST) train on one
pool that mixes every domain — the H&E source images plus each IHC stain's
target images — with a per-image domain label. This tool derives that pool
from the paired ``he_to_*`` shards produced by ``convert_mist`` /
``convert_paired`` without re-scanning the raw images.

Usage::

    python -m puzzlestain.data.tools.convert_domain_pool \
        --shard-root /data/MIST_parquet \
        --raw-root /data/MIST \
        --out /data/MIST_domain_pool

Produces a single shard (source-only rows with ``target_label``, consumed
directly by ``VirtualStainDataset`` in labeled mode)::

    out/
      sample_meta.parquet   # sample_uid, source_path, target_label, ...
      dataset_info.json     # label_mapping / panel_members include the HE domain
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _load_shard_rows(
    shard_root: Path,
    raw_root: Path,
    split_root_name: str,
) -> tuple[list[dict], dict, str]:
    """Collect pool rows from every paired shard under ``shard_root``.

    Returns ``(rows, label_mapping, panel_id)`` where ``rows`` carries
    per-row ``__root_dir`` so the pool shard needs no external patch root.
    """
    top_info_path = shard_root / "dataset_info.json"
    if not top_info_path.is_file():
        raise ValueError(f"Missing top-level dataset_info.json in {shard_root}")
    top_info = json.loads(top_info_path.read_text())
    panel_id = top_info.get("panel_id", "default")
    shards = top_info.get("shards") or []
    if not shards:
        raise ValueError(f"No shards declared in {top_info_path}")

    rows: list[dict] = []
    label_mapping: dict[str, int] = {}
    seen_he: set[str] = set()

    for shard in shards:
        shard_dir = shard_root / shard
        parquet_path = shard_dir / "sample_meta.parquet"
        if not parquet_path.is_file():
            logger.warning("Skipping missing shard parquet: %s", parquet_path)
            continue
        sidecar_path = shard_dir / "dataset_info.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        target_domain = sidecar.get("target_domain") or shard
        label_mapping.update(sidecar.get("label_mapping") or {})
        root = raw_root / target_domain / split_root_name

        df = pd.read_parquet(parquet_path)
        n_he = n_ihc = 0
        for row in df.itertuples(index=False):
            split = getattr(row, "dataset_split", "train")
            # IHC target image of this pair (one per row, keeps its label).
            rows.append(
                {
                    "sample_uid": f"pool_{target_domain}_{Path(row.target_path).stem}",
                    "source_path": str(row.target_path),
                    "target_label": int(row.target_label),
                    "dataset_split": split,
                    "__root_dir": str(root),
                }
            )
            n_ihc += 1
            # H&E source image, deduplicated across shards by resolved path.
            he_key = str(root / row.source_path)
            if he_key in seen_he:
                continue
            seen_he.add(he_key)
            rows.append(
                {
                    "sample_uid": f"pool_{target_domain}_HE_{Path(row.source_path).stem}",
                    "source_path": str(row.source_path),
                    "target_label": -1,  # placeholder; assigned by caller
                    "dataset_split": split,
                    "__root_dir": str(root),
                }
            )
            n_he += 1
        logger.info(
            "Shard %s: %d IHC images, %d new HE images", shard, n_ihc, n_he
        )

    if not rows:
        raise ValueError(f"No pool rows collected from {shard_root}")
    return rows, label_mapping, panel_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a domain-pool shard from paired he_to_* shards"
    )
    parser.add_argument(
        "--shard-root",
        required=True,
        help="Directory with he_to_* shard subdirectories and dataset_info.json",
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Raw dataset root with one subdirectory per target stain",
    )
    parser.add_argument("--out", required=True, help="Output shard directory")
    parser.add_argument(
        "--split-root-name",
        default="TrainValAB",
        help="Intermediate folder between stain dir and trainA/trainB",
    )
    parser.add_argument(
        "--he-name", default="HE", help="Domain name for the H&E images"
    )
    parser.add_argument(
        "--he-label",
        type=int,
        default=None,
        help="Domain label for H&E images (default: number of IHC stains)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    shard_root = Path(args.shard_root)
    raw_root = Path(args.raw_root)
    rows, label_mapping, panel_id = _load_shard_rows(
        shard_root, raw_root, args.split_root_name
    )

    he_label = args.he_label if args.he_label is not None else len(label_mapping)
    if args.he_name in label_mapping:
        raise ValueError(f"HE domain name {args.he_name!r} clashes with an IHC stain")
    for row in rows:
        if row["target_label"] == -1:
            row["target_label"] = he_label

    # Panel members ordered by label index.
    members = [name for name, _ in sorted(label_mapping.items(), key=lambda kv: kv[1])]
    label_mapping[args.he_name] = he_label
    members.append(args.he_name)

    df = pd.DataFrame(rows)
    df["dataset_uid"] = f"{panel_id}_domain_pool"
    df["panel_id"] = panel_id
    if df["sample_uid"].duplicated().any():
        dupes = df.loc[df["sample_uid"].duplicated(), "sample_uid"].tolist()
        raise ValueError(f"Duplicate sample_uid values in pool: {dupes[:5]}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "sample_meta.parquet", index=False)
    info = {
        "dataset_uid": f"{panel_id}_domain_pool",
        "panel_id": panel_id,
        "panel_members": members,
        "panel_policy": "mixed",
        "directory_format": "domain_pool",
        "label_mapping": label_mapping,
        "provenance": {
            "shard_root": str(shard_root),
            "raw_root": str(raw_root),
            "split_root_name": args.split_root_name,
        },
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))

    counts = df.groupby("target_label").size().to_dict()
    logger.info(
        "Wrote %s: %d rows | label counts %s | members %s",
        out_dir,
        len(df),
        counts,
        members,
    )


if __name__ == "__main__":
    main()
