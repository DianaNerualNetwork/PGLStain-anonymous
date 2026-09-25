"""CLI entry point for evaluation metrics.

Everything is configured by the YAML files under
``puzzlestain/configs/evaluate/`` — pick a bundled preset (or pass your own
YAML file) and override values with Hydra-style dotlist arguments::

    puzzlestain-eval image-quality dataset=<name> stain=<stain> model=<model>
    puzzlestain-eval perception dataset=<name> stain=<stain> model=<model>
    puzzlestain-eval pathological-relevance dataset=<name> stain=<stain> model=<model>
    puzzlestain-eval pathfid dataset=<name> stain=<stain> model=<model>
    puzzlestain-eval all dataset=<name> stain=<stain> model=<model> exp_name=<name>
    puzzlestain-eval ./my_eval.yaml

PathFID's extractors are configured in YAML (``pathfid.foundation_models``,
``pathfid.checkpoint``, ``pathfid.hf_id``), e.g.::

    puzzlestain-eval pathfid dataset=<name> stain=<stain> model=<model> \
        pathfid.foundation_models=[conch,uni,univ2]
    puzzlestain-eval pathfid dataset=<name> stain=<stain> model=<model> \
        pathfid.foundation_models=[uni] pathfid.checkpoint=/ckpt/pytorch_model.bin

Results are written as CSV under ``{output_dir} / {dataset} / {stain} / <exp_name> /``.

Directory layout (under ``cache_root``)::

    {dataset}/
        {stain}/
            {model}/
                fake/     <- generated images
                gt/       <- ground-truth images
                source/   <- source images (for pathological metrics)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .config import EvalConfigError, available_configs, load_eval_config
from .metrics import (
    FAMILY_REGISTRY,
    FamilyResult,
    MetricComputationError,
    write_results,
)

logger = logging.getLogger(__name__)


def _resolve_model_dir(cache_root: str, dataset: str, stain: str, model: str) -> str:
    """Build the model result directory path and verify it exists."""
    path = os.path.join(cache_root, dataset, stain, model)
    if not os.path.isdir(path):
        raise ValueError(f"Model result directory not found: {path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puzzlestain-eval",
        description="Evaluate virtual staining results, configured by YAML.",
    )
    parser.add_argument(
        "config",
        help=(
            "Bundled config preset ("
            + ", ".join(available_configs())
            + ") or path to a custom YAML file."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help=(
            "Dotlist overrides, e.g. dataset=breast stain=ER model=pix2pix "
            "pathfid.foundation_models=[uni] pathfid.checkpoint=/ckpt/pytorch_model.bin"
        ),
    )
    return parser


def _run_family(
    family_name: str,
    fake_dir: str,
    gt_dir: str,
    device: str,
    metrics: list[str] | None,
    cfg: DictConfig,
) -> FamilyResult:
    cls = FAMILY_REGISTRY[family_name]
    if family_name == "pathfid":
        pf = OmegaConf.to_container(cfg.pathfid, resolve=True)
        family = cls(
            foundation_models=pf["foundation_models"],
            checkpoint=pf["checkpoint"],
            hf_id=pf["hf_id"],
        )
    else:
        family = cls(metrics)
    print(f"\nRunning {family_name} on {fake_dir} vs {gt_dir}")
    result = family.compute(fake_dir, gt_dir, device)
    return result


def _print_results(results: dict[str, FamilyResult]) -> None:
    """Print formatted summary results to stdout."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for family_name, fres in results.items():
        print(f"\n[{family_name}]")
        for k, v in fres.summary.items():
            print(f"  {k:30s}: {v:.6f}")
    print("=" * 60)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        cfg = load_eval_config(args.config, args.overrides)
    except EvalConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    logging.basicConfig(
        level=logging.INFO if cfg.verbose else logging.WARNING, format="%(message)s"
    )

    cache_root = cfg.cache_root
    if cache_root == ".":
        logger.warning("PATHOLOGY_CACHE_ROOT not set; using current directory")

    try:
        model_dir = _resolve_model_dir(cache_root, cfg.dataset, cfg.stain, cfg.model)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    fake_dir = os.path.join(model_dir, "fake")
    gt_dir = os.path.join(model_dir, "gt")

    if not os.path.isdir(fake_dir):
        print(f"Error: fake directory not found: {fake_dir}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(gt_dir):
        print(f"Error: gt directory not found: {gt_dir}", file=sys.stderr)
        sys.exit(2)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if device != cfg.device:
        logger.warning("CUDA not available, falling back to CPU")

    output_dir = os.path.join(cfg.output_dir, cfg.dataset, cfg.stain)
    exp_name = cfg.exp_name or cfg.model
    metrics = list(cfg.metrics) if cfg.metrics is not None else None

    results: dict[str, FamilyResult] = {}
    try:
        for family_name in list(cfg.families):
            result = _run_family(family_name, fake_dir, gt_dir, device, metrics, cfg)
            results[family_name] = result
    except MetricComputationError as exc:
        print(f"Metric error: {exc}", file=sys.stderr)
        sys.exit(1)

    write_results(results, output_dir, exp_name)
    _print_results(results)
    print(f"\nResults written to {Path(output_dir) / exp_name}")


if __name__ == "__main__":
    main()
