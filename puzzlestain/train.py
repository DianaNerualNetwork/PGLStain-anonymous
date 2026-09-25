"""Training entry point using Hydra for configuration management.

This script composes Hydra configurations, builds the dataset, model, losses,
optimizers, and trainer, and runs training for PGLStain or an included baseline.

Usage:
    python -m puzzlestain.train                    # default config
    python -m puzzlestain.train +experiment=example
    python -m puzzlestain.train data.batch_size=16 training.seed=123
    python -m puzzlestain.train --multirun data.batch_size=8,16
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from puzzlestain.configs.schema import PuzzleStainConfig
from puzzlestain.data.collate import stain_collate_fn
from puzzlestain.data.datasets import (
    AlignedVirtualStainDataset,
    MISTVirtualStainDataset,
    PairedVirtualStainDataset,
    StainPanelDataset,
    VirtualStainDataset,
)
from puzzlestain.data.samplers import SimpleSampler, StainBalancedSampler
from puzzlestain.data.transforms.registry import register_builtin_transforms
from puzzlestain.logging_utils import configure_logging
from puzzlestain.models.loss.base import LossRegistry
from puzzlestain.models import (  # noqa: F401
    cut_family,
    cyclegan_family,
    diffusion_family,
    pix2pix_family,
)
from puzzlestain.models.registry import ModelRegistry
from puzzlestain.training import Trainer

# Register built-in losses
from puzzlestain.models.loss import gan, gauss_pyramid_l1, mrsa, patchnce, pecc  # noqa: F401

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    """Seed all RNGs so data ordering and augmentation are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _plain_paths(value: Any) -> Any:
    """Coerce an OmegaConf container to a plain Python object.

    YAML lists for ``sample_meta`` / ``patch_root`` arrive as OmegaConf
    ``ListConfig``; the dataset expects plain ``list`` (or ``str``) so
    multi-shard configs work natively without a glue script.
    """
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _resolve_patch_root(value: Any) -> Optional[str | list[str]]:
    """Normalize an empty patch_root sentinel to ``None``.

    Hydra defaults use ``""`` to mean "not provided"; the dataset treats
    ``None`` as "use the parquet's ``__root_dir`` column, or the directory
    containing the parquet file when the column is absent". A
    non-empty string or list is passed through unchanged.
    """
    value = _plain_paths(value)
    if value == "":
        return None
    if isinstance(value, list):
        return [v for v in value if v != ""] or None
    return value


def _resolve_training_splits(value: Any) -> tuple[str, ...]:
    """Return the explicit dataset partitions consumed by training.

    Dataset scanners can discover every available split, so the training entry
    point must not inherit their broad discovery defaults. An omitted or empty
    ``data.splits`` is treated as the safe training-only default.
    """
    value = _plain_paths(value)
    if not value:
        return ("train",)
    if isinstance(value, str):
        value = [value]

    splits = tuple(str(split).strip() for split in value)
    if any(not split for split in splits):
        raise ValueError("data.splits must not contain empty split names")
    if len(set(splits)) != len(splits):
        raise ValueError(f"data.splits contains duplicates: {list(splits)}")
    return splits


def _build_transforms(config: PuzzleStainConfig) -> Any:
    """Build the training augmentation pipeline from config.

    Registers built-in transforms first, then composes the configured list.
    Returns ``None`` when no transforms are configured.
    """
    register_builtin_transforms()
    if not config.transforms.train_transforms:
        return None
    from puzzlestain.data.transforms.registry import TransformRegistry

    return TransformRegistry.build_compose(config.transforms.train_transforms)


def _build_dataset(config: PuzzleStainConfig, transforms: Any) -> torch.utils.data.Dataset:
    """Build the stain-level virtual-stain dataset from config.

    The data source can be specified either as a parquet ``sample_meta`` path
    (or list of paths) or as a raw ``dataroot`` directory scanned according to
    ``data.dataset_layout``.
    """
    sample_meta = _plain_paths(config.data.sample_meta)
    patch_root = _resolve_patch_root(config.data.patch_root)
    dataroot = config.data.dataroot
    dataset_layout = config.data.dataset_layout
    training_splits = _resolve_training_splits(config.data.splits)
    aux_columns = list(config.data.aux_columns or [])
    input_keys = list(config.task.input_keys or ["source_image"])
    target_keys = list(config.task.target_keys or ["target_image"])

    has_sample_meta = sample_meta is not None and sample_meta != []
    has_dataroot = dataroot is not None and dataroot != ""

    if not has_sample_meta and not has_dataroot:
        raise ValueError(
            "Either data.sample_meta or data.dataroot must be provided."
        )

    if has_sample_meta:
        # Panel special case: one image per panel class per sample, built
        # from the parquet shards plus the raw image root (for methods
        # whose training step consumes every class at once, e.g. StainExpert).
        if dataset_layout == "panel":
            if not has_dataroot:
                raise ValueError(
                    "dataset_layout='panel' requires both data.sample_meta "
                    "(parquet root) and data.dataroot (raw image root)."
                )
            return StainPanelDataset.from_parquet(
                parquet_root=sample_meta,
                raw_root=dataroot,
                selected_stains=config.data.selected_stains,
                splits=training_splits,
                resize_size=config.data.resize_size,
                crop_size=config.data.crop_size,
                max_dataset_size=config.data.max_dataset_size,
            )
        # MIST special case: pre-built parquet shards plus the raw MIST root.
        # When no patch_root is given, derive per-shard roots from dataroot so
        # users only need to supply the two directory paths.
        if (
            dataset_layout == "mist"
            and patch_root is None
            and has_dataroot
            and isinstance(sample_meta, (str, Path))
            and Path(sample_meta).is_dir()
        ):
            return MISTVirtualStainDataset.from_parquet(
                parquet_root=sample_meta,
                raw_root=dataroot,
                panel_id=config.data.panel_id,
                selected_stains=config.data.selected_stains,
                panel_policy=config.data.panel_policy,
                splits=training_splits,
                return_onehot=config.data.return_onehot,
                transforms=transforms,
                backend=config.data.backend,
                max_dataset_size=config.data.max_dataset_size,
                aux_columns=aux_columns,
                input_keys=input_keys,
                target_keys=target_keys,
            )
        return VirtualStainDataset(
            sample_meta=sample_meta,
            patch_root=patch_root,
            panel_id=config.data.panel_id,
            selected_stains=config.data.selected_stains,
            panel_policy=config.data.panel_policy,
            return_onehot=config.data.return_onehot,
            transforms=transforms,
            splits=list(training_splits),
            row_filter=dict(config.data.row_filter) if config.data.row_filter else None,
            backend=config.data.backend,
            max_dataset_size=config.data.max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )

    if dataset_layout == "mist":
        return MISTVirtualStainDataset(
            dataroot=dataroot,
            panel_id=config.data.panel_id,
            selected_stains=config.data.selected_stains,
            splits=training_splits,
            return_onehot=config.data.return_onehot,
            transforms=transforms,
            backend=config.data.backend,
            max_dataset_size=config.data.max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )
    if dataset_layout == "paired_subdir":
        return PairedVirtualStainDataset(
            dataroot=dataroot,
            panel_id=config.data.panel_id,
            selected_stains=config.data.selected_stains,
            splits=training_splits,
            stain_from_subdir=True,
            return_onehot=config.data.return_onehot,
            transforms=transforms,
            backend=config.data.backend,
            max_dataset_size=config.data.max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )
    if dataset_layout == "paired_flat":
        return PairedVirtualStainDataset(
            dataroot=dataroot,
            panel_id=config.data.panel_id,
            panel_members=config.data.selected_stains,
            selected_stains=config.data.selected_stains,
            splits=training_splits,
            stain_from_subdir=False,
            return_onehot=config.data.return_onehot,
            transforms=transforms,
            backend=config.data.backend,
            max_dataset_size=config.data.max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )
    if dataset_layout == "aligned":
        selected = config.data.selected_stains
        if not selected:
            raise ValueError("data.selected_stains must be set for dataset_layout='aligned'")
        non_train = [s for s in training_splits if s != "train"]
        if non_train:
            raise ValueError(
                f"dataset_layout='aligned' only supports data.splits=['train'] for "
                f"training, got {list(training_splits)}. Run inference/evaluation "
                "entry points for val/test data instead; otherwise those samples "
                "would leak into training."
            )
        return AlignedVirtualStainDataset(
            dataroot=dataroot,
            panel_id=config.data.panel_id,
            target_stain=selected[0],
            splits=training_splits,
            return_onehot=config.data.return_onehot,
            transforms=transforms,
            backend=config.data.backend,
            max_dataset_size=config.data.max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )
    if dataset_layout == "labeled":
        from puzzlestain.data.datasets import LabeledImageFolderVirtualStainDataset

        if len(training_splits) != 1:
            raise ValueError(
                "dataset_layout='labeled' reads one flat directory, so exactly one "
                f"data split must be declared; got {list(training_splits)}. Point "
                "data.dataroot at the directory for the intended split."
            )
        return LabeledImageFolderVirtualStainDataset(
            root=dataroot,
            panel_id=config.data.panel_id,
            panel_members=config.data.selected_stains,
            split=training_splits[0],
            return_onehot=config.data.return_onehot,
            transforms=transforms,
            backend=config.data.backend,
            max_dataset_size=config.data.max_dataset_size,
            aux_columns=aux_columns,
            input_keys=input_keys,
            target_keys=target_keys,
        )

    raise ValueError(
        f"Unknown data.dataset_layout: {dataset_layout}. "
        f"Choose one of {{mist, panel, paired_subdir, paired_flat, aligned, labeled}}."
    )


def _build_sampler(dataset: VirtualStainDataset, config: PuzzleStainConfig):
    """Build a batch sampler appropriate for the configured data."""
    sampler_name = config.data.sampler
    batch_size = config.data.batch_size
    shuffle = config.data.shuffle
    drop_last = config.data.drop_last

    if sampler_name == "stain_balanced":
        target_stains = [entry.get("target_stain", "") for entry in dataset.entries]
        return StainBalancedSampler(
            target_stains=target_stains,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=config.training.seed,
            drop_last=drop_last,
        )
    if sampler_name == "simple":
        return SimpleSampler(
            num_samples=len(dataset),
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=config.training.seed,
        )
    raise ValueError(
        f"Unknown sampler: {sampler_name}. "
        f"Choose one of {{stain_balanced, simple}}."
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: build the data-loading pipeline and verify a batch.

    Hydra composes the YAML config groups (plus any CLI overrides) into ``cfg``
    before this function is called. The body wires up a model-agnostic data
    pipeline: config -> transforms -> dataset -> sampler -> dataloader.

    Args:
        cfg: Hydra-composed config; converted to the typed PuzzleStainConfig.
    """
    # 1. Materialize the typed config. Merging the composed tree onto the
    #    structured PuzzleStainConfig schema validates types and yields a real
    #    dataclass (attribute access), rather than the plain dict that
    #    to_object(cfg) returns for an unregistered schema.
    merged = OmegaConf.merge(OmegaConf.structured(PuzzleStainConfig), cfg)
    config: PuzzleStainConfig = OmegaConf.to_object(merged)

    # 1a. Select the target CUDA device before any CUDA context is created.
    #     ``training.gpu_ids`` mirrors the original pix2pix ``--gpu_ids`` flag:
    #     e.g. "0" or "0,1,2".  Empty means use the default CUDA device.
    gpu_ids = config.training.gpu_ids
    if gpu_ids and gpu_ids.strip():
        device_id = int(gpu_ids.strip().split(",")[0])
        torch.cuda.set_device(device_id)
        # logger is not configured yet; rely on the later config dump.

    # Apply the configured log level to the puzzlestain package logger up front.
    configure_logging(config)

    if gpu_ids and gpu_ids.strip():
        logger.info("CUDA device set to %s", gpu_ids.strip())

    logger.info("Config:\n%s", OmegaConf.to_yaml(merged))

    # 2. Seed all RNGs up front so dataset shuffling and augmentation are
    #    reproducible for this run.
    _set_seed(config.training.seed)

    # 3. Build the augmentation pipeline. register_builtin_transforms() must run
    #    first to populate the registry that build_compose() looks names up in.
    train_transforms = _build_transforms(config)

    # 4. Build training data.
    dataset = _build_dataset(config, train_transforms)
    logger.info(
        "Built dataset | entries=%d | panel_members=%s | selected_stains=%s",
        len(dataset),
        dataset.panel_members,
        dataset.selected_stains,
    )

    sampler = _build_sampler(dataset, config)
    logger.info(
        "Built sampler | type=%s | batches=%d | batch_size=%d",
        config.data.sampler,
        len(sampler),
        config.data.batch_size,
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=stain_collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.data.num_workers > 0,
    )

    # 5. Build the training pipeline.  The trainer itself will load and log the
    #    first batch, and run data-dependent initialization for strategies that
    #    need it (e.g. CUT), so the first batch is transformed exactly once.
    trainer = _build_training_pipeline(config, train_loader)

    if getattr(config.training, "dry_run", False):
        trainer.verify_first_batch()
        logger.info("dry_run=True; stopping after data verification.")
        return

    # 6. Run training.
    trainer.train()


def _build_loss_fn(config: PuzzleStainConfig):
    """Build the composite loss from config.loss.components."""
    loss_cfg = getattr(config, "loss", None)
    if loss_cfg is None or not getattr(loss_cfg, "components", None):
        # Default vanilla GAN.
        return LossRegistry.build_composite(
            {
                "gan": {"gan_mode": "vanilla", "weight": 1.0},
            }
        )
    return LossRegistry.build_composite(dict(loss_cfg.components))


def _build_training_pipeline(config: PuzzleStainConfig, train_loader: DataLoader) -> Trainer:
    """Assemble model, processor, strategy, loss, and trainer."""
    model_name = getattr(config.model, "name", "pix2pix_pyramid")
    # Fields that belong to processor/strategy, not the raw model constructor.
    non_model_fields = {"name", "params", "use_gan", "normalize_to_minus_one_one"}
    model_kwargs = {
        k: v for k, v in vars(config.model).items() if k not in non_model_fields
    }
    model_kwargs.update(dict(getattr(config.model, "params", {}) or {}))

    model = ModelRegistry.build_model(model_name, **model_kwargs)
    processor = ModelRegistry.build_processor(
        model_name,
        normalize_to_minus_one_one=getattr(config.model, "normalize_to_minus_one_one", True),
        # Superset forwarding (dropped per processor signature): lets configs
        # route processor-level options (e.g. uni_checkpoint) via model.params.
        **model_kwargs,
    )
    loss_fn = _build_loss_fn(config)

    strategy = ModelRegistry.build_strategy(
        model_name,
        model=model,
        processor=processor,
        loss_fn=loss_fn,
        config=config,
        use_gan=getattr(config.model, "use_gan", True),
    )

    trainer = Trainer(
        config=config,
        strategy=strategy,
        train_loader=train_loader,
    )
    return trainer


if __name__ == "__main__":
    main()
