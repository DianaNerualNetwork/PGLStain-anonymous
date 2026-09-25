"""Hydra structured-config dataclasses for PuzzleStain.

Each dataclass below mirrors one YAML group under ``configs/`` and is what
gives Hydra/OmegaConf a typed schema: defaults live here, YAML files and CLI
overrides are validated against these types, and ``OmegaConf.to_object`` turns
the merged config back into these typed objects for the rest of the framework.

``MISSING`` marks a field that has no sensible default and *must* be supplied by
a YAML file or the command line (Hydra errors out if it is left unset).
Interpolations such as ``${now:...}`` are resolved by OmegaConf at access time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Data config ──


@dataclass
class DataConfig:
    """Dataset, sampling, and patch-loading parameters.

    Attributes:
        sample_meta: Path(s) to ``sample_meta.parquet`` or a list of paths for
            multi-shard training. Optional when ``dataroot`` is provided.
        patch_root: Root directory containing the patch images. Can be a single
            path (broadcast to all shards) or a list aligned with ``sample_meta``.
            Not needed when the parquet already contains a ``__root_dir`` column.
        dataroot: Root directory for raw dataset layouts such as MIST/IHC4BC.
            Used when ``sample_meta`` is not provided.
        dataset_layout: Layout identifier for ``dataroot`` scanning. One of
            ``"mist"`` (MIST/IHC4BC), ``"paired_subdir"``, ``"paired_flat"``,
            ``"aligned"`` (Pix2PixHD ``{phase}A``/``{phase}B``), ``"labeled"``,
            or ``"panel"`` (per-class panel batches from parquet shards,
            for one-image-per-class methods such as StainExpert).
        panel_id: Panel identifier used to enforce consistency across shards.
        selected_stains: Stains to include; ``None`` means all panel members.
        panel_policy: Optional filter; ``"complete"``/``"partial"``/``"mixed"``.
        sampler: Batch sampler type. ``"stain_balanced"`` round-robins across
            stains; ``"simple"`` yields contiguous batches.
        batch_size: Batch size used by the sampler.
        drop_last: Drop the final incomplete batch.
        shuffle: Shuffle within each stain queue (stain_balanced) or globally
            (simple). Validation usually sets this to False.
        num_workers: DataLoader worker processes.
        backend: Patch image backend name (``"pil"``/``"pillow"``/``"default"``)
            or any name registered with ``BackendFactory``.
        splits: Training ``dataset_split`` values to keep (e.g. ``["train"]``).
            An empty value safely falls back to ``["train"]`` at the training
            entry point; discovery/conversion APIs may still scan all splits.
        row_filter: Column -> value/list equality filter on the rows used for
            training, e.g. ``{"target_stain": ["ER", "PR"]}``.
        return_onehot: Whether to populate ``target_onehot`` on each sample.
        max_dataset_size: Cap dataset entries at this value when positive.
            ``0`` means use all available samples.
        aux_columns: Auxiliary image path columns present in the parquet files.
            A column ``foo_path`` is exposed as sample field ``foo`` and stacked
            by the collator as an image tensor.
        scalar_columns: Numeric parquet columns exposed as scalar supervision
            fields and stacked by the collator.
        resize_size: Optional square whole-image resize consumed by the
            ``panel`` layout before its crop. ``None`` preserves the original
            panel behavior. Stain-level layouts continue to size images through
            their transform specs.
        crop_size: Square random-crop size consumed by the ``panel``
            layout (:class:`StainPanelDataset`); ignored by the stain-level
            layouts, which size images through their transform specs.
        target_mpp: Physical resolution in microns per pixel used to prepare
            training patches. Record this for strict scale-aware inference;
            ``None`` leaves legacy checkpoints and workflows unchanged.
    """

    sample_meta: Any = None
    patch_root: Any = ""
    dataroot: str | None = None
    dataset_layout: str = "mist"
    panel_id: str = "default"
    selected_stains: list[str] | None = None
    panel_policy: str | None = None
    sampler: str = "stain_balanced"
    batch_size: int = 8
    drop_last: bool = False
    shuffle: bool = True
    num_workers: int = 4
    backend: str = "default"
    splits: list[str] = field(default_factory=lambda: ["train"])
    row_filter: dict[str, Any] = field(default_factory=dict)
    return_onehot: bool = False
    max_dataset_size: int = 0
    aux_columns: list[str] = field(default_factory=list)
    scalar_columns: list[str] = field(default_factory=list)
    crop_size: int = 512
    target_mpp: float | None = None
    resize_size: int | None = None


# ── Transform config ──


@dataclass
class TransformConfig:
    """Augmentation pipeline specification.

    Attributes:
        train_transforms: Ordered list of transform specs resolved by
            ``TransformRegistry.build_compose`` into a ``Compose`` pipeline;
            empty means no augmentation. Each spec is a dict with a ``name``
            key (short name registered in ``TransformRegistry``) and any
            constructor kwargs for that transform.
    """

    train_transforms: list[Any] = field(default_factory=list)


# ── Task config ──


@dataclass
class TaskConfig:
    """Per-task field-role contract.

    Attributes:
        input_keys: Field names treated as model inputs.
        target_keys: Field names treated as supervision targets.
    """

    input_keys: list[str] = field(default_factory=lambda: ["source_image"])
    target_keys: list[str] = field(default_factory=lambda: ["target_image"])


# ── Model config ──


@dataclass
class ModelConfig:
    """Model, processor, and strategy selection.

    Model-specific contracts remain in ``params`` and the model constructor.
    For PD-UniST, ``text_conditioning`` defaults to ``legacy_cached`` when
    absent from old checkpoint configs; new presets explicitly select
    ``trainable_projection``. ``text_encoder_path=None`` enables its local
    cache/official-download resolver. These are not shared model fields.
    """

    name: str = "pix2pix_pyramid"
    params: dict[str, Any] = field(default_factory=dict)
    use_gan: bool = True
    normalize_to_minus_one_one: bool = True


# ── Loss config ──


@dataclass
class LossConfig:
    """Composite loss configuration."""

    components: dict[str, Any] = field(default_factory=dict)


# ── Training config ──


@dataclass
class TrainingConfig:
    """Optimization, checkpointing, and reproducibility settings.

    Attributes:
        seed: RNG seed for reproducible dataset ordering and augmentation.
        max_epochs: Epoch cap.
        max_steps: Hard step cap; ``-1`` means unbounded.
        lr: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        gradient_accumulation_steps: Micro-batches per optimizer step.
        beta1: Adam beta1 (MIST/ASP optimizer).
        beta2: Adam beta2 (MIST/ASP optimizer).
        lr_policy: Learning-rate policy, e.g. ``"linear"``.
        lr_decay_iters: Iterations between LR decay drops.
        n_epochs: Number of training epochs at the initial LR.
        n_epochs_decay: Number of epochs for linear decay to zero.
        save_epoch_freq: Checkpoint every N epochs.
        save_latest_freq: Save latest checkpoint every N iterations.
        display_freq: Iterations between display logging.
        print_freq: Iterations between console logging.
        evaluation_freq: Iterations between evaluation runs.
        gpu_ids: Comma-separated CUDA device IDs, e.g. ``"0"`` or ``"0,1,2"``.
            When set, the first ID is selected as the active CUDA device before
            the training pipeline is built. An empty value leaves device
            selection to the runtime.
    """

    seed: int = 42
    max_epochs: int = 50
    max_steps: int = -1
    lr: float = 1e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    beta1: float = 0.5
    beta2: float = 0.999
    lr_policy: str = "linear"
    lr_decay_iters: int = 50
    n_epochs: int = 100
    n_epochs_decay: int = 100
    save_epoch_freq: int = 5
    save_latest_freq: int = 5000
    display_freq: int = 100
    print_freq: int = 100
    evaluation_freq: int = 1000
    mixed_precision: str = "no"
    use_wandb: bool = False
    wandb_project: str = "PuzzleStain"
    log_steps: int = 100
    dry_run: bool = False
    resume_from_checkpoint: str = ""
    gpu_ids: str = ""


# ── Logging config ──


@dataclass
class LoggingConfig:
    """Console logging verbosity.

    Attributes:
        level: Level applied to the ``puzzlestain`` package logger, e.g.
            ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    """

    level: str = "INFO"


# ── Top-level config ──


@dataclass
class PuzzleStainConfig:
    """Root config aggregating every sub-config group.

    This is the object Hydra constructs for ``train.py``. Each field is a
    Hydra config group selectable/overridable from YAML or the command line.

    Attributes:
        data: Dataset, sampling, and patching options.
        transforms: Training augmentation pipeline.
        task: Model input/target field contract.
        training: Optimization and reproducibility options.
        logging: Console log level.
        output_dir: Run output directory; the ``${now:...}`` interpolations
            stamp it with the launch date/time so runs don't collide.
    """

    data: DataConfig = field(default_factory=DataConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output_dir: str = "./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"
