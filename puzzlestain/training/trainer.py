"""Minimal Trainer built on accelerate.

This is the first working trainer for PuzzleStain. It intentionally keeps the
scope small: training loop, checkpointing, and logging. Callbacks, evaluation,
and EMA will be added in follow-up tasks.
"""

from __future__ import annotations

import itertools
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from dominate import document
from dominate.tags import a, h1, h2, img, table, td, tr
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from ..configs.schema import PuzzleStainConfig
from ..models.loss.base import CompositeLoss
from ..models.strategy.base import TrainingStrategy

logger = logging.getLogger(__name__)
_CHECKPOINT_VERSION = 2


def _capture_rng_state() -> dict[str, Any]:
    """Capture process RNG state in a weights-only-loadable representation."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        ),
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore a state produced by _capture_rng_state."""
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(state["cuda"])


def _capture_sampler_state(sampler: Any) -> dict[str, Any]:
    """Capture mutable ordering state from PuzzleStain batch samplers."""
    state: dict[str, Any] = {}
    if hasattr(sampler, "seed"):
        state["seed"] = int(sampler.seed)
    generator = getattr(sampler, "generator", None)
    if isinstance(generator, torch.Generator):
        state["generator"] = generator.get_state()
    return state


def _restore_sampler_state(sampler: Any, state: dict[str, Any] | None) -> None:
    """Restore mutable ordering state captured by _capture_sampler_state."""
    if not state:
        return
    if "seed" in state:
        sampler.seed = int(state["seed"])
    generator = getattr(sampler, "generator", None)
    if "generator" in state and isinstance(generator, torch.Generator):
        generator.set_state(state["generator"])


def _tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``(C, H, W)`` float tensor in ``[-1, 1]`` to ``(H, W, C) uint8``."""
    arr = tensor.detach().cpu().numpy()
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return arr


class Trainer:
    """In-house training loop for virtual-stain models.

    Args:
        config: Resolved experiment config.
        strategy: Training strategy that owns networks/optimizers/step logic.
        train_loader: Training dataloader.
        scheduler: Optional scheduler stepped per optimizer step.
    """

    def __init__(
        self,
        config: PuzzleStainConfig,
        strategy: TrainingStrategy,
        train_loader: DataLoader,
        scheduler: dict[str, Any] | None = None,
    ) -> None:
        self._batch_sampler = train_loader.batch_sampler

        self.config = config
        self.strategy = strategy
        self.train_loader = train_loader
        self.scheduler = scheduler or {}

        self.accelerator = Accelerator(
            mixed_precision=config.training.mixed_precision
            if hasattr(config.training, "mixed_precision")
            else "no",
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            log_with="wandb" if getattr(config.training, "use_wandb", False) else None,
            project_dir=config.output_dir,
        )

        # Build optimizers from the *unprepared* model. The optimizer parameter
        # references are preserved when accelerate prepares the model, so the
        # prepared optimizers still point to the prepared model's parameters.
        optimizers = strategy.get_optimizers(config)

        # Prepare model, optimizers, and dataloader together. Preparing the whole
        # model wrapper (rather than individual netG/netD modules) guarantees that
        # accelerate moves everything to the target device.
        prepared_model, prepared_optimizers, self.train_loader = (
            self.accelerator.prepare(
                strategy.model,
                list(optimizers.values()),
                train_loader,
            )
        )

        # Swap the prepared model back into the strategy so training_step sees
        # the accelerated version on the correct device.
        self.strategy.replace_networks({"model": prepared_model})

        # Loss modules are built on CPU and are not part of the prepared
        # model; move them to the model device so their buffers match the
        # training tensors. No-op on CPU.
        device = next(prepared_model.parameters()).device
        loss_fn = strategy.loss_fn
        if isinstance(loss_fn, CompositeLoss):
            for component, _ in loss_fn.losses:
                if isinstance(component, torch.nn.Module):
                    component.to(device)
        elif isinstance(loss_fn, torch.nn.Module):
            loss_fn.to(device)

        self.networks = strategy.get_networks()
        self.optimizers = dict(zip(optimizers.keys(), prepared_optimizers))
        self._prepared_optimizer_keys = set(self.optimizers.keys())

        # Build schedulers from the *prepared* optimizers so they track the same
        # param_group objects that the training loop steps.
        self.scheduler = scheduler or strategy.get_schedulers(self.optimizers, config)

        self.global_step = 0
        self.current_epoch = 0
        self._batches_completed_in_epoch = 0
        self._epoch_prefix_batches = 0
        self._epoch_start_rng_state: dict[str, Any] | None = None
        self._epoch_start_sampler_state: dict[str, Any] = {}
        self._resume_rng_state: dict[str, Any] | None = None
        self._resume_sampler_state: dict[str, Any] = {}
        self._resume_epoch_start_rng_state: dict[str, Any] | None = None
        self._resume_epoch_start_sampler_state: dict[str, Any] = {}
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Visualization directories (mirrors original PyramidP2P web/ layout).
        self.web_dir = self.output_dir / "web"
        self.img_dir = self.web_dir / "images"
        if self.accelerator.is_main_process:
            self.img_dir.mkdir(parents=True, exist_ok=True)
        self._visuals_saved = False

        # Resume from checkpoint if requested.
        self.resume_checkpoint_dir: Path | None = None
        # Checkpoint state that cannot be loaded yet (e.g. the CUT family's
        # lazily-created netF MLPs and their optimizers) is stashed here and
        # applied by _load_deferred_checkpoint_state once the first-batch
        # warmup in train() has materialized the corresponding modules.
        self._pending_network_state: dict[str, Any] = {}
        self._pending_optimizer_state: dict[str, Any] = {}
        self._pending_scheduler_state: dict[str, Any] = {}
        self._pending_strategy_state: dict[str, Any] | None = None
        self._pending_scaler_state: dict[str, Any] | None = None
        resume_path = getattr(config.training, "resume_from_checkpoint", "")
        if resume_path:
            self.resume_checkpoint_dir = Path(resume_path)
            if not self.resume_checkpoint_dir.is_dir():
                raise FileNotFoundError(
                    f"Resume checkpoint directory not found: {self.resume_checkpoint_dir}"
                )
            self._load_checkpoint(self.resume_checkpoint_dir)

    def verify_first_batch(self) -> None:
        """Load one batch and run a forward pass without training.

        Used by ``training.dry_run=true`` to verify that the data pipeline,
        model, and strategy can produce a batch and pass it through the model.
        """
        batch = next(iter(self.train_loader))
        logger.info(
            "Dry-run: first batch loaded with keys=%s shapes={%s}",
            list(batch.keys()),
            ", ".join(
                f"{k}: {tuple(v.shape)}"
                for k, v in batch.items()
                if hasattr(v, "shape")
            ),
        )
        if hasattr(self.strategy, "sample_step"):
            with torch.no_grad():
                _ = self.strategy.sample_step(batch, self.global_step)
            logger.info("Dry-run: sample_step forward pass succeeded.")
        logger.info("Dry-run complete; data and model pipeline look healthy.")

    def _prepare_new_optimizers(self) -> None:
        """Prepare any optimizers added after ``__init__`` (e.g. lazy netF)."""
        new_keys = [
            k for k in self.optimizers if k not in self._prepared_optimizer_keys
        ]
        if not new_keys:
            return
        new_opts = [self.optimizers[k] for k in new_keys]
        prepared = self.accelerator.prepare(new_opts)
        if not isinstance(prepared, list):
            prepared = [prepared]
        for key, opt in zip(new_keys, prepared):
            self.optimizers[key] = opt
            self._prepared_optimizer_keys.add(key)

    def _load_checkpoint(self, ckpt_dir: Path) -> None:
        """Restore trainer state from a checkpoint directory.

        State for lazily-created modules (e.g. the CUT family's ``netF``
        MLPs, built during the first forward pass in
        ``data_dependent_initialize``) cannot be loaded at this point and is
        stashed for :meth:`_load_deferred_checkpoint_state` instead. This
        mirrors the original CUT/ASP repos, where ``setup()`` (and thus
        ``load_networks``) runs only after ``data_dependent_initialize``.
        """
        state_path = ckpt_dir / "trainer_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing trainer_state.pt in {ckpt_dir}")

        # Load on CPU first; accelerate will move tensors to the right device.
        state = torch.load(state_path, map_location="cpu", weights_only=True)

        self.global_step = int(state.get("global_step", 0))
        resume_state = state.get("resume_state", {})
        self.current_epoch = int(
            resume_state.get("epoch", state.get("current_epoch", 0))
        )
        self._batches_completed_in_epoch = int(
            resume_state.get("batches_completed_in_epoch", 0)
        )
        self._epoch_prefix_batches = int(resume_state.get("epoch_prefix_batches", 0))
        self._resume_rng_state = resume_state.get("rng_state")
        self._resume_sampler_state = resume_state.get("sampler_state", {})
        self._resume_epoch_start_rng_state = resume_state.get("epoch_start_rng_state")
        self._resume_epoch_start_sampler_state = resume_state.get(
            "epoch_start_sampler_state", {}
        )

        defer_all = hasattr(self.strategy, "data_dependent_initialize")
        unwrapped = self.accelerator.unwrap_model
        for name, net_state in state.get("networks", {}).items():
            if defer_all:
                self._pending_network_state[name] = net_state
                continue
            if name not in self.networks:
                self._pending_network_state[name] = net_state
                continue
            net = unwrapped(self.networks[name])
            if set(net.state_dict().keys()) != set(net_state.keys()):
                # The module exists but its parameters are not materialized
                # yet (lazy init); defer the load until after the warmup
                # forward has created them.
                self._pending_network_state[name] = net_state
                continue
            net.load_state_dict(net_state)

        for name, opt_state in state.get("optimizers", {}).items():
            if defer_all:
                self._pending_optimizer_state[name] = opt_state
                continue
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(opt_state)
            else:
                self._pending_optimizer_state[name] = opt_state

        for name, sched_state in state.get("scheduler_state", {}).items():
            if not sched_state:
                continue
            if defer_all:
                self._pending_scheduler_state[name] = sched_state
                continue
            if name in self.scheduler and self.scheduler[name] is not None:
                self.scheduler[name].load_state_dict(sched_state)
            else:
                self._pending_scheduler_state[name] = sched_state

        strategy_state = state.get("strategy_state", {})
        if defer_all:
            self._pending_strategy_state = strategy_state
        else:
            self.strategy.load_state_dict(strategy_state)
        scaler_state = state.get("scaler_state")
        if scaler_state:
            self._pending_scaler_state = scaler_state

        if state.get("checkpoint_version", 1) < _CHECKPOINT_VERSION:
            logger.warning(
                "Checkpoint predates exact-resume state; resuming at the start "
                "of epoch %d without RNG or batch-position restoration",
                self.current_epoch,
            )

        logger.info(
            "Resumed from checkpoint %s at step %d, epoch %d",
            ckpt_dir,
            self.global_step,
            self.current_epoch,
        )
        if self._pending_network_state or self._pending_optimizer_state:
            logger.info(
                "Deferred checkpoint state for networks=%s optimizers=%s "
                "until lazy modules are materialized",
                sorted(self._pending_network_state),
                sorted(self._pending_optimizer_state),
            )

    def _load_deferred_checkpoint_state(self) -> None:
        """Load checkpoint state deferred by :meth:`_load_checkpoint`.

        Must be called after ``data_dependent_initialize`` has materialized
        any lazily-created modules and after their optimizers and schedulers
        have been prepared. A no-op when nothing was deferred (e.g. models
        whose networks and optimizers all exist at construction time).
        """
        if not (
            self._pending_network_state
            or self._pending_optimizer_state
            or self._pending_scheduler_state
            or self._pending_strategy_state is not None
            or self._pending_scaler_state is not None
        ):
            return

        unwrapped = self.accelerator.unwrap_model
        for name, net_state in self._pending_network_state.items():
            if name not in self.networks:
                raise RuntimeError(
                    f"Checkpoint network {name!r} has no counterpart in the "
                    "current model"
                )
            unwrapped(self.networks[name]).load_state_dict(net_state)

        for name, opt_state in self._pending_optimizer_state.items():
            if name not in self.optimizers:
                raise RuntimeError(
                    f"Checkpoint optimizer {name!r} has no counterpart in the "
                    "current model"
                )
            self.optimizers[name].load_state_dict(opt_state)

        for name, sched_state in self._pending_scheduler_state.items():
            if name not in self.scheduler or self.scheduler[name] is None:
                raise RuntimeError(
                    f"Checkpoint scheduler {name!r} has no counterpart in the "
                    "current model"
                )
            self.scheduler[name].load_state_dict(sched_state)

        if self._pending_strategy_state is not None:
            self.strategy.load_state_dict(self._pending_strategy_state)
        if self._pending_scaler_state is not None:
            scaler = getattr(self.accelerator, "scaler", None)
            if scaler is None:
                raise RuntimeError(
                    "Checkpoint contains AMP scaler state but the resumed "
                    "training configuration has no scaler"
                )
            scaler.load_state_dict(self._pending_scaler_state)

        logger.info(
            "Loaded deferred checkpoint state for networks=%s optimizers=%s "
            "schedulers=%s",
            sorted(self._pending_network_state),
            sorted(self._pending_optimizer_state),
            sorted(self._pending_scheduler_state),
        )
        self._pending_network_state = {}
        self._pending_optimizer_state = {}
        self._pending_scheduler_state = {}
        self._pending_strategy_state = None
        self._pending_scaler_state = None

    def train(self) -> None:
        """Run the training loop."""
        cfg = self.config.training
        self.strategy.on_train_begin()

        # Register experiment trackers (main process only, via accelerate's
        # on_main_process decorator). Without this call, accelerator.log()
        # below silently no-ops because no tracker is ever registered.
        if cfg.use_wandb:
            # Generate a descriptive run name: model_name + loss lambdas + timestamp
            model_name = getattr(self.config.model, "name", "unknown")
            model_params = getattr(self.config.model, "params", {}) or {}

            # Extract lambda values from model params
            lambda_parts = []
            for key, value in sorted(model_params.items()):
                if key.startswith("lambda_") and isinstance(value, (int, float)):
                    # Shorten lambda names: lambda_od_graph -> od, lambda_NCE -> nce, lambda_gp -> gp
                    short_name = (
                        key.replace("lambda_", "")
                        .replace("_graph", "")
                        .replace("_", "")
                    )
                    lambda_parts.append(f"{short_name}{value}")

            # Build run name: model_name + lambdas + timestamp
            if lambda_parts:
                lambda_str = "_".join(lambda_parts)
                run_name = f"{model_name}_{lambda_str}_{self.output_dir.name}"
            else:
                run_name = f"{model_name}_{self.output_dir.name}"

            self.accelerator.init_trackers(
                project_name=cfg.wandb_project,
                config=OmegaConf.to_container(
                    OmegaConf.structured(self.config), resolve=True
                ),
                # Keep the wandb local cache inside the run's output dir;
                # otherwise wandb writes ./wandb relative to the cwd.
                init_kwargs={"wandb": {"dir": str(self.output_dir), "name": run_name}},
            )

        # Strategies with data-dependent warmup (e.g. the CUT family's netF
        # materialization / reference forward) grab the first batch once and
        # reuse it for both ``data_dependent_initialize`` and step 1, mirroring
        # the original CUT/ASP repos. Other strategies leave the loader
        # untouched, so every epoch runs exactly ``len(train_loader)`` steps,
        # as in the original CycleGAN/pix2pix repos.
        first_batch = None
        if hasattr(self.strategy, "data_dependent_initialize"):
            first_batch = next(iter(self.train_loader))
            self.strategy.data_dependent_initialize(first_batch, self.optimizers)
            self._prepare_new_optimizers()
            # Attach LR schedulers to optimizers added after __init__ (e.g.
            # CUT's optimizer_F), mirroring the original repos where setup()
            # runs only after data_dependent_initialize creates optimizer_F.
            late_optimizers = {
                k: v for k, v in self.optimizers.items() if k not in self.scheduler
            }
            if late_optimizers:
                self.scheduler.update(
                    self.strategy.get_schedulers(late_optimizers, self.config)
                )

        # Lazily-created modules and their optimizers/schedulers exist now,
        # so any checkpoint state deferred in __init__ can be loaded.
        self._load_deferred_checkpoint_state()

        if 0 < cfg.max_steps <= self.global_step:
            logger.info(
                "Checkpoint is already at global_step=%d (max_steps=%d); "
                "no optimizer step will be repeated",
                self.global_step,
                cfg.max_steps,
            )
            self.strategy.on_train_end()
            if cfg.use_wandb:
                self.accelerator.end_training()
            return

        start_epoch = self.current_epoch
        resumed_run = self.resume_checkpoint_dir is not None
        for epoch in range(start_epoch, cfg.max_epochs):
            self.current_epoch = epoch
            is_resume_epoch = resumed_run and epoch == start_epoch
            resume_batches = self._batches_completed_in_epoch if is_resume_epoch else 0

            # A mid-epoch checkpoint already contains the effects of its
            # epoch-begin hook. Replaying it can mutate strategy state or
            # consume RNG a second time. Boundary resumes still run it.
            if not (is_resume_epoch and resume_batches > 0) and hasattr(
                self.strategy, "on_epoch_begin"
            ):
                self.strategy.on_epoch_begin()
            if hasattr(self.strategy, "set_epoch"):
                self.strategy.set_epoch(epoch)

            epoch_loss_sum = 0.0
            epoch_steps = 0

            if is_resume_epoch and resume_batches > 0:
                if self._resume_epoch_start_rng_state is None:
                    raise RuntimeError(
                        "Checkpoint has an in-epoch batch position but no "
                        "epoch-start RNG state"
                    )
                _restore_rng_state(self._resume_epoch_start_rng_state)
                _restore_sampler_state(
                    self._batch_sampler,
                    self._resume_epoch_start_sampler_state,
                )
            elif is_resume_epoch:
                # An epoch-boundary checkpoint resumes directly from the state
                # saved immediately before the next iterator is constructed.
                _restore_rng_state(self._resume_rng_state)
                _restore_sampler_state(
                    self._batch_sampler,
                    self._resume_sampler_state,
                )

            self._epoch_start_rng_state = _capture_rng_state()
            self._epoch_start_sampler_state = _capture_sampler_state(
                self._batch_sampler
            )
            train_iter = iter(self.train_loader)

            use_cached_first_batch = (
                epoch == start_epoch and first_batch is not None and not resumed_run
            )
            if is_resume_epoch:
                prefix_batches = self._epoch_prefix_batches
                batches_to_skip = resume_batches - prefix_batches
                if batches_to_skip < 0:
                    raise RuntimeError(
                        "Checkpoint batch position is smaller than its "
                        "data-dependent initialization prefix"
                    )
                try:
                    for _ in range(batches_to_skip):
                        next(train_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        "Checkpoint batch position exceeds the current "
                        "training dataloader length"
                    ) from exc

                # Skipping reconstructs the iterator and worker positions.
                # Restore the exact process RNG and sampler counters from the
                # checkpoint before the next real training step.
                if resume_batches > 0:
                    _restore_rng_state(self._resume_rng_state)
                    _restore_sampler_state(
                        self._batch_sampler,
                        self._resume_sampler_state,
                    )
                batches = train_iter
            else:
                self._epoch_prefix_batches = 1 if use_cached_first_batch else 0
                self._batches_completed_in_epoch = 0
                batches = (
                    itertools.chain([first_batch], train_iter)
                    if use_cached_first_batch
                    else train_iter
                )

            self._resume_rng_state = None
            self._resume_sampler_state = {}
            self._resume_epoch_start_rng_state = None
            self._resume_epoch_start_sampler_state = {}

            for batch in batches:
                with self.accelerator.accumulate(*self.networks.values()):
                    losses = self.strategy.training_step(
                        batch, self.global_step, self.optimizers
                    )

                self.global_step += 1
                self._batches_completed_in_epoch += 1
                epoch_loss_sum += losses.get("loss_G", losses.get("loss", 0.0))
                epoch_steps += 1

                if self.global_step % cfg.log_steps == 0:
                    self._log_training_step(losses)

                if self.global_step % cfg.display_freq == 0:
                    self._save_visuals(batch)

                if self.global_step % cfg.save_latest_freq == 0:
                    self._save_checkpoint("latest")

                if 0 < cfg.max_steps <= self.global_step:
                    break

            avg_loss = epoch_loss_sum / max(epoch_steps, 1)
            logger.info("Epoch %d finished, avg loss: %.4f", epoch, avg_loss)

            expected_batches = len(self.train_loader) + self._epoch_prefix_batches
            epoch_completed = self._batches_completed_in_epoch >= expected_batches
            if epoch_completed:
                # Step per-epoch LR schedulers only after a complete epoch. A
                # max_steps stop in the middle of an epoch must not advance LR.
                for scheduler in self.scheduler.values():
                    if scheduler is not None:
                        scheduler.step()

                self._batches_completed_in_epoch = 0
                self._epoch_prefix_batches = 0
                self._epoch_start_rng_state = None
                self._epoch_start_sampler_state = {}
                if (epoch + 1) % cfg.save_epoch_freq == 0:
                    self._save_checkpoint(
                        f"epoch_{epoch + 1}",
                        resume_epoch=epoch + 1,
                    )

            if 0 < cfg.max_steps <= self.global_step:
                # Persist the post-epoch scheduler state when max_steps lands
                # exactly on an epoch boundary, and always save a resumable
                # stopping point even when save_latest_freq did not fire.
                self._save_checkpoint(
                    "latest",
                    resume_epoch=epoch + 1 if epoch_completed else epoch,
                )
                logger.info("Reached max_steps=%d, stopping", cfg.max_steps)
                break

        self.strategy.on_train_end()
        if cfg.use_wandb:
            self.accelerator.end_training()

    def _log_training_step(self, losses: dict[str, float]) -> None:
        if not self.accelerator.is_main_process:
            return
        lr = next(iter(self.optimizers.values())).param_groups[0]["lr"]
        metrics = {
            "train/lr": lr,
            "train/epoch": self.current_epoch,
            "train/global_step": self.global_step,
        }
        metrics.update({f"train/{k}": v for k, v in losses.items()})
        if self.config.training.use_wandb:
            self.accelerator.log(metrics, step=self.global_step)
        logger.info("Step %d: %s", self.global_step, losses)

    def _save_visuals(self, batch: dict[str, Any]) -> None:
        """Save source/prediction/target images and update the HTML gallery."""
        if not self.accelerator.is_main_process:
            return

        sample = self.strategy.sample_step(batch, self.global_step)
        if sample is None:
            return

        epoch = self.current_epoch
        labels = [("source", "source"), ("pred", "fake"), ("target", "gt")]
        saved: list[tuple[str, str]] = []
        for key, label in labels:
            tensor = sample.get(key)
            if tensor is None or tensor.numel() == 0:
                continue
            # Save the first image in the batch.
            image = _tensor_to_uint8(tensor[0])
            image_name = f"epoch{epoch:03d}_{label}.png"
            save_path = self.img_dir / image_name
            Image.fromarray(image).save(save_path)
            saved.append((label, image_name))

        if saved:
            self._update_html(saved, epoch)
            self._visuals_saved = True

    def _update_html(self, saved: list[tuple[str, str]], epoch: int) -> None:
        """Rebuild the HTML gallery with the latest visuals."""
        doc = document(title=f"Experiment {self.output_dir.name}")
        with doc:
            h1(f"Experiment = {self.output_dir.name}")
            h2(f"epoch [{epoch}]")

        rows: list[list[Any]] = []
        for label, image_name in saved:
            link = a(href=f"images/{image_name}")
            link.add(img(src=f"images/{image_name}", style="width:256px;"))
            rows.append([label, link])

        with doc:
            tbl = table()
            for row in rows:
                with tbl.add(tr()):
                    for cell in row:
                        td(cell)

        html_path = self.web_dir / "index.html"
        html_path.write_text(str(doc))

    def _save_checkpoint(
        self,
        tag: str,
        *,
        resume_epoch: int | None = None,
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        ckpt_dir = self.output_dir / f"checkpoint-{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model
        scaler = getattr(self.accelerator, "scaler", None)
        epoch_to_resume = self.current_epoch if resume_epoch is None else resume_epoch
        state = {
            "checkpoint_version": _CHECKPOINT_VERSION,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "resume_state": {
                "epoch": epoch_to_resume,
                "batches_completed_in_epoch": self._batches_completed_in_epoch,
                "epoch_prefix_batches": self._epoch_prefix_batches,
                "rng_state": _capture_rng_state(),
                "sampler_state": _capture_sampler_state(self._batch_sampler),
                "epoch_start_rng_state": self._epoch_start_rng_state,
                "epoch_start_sampler_state": self._epoch_start_sampler_state,
            },
            "networks": {
                name: unwrapped(net).state_dict() for name, net in self.networks.items()
            },
            "optimizers": {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
            "scheduler_state": {
                name: sched.state_dict() if sched is not None else {}
                for name, sched in self.scheduler.items()
            },
            "strategy_state": self.strategy.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else {},
        }
        torch.save(state, ckpt_dir / "trainer_state.pt")

        resolved = OmegaConf.to_container(
            OmegaConf.structured(self.config), resolve=True
        )
        with open(ckpt_dir / "config.yaml", "w") as f:
            OmegaConf.save(resolved, f)

        logger.info("Checkpoint saved: %s", ckpt_dir)

    def _cleanup_old_checkpoints(self, limit: int) -> None:
        if limit <= 0:
            return
        ckpt_dirs = sorted(
            (
                p
                for p in self.output_dir.glob("checkpoint-*")
                if p.name.split("-")[-1].isdigit()
            ),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        while len(ckpt_dirs) > limit:
            old = ckpt_dirs.pop(0)
            shutil.rmtree(old)
            logger.info("Removed old checkpoint: %s", old)
