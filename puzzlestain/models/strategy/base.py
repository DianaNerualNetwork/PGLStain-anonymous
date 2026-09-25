"""Training strategy protocol.

A ``TrainingStrategy`` encapsulates the training dynamics of a particular
algorithm family (e.g. supervised regression, GAN, diffusion). The generic
:class:`~puzzlestain.training.trainer.Trainer` only knows how to call the
strategy's methods; it is not aware of how many networks or optimizers exist,
nor of how gradients are routed between them.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..configs.schema import PuzzleStainConfig


class TrainingStrategy(abc.ABC):
    """Abstract protocol for algorithm-specific training steps.

    A strategy owns the following concerns:
      1. Which networks participate in training (``get_networks``).
      2. Which optimizers/schedulers update those networks (``get_optimizers``,
         ``get_schedulers``).
      3. How a single training iteration is performed (``training_step``).
      4. How samples are generated for visualization/evaluation
         (``sample_step``).
      5. Any algorithm-specific checkpoint state (``state_dict`` /
         ``load_state_dict``).

    The generic :class:`~puzzlestain.training.trainer.Trainer` prepares the
    networks and optimizers with ``accelerate``, then repeatedly calls
    :meth:`training_step` and handles logging, checkpointing, and evaluation
    cadence independently.
    """

    @abc.abstractmethod
    def get_networks(self) -> dict[str, nn.Module]:
        """Return the trainable networks managed by this strategy.

        Returns:
            dict[str, nn.Module]: Mapping from a short name to the network.
            Examples:

            - Supervised: ``{"model": model}``
            - CUT: ``{"G": netG, "F": netF, "D": netD}``
            - Diffusion: ``{"unet": unet}`` (VAE/text encoder are not included
              because they are frozen during training).
        """
        ...

    @abc.abstractmethod
    def get_optimizers(
        self,
        config: "PuzzleStainConfig",
    ) -> dict[str, torch.optim.Optimizer]:
        """Create and return the optimizers for this strategy.

        Args:
            config: The resolved experiment config. Learning rates, betas, and
                weight decay are read from ``config.training``.

        Returns:
            dict[str, torch.optim.Optimizer]: Optimizer names aligned with the
            network names returned by :meth:`get_networks`.
        """
        ...

    @abc.abstractmethod
    def get_schedulers(
        self,
        optimizers: dict[str, torch.optim.Optimizer],
        config: "PuzzleStainConfig",
    ) -> dict[str, Any]:
        """Create and return the LR schedulers for this strategy.

        Args:
            optimizers: The optimizers returned by :meth:`get_optimizers`.
            config: The resolved experiment config.

        Returns:
            dict[str, Any]: Mapping from optimizer name to scheduler. The value
            may be ``None`` for an optimizer that has no scheduler. The trainer
            steps each non-None scheduler per optimizer step.
        """
        ...

    @abc.abstractmethod
    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
        optimizers: dict[str, torch.optim.Optimizer],
    ) -> dict[str, float]:
        """Execute one complete training iteration.

        A complete iteration may involve multiple forward/backward passes
        (e.g. GAN discriminator followed by generator). The strategy is
        responsible for zeroing gradients, detaching tensors, toggling
        ``requires_grad``, and stepping optimizers.

        Args:
            batch: Model-specific tensor inputs produced by the processor.
            global_step: Current global optimizer step counter.
            optimizers: Prepared optimizers (already wrapped by accelerate).

        Returns:
            dict[str, float]: Scalar losses to log. Keys are used as metric
            names (e.g. ``{"loss": ..., "loss_G": ..., "loss_D": ...}``).
        """
        ...

    def sample_step(
        self,
        batch: dict[str, torch.Tensor],
        global_step: int,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Generate visualization samples for the current batch.

        This method is optional. Strategies that support it return a dict of
        image tensors such as ``{"source": ..., "pred": ..., "target": ...}``
        which the trainer/callbacks can log or save.

        Args:
            batch: Model-specific tensor inputs produced by the processor.
            global_step: Current global optimizer step counter.

        Returns:
            Optional dict of image tensors, or ``None`` if sampling is not
            supported by this strategy.
        """
        return None

    def get_ema_networks(self) -> dict[str, nn.Module]:
        """Return the subset of networks that should be tracked by EMA.

        Defaults to all networks returned by :meth:`get_networks`. Strategies
        may override this to exclude auxiliary networks (e.g. the discriminator
        in a GAN).

        Returns:
            dict[str, nn.Module]: Networks to shadow with EMA.
        """
        return self.get_networks()

    def replace_networks(self, networks: dict[str, nn.Module]) -> None:
        """Replace the networks managed by this strategy after acceleration.

        The generic trainer calls ``accelerator.prepare`` on the networks
        returned by :meth:`get_networks`. Because ``prepare`` may return
        wrapped modules (e.g. DDP) that live on the target device, this hook
        gives the strategy a chance to swap those prepared modules back into
        its internal model wrapper so that ``training_step`` uses the
        accelerated versions.

        Args:
            networks: Mapping from network name to the prepared module.
        """
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return algorithm-specific checkpoint state.

        Examples of extra state include diffusion noise scheduler buffers,
        latent mean/std normalizers, or adversarial training counters.

        Returns:
            dict: Extra checkpoint state to merge into the checkpoint.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific checkpoint state.

        Args:
            state: State previously returned by :meth:`state_dict`.
        """
        pass

    def on_train_begin(self) -> None:
        """Hook called once before training starts."""
        pass

    def on_train_end(self) -> None:
        """Hook called once after training finishes."""
        pass

    def on_eval_begin(self) -> None:
        """Hook called before each evaluation block."""
        pass

    def on_eval_end(self) -> None:
        """Hook called after each evaluation block."""
        pass
