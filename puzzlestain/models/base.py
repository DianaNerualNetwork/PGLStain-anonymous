"""Unified protocol and output format for virtual-stain (image-to-image) models.

This module defines contracts for generated images. A model is a pure
tensor-in / tensor-out component, unaware of loss computation and I/O
conversion. Pre- and post-processing live in a matching
:class:`StainProcessor`, and training dynamics live in a
:class:`~puzzlestain.models.strategy.base.TrainingStrategy`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    import numpy as np


@dataclass
class StainOutput:
    """Unified output format for all virtual-stain models.

    Attributes:
        pred_image: Generated image tensor. During training this is typically
            the raw network output (e.g. logits or [-1, 1] RGB); during
            inference it may be rescaled to [0, 1].
        auxiliary: Model-specific extra outputs (attention maps, intermediate
            features, diffusion timestep predictions, etc.) not used by the
            primary loss.
    """

    pred_image: torch.Tensor
    auxiliary: dict[str, Any] = field(default_factory=dict)


class StainModel(abc.ABC):
    """Abstract protocol for virtual-stain models.

    A model is a pure tensor-in / :class:`StainOutput`-out component. It
    deliberately stays unaware of loss computation and pre/post-processing so
    that those concerns can vary independently (loss via ``LossRegistry``,
    I/O conversion via :class:`StainProcessor`, training dynamics via
    ``TrainingStrategy``).

    Conventions:
      1. ``forward()`` returns a :class:`StainOutput`.
      2. The model itself does NOT compute loss; loss is handled externally.
      3. ``get_trainable_params()`` returns parameters for the optimizer.

    Note:
        This is a mixin protocol. Concrete implementations inherit both
        ``nn.Module`` and this protocol, so ``forward`` participates in the
        normal ``nn.Module.__call__`` dispatch.
    """

    @abc.abstractmethod
    def forward(self, **kwargs) -> StainOutput:
        """Run the forward pass.

        Args:
            **kwargs: Model-specific tensor inputs produced by the matching
                :meth:`StainProcessor.preprocess`.

        Returns:
            StainOutput: Raw model predictions.
        """
        ...

    @abc.abstractmethod
    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return the parameters that should be passed to the optimizer.

        Returns:
            list[nn.Parameter]: Parameters with ``requires_grad=True`` (e.g.
            only PEFT/LoRA params when the backbone is frozen).
        """
        ...

    @abc.abstractmethod
    def get_model_info(self) -> dict:
        """Return human-readable model metadata for logging.

        Returns:
            dict: Arbitrary descriptive fields (parameter counts, backbone
            name, trainable ratio, etc.).
        """
        ...


class StainProcessor(abc.ABC):
    """Unified protocol for model preprocessing / postprocessing.

    The processor is the interface between the model and the external world:
      - preprocess: data layer's numpy batch dict -> model-ready tensor dict
      - postprocess: model's StainOutput -> final prediction results

    Design constraints:
      - Processor does NOT inherit nn.Module; no learnable parameters.
      - Processor is NOT wrapped by DDP/FSDP.
      - preprocess runs outside the autocast context.

    For diffusion-based models the frozen VAE encoder may be invoked here as
    part of image-level preprocessing, but the VAE itself is owned and held by
    the model/strategy, not by the processor.
    """

    @abc.abstractmethod
    def preprocess(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Convert data-layer numpy batch to model-consumable tensor dict.

        Responsibilities:
          - numpy -> tensor, move to target device
          - image resize to model input size
          - image normalization (model-specific, often [-1, 1] for GAN/Diffusion
            or [0, 1] for supervised regression)
          - stack variable-length image lists into tensors
          - optionally encode images into latent space via a frozen VAE

        Args:
            batch: numpy batch dict from ``stain_collate_fn``.
            device: Target device.

        Returns:
            dict[str, Tensor]: All inputs for ``model.forward()``, plus target
            tensors such as ``"target_image"`` or ``"target_latent"``.
        """
        ...

    @abc.abstractmethod
    def postprocess(
        self,
        output: StainOutput,
        original_sizes: Optional[list[tuple[int, int]]] = None,
        return_normalized: bool = False,
    ) -> list[dict[str, "np.ndarray"]]:
        """Convert model output to final prediction results.

        Responsibilities:
          - rescale generated tensors to the desired output range
          - resize back to original image dimensions when ``original_sizes``
            is provided
          - convert to ``np.ndarray`` (HWC uint8 or HW float32)

        Args:
            output: Raw model output.
            original_sizes: Per-sample original ``(H, W)`` for resize.
            return_normalized: If True, return floating-point arrays in
                ``[0, 1]`` instead of uint8 ``[0, 255]``.

        Returns:
            list[dict], one per batch sample:
              - ``"image"``: np.ndarray, generated image (HWC uint8 by default).
        """
        ...
