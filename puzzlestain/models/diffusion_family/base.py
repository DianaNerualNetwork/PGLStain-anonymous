"""Shared base classes for the diffusion family of models.

The diffusion family covers bridge-style diffusion models for image-to-image
translation (BBDM, DDBM, ...). A diffusion model wraps a denoiser network
(``denoise_fn``) plus a noise schedule; the full sampling chain is exposed
through ``forward(source_image)`` so that the generic training/inference
pipeline (and the ``puzzlestain-predict`` CLI) stays unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base import StainModel, StainOutput, StainProcessor


class BaseDiffusionModel(nn.Module, StainModel):
    """Common diffusion-family model wrapper.

    Subclasses set schedule hyperparameters and build ``self.denoise_fn`` in
    :meth:`_build_networks`, then register under distinct names in
    :class:`~puzzlestain.models.registry.ModelRegistry`.

    Conventions:
      - ``forward(source_image)`` runs the full sampling chain and returns the
        generated image (inference path; used by the predictor and by
        ``sample_step``).
      - ``denoise(x_t, t, context)`` is a single denoiser evaluation, used by
        the training strategy and by the sampling loop.
      - ``sample_timesteps(b, device)`` draws the timesteps for one training
        batch.

    Args:
        input_nc: Number of source (condition) image channels.
        output_nc: Number of target image channels.
        no_d: Accepted for framework compatibility (diffusion models have no
            discriminator; ignored).
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        no_d: bool = False,
    ) -> None:
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.denoise_fn: nn.Module
        self._build_networks()

    def _build_networks(self) -> None:
        """Construct ``self.denoise_fn`` and register schedule buffers."""
        raise NotImplementedError

    @property
    def netG(self) -> nn.Module:
        """Alias of ``denoise_fn`` for framework compatibility.

        The checkpoint stores the denoiser under the ``networks/G`` key and the
        predictor loads it back via ``model.netG``; both contracts are shared
        with the GAN families, so no predictor change is required.
        """
        return self.denoise_fn

    def forward(self, source_image: torch.Tensor, **kwargs) -> StainOutput:
        """Generate the target-domain image from the source image.

        Runs the full sampling chain starting from the condition image. Use
        :meth:`denoise` for single-step (training-time) evaluations.
        """
        return StainOutput(pred_image=self.sample(source_image))

    def denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one denoiser evaluation."""
        raise NotImplementedError

    def sample(self, y: torch.Tensor) -> torch.Tensor:
        """Run the full sampling chain from the condition image ``y``."""
        raise NotImplementedError

    def sample_timesteps(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Draw the timesteps for one training batch."""
        raise NotImplementedError

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return the denoiser parameters."""
        return list(self.denoise_fn.parameters())

    def get_model_info(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "denoise_fn": self.denoise_fn.__class__.__name__,
        }


class BaseDiffusionProcessor(StainProcessor):
    """Common processor for diffusion-family image-to-image translation.

    Mirrors :class:`BaseCUTProcessor`: batches provide ``source_image`` (the
    condition) and optionally ``target_image``; both are converted to
    ``(B, C, H, W)`` float tensors on the target device in ``[-1, 1]``.

    The constructor accepts and ignores framework kwargs so that
    ``ModelRegistry.build_processor`` can pass a uniform set of kwargs to
    every processor.
    """

    def __init__(self, **kwargs: object) -> None:
        """Accept and ignore framework kwargs."""
        pass

    def preprocess(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        source = self._to_tensor(batch["source_image"], device)
        result: dict[str, torch.Tensor] = {"source_image": source}
        if "target_image" in batch and batch["target_image"] is not None:
            result["target_image"] = self._to_tensor(batch["target_image"], device)
        return result

    def postprocess(
        self,
        output: StainOutput,
        original_sizes: list[tuple[int, int]] | None = None,
        return_normalized: bool = False,
    ) -> list[dict[str, np.ndarray]]:
        pred = output.pred_image.detach().cpu()
        if not return_normalized:
            pred = (pred + 1.0) / 2.0
            pred = pred.clamp(0, 1).numpy()
            pred = (pred * 255).astype(np.uint8)
        else:
            pred = pred.clamp(-1, 1).numpy()

        results = []
        for i in range(pred.shape[0]):
            img = pred[i]
            if img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 1:
                img = img.squeeze(2)
            results.append({"image": img})
        return results

    def _to_tensor(self, value: Any, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device, dtype=torch.float32)
        else:
            arr = np.asarray(value)
            if arr.ndim == 3 and arr.shape[-1] in (1, 3):
                arr = np.transpose(arr, (2, 0, 1))
            if np.issubdtype(arr.dtype, np.integer):
                tensor = torch.from_numpy(arr.astype(np.float32)).to(device) / 255.0
            else:
                tensor = torch.from_numpy(arr.astype(np.float32)).to(device)

        # Ensure the tensor is in [-1, 1].
        t_min, t_max = tensor.min().item(), tensor.max().item()
        if t_min >= -1.0 and t_max <= 1.0 and t_min < 0.0:
            return tensor
        if t_min >= 0.0 and t_max <= 1.0:
            return 2.0 * tensor - 1.0
        if t_max > 1.0:
            tensor = tensor / 255.0
            return 2.0 * tensor - 1.0
        return tensor


__all__ = ["BaseDiffusionModel", "BaseDiffusionProcessor"]
