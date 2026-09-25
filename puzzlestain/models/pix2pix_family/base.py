"""Shared base classes for the Pix2Pix family of models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base import StainModel, StainOutput, StainProcessor


class BasePix2PixModel(nn.Module, StainModel):
    """Common Pix2Pix-family model wrapper.

    All Pix2Pix variants share the same high-level contract:
      - a generator ``netG`` maps source images to target-domain images
      - an optional discriminator ``netD`` is used during training

    Subclasses are responsible for constructing ``netG`` and ``netD`` with the
    specific architecture they need.

    Args:
        input_nc: Number of input channels.
        output_nc: Number of output channels.
        no_d: If True, do not build a discriminator (inference-only).
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
        self.netG: nn.Module
        self.netD: nn.Module | None = None
        self._build_networks(no_d=no_d)

    def _build_networks(self, no_d: bool) -> None:
        """Construct ``self.netG`` and optionally ``self.netD``."""
        raise NotImplementedError

    def forward(self, source_image: torch.Tensor, **kwargs: object) -> StainOutput:
        """Generate the stained image from the source image."""
        pred = self.netG(source_image)
        return StainOutput(pred_image=pred)

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return all trainable parameters (G + D if present)."""
        params: list[nn.Parameter] = list(self.netG.parameters())
        if self.netD is not None:
            params += list(self.netD.parameters())
        return params

    def get_model_info(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "netG": self.netG.__class__.__name__,
            "netD": self.netD.__class__.__name__ if self.netD else None,
        }


class BasePix2PixProcessor(StainProcessor):
    """Common processor for paired image-to-image translation.

    Expects the batch to contain ``source_image`` and optionally ``target_image``.
    Both are converted to ``torch.Tensor`` on the target device, with shape
    ``(B, C, H, W)`` and values in ``[-1, 1]``.

    The constructor accepts and ignores framework kwargs (e.g.
    ``normalize_to_minus_one_one``) so that ``ModelRegistry.build_processor`` can
    pass a uniform set of kwargs to every processor.
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


__all__ = ["BasePix2PixModel", "BasePix2PixProcessor"]
