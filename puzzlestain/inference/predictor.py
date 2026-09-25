"""Inference utilities for virtual-stain models.

This module provides a lightweight, model-agnostic :class:`StainPredictor` that
rehydrates a trained model from a PuzzleStain checkpoint directory and runs
inference on individual images or whole directories. It follows the same
registry-based construction pattern as the training pipeline so that adding a
new model family only requires registering its model/processor/strategy triplet.
"""

from __future__ import annotations

import hashlib
import logging
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from ..configs.schema import PuzzleStainConfig

# Trigger registration of built-in model families.
from ..models import (  # noqa: F401
    cut_family,
    cyclegan_family,
    diffusion_family,
    pix2pix_family,
)
from ..models.base import StainModel, StainProcessor
from ..models.registry import ModelRegistry
from .spatial import (
    PhysicalInferenceSpec,
    prepare_physical_image,
    sliding_window_inference,
)

logger = logging.getLogger(__name__)

# Config-string -> PIL resample filter for input resizing at inference.
_RESAMPLE_MODES = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


class StainPredictor:
    """Rehydrate a virtual-stain checkpoint and run image-to-image inference.

    The predictor is deliberately decoupled from the trainer: it rebuilds the
    model and processor from the checkpoint's ``config.yaml`` using
    :class:`~puzzlestain.models.registry.ModelRegistry`, loads only the
    generator weights (the discriminator is ignored for inference), and exposes
    a small ``predict`` API.

    Args:
        model: Constructed model instance. For Pix2Pix this is the wrapper that
            owns ``netG``.
        processor: Matching processor that handles pre/post-processing.
        device: Torch device to run on.
        crop_size: Spatial size used at training time. Input images are resized
            to this square size before being fed to the model.
        train_mode: If ``True``, leave the model in training mode instead of
            calling ``eval()``. This matches the default behavior of some
            original pix2pix repos when BatchNorm is used with batch size 1;
            it makes BatchNorm normalize using the input image's own statistics
            rather than the accumulated running statistics. Use with caution:
            results become input-dependent and the running statistics may be
            updated during inference.
        seed_per_image: If ``True``, fix the torch random seed from each
            image's content before its forward pass, so repeated runs produce
            identical outputs even with dropout active (``train_mode=True``).
        interpolation: PIL resample filter used when resizing inputs to
            ``crop_size``; resolved from the training transform config by
            :meth:`from_checkpoint` (default bicubic).
        training_resize_size: Explicit resize canvas used by training, when
            present in the transform pipeline.
        training_crop_size: Actual random/center-crop input size used by
            training, when present in the transform pipeline.
        target_mpp: Training physical resolution in microns per pixel, read
            from ``data.target_mpp`` when recorded in the checkpoint config.
    """

    def __init__(
        self,
        model: StainModel,
        processor: StainProcessor,
        device: torch.device,
        crop_size: int = 512,
        train_mode: bool = False,
        seed_per_image: bool = False,
        interpolation: int = Image.BICUBIC,
        training_resize_size: int | None = None,
        training_crop_size: int | None = None,
        target_mpp: float | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.crop_size = crop_size
        self.train_mode = train_mode
        self.seed_per_image = seed_per_image
        self.interpolation = interpolation
        self.training_resize_size = training_resize_size
        self.training_crop_size = training_crop_size
        self.target_mpp = target_mpp
        #: Optional target-domain label for conditional (multi-stain) models.
        #: Set this to the index of the desired stain in ``model.stain_names``
        #: before calling :meth:`predict`; ignored by unconditional models.
        self.target_label: int | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: str | torch.device | None = None,
        strict: bool = True,
        train_mode: bool = False,
        seed_per_image: bool = False,
    ) -> StainPredictor:
        """Build a predictor from a PuzzleStain checkpoint directory.

        The directory must contain ``config.yaml`` and ``trainer_state.pt`` (as
        written by :class:`~puzzlestain.training.trainer.Trainer`).

        Args:
            checkpoint_dir: Checkpoint directory, e.g. an ``checkpoint-epoch_N``
                folder inside a run's ``output_dir``.
            device: Torch device. Defaults to ``cuda`` when available, else ``cpu``.
            strict: Whether to require an exact match when loading generator
                weights. Leave ``True`` unless you are intentionally loading a
                partial checkpoint.
            train_mode: If ``True``, keep the model in training mode for
                inference. See :attr:`StainPredictor.train_mode`.
            seed_per_image: If ``True``, fix the torch random seed per image so
                train-mode inference is reproducible.
                See :attr:`StainPredictor.seed_per_image`.

        Returns:
            A ready-to-use :class:`StainPredictor`.
        """
        checkpoint_dir = Path(checkpoint_dir)
        config_path = checkpoint_dir / "config.yaml"
        state_path = checkpoint_dir / "trainer_state.pt"

        if not config_path.is_file():
            raise FileNotFoundError(f"Missing config in checkpoint dir: {config_path}")
        if not state_path.is_file():
            raise FileNotFoundError(f"Missing state in checkpoint dir: {state_path}")

        raw_cfg = OmegaConf.load(config_path)
        # Resolve without requiring Hydra resolvers such as ${now:...} to be
        # registered. Training writes a fully-resolved config.yaml, but some
        # interpolation types (e.g. now) are Hydra-specific. We only need the
        # model/transforms sub-trees, so unresolved interpolations elsewhere are
        # harmless.
        merged = OmegaConf.merge(OmegaConf.structured(PuzzleStainConfig), raw_cfg)
        config = OmegaConf.to_container(merged, resolve=False)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        model_cfg = config["model"]
        model_name = model_cfg["name"]
        # Fields that belong to processor/strategy, not the raw model constructor.
        non_model_fields = {"name", "params", "use_gan", "normalize_to_minus_one_one"}
        model_kwargs = {k: v for k, v in model_cfg.items() if k not in non_model_fields}
        model_kwargs.update(dict(model_cfg.get("params") or {}))
        # Inference does not need a discriminator.
        model_kwargs["no_d"] = True

        model = ModelRegistry.build_model(model_name, **model_kwargs)
        processor = ModelRegistry.build_processor(
            model_name,
            normalize_to_minus_one_one=model_cfg.get(
                "normalize_to_minus_one_one", True
            ),
            # Superset forwarding (dropped per processor signature): lets the
            # checkpoint config route processor options (e.g. uni_checkpoint).
            **model_kwargs,
        )

        state = torch.load(state_path, map_location=device, weights_only=True)
        if "networks" not in state or "G" not in state["networks"]:
            raise KeyError(
                f"Checkpoint {state_path} does not contain a 'networks/G' state dict"
            )
        g_state = state["networks"]["G"]
        ema = (state.get("strategy_state") or {}).get("ema")
        if ema is not None:
            # Diffusion-family strategies persist an EMA shadow of the denoiser;
            # the reference repos sample with it. Overlay it on the raw state
            # dict so any non-parameter buffers are still supplied.
            g_state = {**g_state, **ema}
            logger.info(
                "Generator weights overlaid with EMA shadow from strategy_state"
            )
        model.netG.load_state_dict(g_state, strict=strict)
        model = model.to(device)
        if train_mode:
            model.train()
            logger.info(
                "Predictor loaded in train mode (BatchNorm uses input statistics)"
            )
        else:
            model.eval()

        crop_size = cls._resolve_inference_size(config)
        training_resize_size, training_crop_size = cls._resolve_training_sizes(config)
        target_mpp = cls._resolve_target_mpp(config)
        interpolation = cls._resolve_inference_interpolation(config)
        logger.info(
            "Loaded predictor from %s | model=%s | crop_size=%d | "
            "training_resize_size=%s | training_crop_size=%s | target_mpp=%s | "
            "device=%s | train_mode=%s",
            checkpoint_dir,
            model_name,
            crop_size,
            training_resize_size,
            training_crop_size,
            target_mpp,
            device,
            train_mode,
        )
        return cls(
            model=model,
            processor=processor,
            device=device,
            crop_size=crop_size,
            train_mode=train_mode,
            seed_per_image=seed_per_image,
            interpolation=interpolation,
            training_resize_size=training_resize_size,
            training_crop_size=training_crop_size,
            target_mpp=target_mpp,
        )

    @staticmethod
    def _resolve_inference_size(config: dict[str, Any]) -> int:
        """Preserve the legacy, order-sensitive inference resize resolution."""
        transforms_cfg = config.get("transforms", {})
        # Compatibility contract: return the first resize/crop encountered.
        # Physical modes separately retain both sizes via
        # ``_resolve_training_sizes``; changing this method would alter existing
        # ``resize`` predictions for resize-then-crop configurations.
        for transform in transforms_cfg.get("train_transforms", []):
            name = transform.get("name", "")
            if (
                name in ("synchronized_random_crop", "synchronized_center_crop")
                and "size" in transform
            ):
                return int(transform["size"])
            if name == "synchronized_resize" and "size" in transform:
                size = transform["size"]
                if isinstance(size, (list, tuple)):
                    return int(size[0])
                return int(size)
        # Fall back to the legacy crop_size field.
        return int(transforms_cfg.get("crop_size", 512))

    @staticmethod
    def _resolve_training_sizes(
        config: dict[str, Any],
    ) -> tuple[int | None, int | None]:
        """Return the explicit square resize canvas and model crop size."""

        resize_size: int | None = None
        crop_size: int | None = None
        transforms = config.get("transforms", {}).get("train_transforms", [])
        for transform in transforms:
            name = transform.get("name", "")
            size = transform.get("size")
            if size is None:
                continue
            if isinstance(size, (list, tuple)):
                if len(size) != 2 or int(size[0]) != int(size[1]):
                    continue
                size = size[0]
            if name == "synchronized_resize" and resize_size is None:
                resize_size = int(size)
            elif (
                name
                in (
                    "synchronized_random_crop",
                    "synchronized_center_crop",
                )
                and crop_size is None
            ):
                crop_size = int(size)
        return resize_size, crop_size

    @staticmethod
    def _resolve_target_mpp(config: dict[str, Any]) -> float | None:
        """Read a positive training MPP when the checkpoint records one."""

        value = config.get("data", {}).get("target_mpp")
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Checkpoint data.target_mpp must be finite and positive")
        return value

    @staticmethod
    def _resolve_inference_interpolation(config: dict[str, Any]) -> int:
        """Infer the resize interpolation used at training time (default bicubic)."""
        for transform in config.get("transforms", {}).get("train_transforms", []):
            if transform.get("name") == "synchronized_resize":
                mode = str(transform.get("interpolation", "bicubic")).lower()
                if mode not in _RESAMPLE_MODES:
                    raise ValueError(
                        f"Unsupported resize interpolation {mode!r}; "
                        f"expected one of {sorted(_RESAMPLE_MODES)}"
                    )
                return _RESAMPLE_MODES[mode]
        return Image.BICUBIC

    def _read_source(self, source: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        """Normalize a variety of inputs to a HWC uint8 or float32 [0,1] array."""
        if isinstance(source, (str, Path)):
            return np.array(Image.open(source).convert("RGB"))
        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))
        arr = np.asarray(source)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr

    def _resize_to_crop(self, arr: np.ndarray) -> np.ndarray:
        """Resize a HWC image to the training crop size."""
        if arr.shape[0] == self.crop_size and arr.shape[1] == self.crop_size:
            return arr
        img = Image.fromarray(arr)
        img = img.resize((self.crop_size, self.crop_size), self.interpolation)
        return np.array(img)

    @torch.inference_mode()
    def predict(
        self,
        source: str | Path | np.ndarray | Image.Image,
        return_normalized: bool = False,
    ) -> np.ndarray:
        """Run inference on a single source image.

        Args:
            source: Input image as a file path, ``PIL.Image``, or HWC numpy array.
                Pixel values may be ``uint8 [0, 255]`` or float ``[0, 1]``.
            return_normalized: If True, return a float32 array in the model
                processor's output range; CUT/PGLStain direct-resize inference
                returns ``[-1, 1]``. Otherwise return uint8 ``[0, 255]``.

        Returns:
            Generated virtual-stain image as a HWC numpy array.
        """
        arr = self._read_source(source)
        arr = self._resize_to_crop(arr)
        return self._predict_array(arr, return_normalized=return_normalized)

    @torch.inference_mode()
    def _predict_array(
        self,
        arr: np.ndarray,
        *,
        return_normalized: bool,
        deterministic_seed: bool = False,
    ) -> np.ndarray:
        """Run the model on an already prepared HWC uint8 RGB image."""

        inputs = self.processor.preprocess({"source_image": arr}, device=self.device)
        # The processor produces per-sample CHW tensors; add the batch dimension.
        inputs["source_image"] = inputs["source_image"].unsqueeze(0)
        if hasattr(self.model, "stain_names"):
            # Conditional (multi-stain) model: route the target-domain label.
            if self.target_label is None:
                raise ValueError(
                    "This is a multi-stain checkpoint; set predictor.target_label "
                    "to the index of the desired stain in model.stain_names "
                    f"({list(self.model.stain_names)}) before calling predict()."
                )
            inputs["target_label"] = torch.tensor(
                [self.target_label], dtype=torch.long, device=self.device
            )
        if deterministic_seed:
            # Strict modes use a stable, low-collision seed while leaving the
            # legacy --seed-per-image behavior byte-for-byte compatible.
            digest = hashlib.sha256(arr.tobytes()).digest()
            seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
            torch.manual_seed(seed)
        elif self.seed_per_image:
            # Seed from the image content so each image gets a fixed dropout
            # mask; repeated runs then produce identical outputs in train mode.
            torch.manual_seed(zlib.crc32(arr.tobytes()))
        output = self.model(**inputs)
        results = self.processor.postprocess(
            output,
            original_sizes=[(arr.shape[0], arr.shape[1])],
            return_normalized=return_normalized,
        )
        return results[0]["image"]

    def predict_mpp(
        self,
        source: str | Path | np.ndarray | Image.Image,
        spec: PhysicalInferenceSpec,
        return_normalized: bool = False,
    ) -> np.ndarray:
        """Run strict physical-scale-aware full-image or sliding inference."""

        arr = prepare_physical_image(
            self._read_source(source), spec, interpolation=self.interpolation
        )
        if spec.mode == "mpp-full":
            return self._predict_array(
                arr,
                return_normalized=return_normalized,
                deterministic_seed=True,
            )

        normalized = sliding_window_inference(
            arr,
            lambda tile: self._predict_array(
                tile,
                return_normalized=True,
                deterministic_seed=True,
            ),
            tile_size=spec.tile_size,
            overlap=spec.overlap,
        )
        if return_normalized:
            return normalized
        return np.rint(normalized * 255.0).clip(0, 255).astype(np.uint8)

    def predict_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        return_normalized: bool = False,
        mpp_spec: PhysicalInferenceSpec | None = None,
    ) -> None:
        """Predict one image and write it to disk."""
        if mpp_spec is None:
            out = self.predict(input_path, return_normalized=return_normalized)
        else:
            out = self.predict_mpp(
                input_path,
                mpp_spec,
                return_normalized=return_normalized,
            )
        if return_normalized:
            np.save(output_path, out)
        else:
            Image.fromarray(out).save(output_path)

    def predict_dir(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        return_normalized: bool = False,
        suffix: str = ".png",
        mpp_spec: PhysicalInferenceSpec | None = None,
    ) -> list[Path]:
        """Predict every image in ``input_dir`` and write results to ``output_dir``.

        Args:
            input_dir: Directory containing source images.
            output_dir: Directory to write generated images into.
            return_normalized: If True, write float32 NumPy arrays. Their range
                depends on the model and inference mode; CUT/PGLStain
                direct-resize inference returns ``[-1, 1]``.
            suffix: Output filename extension.
            mpp_spec: Optional strict physical-scale inference protocol. When
                omitted, the original direct-resize behavior is preserved.

        Returns:
            List of written output paths.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        paths = sorted(
            p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
        )

        written: list[Path] = []
        for src_path in paths:
            if mpp_spec is None:
                pred = self.predict(src_path, return_normalized=return_normalized)
            else:
                pred = self.predict_mpp(
                    src_path,
                    mpp_spec,
                    return_normalized=return_normalized,
                )
            out_path = output_dir / (src_path.stem + suffix)
            if return_normalized:
                np.save(out_path.with_suffix(".npy"), pred)
            else:
                Image.fromarray(pred).save(out_path)
            written.append(out_path)
            logger.info("Predicted %s -> %s", src_path, out_path)

        return written
