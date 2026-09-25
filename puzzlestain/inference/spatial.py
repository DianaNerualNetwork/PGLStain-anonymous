"""Physical-scale-aware preprocessing and tiled inference utilities."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

PhysicalInferenceMode = Literal["mpp-full", "mpp-sliding"]


@dataclass(frozen=True)
class PhysicalInferenceSpec:
    """Validated spatial protocol for research-grade patch inference."""

    mode: PhysicalInferenceMode
    input_mpp: float
    target_mpp: float
    full_size: int | None = None
    tile_size: int = 512
    overlap: int = 128

    def __post_init__(self) -> None:
        if self.mode not in ("mpp-full", "mpp-sliding"):
            raise ValueError(f"Unsupported physical inference mode: {self.mode!r}")
        for name, value in (
            ("input_mpp", self.input_mpp),
            ("target_mpp", self.target_mpp),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite value greater than zero")
        if self.mode == "mpp-full" and (self.full_size is None or self.full_size <= 0):
            raise ValueError("mpp-full requires a positive full_size")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be greater than zero")
        if self.overlap < 0 or self.overlap * 2 >= self.tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size / 2")


def prepare_physical_image(
    image: np.ndarray,
    spec: PhysicalInferenceSpec,
    interpolation: int = Image.BICUBIC,
) -> np.ndarray:
    """Resample an RGB patch to ``target_mpp`` and enforce full-mode size."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image data, got {image.dtype}")

    scale = spec.input_mpp / spec.target_mpp
    height, width = image.shape[:2]
    out_width = max(1, math.floor(width * scale + 0.5))
    out_height = max(1, math.floor(height * scale + 0.5))
    if (out_width, out_height) == (width, height):
        prepared = image
    else:
        prepared = np.asarray(
            Image.fromarray(image).resize((out_width, out_height), interpolation)
        )

    if spec.mode == "mpp-full" and prepared.shape[:2] != (
        spec.full_size,
        spec.full_size,
    ):
        raise ValueError(
            "MPP resampling produced "
            f"{prepared.shape[1]}x{prepared.shape[0]}, but mpp-full requires "
            f"{spec.full_size}x{spec.full_size}; refusing a second resize because "
            "it would change the physical scale"
        )
    return prepared


def sliding_window_inference(
    image: np.ndarray,
    predict_tile: Callable[[np.ndarray], np.ndarray],
    *,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """Predict overlapping tiles and blend them into one float32 RGB image."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")
    if tile_size <= 0 or overlap < 0 or overlap * 2 >= tile_size:
        raise ValueError("Expected tile_size > 0 and 0 <= overlap < tile_size / 2")

    height, width = image.shape[:2]
    pad_height = max(0, tile_size - height)
    pad_width = max(0, tile_size - width)
    if pad_height or pad_width:
        pad_mode = "reflect" if height > 1 and width > 1 else "edge"
        image = np.pad(
            image,
            ((0, pad_height), (0, pad_width), (0, 0)),
            mode=pad_mode,
        )

    padded_height, padded_width = image.shape[:2]
    stride = tile_size - overlap

    def origins(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, length - tile_size + 1, stride))
        final = length - tile_size
        if values[-1] != final:
            values.append(final)
        return values

    axis_weight = np.ones(tile_size, dtype=np.float32)
    if overlap:
        ramp = (
            np.sin(np.linspace(0.0, math.pi / 2.0, overlap + 2, dtype=np.float32)[1:-1])
            ** 2
        )
        axis_weight[:overlap] = ramp
        axis_weight[-overlap:] = ramp[::-1]
    axis_weight = np.maximum(axis_weight, np.float32(1e-3))
    weight = axis_weight[:, None] * axis_weight[None, :]

    accumulated = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    normalizer = np.zeros((padded_height, padded_width, 1), dtype=np.float32)
    for top in origins(padded_height):
        for left in origins(padded_width):
            tile = image[top : top + tile_size, left : left + tile_size]
            prediction = np.asarray(predict_tile(tile), dtype=np.float32)
            if prediction.shape != (tile_size, tile_size, 3):
                raise ValueError(
                    "Tile predictor must preserve spatial shape; expected "
                    f"{(tile_size, tile_size, 3)}, got {prediction.shape}"
                )
            accumulated[top : top + tile_size, left : left + tile_size] += (
                prediction * weight[..., None]
            )
            normalizer[top : top + tile_size, left : left + tile_size] += weight[
                ..., None
            ]

    blended = accumulated / normalizer
    return np.clip(blended[:height, :width], 0.0, 1.0)
