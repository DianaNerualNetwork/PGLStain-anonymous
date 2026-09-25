"""Torchvision-based synchronized spatial transforms.

Adapted from ASP/PGVMS ``AlignedDataset``: the same random seed is applied to
all image fields of a ``StainSample`` (source, target, and auxiliary images) so
that spatial augmentation stays aligned.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from ..fields import iter_image_field_names
from ..sample import StainSample
from .base import Transform


def _to_pil(img: Any):
    """Convert a numpy array/tensor to a PIL Image for torchvision transforms."""
    from PIL import Image

    if isinstance(img, torch.Tensor):
        return img
    if isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 1:
        return Image.fromarray(arr.squeeze(2), mode="L")
    return Image.fromarray(arr)


def _capture_rng_state() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None]:
    """Capture RNGs a generic transform may use."""
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        cuda_state,
    )


def _restore_rng_state(
    state: tuple[Any, Any, torch.Tensor, list[torch.Tensor] | None],
) -> None:
    """Restore Python, NumPy, CPU Torch, and initialized CUDA RNGs."""
    python_state, numpy_state, torch_state, cuda_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _present_image_fields(sample: StainSample) -> list[str]:
    """Return registered image fields whose values are present."""
    return [
        name
        for name in iter_image_field_names(sample)
        if getattr(sample, name) is not None
    ]


class SynchronizedTorchvisionTransform(Transform):
    """Apply a torchvision transform to every image field with the same random state.


    Args:
        transform: A callable that accepts a PIL Image or tensor and returns a
            transformed tensor/image. It is applied once per registered image
            field, re-seeding the RNG each time so source, target, and auxiliary
            images receive identical geometric operations.
    """

    def __init__(self, transform: Callable):
        self.transform = transform

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = _present_image_fields(sample)
        if not image_fields:
            return sample

        # Replay one captured RNG state for every field, then leave the global
        # RNGs at the state produced by a single transform invocation. This
        # gives every paired field identical random parameters without making
        # later samples depend on how many image fields this sample contains.
        initial_state = _capture_rng_state()
        advanced_state = initial_state
        updates: dict[str, Any] = {}
        try:
            for index, name in enumerate(image_fields):
                _restore_rng_state(initial_state)
                updates[name] = self.transform(_to_pil(getattr(sample, name)))
                if index == 0:
                    advanced_state = _capture_rng_state()
        finally:
            _restore_rng_state(advanced_state)

        return sample.replace(**updates)


class SynchronizedRandomCrop(Transform):
    """Crop all image fields at the same random location.

    Uses Python's ``random`` module (matching the original PyramidPix2pix
    ``get_params`` implementation) rather than PyTorch's RNG, so that the
    same global seed produces identical crop coordinates in both codebases.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = _present_image_fields(sample)
        if not image_fields:
            return sample

        # Pick one crop position using Python random, exactly like
        # PyramidPix2pix: random.randint(0, max(0, width - crop_size)).
        first = _to_pil(getattr(sample, image_fields[0]))
        w, h = first.size
        max_x = max(0, w - self.size)
        max_y = max(0, h - self.size)
        left = random.randint(0, max_x)
        top = random.randint(0, max_y)

        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            updates[name] = TF.crop(_to_pil(img), top, left, self.size, self.size)
        return sample.replace(**updates)


class SynchronizedRandomHorizontalFlip(Transform):
    """Horizontally flip all image fields with the same random decision.

    ``p`` is the probability that the flip is applied. One Python RNG draw
    is shared by every paired image field.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = _present_image_fields(sample)
        if not image_fields:
            return sample

        flip = random.random() < self.p
        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            pil_img = _to_pil(img)
            updates[name] = TF.hflip(pil_img) if flip else pil_img
        return sample.replace(**updates)


class SynchronizedRandomVerticalFlip(Transform):
    """Vertically flip all image fields with the same random decision.

    ``p`` is the probability that all paired fields are flipped.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = _present_image_fields(sample)
        if not image_fields:
            return sample

        flip = random.random() < self.p
        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            pil_img = _to_pil(img)
            updates[name] = TF.vflip(pil_img) if flip else pil_img
        return sample.replace(**updates)


class SynchronizedRandomRot90(Transform):
    """Rotate all image fields by the same random multiple of 90 degrees.

    Mirrors the UNIStainNet ``AlignedDataset`` rot90 augmentation: with
    probability ``p``, rotate by ``k * 90`` degrees with
    ``k = random.choice([1, 2, 3])``.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = _present_image_fields(sample)
        if not image_fields:
            return sample

        rotate = random.random() < self.p
        k = random.choice([1, 2, 3]) if rotate else 0
        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            pil_img = _to_pil(img)
            updates[name] = TF.rotate(pil_img, k * 90) if rotate else pil_img
        return sample.replace(**updates)


class SynchronizedRandomAffine(Transform):
    """Apply the same random affine transform to all image fields.

    Mirrors the UNIStainNet ``AlignedDataset`` affine augmentation: with
    probability ``p`` (``random.random() > 1 - p``), sample
    ``angle ~ U(-degrees, degrees)``, ``translate ~ U(-translate, translate)``
    times the per-axis image size, and ``scale ~ U(*scale_range)`` — once per
    sample — applied with bilinear interpolation and zero shear.
    """

    def __init__(
        self,
        p: float = 0.3,
        degrees: float = 15.0,
        translate: float = 0.05,
        scale_range: tuple[float, float] = (0.9, 1.1),
    ) -> None:
        self.p = p
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = list(iter_image_field_names(sample))
        if not image_fields:
            return sample

        apply = random.random() > (1.0 - self.p)
        if apply:
            first = _to_pil(getattr(sample, image_fields[0]))
            w, h = first.size
            angle = random.uniform(-self.degrees, self.degrees)
            translate = [
                random.uniform(-self.translate, self.translate) * w,
                random.uniform(-self.translate, self.translate) * h,
            ]
            scale = random.uniform(*self.scale_range)

        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            pil_img = _to_pil(img)
            if not apply:
                updates[name] = pil_img
                continue
            updates[name] = TF.affine(
                pil_img,
                angle,
                translate,
                scale,
                shear=0,
                interpolation=T.InterpolationMode.BILINEAR,
            )
        return sample.replace(**updates)


class SynchronizedScaleWidth(Transform):
    """Resize so that width matches ``target_width`` while preserving aspect ratio.

    The height is scaled proportionally but clamped to be at least
    ``crop_size`` so that a subsequent random crop is always valid. This
    mirrors the ``scale_width_and_crop`` behaviour of the original MIST/ASP
    ``base_dataset.py``.
    """

    def __init__(self, target_width: int, crop_size: int) -> None:
        self.target_width = target_width
        self.crop_size = crop_size

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = list(iter_image_field_names(sample))
        if not image_fields:
            return sample

        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            pil_img = _to_pil(img)
            ow, oh = pil_img.size
            w = self.target_width
            h = int(max(self.target_width * oh / ow, self.crop_size))
            updates[name] = pil_img.resize((w, h), Image.BICUBIC)
        return sample.replace(**updates)


class SynchronizedMakePower2(Transform):
    """Round the spatial dimensions of every image field to a multiple of ``base``.

    Mirrors the ``make_power_2`` post-processing step in MIST/ASP's
    ``base_dataset.py``; useful for architectures that require sizes divisible
    by a power of two.
    """

    def __init__(self, base: int = 4) -> None:
        self.base = base

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = list(iter_image_field_names(sample))
        if not image_fields:
            return sample

        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            pil_img = _to_pil(img)
            ow, oh = pil_img.size
            w = int(round(ow / self.base) * self.base)
            h = int(round(oh / self.base) * self.base)
            if (w, h) != (ow, oh):
                pil_img = pil_img.resize((w, h), Image.BICUBIC)
            updates[name] = pil_img
        return sample.replace(**updates)


class SynchronizedRandomZoom(Transform):
    """Randomly zoom all image fields with the same scale factor.

    The zoom factor is sampled uniformly from ``factor_range`` and applied to
    both spatial dimensions. This mirrors MIST/ASP's ``_random_zoom``.
    """

    def __init__(
        self,
        crop_size: int,
        factor_range: tuple[float, float] = (0.8, 1.0),
    ) -> None:
        self.crop_size = crop_size
        self.factor_range = factor_range

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = list(iter_image_field_names(sample))
        if not image_fields:
            return sample

        fx = float(np.random.uniform(*self.factor_range))
        fy = float(np.random.uniform(*self.factor_range))
        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            pil_img = _to_pil(img)
            iw, ih = pil_img.size
            zoomw = max(self.crop_size, round(iw * fx))
            zoomh = max(self.crop_size, round(ih * fy))
            updates[name] = pil_img.resize((zoomw, zoomh), Image.BICUBIC)
        return sample.replace(**updates)


class SynchronizedRandomPatch(Transform):
    """Extract a random grid patch of ``size`` from all image fields.

    Mirrors MIST/ASP's ``_patch``: the image is divided into an ``nw x nh``
    grid of ``size x size`` patches, one cell is chosen uniformly, and the
    same cell is cropped from every field.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = list(iter_image_field_names(sample))
        if not image_fields:
            return sample

        first = _to_pil(getattr(sample, image_fields[0]))
        ow, oh = first.size
        nw = max(1, ow // self.size)
        nh = max(1, oh // self.size)
        roomx = max(0, ow - nw * self.size)
        roomy = max(0, oh - nh * self.size)
        startx = random.randint(0, int(roomx))
        starty = random.randint(0, int(roomy))
        index = random.randint(0, nw * nh - 1)
        ix = index // nh
        iy = index % nh
        left = startx + ix * self.size
        top = starty + iy * self.size

        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            updates[name] = TF.crop(_to_pil(img), top, left, self.size, self.size)
        return sample.replace(**updates)


class SynchronizedRandomTrim(Transform):
    """Randomly trim all image fields to a square of ``trim_size``.

    Mirrors MIST/ASP's ``_trim``: a random ``trim_size x trim_size`` window
    is chosen and applied identically to every field.
    """

    def __init__(self, trim_size: int) -> None:
        self.trim_size = trim_size

    def __call__(self, sample: StainSample) -> StainSample:
        image_fields = list(iter_image_field_names(sample))
        if not image_fields:
            return sample

        first = _to_pil(getattr(sample, image_fields[0]))
        ow, oh = first.size
        max_x = max(0, ow - self.trim_size)
        max_y = max(0, oh - self.trim_size)
        left = random.randint(0, max_x)
        top = random.randint(0, max_y)

        updates: dict[str, Any] = {}
        for name in image_fields:
            img = getattr(sample, name)
            if img is None:
                continue
            updates[name] = TF.crop(
                _to_pil(img), top, left, self.trim_size, self.trim_size
            )
        return sample.replace(**updates)


class SourceColorJitter(Transform):
    """Apply color jitter to the source image only (staining variability).

    With ``brightness``, ``contrast``, ``saturation`` or ``hue`` configured,
    this delegates to :class:``torchvision.transforms.ColorJitter``, matching
    PSPStain's source-only augmentation (including its randomized operation
    order). Otherwise it preserves the legacy UNIStainNet behavior:
    brightness, contrast and saturation are each adjusted with probability
    ``p`` by a factor drawn from ``U(*factor_range)``, in this order. Target
    and auxiliary images are always left untouched.
    """

    def __init__(
        self,
        p: float = 0.5,
        factor_range: tuple[float, float] = (0.9, 1.1),
        brightness: float | tuple[float, float] | None = None,
        contrast: float | tuple[float, float] | None = None,
        saturation: float | tuple[float, float] | None = None,
        hue: float | tuple[float, float] | None = None,
    ) -> None:
        self.p = p
        self.factor_range = factor_range
        components = (brightness, contrast, saturation, hue)
        self._torchvision_jitter = (
            T.ColorJitter(
                brightness=brightness or 0.0,
                contrast=contrast or 0.0,
                saturation=saturation or 0.0,
                hue=hue or 0.0,
            )
            if any(component is not None for component in components)
            else None
        )

    def __call__(self, sample: StainSample) -> StainSample:
        img = getattr(sample, "source_image", None)
        if img is None:
            return sample

        pil_img = _to_pil(img)
        if self._torchvision_jitter is not None:
            return sample.replace(source_image=self._torchvision_jitter(pil_img))

        if random.random() > self.p:
            pil_img = TF.adjust_brightness(pil_img, random.uniform(*self.factor_range))
        if random.random() > self.p:
            pil_img = TF.adjust_contrast(pil_img, random.uniform(*self.factor_range))
        if random.random() > self.p:
            pil_img = TF.adjust_saturation(pil_img, random.uniform(*self.factor_range))
        return sample.replace(source_image=pil_img)


class ToTensorNormalize(Transform):
    """Convert all image fields to tensors and optionally normalize.

    This produces the HistDiST-style output: ``(C, H, W)`` float tensors in
    ``[-1, 1]`` when ``scale_to_minus_one_one=True``, or ``[0, 1]`` otherwise.
    """

    def __init__(self, scale_to_minus_one_one: bool = False):
        self.scale_to_minus_one_one = scale_to_minus_one_one

    def __call__(self, sample: StainSample) -> StainSample:
        def _convert(img: Any) -> torch.Tensor:
            if isinstance(img, torch.Tensor):
                tensor = img
            else:
                tensor = TF.to_tensor(img)
            if self.scale_to_minus_one_one:
                tensor = 2.0 * tensor - 1.0
            return tensor

        updates = {
            name: _convert(getattr(sample, name))
            for name in iter_image_field_names(sample)
            if getattr(sample, name) is not None
        }
        return sample.replace(**updates)
