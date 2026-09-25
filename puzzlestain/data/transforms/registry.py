"""Transform registry driven by short names in YAML configs.

YAML lists of transform specs are
resolved into a ``Compose`` pipeline by short name, so switching augmentation
strategies is a config change rather than a code change.
"""

from __future__ import annotations

from typing import Any

import torchvision.transforms as T

from .base import Compose, Transform
from .torchvision_transforms import (
    SourceColorJitter,
    SynchronizedMakePower2,
    SynchronizedRandomAffine,
    SynchronizedRandomCrop,
    SynchronizedRandomHorizontalFlip,
    SynchronizedRandomPatch,
    SynchronizedRandomRot90,
    SynchronizedRandomTrim,
    SynchronizedRandomVerticalFlip,
    SynchronizedRandomZoom,
    SynchronizedScaleWidth,
    SynchronizedTorchvisionTransform,
    ToTensorNormalize,
)


class TransformRegistry:
    """Maps short config names to transform builders for virtual staining.

    Attributes:
        _registry: name -> callable that accepts kwargs and returns a Transform.
    """

    _registry: dict[str, type[Transform] | Any] = {}

    @classmethod
    def register(cls, name: str, builder: Any) -> None:
        """Register a transform builder under a short name.

        Args:
            name: Config-facing short name (must be unique).
            builder: A callable that accepts transform kwargs and returns a
                ``Transform`` instance, or a transform class whose constructor
                accepts those kwargs directly.
        """
        cls._registry[name] = builder

    @classmethod
    def build(cls, name: str, **kwargs) -> Transform:
        """Instantiate a single registered transform by name.

        Args:
            name: Registered short name.
            **kwargs: Constructor arguments forwarded to the transform builder.

        Returns:
            A ready-to-use ``Transform`` instance.

        Raises:
            ValueError: If ``name`` is not registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown transform: {name}. "
                f"Registered: {sorted(cls._registry.keys())}"
            )
        builder = cls._registry[name]
        return builder(**kwargs)

    @classmethod
    def build_compose(cls, transform_configs: list[dict[str, Any]]) -> Compose:
        """Build a ``Compose`` pipeline from a list of config dicts.

        Each dict must contain a ``"name"`` key; remaining keys are forwarded as
        constructor kwargs.

        Args:
            transform_configs: Ordered list of per-transform config dicts.

        Returns:
            A ``Compose`` chaining the built transforms in the given order.
        """
        transforms: list[Transform] = []
        for cfg in transform_configs:
            cfg = dict(cfg)
            name = cfg.pop("name")
            transforms.append(cls.build(name, **cfg))
        return Compose(transforms)

    @classmethod
    def list_transforms(cls) -> list[str]:
        """Return all registered transform names, sorted alphabetically."""
        return sorted(cls._registry.keys())


def _synchronized(name: str, tv_builder: Any) -> None:
    """Register a synchronized torchvision wrapper."""

    def _builder(**kwargs) -> SynchronizedTorchvisionTransform:
        return SynchronizedTorchvisionTransform(tv_builder(**kwargs))

    TransformRegistry.register(name, _builder)


def register_builtin_transforms() -> None:
    """Register all built-in transforms. Call once at framework init.

    Populates the registry with synchronized spatial transforms (resize, crop,
    flip, zoom, patch, trim, power-of-2 rounding) and the ``ToTensorNormalize``
    utility used by virtual-staining pipelines.
    """
    TransformRegistry.register("synchronized_random_crop", SynchronizedRandomCrop)
    _synchronized(
        "synchronized_resize",
        lambda size, interpolation="bicubic": T.Resize(
            size, interpolation=T.InterpolationMode(interpolation)
        ),
    )
    TransformRegistry.register(
        "synchronized_random_horizontal_flip", SynchronizedRandomHorizontalFlip
    )
    TransformRegistry.register(
        "synchronized_random_vertical_flip", SynchronizedRandomVerticalFlip
    )
    TransformRegistry.register("synchronized_random_rot90", SynchronizedRandomRot90)
    TransformRegistry.register("synchronized_random_affine", SynchronizedRandomAffine)
    TransformRegistry.register("source_color_jitter", SourceColorJitter)
    _synchronized(
        "synchronized_random_rotation",
        lambda degrees: T.RandomRotation(degrees),
    )
    TransformRegistry.register("synchronized_scale_width", SynchronizedScaleWidth)
    TransformRegistry.register("synchronized_make_power_2", SynchronizedMakePower2)
    TransformRegistry.register("synchronized_random_zoom", SynchronizedRandomZoom)
    TransformRegistry.register("synchronized_random_patch", SynchronizedRandomPatch)
    TransformRegistry.register("synchronized_random_trim", SynchronizedRandomTrim)
    TransformRegistry.register("to_tensor_normalize", ToTensorNormalize)


__all__ = [
    "TransformRegistry",
    "register_builtin_transforms",
]
