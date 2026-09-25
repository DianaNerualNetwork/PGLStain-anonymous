"""Unified protocol for reading single patch images."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


class ImageBackend(abc.ABC):
    """Abstract base for patch image readers."""

    @abc.abstractmethod
    def read(self, path: PathLike) -> np.ndarray:
        """Read an image as a channels-last ``(H, W, C)`` numpy array.

        Grayscale images are expanded to ``(H, W, 1)``; color images are
        converted to RGB ``(H, W, 3)``.
        """
        ...


class BackendFactory:
    """Registry-backed factory for patch image backends."""

    _registry: dict[str, type[ImageBackend]] = {}

    @classmethod
    def register(cls, name: str, backend_cls: type[ImageBackend]) -> None:
        cls._registry[name.lower()] = backend_cls

    @classmethod
    def create(cls, name: str) -> ImageBackend:
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(
                f"Unknown backend: {name}. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name]()
