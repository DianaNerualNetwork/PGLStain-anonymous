"""Patch image reading backends."""

from .base import BackendFactory, ImageBackend, PathLike
from .pil_backend import PILBackend

BackendFactory.register("pil", PILBackend)
BackendFactory.register("pillow", PILBackend)
BackendFactory.register("default", PILBackend)

__all__ = ["BackendFactory", "ImageBackend", "PathLike", "PILBackend"]
