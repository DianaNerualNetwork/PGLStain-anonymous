"""Offline auxiliary-image preprocessors."""

from .base import (
    Preprocessor,
    PreprocessorRegistry,
    run_preprocessors,
)

try:
    from .dab import DABPreprocessor
except ImportError:  # pragma: no cover
    DABPreprocessor = None  # type: ignore[misc, assignment]

try:
    from .nuclei import NucleiPreprocessor
except ImportError:  # pragma: no cover
    NucleiPreprocessor = None  # type: ignore[misc, assignment]

__all__ = [
    "DABPreprocessor",
    "NucleiPreprocessor",
    "Preprocessor",
    "PreprocessorRegistry",
    "run_preprocessors",
]
