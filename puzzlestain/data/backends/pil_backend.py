"""PIL-based patch image reader."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import ImageBackend, PathLike

#: Single-band PIL modes preserved as single-channel ``(H, W, 1)`` arrays.
#: Palette ("P") is single-band too, but encodes color, so it is not included.
_GRAYSCALE_MODES = ("1", "L", "I", "I;16", "F")


class PILBackend(ImageBackend):
    """Read images with Pillow, always returning channels-last arrays.

    Grayscale images (e.g. auxiliary masks saved as single-channel PNGs) are
    preserved as ``(H, W, 1)``; all other images are converted to RGB
    ``(H, W, 3)``.
    """

    def read(self, path: PathLike) -> np.ndarray:
        img = Image.open(str(path))
        if img.mode in _GRAYSCALE_MODES:
            return np.asarray(img)[..., None]
        return np.asarray(img.convert("RGB"))
