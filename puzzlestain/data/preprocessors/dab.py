"""DAB channel and DAB mask extraction.

Ported from TDKStain's ``get_dab_mask.py``. The preprocessor separates the DAB
chromogen from an IHC RGB patch and produces a binary/soft mask of the stained
regions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Preprocessor


class DABPreprocessor(Preprocessor):
    """Extract DAB RGB channel and DAB saturation mask from IHC images.

    Args:
        threshold: Saturation threshold used to binarize the DAB mask.
        blur: Whether to apply a small Gaussian blur to the mask.
    """

    def __init__(self, threshold: float = 0.15, blur: bool = True) -> None:
        self.threshold = threshold
        self.blur = blur

    @property
    def name(self) -> str:
        return "dab"

    @property
    def output_columns(self) -> list[str]:
        return ["dab_image", "dab_mask"]

    def process(self, image: np.ndarray) -> dict[str, np.ndarray]:
        from skimage import color

        # Expect channels-last RGB in [0, 255].
        rgb = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)

        hed = color.rgb2hed(rgb)
        dab = hed[:, :, 2]

        # DAB-only RGB for visualization and downstream use.
        null_channel = np.zeros_like(dab)
        dab_rgb = color.hed2rgb(np.stack([null_channel, null_channel, dab], axis=-1))
        dab_rgb = (np.clip(dab_rgb, 0, 1) * 255).astype(np.uint8)

        # Saturation-based mask.
        dab_hsv = color.rgb2hsv(dab_rgb.astype(np.float32) / 255.0)
        dab_s = dab_hsv[:, :, 1]
        mask = (dab_s > self.threshold).astype(np.uint8) * 255

        if self.blur:
            # TDKStain's get_dab_mask.py uses cv2.GaussianBlur with a 9x9
            # kernel and sigma 3 on the uint8 mask (soft edges).
            import cv2

            mask = cv2.GaussianBlur(mask, (9, 9), sigmaX=3, sigmaY=3)

        return {"dab_image": dab_rgb, "dab_mask": mask}
