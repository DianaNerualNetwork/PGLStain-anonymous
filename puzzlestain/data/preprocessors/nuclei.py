"""Nuclei map extraction from IHC or H&E images.

Ported from TDKStain's ``get_nuclei_map.py``. The preprocessor detects nuclei
using the hematoxylin channel (and optionally DAB-stained regions), renders them
as a centroid density map, and returns a single-channel ``nuclei_map``.

Cellpose is optional at import time; the runner only needs it when this
preprocessor is actually requested.
"""

from __future__ import annotations

import numpy as np

from .base import Preprocessor
from .dab import DABPreprocessor


def _cuda_available() -> bool:
    """Return True when a CUDA device is available for Cellpose (lazy torch)."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _make_cellpose(model_type: str):
    """Instantiate a Cellpose segmentation model across cellpose versions.

    TDKStain pins cellpose 2.2.3 (``models.Cellpose``); cellpose 3+/4+
    renamed it to ``models.CellposeModel``. The ``nuclei``/``cyto2`` model
    weights only exist in cellpose 2.x, so on newer majors without them a
    clear error is raised instead of an obscure download/lookup failure.
    """
    from cellpose import models

    cls = getattr(models, "Cellpose", None) or getattr(models, "CellposeModel")
    try:
        return cls(model_type=model_type, gpu=_cuda_available())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Cellpose model {model_type!r}. TDKStain's "
            "nuclei/cyto2 weights ship with cellpose 2.2.x; install "
            "'cellpose<3' to reproduce the original preprocessing."
        ) from exc


class NucleiPreprocessor(Preprocessor):
    """Generate a nuclei density map from a pathology patch.

    Args:
        use_dab: Also segment nuclei in the DAB-stained regions and merge them
            with the hematoxylin nuclei (matches TDKStain's behavior: the
            cyto2 Cellpose model runs on the blurred DAB mask produced by
            :class:`DABPreprocessor`).
        sigma: Gaussian sigma for smoothing the centroid density map.
        cellpose_diameter: Approximate nuclei diameter passed to Cellpose.
            ``None`` (the default) keeps Cellpose's per-model default
            (``diam_mean``: 17 for ``nuclei``, 30 for ``cyto2``), matching
            TDKStain, which passes no diameter.
    """

    def __init__(
        self,
        use_dab: bool = True,
        sigma: float = 11.0,
        cellpose_diameter: int | None = None,
    ) -> None:
        self.use_dab = use_dab
        self.sigma = sigma
        self.cellpose_diameter = cellpose_diameter

    @property
    def name(self) -> str:
        return "nuclei"

    @property
    def output_columns(self) -> list[str]:
        return ["nuclei_map"]

    def process(self, image: np.ndarray) -> dict[str, np.ndarray]:
        from skimage import color

        rgb = image.astype(np.float32) / 255.0 if image.max() > 1 else image.astype(np.float32)
        hed = color.rgb2hed(rgb)
        hematoxylin = hed[:, :, 0]

        h_nuclei = self._get_h_nuclei(hematoxylin)
        nuclei = list(h_nuclei)

        if self.use_dab:
            # TDKStain runs the cyto2 model on the blurred DAB mask from the
            # get_dab_mask step, not on an Otsu-thresholded DAB channel.
            dab_mask = DABPreprocessor().process(image)["dab_mask"]
            dab_nuclei = self._get_dab_nuclei(dab_mask)
            nuclei.extend(dab_nuclei)

        nuclei_map = self._nuclei_to_map(rgb.shape[:2], nuclei)
        return {"nuclei_map": nuclei_map}

    def _get_h_nuclei(self, hematoxylin: np.ndarray) -> list[tuple[int, int]]:
        from skimage import exposure, filters, morphology

        h_scaled = exposure.rescale_intensity(
            hematoxylin,
            in_range=(hematoxylin.min(), np.percentile(hematoxylin, 99)),
            out_range=(0, 255),
        ).astype(np.uint8)

        threshold = filters.threshold_otsu(h_scaled)
        h_seg = (h_scaled > threshold).astype(np.uint8) * 255

        h_seg = (
            morphology.remove_small_holes(h_seg.astype(bool), area_threshold=400)
            * 255
        ).astype(np.uint8)
        h_seg = morphology.dilation(h_seg, morphology.disk(1)).astype(np.uint8)

        model = _make_cellpose("nuclei")
        masks, _, _, _ = model.eval(
            h_seg,
            channels=[0, 0],
            diameter=self.cellpose_diameter,
        )
        return self._mask_to_nuclei(masks)

    def _get_dab_nuclei(self, dab_mask: np.ndarray) -> list[tuple[int, int]]:
        # TDKStain feeds the blurred DAB mask directly to cyto2 (grayscale).
        model = _make_cellpose("cyto2")
        masks, _, _, _ = model.eval(
            dab_mask,
            channels=[0, 0],
            diameter=self.cellpose_diameter,
        )
        return self._mask_to_nuclei(masks)

    @staticmethod
    def _mask_to_nuclei(mask: np.ndarray) -> list[tuple[int, int]]:
        from skimage import measure

        labels = measure.label(mask)
        return [
            (int(prop.centroid[1]), int(prop.centroid[0]))
            for prop in measure.regionprops(labels)
        ]

    def _nuclei_to_map(
        self, shape: tuple[int, int], nuclei: list[tuple[int, int]]
    ) -> np.ndarray:
        from skimage import filters

        canvas = np.zeros(shape, dtype=np.float32)
        for x, y in nuclei:
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                canvas[y, x] += 1.0

        canvas = filters.gaussian(canvas, sigma=self.sigma)
        if canvas.max() > 0:
            canvas = (canvas - canvas.min()) / (canvas.max() - canvas.min())
        return (canvas * 255).astype(np.uint8)
