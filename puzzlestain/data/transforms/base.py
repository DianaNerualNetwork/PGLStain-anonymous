"""Unified transform protocol and Compose for virtual staining."""

from __future__ import annotations

import abc
from dataclasses import dataclass

from ..sample import StainSample


class Transform(abc.ABC):
    """Base class for data augmentation.

    Conventions:
      - ``__call__`` receives a ``StainSample`` and returns a new one.
      - Spatial transforms must apply the same geometric operation to both
        ``he_image`` and ``target_image``.
      - Color/intensity transforms only affect the appropriate image.
    """

    @abc.abstractmethod
    def __call__(self, sample: StainSample) -> StainSample: ...


@dataclass
class Compose:
    """Chain multiple transforms sequentially."""

    transforms: list[Transform]

    def __call__(self, sample: StainSample) -> StainSample:
        for t in self.transforms:
            sample = t(sample)
        return sample
