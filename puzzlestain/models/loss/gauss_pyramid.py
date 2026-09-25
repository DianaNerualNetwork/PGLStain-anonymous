"""Gaussian pyramid convolution module from the ASP repository.

Original source: https://github.com/csjliang/LPTN
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Gauss_Pyramid_Conv(nn.Module):
    """Build a Gaussian pyramid with learnable/convolutional downsampling.

    Args:
        num_high: Number of high-resolution pyramid levels to produce before
            the final low-resolution residual.
    """

    def __init__(self, num_high: int = 5) -> None:
        super().__init__()
        self.num_high = num_high
        kernel = self.gauss_kernel()
        self.register_buffer("kernel", kernel)

    def gauss_kernel(
        self,
        device: torch.device | None = None,
        channels: int = 3,
    ) -> torch.Tensor:
        """Return a 5x5 Gaussian kernel repeated for ``channels``."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kernel = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )
        kernel /= 256.0
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel.to(device)

    def downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample by a factor of two in each spatial dimension."""
        return x[:, :, ::2, ::2]

    def conv_gauss(self, img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian kernel with reflective padding."""
        img = F.pad(img, (2, 2, 2, 2), mode="reflect")
        out = F.conv2d(img, kernel, groups=img.shape[1])
        return out

    def forward(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Build the pyramid.

        Returns:
            A list of ``num_high + 1`` tensors. The first ``num_high`` entries
            are blurred versions of the current octave; the last entry is the
            final downsampled residual.
        """
        current = img
        pyr: list[torch.Tensor] = []
        kernel = self.kernel
        if kernel.device != img.device:
            kernel = kernel.to(img.device)
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, kernel)
            pyr.append(filtered)
            current = self.downsample(filtered)
        pyr.append(current)
        return pyr


__all__ = ["Gauss_Pyramid_Conv"]
