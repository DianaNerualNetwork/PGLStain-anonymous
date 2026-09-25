"""Generated-image replay buffer adapted from pix2pixHD.

Original source: https://github.com/NVIDIA/pix2pixHD
See ATTRIBUTION.md for the retained third-party provenance and license.
"""

from __future__ import annotations

import torch


class ImagePool:
    """Buffer that stores previously generated images, adapted from pix2pixHD.

    When ``pool_size > 0``, the discriminator is trained on images sampled
    from this buffer 50% of the time, which stabilizes training. A size of
    ``0`` disables the buffer and always uses the freshly generated image.
    """

    def __init__(self, pool_size: int) -> None:
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """Return images from the pool, possibly replacing some with ``images``."""
        if self.pool_size == 0:
            return images
        return_images: list[torch.Tensor] = []
        for image in images:
            # Detach before storing, mirroring the original repos'
            # ``torch.unsqueeze(image.data, 0)`` so pooled images never keep
            # the generator's autograd graph alive.
            image = image.detach().unsqueeze(0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                import random

                if random.random() > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, dim=0)
