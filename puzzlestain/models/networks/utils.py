"""Network building utilities adapted from PyramidPix2pix / pix2pix.

Original source: https://github.com/bupt-ai-cz/BCI (PyramidPix2pix)
Licensed under the project license; see ATTRIBUTION.md.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
from torch.nn import init


class Identity(nn.Module):
    """Identity mapping module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x


def get_norm_layer(norm_type: str = "instance") -> type[nn.Module] | functools.partial:
    """Return a normalization layer constructor.

    Args:
        norm_type: One of ``batch``, ``instance``, or ``none``.

    Returns:
        A partial ``nn.BatchNorm2d``, ``nn.InstanceNorm2d``, or ``Identity``.
    """
    if norm_type == "batch":
        return functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    if norm_type == "none":
        return Identity
    raise NotImplementedError(f"normalization layer [{norm_type}] is not found")


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02) -> None:
    """Initialize network weights.

    Args:
        net: Network to initialize.
        init_type: Initialization method: ``normal`` | ``xavier`` | ``kaiming``
            | ``orthogonal``.
        init_gain: Scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(
    net: nn.Module,
    init_type: str = "normal",
    init_gain: float = 0.02,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Initialize a network and move it to the target device.

    To reproduce the original PyramidPix2pix / pix2pix training curves, weights
    are initialized on the same device they will run on (CUDA when available).
    PyTorch's ``normal_`` implementation can yield different values on CPU vs
    GPU even with the same seed, so initializing on the target device matters
    for bit-for-bit alignment with the reference repo.

    Note:
        Unlike the original pix2pix code, this helper does **not** wrap the
        network in ``DataParallel``. Distributed wrapping is handled by
        ``accelerate`` in the trainer.

    Args:
        net: Network to initialize.
        init_type: Weight initialization method.
        init_gain: Scaling factor.
        device: Target device. Defaults to ``cuda`` if available, else ``cpu``.

    Returns:
        The initialized network (already on ``device``).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net


__all__ = ["Identity", "get_norm_layer", "init_weights", "init_net"]
