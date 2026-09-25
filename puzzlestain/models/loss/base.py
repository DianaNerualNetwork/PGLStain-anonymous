"""Loss function protocol, CompositeLoss, and LossRegistry for virtual staining."""

from __future__ import annotations

import abc
from typing import Any

import torch


class StainLoss(abc.ABC):
    """Unified protocol for image-to-image losses.

    Unlike segmentation losses, image-to-image losses take a variety of input
    shapes (raw tensors, boolean flags for GAN targets, paired images, etc.).
    Each concrete loss therefore receives its inputs as keyword arguments and
    returns a scalar reduced tensor.
    """

    @abc.abstractmethod
    def __call__(self, **kwargs: Any) -> torch.Tensor:
        """Compute the loss.

        Args:
            **kwargs: Loss-specific inputs (e.g. ``pred``, ``target``,
                ``prediction``, ``target_is_real``).

        Returns:
            Scalar loss tensor, already reduced (default: mean).
        """
        ...

    @abc.abstractmethod
    def name(self) -> str:
        """Return the short loss name used as a logging key."""
        ...

    @abc.abstractmethod
    def required_kwargs(self) -> set[str]:
        """Return the keyword argument names this loss expects.

        :class:`CompositeLoss` uses this to forward only the relevant subset of
        kwargs to each component.
        """
        ...


class CompositeLoss(StainLoss):
    """Weighted sum of several independent stain losses.

    Each component loss receives only the keyword arguments it declares via
    :meth:`StainLoss.required_kwargs`. This allows GAN losses (which need
    ``prediction`` and ``target_is_real``) and pixel losses (which need ``pred``
    and ``target``) to coexist in one composite.

    Args:
        losses: Ordered list of ``(loss_fn, weight)`` pairs.

    Attributes:
        losses: The configured ``(loss_fn, weight)`` pairs.
        _last_breakdown: Maps ``"loss/<name>"`` -> float value from the most
            recent ``__call__`` (plus ``"loss/total"``).
    """

    def __init__(self, losses: list[tuple[StainLoss, float]]):
        self.losses = losses
        self._last_breakdown: dict[str, float] = {}

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        """Evaluate all component losses and return their weighted sum."""
        device = self._get_device(kwargs)
        total = torch.tensor(0.0, device=device)
        self._last_breakdown = {}

        for loss_fn, weight in self.losses:
            required = loss_fn.required_kwargs()
            filtered = {k: v for k, v in kwargs.items() if k in required}
            if set(filtered) != required:
                # Not all required arguments are present; skip this component.
                # This allows a single CompositeLoss to be reused across different
                # call sites (e.g. GAN loss vs. pixel loss) in a training step.
                continue
            val = loss_fn(**filtered)
            self._last_breakdown[f"loss/{loss_fn.name()}"] = val.item()
            total = total + weight * val

        self._last_breakdown["loss/total"] = total.item()
        return total

    def get_last_breakdown(self) -> dict[str, float]:
        """Return per-component loss values from the most recent call."""
        return self._last_breakdown

    def name(self) -> str:
        """Return the logging name of the composite loss."""
        return "composite"

    def required_kwargs(self) -> set[str]:
        """Return the union of all component required kwargs."""
        keys: set[str] = set()
        for loss_fn, _ in self.losses:
            keys |= loss_fn.required_kwargs()
        return keys

    @staticmethod
    def _get_device(kwargs: dict[str, Any]) -> torch.device:
        """Infer the device from the first tensor in kwargs."""
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")


class LossRegistry:
    """Registry for building losses (and composites) from string names."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, loss_cls: type) -> None:
        """Register a loss class under a short name."""
        cls._registry[name] = loss_cls

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> StainLoss:
        """Instantiate a single registered loss."""
        if name not in cls._registry:
            raise ValueError(
                f"Unknown loss: {name}. Registered: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def build_composite(cls, config: dict[str, Any]) -> CompositeLoss:
        """Build a :class:`CompositeLoss` from a name -> params config mapping.

        Args:
            config: Maps loss name -> kwargs dict. Each entry's optional
                ``"weight"`` key (default ``1.0``) becomes the composite weight;
                the remaining kwargs are passed to the loss constructor.
        """
        losses = []
        for name, params in config.items():
            params = dict(params)
            weight = params.pop("weight", 1.0)
            loss_fn = cls.build(name, **params)
            losses.append((loss_fn, weight))
        return CompositeLoss(losses)


__all__ = ["StainLoss", "CompositeLoss", "LossRegistry"]
