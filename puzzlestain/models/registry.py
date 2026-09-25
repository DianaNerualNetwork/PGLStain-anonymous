"""Global model registry: registers model_cls + processor_cls + strategy_cls together.

This registry pairs each model name with its model, processor, and training
strategy classes. Looking them up under one short name guarantees that the
pre/post-processing logic and training dynamics stay matched to the model they
were written for.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

from .base import StainModel, StainProcessor
from .strategy.base import TrainingStrategy


class ModelRegistry:
    """Registry that pairs each model name with its model, processor, and strategy.

    A model, its :class:`StainProcessor`, and its :class:`TrainingStrategy` are
    always registered and looked up together under one short name.

    Attributes:
        _registry: Maps name to a dict with keys ``"model"``, ``"processor"``,
            and ``"strategy"``. Class-level, populated at import time by model
            subpackages.
    """

    _registry: dict[str, dict[str, type]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_cls: type,
        processor_cls: type,
        strategy_cls: Optional[type] = None,
    ) -> None:
        """Register a model, processor, and optional strategy under a short name.

        Args:
            name: Lookup key used by config files and the build methods.
            model_cls: Concrete :class:`StainModel` subclass.
            processor_cls: Matching :class:`StainProcessor` subclass.
            strategy_cls: Matching :class:`TrainingStrategy` subclass. If None,
                :meth:`build_strategy` will raise for this name until a strategy
                is registered.
        """
        cls._registry[name] = {
            "model": model_cls,
            "processor": processor_cls,
            "strategy": strategy_cls,
        }

    @classmethod
    def build_model(cls, name: str, **kwargs: Any) -> StainModel:
        """Instantiate the model registered under ``name``.

        Args:
            name: Registered model name.
            **kwargs: Forwarded to the model constructor after filtering.

        Returns:
            StainModel: A newly constructed model instance.

        Raises:
            ValueError: If ``name`` was never registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown model: {name}. Registered: {list(cls._registry.keys())}"
            )
        model_cls = cls._registry[name]["model"]
        filtered = cls._filter_kwargs(model_cls, kwargs)
        return model_cls(**filtered)

    @classmethod
    def build_processor(cls, name: str, **kwargs: Any) -> StainProcessor:
        """Instantiate the processor registered under ``name``.

        Callers may pass a *superset* of kwargs uniformly across all models:
        kwargs the target processor's ``__init__`` does not accept are dropped
        here via signature introspection, so a processor that doesn't take them
        is unaffected. Processors whose ``__init__`` declares ``**kwargs``
        receive every kwarg unfiltered.

        Args:
            name: Registered model name (same key used for the model).
            **kwargs: Forwarded to the processor constructor after filtering.

        Returns:
            StainProcessor: A newly constructed processor instance.

        Raises:
            ValueError: If ``name`` was never registered.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown model: {name}. Registered: {list(cls._registry.keys())}"
            )
        processor_cls = cls._registry[name]["processor"]
        filtered = cls._filter_kwargs(processor_cls, kwargs)
        return processor_cls(**filtered)

    @classmethod
    def build_strategy(cls, name: str, **kwargs: Any) -> TrainingStrategy:
        """Instantiate the strategy registered under ``name``.

        Args:
            name: Registered model name (same key used for the model).
            **kwargs: Forwarded to the strategy constructor after filtering.

        Returns:
            TrainingStrategy: A newly constructed strategy instance.

        Raises:
            ValueError: If ``name`` was never registered or no strategy class
            was associated with it.
        """
        if name not in cls._registry:
            raise ValueError(
                f"Unknown model: {name}. Registered: {list(cls._registry.keys())}"
            )
        strategy_cls = cls._registry[name].get("strategy")
        if strategy_cls is None:
            raise ValueError(
                f"Model '{name}' has no registered strategy. "
                f"Registered models: {list(cls._registry.keys())}"
            )
        filtered = cls._filter_kwargs(strategy_cls, kwargs)
        return strategy_cls(**filtered)

    @classmethod
    def list_models(cls) -> list[str]:
        """Return the names of all registered models.

        Returns:
            list[str]: Registered keys.
        """
        return list(cls._registry.keys())

    @classmethod
    def has_strategy(cls, name: str) -> bool:
        """Return whether a strategy class is registered for ``name``."""
        return name in cls._registry and cls._registry[name].get("strategy") is not None

    @classmethod
    def _filter_kwargs(cls, target_cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs that the target constructor cannot accept.

        If the constructor declares ``**kwargs``, all kwargs are passed through.
        """
        sig = inspect.signature(target_cls.__init__)
        params = sig.parameters.values()
        accepts_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params
        )
        if accepts_var_kwargs:
            return kwargs
        accepted = {p.name for p in params}
        return {k: v for k, v in kwargs.items() if k in accepted}
