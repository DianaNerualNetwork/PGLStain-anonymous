"""YAML config loading for ``puzzlestain-eval``.

Evaluation is configured by the YAML files under
``puzzlestain/configs/evaluate/``:

- ``default.yaml`` holds the base defaults (including the ``pathfid``
  foundation-model settings);
- the family presets (``image-quality``, ``perception``,
  ``pathological-relevance``, ``pathfid``, ``all``) only select which
  metric families run;
- a user-supplied YAML file overrides the bundled defaults.

Merge order (lowest to highest priority)::

    default.yaml  <  preset / custom YAML  <  command-line dotlist overrides
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .metrics import FAMILY_REGISTRY, _normalize_per_model

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "evaluate"

FOUNDATION_MODELS = ("conch", "uni", "univ2-h", "univ2", "stainnet", "inception")

_REQUIRED_FIELDS = ("dataset", "stain", "model")


class EvalConfigError(ValueError):
    """Raised when the eval config is invalid or incomplete."""


def available_configs() -> list[str]:
    """Return the names of the bundled config presets."""
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def resolve_config_path(name_or_path: str) -> Path:
    """Resolve a config name or filesystem path to a YAML file.

    Args:
        name_or_path: Bundled preset name (e.g. ``"pathfid"``) or a path
            to a YAML file on disk.

    Returns:
        Path to the resolved YAML file.

    Raises:
        EvalConfigError: If neither a file nor a bundled preset matches.
    """
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    bundled = CONFIG_DIR / f"{name_or_path}.yaml"
    if bundled.is_file():
        return bundled
    raise EvalConfigError(
        f"Unknown eval config: {name_or_path!r}. "
        f"Available presets: {', '.join(available_configs())}; "
        f"or pass a path to a YAML file."
    )


def load_eval_config(name_or_path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load and validate the evaluation config.

    Args:
        name_or_path: Bundled preset name or path to a custom YAML file.
        overrides: Hydra-style dotlist, e.g. ``["dataset=x", "pathfid.hf_id=foo/bar"]``.

    Returns:
        The merged, validated config.

    Raises:
        EvalConfigError: On unknown config, missing required fields, or
            invalid values.
    """
    base = OmegaConf.load(CONFIG_DIR / "default.yaml")
    cfg_path = resolve_config_path(name_or_path)
    if cfg_path.name != "default.yaml":
        base = OmegaConf.merge(base, OmegaConf.load(cfg_path))
    if overrides:
        try:
            base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
        except Exception as e:
            raise EvalConfigError(f"Invalid override {overrides}: {e}") from e
    _validate(base)
    return base


def _validate(cfg: DictConfig) -> None:
    """Check required fields and value ranges."""
    missing = [f for f in _REQUIRED_FIELDS if OmegaConf.is_missing(cfg, f)]
    if missing:
        raise EvalConfigError(
            f"Missing required field(s): {', '.join(missing)}. "
            f"Set them via overrides, e.g. "
            f"'puzzlestain-eval pathfid dataset=... stain=... model=...'."
        )

    families = list(cfg.families or [])
    unknown = [f for f in families if f not in FAMILY_REGISTRY]
    if not families or unknown:
        raise EvalConfigError(
            f"Invalid families: {families}. "
            f"Valid values: {', '.join(sorted(FAMILY_REGISTRY))}."
        )

    pathfid = cfg.pathfid
    if "foundation_model" in pathfid:
        raise EvalConfigError(
            "pathfid.foundation_model was renamed to pathfid.foundation_models "
            "(a list), e.g. 'pathfid.foundation_models=[conch,uni]'."
        )
    pf = OmegaConf.to_container(pathfid, resolve=True)
    models = list(pf.get("foundation_models") or [])
    invalid = [m for m in models if m not in FOUNDATION_MODELS]
    if not models or invalid:
        raise EvalConfigError(
            f"Invalid pathfid.foundation_models: {models}. "
            f"Valid values: {', '.join(FOUNDATION_MODELS)}."
        )
    for field in ("checkpoint", "hf_id"):
        try:
            _normalize_per_model(pf.get(field), models, field)
        except ValueError as e:
            raise EvalConfigError(str(e)) from e
