"""Offline checks for the exported package, method registry, and presets."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import puzzlestain
from puzzlestain.models import (  # noqa: F401: registration side effects
    cut_family,
    cyclegan_family,
    diffusion_family,
    pix2pix_family,
)
from puzzlestain.models.loss.base import LossRegistry
from puzzlestain.models.registry import ModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "puzzlestain" / "configs"
EXPERIMENT_ROOT = CONFIG_ROOT / "experiment"
METHODS = (
    "cut", "cyclegan", "unsb", "pix2pix_pyramid", "asp", "ppt",
    "pspstain", "mdcl", "simgan", "usigan", "m2plgan", "pglstain",
)
PRESETS = sorted(
    path.relative_to(EXPERIMENT_ROOT).with_suffix("").as_posix()
    for path in EXPERIMENT_ROOT.rglob("*.yaml")
)


def test_import_uses_this_release():
    """An installed sibling project must not silently satisfy these tests."""
    assert Path(puzzlestain.__file__).resolve() == (
        PROJECT_ROOT / "puzzlestain" / "__init__.py"
    ).resolve()


@pytest.mark.parametrize("method", METHODS)
def test_requested_method_has_training_strategy(method):
    assert method in ModelRegistry.list_models()
    assert ModelRegistry.has_strategy(method)


def test_presets_are_present():
    assert PRESETS, "The export must include runnable experiment presets"
    assert "mist/pglstain/full" in PRESETS


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_composes_with_registered_components(preset, tmp_path):
    """Resolve configuration only; do not read datasets or construct weights."""
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment={preset}",
                f"data.dataroot={tmp_path.as_posix()}",
                "data.selected_stains=[ER]",
            ],
        )
        resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert resolved["model"]["name"] in ModelRegistry.list_models()
    assert ModelRegistry.has_strategy(resolved["model"]["name"])
    assert resolved["loss"]["components"], f"No losses configured for {preset}"
    for name in resolved["loss"]["components"]:
        assert name in LossRegistry._registry, f"{preset}: unregistered loss {name}"


def test_training_and_inference_entry_points_import():
    import puzzlestain.inference.predictor  # noqa: F401
    import puzzlestain.train  # noqa: F401
