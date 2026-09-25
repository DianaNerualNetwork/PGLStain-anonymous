"""Offline export checks using only synthetic checkpoint tensors."""

from collections import OrderedDict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_inference_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


@pytest.fixture
def source_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "model": {
            "name": "od_guided_higraph_v3_softgate_odset_odugw",
            "normalize_to_minus_one_one": True,
            "params": {
                "netG": "resnet_6blocks", "ngf": 32, "normG": "instance",
                "weight_norm": "spectral", "no_dropout": True,
                "lambda_gbclm": 0.2, "gbclm_layers": "8,12,16",
                "norm": "batch", "use_dropout": False,
            },
        },
        "transforms": {
            "train_transforms": [
                {"name": "synchronized_resize", "size": [512, 512], "interpolation": "bicubic"},
                {"name": "synchronized_random_crop", "size": 512},
            ]
        },
        "data": {"target_mpp": 0.5, "dataroot": "/private/example-data"},
        "training": {"name": "private-run"},
        "logging": {"owner": "private@example.invalid"},
        "output_dir": "${now:%Y-%m-%d}/private-run",
        "loss": {"components": {"od_guided_gbclm_v3": {"weight": 1.0}}},
    }
    OmegaConf.save(OmegaConf.create(config), source / "config.yaml")
    generator = OrderedDict([
        ("model.1.weight_orig", torch.arange(12, dtype=torch.float32).reshape(3, 4)),
        ("model.1.weight_u", torch.ones(3)),
        ("model.1.weight_v", torch.ones(4)),
    ])
    generator._metadata = OrderedDict([("", {"version": 1}), ("model.1", {"version": 1})])
    torch.save({
        "networks": {"G": generator, "D": {"weight": torch.zeros(2)}},
        "optimizers": {"private-training-state": {}},
        "current_epoch": 100,
        "strategy_state": {"unrelated": "private-run"},
    }, source / "trainer_state.pt")
    return source


def _load(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exports_only_generator_and_minimal_config_without_changing_source(source_checkpoint, tmp_path):
    before = {path.name: _hash(path) for path in source_checkpoint.iterdir()}
    destination = tmp_path / "exported"
    metadata = EXPORT.export_checkpoint(source_checkpoint, destination)
    state = _load(destination / "trainer_state.pt")
    original = _load(source_checkpoint / "trainer_state.pt")["networks"]["G"]
    assert set(state) == {"networks"}
    assert set(state["networks"]) == {"G"}
    assert state["networks"]["G"]._metadata == original._metadata
    for name, tensor in state["networks"]["G"].items():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    config = OmegaConf.to_container(OmegaConf.load(destination / "config.yaml"), resolve=True)
    assert set(config) == {"model", "transforms", "data"}
    assert config["data"] == {"target_mpp": 0.5}
    assert config["model"]["name"] == "pglstain"
    params = config["model"]["params"]
    assert params["ngf"] == 32
    assert params["lambda_pecc"] == 0.2
    assert params["pecc_layers"] == "8,12,16"
    assert "lambda_gbclm" not in params and "gbclm_layers" not in params
    assert "norm" not in params and "use_dropout" not in params
    assert config["transforms"]["train_transforms"][0]["size"] == [512, 512]
    assert set(metadata) == {"generator_key_count", "files"}
    assert metadata["generator_key_count"] == 3
    for name, info in metadata["files"].items():
        assert info == {"size_bytes": (destination / name).stat().st_size, "sha256": _hash(destination / name)}
    metadata_text = (destination / "export_metadata.json").read_text()
    assert json.loads(metadata_text) == metadata
    assert str(source_checkpoint) not in metadata_text
    assert "private" not in (destination / "config.yaml").read_text()
    assert before == {path.name: _hash(path) for path in source_checkpoint.iterdir()}


def test_ema_is_preserved_without_other_strategy_state(source_checkpoint, tmp_path):
    state = _load(source_checkpoint / "trainer_state.pt")
    ema = {"model.1.weight_orig": torch.full((3, 4), 0.25)}
    state["strategy_state"]["ema"] = ema
    torch.save(state, source_checkpoint / "trainer_state.pt")
    destination = tmp_path / "exported"
    EXPORT.export_checkpoint(source_checkpoint, destination)
    exported = _load(destination / "trainer_state.pt")
    assert set(exported["strategy_state"]) == {"ema"}
    torch.testing.assert_close(exported["strategy_state"]["ema"]["model.1.weight_orig"], ema["model.1.weight_orig"])


def test_refuses_existing_destination(source_checkpoint, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("unchanged")
    with pytest.raises(FileExistsError):
        EXPORT.export_checkpoint(source_checkpoint, destination)
    assert marker.read_text() == "unchanged"
    assert list(destination.iterdir()) == [marker]


@pytest.mark.parametrize("bad_state", [
    {}, {"networks": {}}, {"networks": {"G": {}}},
    {"networks": {"G": {"weight": "not-a-tensor"}}},
    {"networks": {"G": {"weight": torch.ones(2)}}, "strategy_state": {"ema": {"other": torch.ones(2)}}},
    {"networks": {"G": {"weight": torch.ones(2)}}, "strategy_state": {"ema": {"weight": torch.ones(3)}}},
])
def test_invalid_weights_are_rejected_before_destination_creation(source_checkpoint, tmp_path, bad_state):
    torch.save(bad_state, source_checkpoint / "trainer_state.pt")
    destination = tmp_path / "exported"
    with pytest.raises(ValueError):
        EXPORT.export_checkpoint(source_checkpoint, destination)
    assert not destination.exists()


@pytest.mark.parametrize("mutation", [
    {"model": {"name": "cyclegan"}},
    {"model": {"params": {"lambda_pecc": 0.9}}},
    {"model": {"params": {"unknown_parameter": 1}}},
    {"model": {"params": {"netG": "/private/model"}}},
    {"data": {"target_mpp": -1}},
    {"transforms": {"train_transforms": [{"name": "resize", "file": "private@example.invalid"}]}},
])
def test_unsafe_or_ambiguous_config_is_rejected(source_checkpoint, tmp_path, mutation):
    config = OmegaConf.merge(OmegaConf.load(source_checkpoint / "config.yaml"), mutation)
    OmegaConf.save(config, source_checkpoint / "config.yaml")
    destination = tmp_path / "exported"
    with pytest.raises(ValueError):
        EXPORT.export_checkpoint(source_checkpoint, destination)
    assert not destination.exists()


def test_clone_is_detached_and_does_not_share_storage():
    tensor = torch.ones(3, requires_grad=True)
    cloned = EXPORT._clone_tensor_state({"weight": tensor}, "G")["weight"]
    assert cloned.data_ptr() != tensor.data_ptr()
    assert not cloned.requires_grad
    assert cloned.device.type == "cpu"


def test_cli_accepts_source_and_output_directories(source_checkpoint, tmp_path):
    output = tmp_path / "cli-export"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(source_checkpoint), str(output)],
        text=True, capture_output=True, check=True,
    )
    assert "Exported inference checkpoint" in completed.stdout
    assert (output / "trainer_state.pt").is_file()
    assert (output / "config.yaml").is_file()
