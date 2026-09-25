"""Export a minimal PGLStain inference checkpoint without training metadata.

Usage:
    python scripts/export_inference_checkpoint.py SOURCE_DIR OUTPUT_DIR

The destination must not exist. The source checkpoint is read without changes.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from omegaconf import OmegaConf
import torch


SOURCE_MODEL_NAMES = {
    "pglstain",
    "od_guided_higraph_v3",
    "od_guided_higraph_v3_softgate_odset_odugw",
}
PARAMETER_ALIASES = {
    "lambda_gbclm": "lambda_pecc",
    "gbclm_layers": "pecc_layers",
}
MODEL_PARAMETERS = {
    "input_nc", "output_nc", "ngf", "ndf", "netG", "netD", "netF",
    "n_layers_D", "normG", "normD", "init_type", "init_gain", "no_dropout",
    "no_antialias", "no_antialias_up", "nce_idt", "lambda_GAN", "lambda_NCE",
    "nce_layers", "nce_T", "num_patches", "netF_nc", "flip_equivariance",
    "lambda_gp", "gp_weights", "lambda_pecc", "lambda_od_struct",
    "lambda_od_set", "lambda_od_graph", "pecc_layers", "crop_size",
    "gpu_ids", "no_d", "weight_norm", "d_lr_factor",
}
# These inherited defaults are ignored by both original and exported CUT models.
UNUSED_SHARED_PARAMETERS = {"norm", "use_dropout"}


def _validate_public_values(value: Any, location: str) -> None:
    """Reject paths, URLs, email-like values, and unresolved interpolations."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} must use string keys")
            _validate_public_values(key, location)
            _validate_public_values(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_public_values(item, f"{location}[{index}]")
    elif isinstance(value, str):
        if (
            any(marker in value for marker in ("/", "\\", "@", "${"))
            or value.startswith("~")
            or re.search(r"[A-Za-z]:", value)
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(f"Potential private or unresolved string in {location}")
    elif value is None or isinstance(value, (bool, int)):
        return
    elif isinstance(value, float) and math.isfinite(value):
        return
    else:
        raise ValueError(f"Unsupported configuration value in {location}")


def _inference_config(config_path: Path) -> dict[str, Any]:
    config = OmegaConf.load(config_path)
    if "model" not in config or "transforms" not in config:
        raise ValueError("Checkpoint config requires model and transforms sections")
    # Resolve only retained sections: private training/output fields may use
    # Hydra resolvers that are irrelevant to inference and unavailable here.
    model = OmegaConf.to_container(config.model, resolve=True, throw_on_missing=True)
    transforms = OmegaConf.to_container(
        config.transforms, resolve=True, throw_on_missing=True
    )
    if not isinstance(model, dict) or model.get("name") not in SOURCE_MODEL_NAMES:
        raise ValueError("Only PGLStain and its supported development names can be exported")
    if not isinstance(transforms, dict) or not isinstance(
        transforms.get("train_transforms"), list
    ):
        raise ValueError("transforms.train_transforms must be a list")
    nested = model.get("params", {})
    if not isinstance(nested, dict):
        raise ValueError("model.params must be a mapping")
    excluded = {"name", "params", "use_gan", "normalize_to_minus_one_one"}
    params = {key: value for key, value in model.items() if key not in excluded}
    params.update(nested)
    for old, new in PARAMETER_ALIASES.items():
        if old in params:
            if new in params and params[new] != params[old]:
                raise ValueError(f"Conflicting model parameters: {old} and {new}")
            params[new] = params.pop(old)
    unknown = set(params) - MODEL_PARAMETERS - UNUSED_SHARED_PARAMETERS
    if unknown:
        raise ValueError(f"Unrecognized model parameters: {sorted(unknown)}")
    params = {key: value for key, value in params.items() if key in MODEL_PARAMETERS}
    result: dict[str, Any] = {
        "model": {
            "name": "pglstain",
            "params": params,
            "normalize_to_minus_one_one": model.get("normalize_to_minus_one_one", True),
        },
        "transforms": transforms,
    }
    target_mpp = OmegaConf.select(config, "data.target_mpp", default=None)
    if target_mpp is not None:
        if isinstance(target_mpp, bool) or not isinstance(target_mpp, (int, float)):
            raise ValueError("data.target_mpp must be a positive number")
        if not math.isfinite(target_mpp) or target_mpp <= 0:
            raise ValueError("data.target_mpp must be a positive finite number")
        result["data"] = {"target_mpp": target_mpp}
    _validate_public_values(result, "config")
    return result


def _clone_tensor_state(value: Any, label: str, *, allow_empty: bool = False):
    if not isinstance(value, Mapping) or (not value and not allow_empty):
        raise ValueError(f"{label} must be a {'nonempty ' if not allow_empty else ''}mapping")
    cloned = OrderedDict()
    for key, tensor in value.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", key):
            raise ValueError(f"{label} contains an invalid state key")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{label}.{key} must be a tensor")
        if tensor.device.type == "meta":
            raise ValueError(f"{label}.{key} has no materialized tensor data")
        cloned[key] = tensor.detach().cpu().clone()
    # PyTorch state_dict version metadata is needed by some load hooks (for
    # example spectral normalization); keep it without serializing modules.
    if hasattr(value, "_metadata"):
        metadata = copy.deepcopy(value._metadata)
        _validate_public_values(metadata, f"{label}._metadata")
        cloned._metadata = metadata
    return cloned


def _file_info(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def export_checkpoint(source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("The export destination already exists; choose a new directory")
    config = _inference_config(source_dir / "config.yaml")
    state = torch.load(
        source_dir / "trainer_state.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(state, Mapping) or not isinstance(state.get("networks"), Mapping):
        raise ValueError("Checkpoint requires a networks mapping")
    generator = _clone_tensor_state(state["networks"].get("G"), "networks.G")
    exported: dict[str, Any] = {"networks": {"G": generator}}
    strategy = state.get("strategy_state")
    if strategy is not None and not isinstance(strategy, Mapping):
        raise ValueError("strategy_state must be a mapping when present")
    ema = strategy.get("ema") if strategy is not None else None
    if ema is not None:
        ema = _clone_tensor_state(ema, "strategy_state.ema", allow_empty=True)
        for key, tensor in ema.items():
            if key not in generator or tensor.shape != generator[key].shape:
                raise ValueError(f"EMA key/shape does not match the generator: {key}")
        exported["strategy_state"] = {"ema": ema}
    # All validation precedes creating the destination. mkdir is exclusive:
    # even a destination created concurrently is never overwritten.
    output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "trainer_state.pt"
    config_path = output_dir / "config.yaml"
    torch.save(exported, state_path)
    OmegaConf.save(OmegaConf.create(config), config_path)
    metadata = {
        "generator_key_count": len(generator),
        "files": {
            "trainer_state.pt": _file_info(state_path),
            "config.yaml": _file_info(config_path),
        },
    }
    (output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.source_dir, args.output_dir)
    print(f"Exported inference checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
