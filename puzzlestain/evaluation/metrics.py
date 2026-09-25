"""Unified evaluation metrics for virtual staining.

Four metric families:

* **ImageQuality** — pixel-level fidelity (SSIM, PSNR, HistCorr)
* **Perception** — human-perceptual similarity (PHV, DISTS, FID, KIDx1k)
* **PathologicalRelevance** — clinical relevance (mIoD, PearsonR, DAB_KL)
* **PathFID** — pathology foundation model FID (CONCH, UNI, StainNet, etc.)

Each family is a callable that accepts ``(fake_dir, gt_dir, device)`` and returns
``(summary_dict, per_sample_list)``.
"""

from __future__ import annotations

import abc
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from scipy.stats import entropy, pearsonr
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class MetricComputationError(RuntimeError):
    """Raised when a metric cannot produce a valid benchmark value.

    A failed metric must never be represented by a numeric sentinel such as
    ``0.0``: for distance metrics that value looks like a perfect score and can
    silently contaminate benchmark tables.
    """


def _require_finite(name: str, value: float) -> float:
    """Return a finite metric value or fail the benchmark run."""
    score = float(value)
    if not np.isfinite(score):
        raise MetricComputationError(f"{name} produced a non-finite value: {score}")
    return score


# ---------------------------------------------------------------------------
# Base protocol
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Per-sample metric record."""

    filename: str
    values: dict[str, float] = field(default_factory=dict)


@dataclass
class FamilyResult:
    """Result from one metric family."""

    summary: dict[str, float]
    per_sample: list[MetricResult]


class MetricFamily(abc.ABC):
    """Abstract metric family."""

    name: str = ""

    @abc.abstractmethod
    def compute(self, fake_dir: str, gt_dir: str, device: str) -> FamilyResult:
        """Compute metrics for all matched image pairs.

        Args:
            fake_dir: Directory with generated images.
            gt_dir: Directory with ground-truth images.
            device: torch device string.

        Returns:
            FamilyResult with summary averages and per-sample records.
        """
        ...


# ---------------------------------------------------------------------------
# ImageQuality family
# ---------------------------------------------------------------------------


class ImageQualityFamily(MetricFamily):
    """Pixel-level fidelity metrics: SSIM, PSNR, and HistCorr."""

    name = "image_quality"

    def __init__(self, metrics: list[str] | None = None) -> None:
        """Initialize.

        Args:
            metrics: Subset of ["ssim", "psnr", "hist_corr"].
                None means all.
        """
        all_metrics = {"ssim", "psnr", "hist_corr"}
        requested = set(metrics) if metrics else all_metrics
        if "vif" in requested:
            raise MetricComputationError(
                "VIF is currently excluded from the image_quality family because "
                "its benchmark protocol is under review."
            )
        self.metrics = requested

    def compute(self, fake_dir: str, gt_dir: str, device: str) -> FamilyResult:
        pairs = _match_image_pairs(fake_dir, gt_dir)
        per_sample: list[MetricResult] = []
        ssim_vals, psnr_vals, vif_vals = [], [], []
        hist_vals = []

        for fake_path, gt_path, name in tqdm(pairs, desc="image-quality"):
            fake = _load_gray(fake_path)
            gt = _load_gray(gt_path)
            if fake is None or gt is None:
                raise MetricComputationError(
                    f"Image-quality input could not be decoded: {name}"
                )
            if fake.shape != gt.shape:
                gt = cv2.resize(gt, (fake.shape[1], fake.shape[0]))

            rec = MetricResult(filename=name)

            if "ssim" in self.metrics:
                val = ssim(fake, gt, data_range=255.0)
                rec.values["ssim"] = val
                ssim_vals.append(val)

            if "psnr" in self.metrics:
                val = psnr(fake, gt, data_range=255.0)
                rec.values["psnr"] = val
                psnr_vals.append(val)

            if "vif" in self.metrics:
                val = _vif_pineapple(fake, gt)
                rec.values["vif"] = val
                vif_vals.append(val)

            if "hist_corr" in self.metrics:
                val = _hist_correlation(fake_path, gt_path)
                rec.values["hist_corr"] = val
                hist_vals.append(val)

            per_sample.append(rec)

        summary: dict[str, float] = {}
        if ssim_vals:
            summary["ssim_mean"] = float(np.mean(ssim_vals))
        if psnr_vals:
            summary["psnr_mean"] = float(np.mean(psnr_vals))
        if vif_vals:
            summary["vif_mean"] = float(np.mean(vif_vals))
        if hist_vals:
            summary["hist_corr_mean"] = float(np.mean(hist_vals))

        return FamilyResult(summary=summary, per_sample=per_sample)


# ---------------------------------------------------------------------------
# Perception family
# ---------------------------------------------------------------------------


class PerceptionFamily(MetricFamily):
    """Human-perceptual similarity: PHV, DISTS, FID, and KID.

    Inception Score is intentionally disabled for benchmark runs. The legacy
    implementation remains in this module for reference, but it is not part of
    ``self.metrics`` and is never invoked by this family.
    """

    name = "perception"

    def __init__(self, metrics: list[str] | None = None) -> None:
        all_metrics = {"phv", "dists", "fid", "kid"}
        requested = set(metrics) if metrics else all_metrics
        if "is" in requested:
            logger.warning(
                "Inception Score is disabled and will not be computed; "
                "use PHV/DISTS/FID/KID for the perception family."
            )
            requested.remove("is")
        self.metrics = requested

    def compute(self, fake_dir: str, gt_dir: str, device: str) -> FamilyResult:
        pairs = _match_image_pairs(fake_dir, gt_dir)
        per_sample: list[MetricResult] = []
        phv_vals: dict[str, list[float]] = {
            "phv_1": [],
            "phv_2": [],
            "phv_3": [],
            "phv_4": [],
            "phv_avg": [],
        }
        dists_vals = []

        # Lazy-init PHV model on first use
        _phv_model = None
        if "phv" in self.metrics:
            _phv_model = _PerceptualHashValue(
                T=0.01,
                network="resnet50",
                layers=["layer_1", "layer_2", "layer_3", "layer_4"],
                resize=False,
                resize_mode="bilinear",
                instance_normalized=False,
            ).to(device)

        # DISTS is a heavyweight VGG16-based model. Construct it once per
        # evaluation run instead of once per image pair.
        _dists_model = None
        _dists_transform = None
        if "dists" in self.metrics:
            try:
                from DISTS_pytorch import DISTS as DISTSModel
                from torchvision import transforms

                _dists_model = DISTSModel().to(device).eval()
                _dists_transform = transforms.Compose(
                    [
                        transforms.Resize((1024, 1024)),
                        transforms.ToTensor(),
                    ]
                )
            except Exception as exc:
                raise MetricComputationError(
                    f"DISTS model initialization failed: {exc}"
                ) from exc

        for fake_path, gt_path, name in tqdm(pairs, desc="perception"):
            rec = MetricResult(filename=name)

            if "phv" in self.metrics and _phv_model is not None:
                phv_dict = _phv_perceptual_hash(_phv_model, fake_path, gt_path, device)
                rec.values.update(phv_dict)
                for k, values in phv_vals.items():
                    values.append(phv_dict[k])

            if (
                "dists" in self.metrics
                and _dists_model is not None
                and _dists_transform is not None
            ):
                val = _dists(
                    fake_path,
                    gt_path,
                    device,
                    _dists_model,
                    _dists_transform,
                )
                rec.values["dists"] = val
                dists_vals.append(val)

            per_sample.append(rec)

        summary: dict[str, float] = {}
        for k, vals in phv_vals.items():
            if vals:
                summary[f"{k}_mean"] = _require_finite(k, np.mean(vals))
        if dists_vals:
            summary["dists_mean"] = _require_finite("DISTS", np.mean(dists_vals))

        # FID / KID are dataset-level. Inception Score is intentionally not
        # called; _compute_inception_score is retained below as legacy code.
        if "fid" in self.metrics:
            summary["fid"] = _compute_fid(fake_dir, gt_dir, device)
        if "kid" in self.metrics:
            summary["kid"], summary["kid_std"] = _compute_kid(fake_dir, gt_dir, device)

        return FamilyResult(summary=summary, per_sample=per_sample)


# ---------------------------------------------------------------------------
# PathologicalRelevance family
# ---------------------------------------------------------------------------


class DABExtractor:
    """Extract DAB stain intensity from IHC images using color deconvolution.

    Reference: Ruifrok & Johnston, "Quantification of histochemical
    staining by color deconvolution", Anal Quant Cytol Histol 2001
    """

    def __init__(self, device="cpu"):
        self.device = device
        # Standard H-DAB stain matrix (Ruifrok & Johnston)
        self.stain_matrix = torch.tensor(
            [
                [0.268, 0.570, 0.776],  # DAB (brown)
                [0.650, 0.704, 0.286],  # Hematoxylin (blue)
            ],
            device=device,
            dtype=torch.float32,
        )
        self.deconv_matrix = torch.linalg.pinv(self.stain_matrix.T)

    def rgb_to_od(self, rgb_images: torch.Tensor) -> torch.Tensor:
        """Convert RGB [0,1] to optical density: OD = -log10(I/I0)."""
        rgb_images = rgb_images.clamp(1e-6, 1.0)
        return -torch.log10(rgb_images + 1e-6)

    def extract_dab_intensity(
        self, images: torch.Tensor, normalize: str = "none"
    ) -> torch.Tensor:
        """Extract DAB stain intensity from IHC images.

        Args:
            images: [B, 3, H, W] RGB images in [-1, 1] or [0, 1]
            normalize: "none", "max", or "meanstd"

        Returns:
            dab_intensity: [B, H, W] DAB intensity map
        """
        B, C, H, W = images.shape
        assert C == 3, "Input must be RGB images"

        # Auto-convert [-1, 1] -> [0, 1] if needed
        if images.min() < 0:
            images = (images + 1.0) / 2.0

        od = self.rgb_to_od(images)
        od_flat = od.permute(0, 2, 3, 1).reshape(-1, 3)

        deconv_matrix = self.deconv_matrix.to(od_flat.device)
        concentrations = od_flat @ deconv_matrix.T
        dab_flat = concentrations[:, 0]  # DAB channel

        dab_intensity = dab_flat.reshape(B, H, W)
        dab = F.softplus(dab_intensity, beta=5.0)

        if normalize == "max":
            mx = dab.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            dab = dab / mx
        elif normalize == "meanstd":
            mean = dab.mean(dim=(1, 2), keepdim=True)
            std = dab.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            dab = (dab - mean) / std
        # "none" -> pass
        return dab


class PathologicalRelevanceFamily(MetricFamily):
    """Clinical relevance: mIoD, PearsonR, DAB_KL.

    Args:
        metrics: Subset of ["miod", "pearsonr", "dab_kl"].
            None means all.
    """

    name = "pathological_relevance"

    def __init__(self, metrics: list[str] | None = None) -> None:
        all_metrics = {"miod", "pearsonr", "dab_kl"}
        self.metrics = set(metrics) if metrics else all_metrics

    def compute(self, fake_dir: str, gt_dir: str, device: str) -> FamilyResult:
        pairs = _match_image_pairs(fake_dir, gt_dir)
        per_sample: list[MetricResult] = []
        fake_miods, gt_miods = [], []
        per_sample_kls = []

        extractor = DABExtractor(device=device)

        for fake_path, gt_path, name in tqdm(pairs, desc="pathological-relevance"):
            rec = MetricResult(filename=name)

            fake_t = _load_tensor(fake_path, device)
            gt_t = _load_tensor(gt_path, device)
            if fake_t is None or gt_t is None:
                raise MetricComputationError(
                    f"Pathological-relevance input could not be decoded: {name}"
                )

            if self.metrics & {"miod", "pearsonr"}:
                miod_f, miod_g = _compute_miod_pair(fake_t, gt_t)
                fake_miods.append(miod_f)
                gt_miods.append(miod_g)
                if "miod" in self.metrics:
                    rec.values["fake_miod"] = miod_f
                    rec.values["gt_miod"] = miod_g

            if "dab_kl" in self.metrics:
                kl = _compute_dab_kl(fake_t, gt_t, extractor)
                rec.values["dab_kl"] = kl
                per_sample_kls.append(kl)

            per_sample.append(rec)

        summary: dict[str, float] = {}
        if "pearsonr" in self.metrics:
            if len(fake_miods) < 2:
                raise MetricComputationError(
                    "PearsonR requires at least two matched image pairs"
                )
            r, _ = pearsonr(np.array(fake_miods), np.array(gt_miods))
            summary["pearsonr"] = _require_finite("PearsonR", r)
        if "miod" in self.metrics and fake_miods:
            summary["miod_mean_diff"] = _require_finite(
                "mIoD mean difference",
                np.mean(np.array(fake_miods) - np.array(gt_miods)),
            )

        if per_sample_kls:
            summary["dab_kl_mean"] = _require_finite(
                "DAB KL mean", np.mean(per_sample_kls)
            )
            summary["dab_kl_std"] = _require_finite(
                "DAB KL std", np.std(per_sample_kls)
            )

        return FamilyResult(summary=summary, per_sample=per_sample)


# ---------------------------------------------------------------------------
# PathFID family
# ---------------------------------------------------------------------------


def _normalize_per_model(
    value: str | dict[str, str] | None,
    models: list[str],
    field: str,
) -> dict[str, str | None]:
    """Normalize a scalar-or-map config field into a per-model mapping.

    Args:
        value: ``None`` (every model uses its built-in default), a single
            string (only valid when exactly one model is configured), or a
            mapping of model name -> value.
        models: Configured foundation model names.
        field: Config field name, used in error messages.

    Returns:
        ``{model_name: value or None}`` for every configured model.

    Raises:
        ValueError: If a scalar is given for multiple models, or the map
            contains models that are not configured.
    """
    if value is None:
        return {m: None for m in models}
    if isinstance(value, str):
        if len(models) != 1:
            raise ValueError(
                f"pathfid.{field} is a single value but multiple foundation "
                f"models are configured {models}; use a mapping instead, e.g. "
                f"{field}: {{{models[0]}: ...}}"
            )
        return {models[0]: value}
    unknown = sorted(set(value) - set(models))
    if unknown:
        raise ValueError(
            f"pathfid.{field} has entries for unconfigured model(s): "
            f"{', '.join(unknown)} (configured: {', '.join(models)})"
        )
    return {m: value.get(m) for m in models}


class PathFIDFamily(MetricFamily):
    """Pathology foundation model FID, one score per feature extractor.

    Uses pathology-specific foundation models (CONCH, UNI, UNI2-h, StainNet)
    instead of generic ImageNet InceptionV3 for feature extraction. Each
    configured extractor produces its own ``pathfid_<name>`` summary entry.
    Default: CONCH.

    Args:
        foundation_models: Extractor names, any of ["conch", "uni", "univ2-h",
            "stainnet", "inception"] ("univ2" is accepted as an alias of
            "univ2-h"). Default: ["conch"].
        checkpoint: Local weights for the extractors — a single path (only
            when one model is configured) or a ``{model: path}`` mapping.
            Takes priority over ``hf_id`` and over the ``UNI_CHECKPOINT`` /
            ``PATHOLOGY_CHECKPOINT_ROOT`` env vars.
        hf_id: HuggingFace Hub model id(s) — a single id or a ``{model: id}``
            mapping. ``None`` uses each model's built-in default. Ignored
            when ``checkpoint`` is set for that model.
    """

    name = "pathfid"

    def __init__(
        self,
        foundation_models: list[str] | None = None,
        checkpoint: str | dict[str, str] | None = None,
        hf_id: str | dict[str, str] | None = None,
    ) -> None:
        self.foundation_models = (
            list(foundation_models) if foundation_models else ["conch"]
        )
        self.checkpoints = _normalize_per_model(
            checkpoint, self.foundation_models, "checkpoint"
        )
        self.hf_ids = _normalize_per_model(hf_id, self.foundation_models, "hf_id")

    def compute(self, fake_dir: str, gt_dir: str, device: str) -> FamilyResult:
        _match_image_pairs(fake_dir, gt_dir)
        summary: dict[str, float] = {}
        for model_name in self.foundation_models:
            summary[f"pathfid_{model_name}"] = _compute_pathfid(
                fake_dir,
                gt_dir,
                device,
                model_name,
                checkpoint=self.checkpoints[model_name],
                hf_id=self.hf_ids[model_name],
            )
        return FamilyResult(summary=summary, per_sample=[])


# ---------------------------------------------------------------------------
# Family registry
# ---------------------------------------------------------------------------


FAMILY_REGISTRY: dict[str, type[MetricFamily]] = {
    "image_quality": ImageQualityFamily,
    "perception": PerceptionFamily,
    "pathological_relevance": PathologicalRelevanceFamily,
    "pathfid": PathFIDFamily,
}


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def write_results(
    results: dict[str, FamilyResult],
    output_dir: str,
    exp_name: str,
) -> None:
    """Write per-family summaries and unified per-sample CSV.

    Args:
        results: family_name -> FamilyResult.
        output_dir: Root directory for metrics (e.g. ``PATHOLOGY_METRIC_ROOT``).
        exp_name: Experiment name used in filenames.
    """
    out = Path(output_dir) / exp_name
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Per-family summary CSV
    for family_name, fres in results.items():
        summary_path = out / f"{family_name}_{timestamp}_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in fres.summary.items():
                writer.writerow([k, f"{v:.6f}"])
        logger.info("Wrote summary %s", summary_path)

    # Unified per-sample CSV (all families merged by filename)
    unified_path = out / f"per_sample_{timestamp}.csv"
    all_filenames = set()
    for fres in results.values():
        for s in fres.per_sample:
            all_filenames.add(s.filename)

    # Collect all metric keys
    metric_keys: list[str] = []
    for fres in results.values():
        if fres.per_sample:
            metric_keys.extend(fres.per_sample[0].values.keys())
    metric_keys = sorted(set(metric_keys))

    with open(unified_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + metric_keys)
        for fname in sorted(all_filenames):
            row: dict[str, float] = {}
            for fres in results.values():
                for s in fres.per_sample:
                    if s.filename == fname:
                        row.update(s.values)
            writer.writerow(
                [fname]
                + [f"{row.get(k, ''):.6f}" if k in row else "" for k in metric_keys]
            )
    logger.info("Wrote per-sample %s", unified_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match_image_pairs(fake_dir: str, gt_dir: str) -> list[tuple[str, str, str]]:
    """Match images one-to-one by stem, failing on incomplete or ambiguous sets."""

    def index(directory: str, role: str) -> dict[str, str]:
        files: dict[str, str] = {}
        for filename in sorted(os.listdir(directory)):
            path = Path(filename)
            if path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            if path.stem in files:
                raise MetricComputationError(
                    f"{role} contains duplicate image stem {path.stem!r}: "
                    f"{Path(files[path.stem]).name!r} and {filename!r}"
                )
            files[path.stem] = os.path.join(directory, filename)
        if not files:
            raise MetricComputationError(
                f"{role} directory contains no supported images: {directory}"
            )
        return files

    fake_files = index(fake_dir, "fake")
    gt_files = index(gt_dir, "gt")
    fake_stems = set(fake_files)
    gt_stems = set(gt_files)
    missing_gt = sorted(fake_stems - gt_stems)
    missing_fake = sorted(gt_stems - fake_stems)
    if missing_gt or missing_fake:
        preview = 10
        raise MetricComputationError(
            "fake/gt image sets do not match by filename stem: "
            f"missing_gt={missing_gt[:preview]} "
            f"missing_fake={missing_fake[:preview]} "
            f"(fake={len(fake_files)}, gt={len(gt_files)})"
        )

    return [(fake_files[stem], gt_files[stem], stem) for stem in sorted(fake_stems)]


def _load_gray(path: str) -> np.ndarray | None:
    """Load image as grayscale uint8 numpy array."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _load_tensor(path: str, device: str) -> torch.Tensor | None:
    """Load image as [1, 3, H, W] float tensor in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def _vif_pineapple(ref: np.ndarray, dist: np.ndarray) -> float:
    """Compute four-scale pixel-domain Visual Information Fidelity (VIFp).

    This follows the multi-scale Gaussian local-statistics formulation from
    Sheikh and Bovik, *Image Information and Visual Quality*, IEEE TIP 2006.
    Inputs are grayscale images in the same intensity scale (normally uint8
    [0, 255]). A perfect non-constant reconstruction scores approximately
    1.0; VIF may legitimately exceed one for contrast-enhanced images.

    The historical function name is retained for API compatibility.
    """
    from scipy.ndimage import gaussian_filter

    ref = np.asarray(ref, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    if ref.ndim != 2 or dist.ndim != 2:
        raise ValueError(
            f"VIF expects two grayscale 2-D arrays, got {ref.shape} and {dist.shape}"
        )
    if ref.shape != dist.shape:
        raise ValueError(
            f"VIF inputs must have the same shape, got {ref.shape} and {dist.shape}"
        )
    if ref.size == 0:
        raise ValueError("VIF inputs must not be empty")
    if not np.isfinite(ref).all() or not np.isfinite(dist).all():
        raise ValueError("VIF inputs must contain only finite values")

    sigma_nsq = 2.0
    eps = 1e-10
    numerator = 0.0
    denominator = 0.0

    ref_scale = ref
    dist_scale = dist
    for scale in range(1, 5):
        window_size = 2 ** (4 - scale + 1) + 1
        sigma = window_size / 5.0

        if scale > 1:
            ref_scale = gaussian_filter(ref_scale, sigma, mode="reflect")[::2, ::2]
            dist_scale = gaussian_filter(dist_scale, sigma, mode="reflect")[::2, ::2]

        mu_ref = gaussian_filter(ref_scale, sigma, mode="reflect")
        mu_dist = gaussian_filter(dist_scale, sigma, mode="reflect")
        sigma_ref_sq = (
            gaussian_filter(ref_scale * ref_scale, sigma, mode="reflect")
            - mu_ref * mu_ref
        )
        sigma_dist_sq = (
            gaussian_filter(dist_scale * dist_scale, sigma, mode="reflect")
            - mu_dist * mu_dist
        )
        sigma_cross = (
            gaussian_filter(ref_scale * dist_scale, sigma, mode="reflect")
            - mu_ref * mu_dist
        )

        sigma_ref_sq = np.maximum(sigma_ref_sq, 0.0)
        sigma_dist_sq = np.maximum(sigma_dist_sq, 0.0)
        gain = sigma_cross / (sigma_ref_sq + eps)
        residual_var = sigma_dist_sq - gain * sigma_cross

        flat_reference = sigma_ref_sq < eps
        gain[flat_reference] = 0.0
        residual_var[flat_reference] = sigma_dist_sq[flat_reference]

        flat_distortion = sigma_dist_sq < eps
        gain[flat_distortion] = 0.0
        residual_var[flat_distortion] = 0.0

        negative_gain = gain < 0.0
        gain[negative_gain] = 0.0
        residual_var[negative_gain] = sigma_dist_sq[negative_gain]
        residual_var = np.maximum(residual_var, eps)

        numerator += float(
            np.log1p(gain * gain * sigma_ref_sq / (residual_var + sigma_nsq)).sum()
        )
        denominator += float(np.log1p(sigma_ref_sq / sigma_nsq).sum())

    if denominator <= eps:
        return 1.0 if np.allclose(ref, dist, rtol=0.0, atol=eps) else 0.0

    score = numerator / denominator
    if not np.isfinite(score):
        raise MetricComputationError("VIF produced a non-finite value")
    return float(score)


def _phv_perceptual_hash(
    model, fake_path: str, gt_path: str, device: str
) -> dict[str, float]:
    """Compute Perceptual Hash Value (PHV) using ResNet50 layer features.

    Returns dict with keys phv_1, phv_2, phv_3, phv_4, phv_avg.
    """
    fake = cv2.imread(fake_path)
    gt = cv2.imread(gt_path)
    if fake is None or gt is None:
        raise MetricComputationError(
            f"PHV input could not be decoded: {fake_path!r}, {gt_path!r}"
        )

    # Convert BGR to RGB and to tensor [0, 1]
    fake_rgb = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    fake_t = to_tensor(fake_rgb).unsqueeze(0).to(device)
    gt_t = to_tensor(gt_rgb).unsqueeze(0).to(device)

    # PHV consumes tensors in [0, 1] and applies ImageNet normalization.
    with torch.no_grad():
        phv_list = model(fake_t, gt_t)

    result: dict[str, float] = {}
    for i, val in enumerate(phv_list, start=1):
        result[f"phv_{i}"] = _require_finite(f"PHV layer {i}", val)
    result["phv_avg"] = _require_finite("PHV average", np.mean(phv_list))
    return result


class _PerceptualHashValue(nn.Module):
    """Perceptual Hash Value using ResNet50 features.

    Reference: EasyVirtualStain/patch_metrics/perceptual.py
    """

    def __init__(
        self,
        T: float = 0.01,
        network: str = "resnet50",
        layers: list[str] | None = None,
        resize: bool = False,
        resize_mode: str = "bilinear",
        instance_normalized: bool = False,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = ["layer_1", "layer_2", "layer_3", "layer_4"]
        self.T = T
        self.layers = layers
        self.resize = resize
        self.resize_mode = resize_mode
        self.instance_normalized = instance_normalized

        resnet50 = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        network = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3,
            resnet50.layer4,
            resnet50.avgpool,
        )
        layer_name_mapping = {4: "layer_1", 5: "layer_2", 6: "layer_3", 7: "layer_4"}
        self.model = _PerceptualNetwork(network, layer_name_mapping, layers)

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> list[float]:
        self.model.eval()
        inp, target = (
            _apply_imagenet_normalization(inp),
            _apply_imagenet_normalization(target),
        )
        if self.resize:
            inp = F.interpolate(
                inp, mode=self.resize_mode, size=(224, 224), align_corners=False
            )
            target = F.interpolate(
                target, mode=self.resize_mode, size=(224, 224), align_corners=False
            )

        input_features = self.model(inp)
        target_features = self.model(target)

        hpv_list = []
        for layer in self.layers:
            input_feature = input_features[layer]
            target_feature = target_features[layer].detach()
            if self.instance_normalized:
                input_feature = F.instance_norm(input_feature)
                target_feature = F.instance_norm(target_feature)

            B, C = input_feature.shape[:2]
            inp_avg = torch.mean(input_feature.view(B, C, -1), -1)
            tgt_avg = torch.mean(target_feature.view(B, C, -1), -1)
            abs_dif = torch.abs(inp_avg - tgt_avg)
            hpv = torch.sum(abs_dif > self.T).item() / (B * C)
            hpv_list.append(hpv)

        return hpv_list


class _PerceptualNetwork(nn.Module):
    """Feature extraction network for perceptual metrics."""

    def __init__(
        self,
        network: nn.Sequential,
        layer_name_mapping: dict[int, str],
        layers: list[str],
    ) -> None:
        super().__init__()
        self.network = network
        self.layer_name_mapping = layer_name_mapping
        self.layers = layers
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        for i, layer in enumerate(self.network):
            x = layer(x)
            layer_name = self.layer_name_mapping.get(i, None)
            if layer_name in self.layers:
                output[layer_name] = x
        return output


def _apply_imagenet_normalization(input: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to an RGB tensor in [0, 1]."""
    mean = input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (input - mean) / std


def _hist_correlation(fake_path: str, gt_path: str) -> float:
    """HSV histogram correlation between two images."""
    fake = cv2.imread(fake_path)
    gt = cv2.imread(gt_path)
    if fake is None or gt is None:
        raise MetricComputationError(
            f"Histogram input could not be decoded: {fake_path!r}, {gt_path!r}"
        )
    if fake.shape != gt.shape:
        gt = cv2.resize(gt, (fake.shape[1], fake.shape[0]))
    fake_hsv = cv2.cvtColor(fake, cv2.COLOR_BGR2HSV)
    gt_hsv = cv2.cvtColor(gt, cv2.COLOR_BGR2HSV)
    fh = cv2.calcHist([fake_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    gh = cv2.calcHist([gt_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    fh = cv2.normalize(fh, fh).flatten()
    gh = cv2.normalize(gh, gh).flatten()
    return _require_finite(
        "Histogram correlation", cv2.compareHist(fh, gh, cv2.HISTCMP_CORREL)
    )


def _compute_niqe(dir_path: str) -> float:
    """Compute NIQE (Natural Image Quality Evaluator) for a directory.

    Uses the official niqe.py implementation.
    """
    try:
        import niqe

        scores = []
        for f in sorted(os.listdir(dir_path)):
            if Path(f).suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            img = cv2.imread(os.path.join(dir_path, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                scores.append(niqe.calculate_niqe(img))
        if not scores:
            raise MetricComputationError(
                f"NIQE found no decodable images in {dir_path}"
            )
        return _require_finite("NIQE", np.mean(scores))
    except ImportError as exc:
        raise MetricComputationError(
            "NIQE requires the optional 'niqe' package"
        ) from exc


def _compute_inception_score(
    dir_path: str, device: str, splits: int = 10
) -> tuple[float, float]:
    """Legacy Inception Score implementation, disabled for benchmarks.

    This function is intentionally retained but is not called by
    PerceptionFamily.
    """
    try:
        from torchvision import models, transforms

        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()

        trans = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        preds = []
        for f in sorted(os.listdir(dir_path)):
            if Path(f).suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            img = Image.open(os.path.join(dir_path, f)).convert("RGB")
            t = trans(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred = F.softmax(model(t), dim=1)
            preds.append(pred.cpu().numpy())

        if not preds:
            raise MetricComputationError(
                f"Inception Score found no supported images in {dir_path}"
            )

        preds = np.concatenate(preds, axis=0)
        split_scores = []
        for k in range(splits):
            part = preds[k * len(preds) // splits : (k + 1) * len(preds) // splits, :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return float(np.mean(split_scores)), float(np.std(split_scores))
    except MetricComputationError:
        raise
    except Exception as exc:
        raise MetricComputationError(
            f"Inception Score computation failed: {exc}"
        ) from exc


def _dists(
    fake_path: str,
    gt_path: str,
    device: str,
    model: nn.Module,
    transform: torchvision.transforms.Compose,
) -> float:
    """DISTS perceptual metric."""
    try:
        fake = Image.open(fake_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        f = transform(fake).unsqueeze(0).to(device)
        g = transform(gt).unsqueeze(0).to(device)
        with torch.inference_mode():
            score = model(f, g).item()
        return _require_finite("DISTS", score)
    except Exception as exc:
        raise MetricComputationError(f"DISTS computation failed: {exc}") from exc


def _compute_fid(fake_dir: str, gt_dir: str, device: str) -> float:
    """Compute FID using InceptionV3 features."""
    try:
        from torchvision import models, transforms

        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()

        trans = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def extract(dir_path: str) -> np.ndarray:
            feats = []
            for f in sorted(os.listdir(dir_path)):
                if Path(f).suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                img = Image.open(os.path.join(dir_path, f)).convert("RGB")
                t = trans(img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = model(t)
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    feats.append(feat.cpu().numpy())
            return np.concatenate(feats, axis=0)

        fake_feats = extract(fake_dir)
        gt_feats = extract(gt_dir)
        if len(fake_feats) < 2 or len(gt_feats) < 2:
            raise MetricComputationError(
                "FID requires at least two images in both fake and gt "
                f"(fake={len(fake_feats)}, gt={len(gt_feats)})"
            )

        mu1, sigma1 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
        mu2, sigma2 = gt_feats.mean(axis=0), np.cov(gt_feats, rowvar=False)

        from scipy import linalg

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        score = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return _require_finite("FID", score)
    except Exception as exc:
        raise MetricComputationError(f"FID computation failed: {exc}") from exc


def _compute_kid(fake_dir: str, gt_dir: str, device: str) -> tuple[float, float]:
    """Compute KID using InceptionV3 features."""
    try:
        from torchvision import models, transforms

        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()

        trans = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def extract(dir_path: str) -> np.ndarray:
            feats = []
            for f in sorted(os.listdir(dir_path)):
                if Path(f).suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                img = Image.open(os.path.join(dir_path, f)).convert("RGB")
                t = trans(img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = model(t)
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    feats.append(feat.cpu().numpy())
            return np.concatenate(feats, axis=0)

        x = extract(fake_dir)
        y = extract(gt_dir)
        return _kid_from_features(x, y)
    except Exception as exc:
        raise MetricComputationError(f"KID computation failed: {exc}") from exc


def _kid_from_features(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Polynomial-kernel MMD^2 (KID) from feature arrays."""
    degree, gamma, coef0 = 3, None, 1
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    def k(a, b):
        return (gamma * np.dot(a, b.T) + coef0) ** degree

    n_x, n_y = x.shape[0], y.shape[0]
    if n_x < 2 or n_y < 2:
        raise MetricComputationError(
            "KID requires at least two images in both fake and gt "
            f"(fake={n_x}, gt={n_y})"
        )
    k_xx = k(x, x)
    k_yy = k(y, y)
    k_xy = k(x, y)

    m_xx = (np.sum(k_xx) - np.sum(np.diag(k_xx))) / (n_x * (n_x - 1))
    m_yy = (np.sum(k_yy) - np.sum(np.diag(k_yy))) / (n_y * (n_y - 1))
    m_xy = np.sum(k_xy) / (n_x * n_y)
    mmd2 = m_xx + m_yy - 2 * m_xy

    # std estimation
    k_xx_diag = np.diag(k_xx)
    var_xx_term = (np.sum(k_xx, axis=1) - k_xx_diag) / (n_x - 1)
    var_xx = np.sum((var_xx_term - m_xx) ** 2) / (n_x - 1)

    k_yy_diag = np.diag(k_yy)
    var_yy_term = (np.sum(k_yy, axis=1) - k_yy_diag) / (n_y - 1)
    var_yy = np.sum((var_yy_term - m_yy) ** 2) / (n_y - 1)

    k_xy_row = np.sum(k_xy, axis=1) / n_y
    k_xy_col = np.sum(k_xy, axis=0) / n_x
    var_xy = np.sum((k_xy_row - m_xy) ** 2) / n_x + np.sum((k_xy_col - m_xy) ** 2) / n_y

    var = var_xx / n_x + var_yy / n_y + 2 * var_xy / (n_x * n_y)
    std = np.sqrt(max(var, 0))
    return (
        _require_finite("KID", mmd2 * 1000),
        _require_finite("KID std", std * 1000),
    )


def _compute_miod_pair(fake_t: torch.Tensor, gt_t: torch.Tensor) -> tuple[float, float]:
    """Compute mean optical density for a pair."""
    gen_255 = (fake_t * 255.0).clamp(min=1.0)
    real_255 = (gt_t * 255.0).clamp(min=1.0)
    od_gen = -torch.log10(gen_255 / 255.0)
    od_real = -torch.log10(real_255 / 255.0)
    return (
        _require_finite("generated mIoD", od_gen.mean().cpu()),
        _require_finite("ground-truth mIoD", od_real.mean().cpu()),
    )


def _compute_dab_kl(
    fake_t: torch.Tensor, gt_t: torch.Tensor, extractor: DABExtractor
) -> float:
    """Compute DAB KL divergence for a pair."""
    with torch.no_grad():
        dab_gen = extractor.extract_dab_intensity(fake_t, normalize="none")
        dab_real = extractor.extract_dab_intensity(gt_t, normalize="none")

    g = dab_gen[0].flatten().cpu().numpy()
    r = dab_real[0].flatten().cpu().numpy()

    n_bins = 256
    eps = 1e-10
    hist_range = (0, max(g.max(), r.max()) + 1e-6)
    hg, _ = np.histogram(g, bins=n_bins, range=hist_range, density=True)
    hr, _ = np.histogram(r, bins=n_bins, range=hist_range, density=True)
    hg = hg + eps
    hr = hr + eps
    hg = hg / hg.sum()
    hr = hr / hr.sum()
    return _require_finite("DAB KL", entropy(hg, hr))


def _compute_pathfid(
    fake_dir: str,
    gt_dir: str,
    device: str,
    foundation_model: str = "inception",
    checkpoint: str | None = None,
    hf_id: str | None = None,
) -> float:
    """Compute PathFID using pathology foundation model features.

    Args:
        fake_dir: Directory with generated images.
        gt_dir: Directory with ground-truth images.
        device: torch device string.
        foundation_model: One of ["inception", "uni", "univ2-h", "conch",
            "stainnet"] ("univ2" is accepted as an alias of "univ2-h").
        checkpoint: Local weights path; takes priority over ``hf_id``.
        hf_id: HuggingFace Hub model id; ``None`` uses the model default.

    Returns:
        PathFID value.
    """
    foundation_model = _EXTRACTOR_ALIASES.get(foundation_model, foundation_model)
    try:
        fake_feats, gt_feats = _extract_foundation_features(
            fake_dir, gt_dir, device, foundation_model, checkpoint, hf_id
        )

        if fake_feats is None or gt_feats is None:
            raise MetricComputationError("PathFID produced no valid feature arrays")

        if len(fake_feats) < 2 or len(gt_feats) < 2:
            raise MetricComputationError(
                "PathFID requires at least two images in both fake and gt "
                f"(fake={len(fake_feats)}, gt={len(gt_feats)})"
            )
        mu1, sigma1 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
        mu2, sigma2 = gt_feats.mean(axis=0), np.cov(gt_feats, rowvar=False)

        from scipy import linalg

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        score = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return _require_finite("PathFID", score)
    except MetricComputationError:
        raise
    except Exception as exc:
        raise MetricComputationError(f"PathFID computation failed: {exc}") from exc


# Registry of timm-based pathology extractors: the fallback architecture
# (used when no local config.json describes the checkpoint), the default
# HuggingFace Hub id, the env var fallback for the weights path, and the
# default directory (relative to PATHOLOGY_CHECKPOINT_ROOT) for offline
# checkpoints.
_TIMM_EXTRACTORS: dict[str, dict] = {
    "uni": {
        "arch": "vit_large_patch16_224",
        "default_hf_id": "MahmoodLab/uni",
        "env_var": "UNI_CHECKPOINT",
        "default_subdir": "uni",
        "init_values": 1e-5,
        "create_kwargs": {},
    },
    "univ2-h": {
        # UNI2-h is a custom ViT-H/14 (1536/24, SwiGLU-packed MLP, 8 register
        # tokens); plain timm "vit_giant_patch14_224" (1408/40) does NOT match.
        # Overrides follow the official mahmoodlab/UNI loading recipe:
        # https://github.com/mahmoodlab/UNI#2-downloading-weights--creating-model
        "arch": "vit_giant_patch14_224",
        "default_hf_id": "MahmoodLab/UNI2-h",
        "env_var": "UNI2_CHECKPOINT",
        "default_subdir": "univ2-h",
        "init_values": 1e-5,
        "create_kwargs": {
            "embed_dim": 1536,
            "depth": 24,
            "num_heads": 24,
            "mlp_ratio": 2.66667 * 2,
            "no_embed_class": True,
            "mlp_layer": "SwiGLUPacked",
            "act_layer": "silu",
            "reg_tokens": 8,
        },
    },
    "stainnet": {
        # StainNet-Small (IHC/special-stain DINO ViT-S/16),
        # https://github.com/WonderLandxD/StainNet
        # DINO ViT-S has no LayerScale -> init_values must be None.
        "arch": "vit_small_patch16_224",
        "default_hf_id": "JWonderLand/StainNet",
        "env_var": "STAINNET_CHECKPOINT",
        "default_subdir": "stainnet",
        "init_values": None,
        "create_kwargs": {},
    },
}

# Backward-compatible name aliases (canonical name on the right).
_EXTRACTOR_ALIASES: dict[str, str] = {"univ2": "univ2-h"}

# CONCH (CoCa ViT-B/16 visual tower) loading settings. CONCH is not a timm
# arch; it is loaded by :class:`_ConchVisualEncoder`.
_CONCH_DEFAULT_HF_ID = "MahmoodLab/CONCH"
_CONCH_ENV_VAR = "CONCH_CHECKPOINT"
_CONCH_DEFAULT_SUBDIR = "conch"
_CONCH_MEAN = (0.48145466, 0.4578275, 0.40821073)  # OpenAI CLIP stats
_CONCH_STD = (0.26862954, 0.26130258, 0.27577711)


def _load_state_dict_file(path: str) -> dict:
    """Load a ``.bin`` / ``.pt`` or ``.safetensors`` state dict from disk."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _as_state_dict_path(path: str) -> str | None:
    """Return the state-dict file for a checkpoint file or directory."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for fname in ("pytorch_model.bin", "model.safetensors"):
            candidate = os.path.join(path, fname)
            if os.path.isfile(candidate):
                return candidate
    return None


def _resolve_local_state_dict(
    checkpoint: str | None,
    env_var: str | None,
    default_subdir: str | None,
) -> str | None:
    """Resolve a local state-dict path for an extractor.

    Resolution order:

    1. ``checkpoint`` argument (a file, or a directory containing
       ``pytorch_model.bin`` / ``model.safetensors``).
    2. The extractor's env var (``UNI_CHECKPOINT`` / ``UNI2_CHECKPOINT`` /
       ``STAINNET_CHECKPOINT`` / ``CONCH_CHECKPOINT``).
    3. The default directory under ``PATHOLOGY_CHECKPOINT_ROOT``
       (e.g. ``{PATHOLOGY_CHECKPOINT_ROOT}/conch``).

    Returns:
        Path to the state dict, or ``None`` if no local checkpoint was found
        (the caller should then fall back to HuggingFace Hub).
    """
    if checkpoint:
        state_dict = _as_state_dict_path(checkpoint)
        if state_dict:
            return state_dict
        logger.warning("pathfid.checkpoint is not a valid checkpoint: %s", checkpoint)

    env_checkpoint = os.environ.get(env_var) if env_var else None
    if env_checkpoint:
        state_dict = _as_state_dict_path(env_checkpoint)
        if state_dict:
            return state_dict
        logger.warning("%s points to a missing checkpoint: %s", env_var, env_checkpoint)

    checkpoint_root = os.environ.get("PATHOLOGY_CHECKPOINT_ROOT")
    if checkpoint_root and default_subdir:
        default_dir = os.path.join(checkpoint_root, default_subdir)
        state_dict = _as_state_dict_path(default_dir)
        if state_dict:
            return state_dict
    return None


def _local_timm_config(
    state_dict_path: str, spec: dict[str, str | None]
) -> tuple[str, tuple, tuple, str]:
    """Read architecture and preprocessing from the checkpoint's config.json.

    Falls back to the registry spec (architecture) and ImageNet statistics
    when the file is absent or malformed.
    """
    arch = str(spec["arch"])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = "bilinear"
    config_path = os.path.join(os.path.dirname(state_dict_path), "config.json")
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            arch = cfg.get("architecture", arch)
            pre = cfg.get("pretrained_cfg", {})
            mean = tuple(pre.get("mean", mean))
            std = tuple(pre.get("std", std))
            interpolation = pre.get("interpolation", interpolation)
        except Exception as e:  # noqa: BLE001 - malformed optional config falls back
            logger.warning("Failed to parse %s (%s); using defaults", config_path, e)
    return arch, mean, std, interpolation


def _load_timm_extractor(
    model_name: str,
    device: str,
    checkpoint: str | None = None,
    hf_id: str | None = None,
) -> tuple[nn.Module, torchvision.transforms.Compose]:
    """Load a timm-based extractor (UNI / UNI2-h / StainNet) and its transform.

    Supports both local checkpoints and HuggingFace Hub download. The local
    state-dict path is resolved by :func:`_resolve_local_state_dict` (config
    checkpoint first, then env vars, then ``PATHOLOGY_CHECKPOINT_ROOT``).
    Architecture and preprocessing are read from the ``config.json`` sitting
    next to the weights when available. When no local checkpoint is found,
    weights are loaded from ``hf-hub:{hf_id or <model default>}``.

    To prepare a local checkpoint, download ``pytorch_model.bin`` and
    ``config.json`` from the model's HuggingFace repo (e.g. with
    ``huggingface-cli download`` or ``hf_hub_download``) and point
    ``pathfid.checkpoint`` at the directory.

    Args:
        model_name: Key in :data:`_TIMM_EXTRACTORS` ("uni", "univ2-h",
            "stainnet").
        device: torch device string.
        checkpoint: Local weights path (file or directory); highest priority.
        hf_id: HuggingFace Hub model id used when no local checkpoint exists.

    Returns:
        (model, transform) ready for inference on ``device``.
    """
    from torchvision import transforms

    spec = _TIMM_EXTRACTORS[model_name]
    state_dict_path = _resolve_local_state_dict(
        checkpoint, spec["env_var"], spec["default_subdir"]
    )

    if state_dict_path is not None:
        arch, mean, std, interpolation = _local_timm_config(state_dict_path, spec)
        logger.info(
            "Loading %s from local checkpoint: %s (arch=%s)",
            model_name,
            state_dict_path,
            arch,
        )
        create_kwargs = dict(spec.get("create_kwargs") or {})
        if isinstance(create_kwargs.get("mlp_layer"), str):
            create_kwargs["mlp_layer"] = getattr(
                timm.layers, create_kwargs["mlp_layer"]
            )
        model = timm.create_model(
            arch,
            img_size=224,
            init_values=spec.get("init_values", 1e-5),
            num_classes=0,
            dynamic_img_size=True,
            **create_kwargs,
        )
        model.load_state_dict(_load_state_dict_file(state_dict_path), strict=True)
        interp_mode = getattr(
            transforms.InterpolationMode,
            interpolation.upper(),
            transforms.InterpolationMode.BILINEAR,
        )
        trans = transforms.Compose(
            [
                transforms.Resize(224, interpolation=interp_mode),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        hub_id = f"hf-hub:{hf_id or spec['default_hf_id']}"
        logger.info("Loading %s from HuggingFace Hub: %s", model_name, hub_id)
        model = timm.create_model(
            hub_id,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            num_classes=0,
        )
        trans = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

    return model.to(device).eval(), trans


class _ConchAttentionalPooler(nn.Module):
    """Attentional pooler from MahmoodLab/CONCH.

    Adapted from ``open_clip_custom.transformer.AttentionalPooler``
    (https://github.com/mahmoodlab/CONCH).
    """

    def __init__(
        self,
        d_model: int = 512,
        context_dim: int = 768,
        n_head: int = 8,
        n_queries: int = 1,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim
        )
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(context_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        n = x.shape[1]
        q = self.ln_q(self.query).unsqueeze(1).repeat(1, n, 1)
        out = self.attn(q, x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class _ConchVisualEncoder(nn.Module):
    """CONCH visual tower: ViT-B/16 trunk + attentional pooler -> 512-d embedding.

    Reimplements the image path of MahmoodLab/CONCH's
    ``create_model_from_pretrained('conch_ViT-B-16', ...)`` followed by
    ``encode_image(..., proj_contrast=False, normalize=False)`` so PathFID can
    run without installing the ``conch`` package. The full CoCa checkpoint
    (visual + text) is accepted; only the visual trunk, the contrastive
    attentional pooler and the final LayerNorm are kept.
    """

    def __init__(self, state_dict_path: str) -> None:
        super().__init__()
        self.trunk = timm.create_model(
            "vit_base_patch16_224",
            img_size=448,  # checkpoint pos_embed has 28x28 patches
            num_classes=0,
            global_pool="",  # forward returns the full token sequence
            dynamic_img_size=True,
        )
        self.attn_pool_contrast = _ConchAttentionalPooler()
        self.ln_contrast = nn.LayerNorm(512)
        self._load_visual_weights(state_dict_path)

    def _load_visual_weights(self, state_dict_path: str) -> None:
        sd = _load_state_dict_file(state_dict_path)
        for prefix, module in (
            ("visual.trunk.", self.trunk),
            ("visual.attn_pool_contrast.", self.attn_pool_contrast),
            ("visual.ln_contrast.", self.ln_contrast),
        ):
            sub = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
            module.load_state_dict(sub, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.trunk(x)  # [N, L, 768]
        pooled = self.attn_pool_contrast(tokens)[:, 0]
        return self.ln_contrast(pooled)


def _load_conch_extractor(
    device: str,
    checkpoint: str | None = None,
    hf_id: str | None = None,
) -> tuple[nn.Module, torchvision.transforms.Compose]:
    """Load the CONCH visual encoder and its preprocessing transform.

    Local weights (config ``checkpoint`` first, then ``CONCH_CHECKPOINT``,
    then ``{PATHOLOGY_CHECKPOINT_ROOT}/conch``) take priority; otherwise
    ``pytorch_model.bin`` is downloaded from the HuggingFace Hub repo
    (``hf_id`` or the default ``MahmoodLab/CONCH`` — gated, requires an
    accepted license and an auth token).

    Returns:
        (model, transform) ready for inference on ``device``.
    """
    from torchvision import transforms

    state_dict_path = _resolve_local_state_dict(
        checkpoint, _CONCH_ENV_VAR, _CONCH_DEFAULT_SUBDIR
    )
    if state_dict_path is None:
        hub_id = hf_id or _CONCH_DEFAULT_HF_ID
        logger.info("Downloading CONCH weights from HuggingFace Hub: %s", hub_id)
        from huggingface_hub import hf_hub_download

        state_dict_path = hf_hub_download(hub_id, "pytorch_model.bin")

    logger.info("Loading CONCH visual encoder from %s", state_dict_path)
    model = _ConchVisualEncoder(state_dict_path)
    trans = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CONCH_MEAN, std=_CONCH_STD),
        ]
    )
    return model.to(device).eval(), trans


def _extract_foundation_features(
    fake_dir: str,
    gt_dir: str,
    device: str,
    model_name: str,
    checkpoint: str | None = None,
    hf_id: str | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract features using pathology foundation models.

    Args:
        fake_dir: Directory with generated images.
        gt_dir: Directory with ground-truth images.
        device: torch device string.
        model_name: One of ["inception", "uni", "univ2-h", "conch",
            "stainnet"] ("univ2" is accepted as an alias of "univ2-h").
        checkpoint: Local weights path (file or directory); takes priority
            over ``hf_id``. Used by the timm ("uni"/"univ2-h"/"stainnet")
            and "conch" extractors.
        hf_id: HuggingFace Hub model id; ``None`` uses each model's
            built-in default.

    Returns:
        (fake_features, gt_features) or (None, None) on failure.
    """
    from torchvision import transforms

    model_name = _EXTRACTOR_ALIASES.get(model_name, model_name)

    # Model-specific preprocessing
    if model_name == "inception":
        from torchvision import models

        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        trans = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    elif model_name in _TIMM_EXTRACTORS:
        model, trans = _load_timm_extractor(
            model_name, device, checkpoint=checkpoint, hf_id=hf_id
        )
    elif model_name == "conch":
        model, trans = _load_conch_extractor(device, checkpoint=checkpoint, hf_id=hf_id)
    else:
        raise MetricComputationError(f"Unknown PathFID foundation model: {model_name}")

    model = model.to(device).eval()

    def extract(dir_path: str) -> np.ndarray | None:
        feats = []
        for f in sorted(os.listdir(dir_path)):
            if Path(f).suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            try:
                img = Image.open(os.path.join(dir_path, f)).convert("RGB")
                t = trans(img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = model(t)
                    if isinstance(feat, tuple):
                        feat = feat[0]
                feats.append(feat.cpu().numpy())
            except Exception as exc:
                raise MetricComputationError(
                    f"PathFID feature extraction failed for {f}: {exc}"
                ) from exc
        if not feats:
            return None
        return np.concatenate(feats, axis=0)

    fake_feats = extract(fake_dir)
    gt_feats = extract(gt_dir)
    return fake_feats, gt_feats
