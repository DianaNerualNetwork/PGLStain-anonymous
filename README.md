# PGLStain — Anonymous Review Artifact

Research implementation accompanying **Beyond Spatial Correspondence:
Optical-Density-Guided Marginal and Relational Learning for Weakly-Paired
H&E-to-IHC Virtual Staining**.

This standalone source snapshot contains PGLStain and eleven comparison
methods: **CUT, CycleGAN, UNSB, PyramidP2P, ASP, PPT, PSPStain, MDCL, SIMGAN,
USIGAN, and M2PL-GAN**. The Python package and console commands retain the
`puzzlestain` name for compatibility. Two inference-only Full PGLStain
checkpoints are included for MIST–ER and ACROBAT–ER. No comparison-method
weights, author identities, development Git history, datasets, or private
training metadata are included.

## Method and organization

- **PECC** combines feature-context and OD-calibrated graphs, continuous
  candidate weighting, and a separate OD-only graph-summary constraint.
- **MRSA** aligns tissue-block FOD marginals and intra-image OD-state relations
  using quantile matching and unbalanced Gromov–Wasserstein transport.

```text
puzzlestain/
  models/cut_family/pglstain.py     PGLStain model and training strategy
  models/loss/pecc.py               PECC correspondence and ODStruct losses
  models/loss/mrsa.py               MRSA marginal and relational losses
  models/{cut,cyclegan,pix2pix,diffusion}_family/
  configs/experiment/mist/          MIST comparison and PGLStain recipes
  configs/experiment/acrobat/       ACROBAT comparison recipes
  data/                            Patch loading and preprocessing
  training/                        Shared training loop
  inference/                       Checkpoint inference
  evaluation/                      Image-quality and staining metrics
checkpoints/{mist_er_full,acrobat_er_full}/  Released Full generators
docs/CONFIGURATIONS.md              Recipe details and external weights
docs/CHECKPOINTS.md                 Checkpoint provenance and prediction
scripts/export_inference_checkpoint.py      Minimal checkpoint exporter
tests/                             Offline synthetic validation
```

## Installation

Use a new Python 3.10+ environment, separately from other installations of
the same Python package. Choose a PyTorch/CUDA build appropriate for your
machine before installing if necessary.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[evaluation,dev]'
puzzlestain-run --help
```

GPU training is recommended. The selected methods do not require DGL,
PyTorch Geometric, POT, or diffusers. Optional `nuclei` and `tracking` extras
are not needed for the supplied training recipes. Online experiment tracking
is disabled by default. See [VALIDATION.md](VALIDATION.md) for the scope of
local checks; installation from an entirely clean environment is a separate
check from those tests.

## Data

Obtain MIST and ACROBAT from their authorized providers. This export does not
redistribute slides, patches, or original split manifests. For the paired
patch recipes, prepare matching source/reference filenames in this layout:

```text
ER/TrainValAB/
  trainA/   H&E training patches
  trainB/   corresponding weakly paired IHC patches
  valA/     H&E validation patches
  valB/     corresponding weakly paired IHC patches
```

Keep the intended data partitions fixed. A patch-level MIST split must not be
described as WSI-disjoint. The released checkpoints alone do not establish exact paper reproduction:
the original split lists, preprocessing provenance, and matching evaluation
settings are also required. Original data and split manifests are not bundled.

## Training

PGLStain on MIST–ER:

```bash
puzzlestain-run mist/pglstain/full \
  data.dataroot=/path/to/MIST/ER/TrainValAB \
  'data.selected_stains=[ER]'
```

The other canonical own-method presets are `mist/pglstain/baseline`,
`mist/pglstain/pecc_only`, and `mist/pglstain/mrsa_only`.

Each requested comparison method has both a `mist/` and an `acrobat/` recipe:

```bash
puzzlestain-run mist/pspstain \
  data.dataroot=/path/to/MIST/ER/TrainValAB \
  'data.selected_stains=[ER]'

puzzlestain-run acrobat/usigan \
  data.dataroot=/path/to/ACROBAT/ER/TrainValAB \
  'data.selected_stains=[ER]'
```

Available baseline suffixes are `cut`, `cyclegan`, `unsb`, `pyramid_p2p`,
`asp`, `ppt`, `pspstain`, `mdcl`, `simgan`, `usigan`, and `m2plgan`.
PyramidP2P's model registry ID is `pix2pix_pyramid`. CPT is retained as an
internal backbone dependency, not an additional requested comparison.

The copied recipes preserve their original numerical settings. They are not
all identical: for example, SIMGAN/USIGAN use batch size two, and PSPStain
uses a different learning rate. Do not infer matched hyperparameters merely
from the shared framework. See [configuration details](docs/CONFIGURATIONS.md).
No ACROBAT PGLStain preset was present in the reference snapshot, so this
export does not invent one. Newer research-only ablation variants are not
included in the canonical PGLStain snapshot.

## External weights

PSPStain training requires a separately obtained segmentation checkpoint.
Set `PATHOLOGY_CHECKPOINT_ROOT` so that the file is available at
`$PATHOLOGY_CHECKPOINT_ROOT/pspstain/MIST_unet_seg.pth`, or override the
configured `loss.components.psp_pathology.seg_pretrained_path`.
PPT initializes pretrained torchvision VGG19 and may download its weights.
Perceptual metrics and PathFID also require pretrained feature extractors;
obtain them under the relevant upstream terms. These external dependency
weights are not bundled; only the two PGLStain Full generators are supplied.

## Inference and evaluation

Use the released MIST–ER checkpoint directly:

```bash
puzzlestain-predict checkpoints/mist_er_full MIST ER pglstain \
  /path/to/MIST/ER/TrainValAB/valA \
  --gt-dir /path/to/MIST/ER/TrainValAB/valB \
  --cache-root ./results

puzzlestain-eval image-quality \
  dataset=MIST stain=ER model=pglstain cache_root=./results
```

For ACROBAT–ER, use `checkpoints/acrobat_er_full` and the corresponding
ACROBAT source/reference directories. Both checkpoints use the default
512 × 512 direct-resize protocol and evaluation mode. Their minimal configs
are for inference, not training resumption. See [checkpoint details](docs/CHECKPOINTS.md)
for full commands, hashes, and the ACROBAT training-recovery caveat.

Other evaluation presets are `perception`, `pathological-relevance`,
`pathfid`, and `all`. Configure the desired extractor and weights explicitly
for PathFID; the base evaluation configuration lists multiple extractors,
whereas the paper's PathFID uses StainNet. Use the checkpoint's intended
resolution and preprocessing rather than silently changing inference mode.

Fiji H-score/positive-nuclei/area measurement macros, the 55-case aggregation
scripts, WSI confidence-interval analyses, and paper-result CSVs are **not**
included in the two source snapshots exported here. The image-metric CLI
should not be mistaken for that separate Fiji analysis pipeline.

## Validation and release checklist

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

The offline suite uses synthetic tensors and configuration checks. In addition,
both released checkpoints were tested on three real validation patches each:
the six floating-point predictions were bitwise identical to the original
implementation, and the prediction CLI produced six valid PNGs. This is an
inference-equivalence check, not a new full-dataset evaluation or clinical
validation. See [VALIDATION.md](VALIDATION.md).

Before distributing, inspect all links and files, confirm redistribution
permissions, and verify downloads in a logged-out browser. GitHub hosting and
Anonymous GitHub proxy access are separate: confirm that the proxy permits
complete checkpoint downloads and compare their SHA-256 hashes. Keep the
reviewed source snapshot fixed for submission.

## Attribution and license

See [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE).
Third-party code, methods, and pretrained weights retain their upstream
terms; the framework license does not override method-specific restrictions.
