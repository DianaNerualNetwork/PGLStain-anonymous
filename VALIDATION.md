# Validation of the anonymous release

These checks validate packaging, selected implementation behavior, and the
released generators' inference outputs. They do not constitute a new
reproduction of the paper's full-dataset numerical results.

## Actual checkpoint inference

The two released Full checkpoints were tested on real held-out inputs:

| Checkpoint | Test patches | Output | Strict generator loading | Maximum absolute difference from original float prediction |
| --- | ---: | --- | --- | ---: |
| MIST–ER, saved epoch 100 | 3 | 512 × 512 RGB PNG | Passed | 0.0 |
| ACROBAT–ER, saved epoch 20 | 3 | 512 × 512 RGB PNG | Passed | 0.0 |

For each dataset, the first, middle, and last matched validation filenames
in sorted order were selected, rather than choosing outputs by visual quality.
The source validation directories contained 1,000 MIST pairs and 8,314 ACROBAT
pairs; only three pairs from each were used for this release smoke test.

- The real `puzzlestain-predict` console command completed with three
  predictions per dataset. Each `source/`, `fake/`, and `gt/` output directory
  contained exactly three images.
- The generated PNGs reopened as RGB, had the expected dimensions, and were
  nonconstant. All floating-point outputs were finite. Two representative
  generated images were also inspected visually; this is not pathology scoring.
- Original-project predictions were computed with the original full training
  checkpoints, while release predictions used the anonymized package and
  generator-only checkpoints. Both used the same per-dataset GPU, direct
  resize, evaluation mode, and preprocessing.
- All six floating-point outputs were bitwise identical in this environment.
  The actual CLI PNGs also matched the processor's conversion of those floats.
  CUT/PGLStain's `return_normalized=True` in the default resize mode preserves
  the model-space range `[-1, 1]`; PNG export maps this to `[0, 255]`.
- Each export retained all 76 generator state tensors and their PyTorch
  version metadata, including spectral-normalization buffers. Keys, dtypes,
  tensor values, and metadata matched the original checkpoint exactly.
  Neither selected checkpoint contained an EMA shadow.
- Only the generator was retained. Each `trainer_state.pt` is 31,576,491 bytes
  (approximately 30.1 MiB), with a per-file SHA-256 record in
  `export_metadata.json`. Optimizers, discriminators, feature heads, graph
  modules, private paths, and training logs were not exported.

The real input images, reference images, generated images, and local execution
logs are not included in this repository. See [CHECKPOINTS.md](docs/CHECKPOINTS.md)
for the public prediction commands and checkpoint provenance, including the
ACROBAT training-recovery caveat. Matching outputs on six patches does not
establish cross-device bitwise reproducibility or dataset-wide accuracy.

## Offline tests and packaging

- All twelve requested method IDs (PGLStain plus eleven baselines) register
  with a training strategy, and the training/inference entry modules import.
- All 27 training presets compose with Hydra, resolve their configuration,
  and refer to registered models and losses. A separate audit checked them
  against the structured configuration schema. The six evaluation
  configuration files also resolve successfully.
- Synthetic PECC/MRSA tests cover finite losses, backward propagation,
  reference detachment where implemented, and the tested permutation-invariant
  objectives. They do not imply invariance to arbitrary image perturbations.
- Checkpoint-export tests cover the minimal schema, parameter aliases, EMA,
  source preservation, tensor cloning, invalid configurations/states, overwrite
  protection, and the export CLI.
- The combined offline suite completed with **67 passed**. The 43 warnings
  were upstream PyTorch deprecation warnings for `torch.jit.script`.
- An offline wheel build succeeded and included all 57 YAML configurations
  and the PGLStain, PECC, and MRSA implementation modules. Checkpoints are
  repository assets, not Python wheel package data.
- Baseline numerical training settings were preserved from their supplied
  source recipes; private paths and online tracking settings were adjusted.
- Package imports were explicitly checked to originate from this release.
- A clean clone of the prepared Git commit also passed all 67 tests and both
  real prediction commands (three patches per dataset). All six PNG files were
  byte-for-byte identical to the previously checked outputs, confirming that
  required source files and weights were included in Git.

Run the offline tests from the project root after installing the development
extra:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

## Validation environment

The checks used an existing Python 3.11 environment with PyTorch 2.12.1,
torchvision 0.27.1, NumPy 1.26.4, Hydra 1.3.3, OmegaConf 2.3.1, and pytest 9.1.1.
Real-image inference used NVIDIA RTX 4090 GPUs with four CPU execution threads
per process. The wheel was built with setuptools 81.0.0. These are release
validation versions, not a claim about the environment of every paper run.

## Not validated or not bundled

- Installation and dependency resolution in a completely fresh environment.
- Full retraining, full-dataset metric recomputation, comparison-method
  checkpoint inference, or compatibility with other GPU/CUDA configurations.
- Pretrained-weight downloads or optional evaluation backbones. PSPStain and
  PPT have external weight requirements; consult the configuration guide.
- Datasets, exact split manifests, comparison-method weights, and paper-result
  CSVs; the separate Fiji/55-case aggregation and WSI confidence-interval
  pipelines; and newer research-only ablation branches.
- An ACROBAT PGLStain training preset: the released inference checkpoint does
  not supply a complete training or within-epoch resume configuration.
- Anonymous-proxy checkpoint download availability or complete legal clearance
  of every third-party component. Review the retained upstream notices.

## Anonymization scope

The release omits development Git history, first-party author/contact metadata,
private absolute paths, local environments, and private training metadata.
Third-party attribution and the existing `puzzlestain` package namespace are
retained. Automated text checks are not a guarantee against every possible
identity link. Inspect the final files and hosting account before submission,
and validate downloads through the actual anonymous reviewer link.
