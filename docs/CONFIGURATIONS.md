# Experiment configurations

This release contains 22 comparison presets from the comparison framework
snapshot, four canonical PGLStain variants from the PGLStain reference snapshot,
and one CPT preset needed by the PGLStain baseline. The exported recipes preserve
their source parameter values; portability changes affect paths and comments.

## Comparison methods

Each selector below is available under both `mist/` and `acrobat/`.
The model name is the registry key stored in checkpoints.

| Method | Selector suffix | Model name |
|---|---|---|
| CUT | `cut` | `cut` |
| CycleGAN | `cyclegan` | `cyclegan` |
| UNSB | `unsb` | `unsb` |
| PyramidP2P | `pyramid_p2p` | `pix2pix_pyramid` |
| ASP | `asp` | `asp` |
| PPT | `ppt` | `ppt` |
| PSPStain | `pspstain` | `pspstain` |
| MDCL | `mdcl` | `mdcl` |
| SIMGAN | `simgan` | `simgan` |
| USIGAN | `usigan` | `usigan` |
| M2PLGAN | `m2plgan` | `m2plgan` |

For example:

```bash
puzzlestain-run mist/cut \
  data.dataroot=/path/to/MIST/ER/TrainValAB \
  'data.selected_stains=[ER]'

puzzlestain-run acrobat/cut \
  data.dataroot=/path/to/ACROBAT/ER/TrainValAB \
  'data.selected_stains=[ER]'
```

The aligned data root contains `trainA/` and `trainB/`; held-out inference may
use `valA/` and `valB/`. Source and target filenames must meet the data loader's
pairing convention. These folders do not supply or define a patient-level split.

## Preserved training schedules

All comparison recipes use a 512 × 512 training canvas and inherit seed 42.
The epoch schedule is linear; the two schedule columns specify epochs at the
initial learning rate and epochs assigned to decay.

| Dataset and methods | Total epochs | Constant | Decay |
|---|---:|---:|---:|
| MIST: all eleven methods | 100 | 50 | 50 |
| ACROBAT: UNSB, ASP, PPT | 20 | 0 | 20 |
| ACROBAT: the other eight methods | 20 | 10 | 10 |

The comparison recipes use learning rate `0.0002`, except PSPStain, which uses
`0.0001`. Batch size is 1, except SIMGAN and USIGAN, which use 2. These exceptions
apply to both datasets. Model-specific discriminator rates, transforms, losses,
dropout, and normalization remain as specified in each YAML; the table is not a
replacement for the full recipe.

MIST comparison presets use panel `breast_mist_4plex_20x` and target resolution
`0.9322` microns per pixel. ACROBAT presets use `breast_acrobat_4plex_10x` and
`1.84` microns per pixel. These values describe the configured patch protocol;
confirm that prepared data matches it before using scale-aware inference.

## PGLStain variants

| Selector | Model | Active additions to CPT |
|---|---|---|
| `mist/pglstain/baseline` | `cpt` | None |
| `mist/pglstain/pecc_only` | `pglstain` | PECC |
| `mist/pglstain/mrsa_only` | `pglstain` | MRSA |
| `mist/pglstain/full` | `pglstain` | PECC and MRSA |

Use the same data overrides shown above, for example:

```bash
puzzlestain-run mist/pglstain/full \
  data.dataroot=/path/to/MIST/ER/TrainValAB \
  'data.selected_stains=[ER]'
```

The four variants retain 512 × 512 training, seed 42, batch size 1, learning rate
`0.0002`, and 100 epochs with a 50 + 50 schedule. They use `nce_idt=true` and
`lambda_gp=10`. The MRSA variants retain `sinkhorn_epsilon=0.05`.
The additional `mist/cpt` selector is the baseline's inherited
configuration dependency, rather than a twelfth comparison method.

The PGLStain reference snapshot contains legacy metadata: `full` and `pecc_only`
use panel `breast_ihc_4plex_40x` and leave `target_mpp` unset; `mrsa_only` inherits
these fields from `full`. The `baseline` inherits the MIST metadata from CPT.
These differences were preserved. Supply a verified `--target-mpp` when using
strict scale-aware inference with a checkpoint that lacks that metadata.

No ACROBAT PGLStain preset was present in the supplied reference snapshot, so
none is included or represented as a validated experiment.

## Weights, outputs, and evaluation

Five Full PGLStain generators are bundled for inference; see
[CHECKPOINTS.md](CHECKPOINTS.md). Comparison-method weights and third-party
pretrained weights are not bundled. PSPStain requires its third-party
segmentation checkpoint. Its default is
`./checkpoints/pspstain/MIST_unet_seg.pth`, or
`$PATHOLOGY_CHECKPOINT_ROOT/pspstain/MIST_unet_seg.pth` when that environment
variable is set. Override
`loss.components.psp_pathology.seg_pretrained_path` to use another verified file.
The upstream weight source is linked in `configs/loss/pspstain.yaml`.

PathFID defaults to UNI, UNI v2-H, CONCH, and StainNet extractors. Obtain the
appropriate weights and any required access separately. Its local directories
are `uni/`, `univ2-h/`, `conch/`, and `stainnet/` under
`PATHOLOGY_CHECKPOINT_ROOT` (default `./checkpoints`); override
`pathfid.checkpoint` for a different arrangement. Additional perceptual metrics
may also require their own pretrained weights.

`PATHOLOGY_CACHE_ROOT` selects inference/evaluation image storage;
`PATHOLOGY_METRIC_ROOT` selects metric output storage. Both can also be supplied
through their CLI configuration options. Online experiment tracking is disabled
by default in every training preset.

The six evaluation presets are `default`, `image-quality`, `perception`,
`pathological-relevance`, `pathfid`, and `all`. Image-quality evaluation is a
useful initial check without PathFID foundation-model weights:

```bash
puzzlestain-eval image-quality dataset=MIST stain=ER model=cut \
  cache_root=./results output_dir=./metrics
```

Hydra composition and structured-schema validation passed for all 27 training
selectors; all six evaluation presets resolve. These checks do not establish
training convergence or agreement with published scores. Dataset split
manifests, data, and training-resumption state are not distributed here; exact
paper replication also depends on matching preprocessing, evaluation, and
runtime conditions.
