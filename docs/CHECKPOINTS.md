# Released PGLStain checkpoints

This release includes five Full PGLStain generators. No comparison-method
weights are distributed.

| Directory | Dataset and stain | Saved epoch | Training UGW mass penalty (`rho`) |
|---|---|---:|---:|
| `checkpoints/mist_er_full` | MIST–ER | 100 | 2.0 |
| `checkpoints/mist_her2_full` | MIST–HER2 | 100 | 1.0 |
| `checkpoints/mist_pr_full` | MIST–PR | 100 | 1.0 |
| `checkpoints/mist_ki67_full` | MIST–Ki67 | 100 | 1.0 |
| `checkpoints/acrobat_er_full` | ACROBAT–ER | 20 | 2.0 |

Each directory contains the generator weights, an anonymized `config.yaml`
with the fields needed for inference, and `export_metadata.json` recording
file sizes and SHA-256 hashes. Verify downloaded files against these hashes.
The supplied configurations describe the released generators; they are not
complete training configurations. Optimizer state and other training networks
are omitted, so these checkpoints cannot resume training.

## Training provenance

All five generators were trained with the Full C+S+M+R objective: PECC
correspondence and structure terms plus MRSA marginal and relational terms.
Their shared settings were a 512 × 512 training canvas, seed 42, batch size 1,
learning rate `0.0002`, `nce_idt=true`, and `lambda_gp=10`. The four module
weights were `0.1`, `0.1`, `5.0`, and `1.0`, respectively. Relational transport
was unbalanced, with `sinkhorn_epsilon=0.05`.

The MIST generators used 100 epochs: 50 constant-learning-rate epochs followed
by 50 decay epochs. The MIST–ER training recipe is `mist/pglstain/full`.
The released MIST–HER2, PR, and Ki67 checkpoints correspond to the reported
cross-stain results and used **`mass_penalty=1.0`**, not the ER default of 2.0.
Later `rho=2` cross-stain reruns are not substituted for these checkpoints.
When adapting the Full training recipe to these three stains, set
`loss.components.mrsa_relational.mass_penalty=1.0` and use the matching stain
and dataset directories. Changing an inference config cannot retroactively
change the objective used to train a generator. The HER2 checkpoint here is
from MIST, not BCI. See [configuration details](CONFIGURATIONS.md).

ACROBAT–ER was configured for 20 epochs: 10 constant-learning-rate epochs
followed by 10 decay epochs. Training stopped during epoch 16 and was resumed
from a saved checkpoint. The trainer restarted that epoch's data traversal,
replaying part of epoch 16. This checkpoint therefore does not represent an
uninterrupted 20-epoch run. There is no ACROBAT PGLStain training preset in this
release; the inference configuration does not replace that missing recipe.

## Prediction

Run the appropriate command from the release root after installation:

```bash
puzzlestain-predict checkpoints/mist_er_full MIST ER pglstain \
  /path/to/MIST/ER/TrainValAB/valA --gt-dir /path/to/MIST/ER/TrainValAB/valB \
  --cache-root ./results --device cuda:0

puzzlestain-predict checkpoints/mist_her2_full MIST HER2 pglstain \
  /path/to/MIST/HER2/TrainValAB/valA --gt-dir /path/to/MIST/HER2/TrainValAB/valB \
  --cache-root ./results --device cuda:0

puzzlestain-predict checkpoints/mist_pr_full MIST PR pglstain \
  /path/to/MIST/PR/TrainValAB/valA --gt-dir /path/to/MIST/PR/TrainValAB/valB \
  --cache-root ./results --device cuda:0

puzzlestain-predict checkpoints/mist_ki67_full MIST Ki67 pglstain \
  /path/to/MIST/Ki67/TrainValAB/valA --gt-dir /path/to/MIST/Ki67/TrainValAB/valB \
  --cache-root ./results --device cuda:0

puzzlestain-predict checkpoints/acrobat_er_full ACROBAT ER pglstain \
  /path/to/ACROBAT/ER/TrainValAB/valA --gt-dir /path/to/ACROBAT/ER/TrainValAB/valB \
  --cache-root ./results --device cuda:0
```

Replace the source and reference directories with the corresponding dataset's
held-out patches. For CPU inference, replace `--device cuda:0` with
`--device cpu`. The `--gt-dir` option may be omitted for prediction without
reference IHC. The positional stain name labels the output directory; select
the correct trained checkpoint rather than expecting that label to change the
generator's target stain.

The default prediction mode directly resizes images to the checkpoint's
512 × 512 canvas and runs the generator in evaluation mode. Keep these defaults
for the released inference protocol; do not add `--train-mode`. Other physical
resolution or sliding-window modes define a different preprocessing protocol.

Outputs are organized under `results/<dataset>/<stain>/pglstain/`, with
`fake/`, `source/`, and (when supplied) `gt/` subdirectories. For example:

```bash
puzzlestain-eval image-quality dataset=MIST stain=HER2 model=pglstain \
  cache_root=./results
```

See [VALIDATION.md](../VALIDATION.md) for the checks actually performed on this
release. Dataset images and original split manifests are not included; access
to weights alone does not establish an exact reproduction of published results.
