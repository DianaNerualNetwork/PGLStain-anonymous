# Released PGLStain checkpoints

This release includes two Full PGLStain generators. No comparison-method
weights are distributed.

| Directory | Dataset and stain | Saved epoch |
|---|---|---:|
| `checkpoints/mist_er_full` | MIST–ER | 100 |
| `checkpoints/acrobat_er_full` | ACROBAT–ER | 20 |

Each directory contains the generator weights, an anonymized `config.yaml`
with the fields needed for inference, and exporter-generated metadata recording
file hashes. Verify the files against those hashes when checking an archive or
copy. The supplied configurations describe the released generators; they are
not complete training configurations. Optimizer state and the other training
networks are omitted, so these checkpoints cannot resume training.

## Training provenance

Both generators were trained with the Full C+S+M+R objective: PECC correspondence
and structure terms plus MRSA marginal and relational terms. Their shared
settings were a 512 × 512 training canvas, seed 42, batch size 1, learning rate
`0.0002`, `nce_idt=true`, and `lambda_gp=10`. The four module weights were
`0.1`, `0.1`, `5.0`, and `1.0`, respectively. Relational transport used
`mass_penalty=2.0` and `sinkhorn_epsilon=0.05` with unbalanced transport.

MIST–ER used 100 epochs, with 50 constant-learning-rate epochs followed by
50 decay epochs. The corresponding complete training recipe is
`mist/pglstain/full`; see [configuration details](CONFIGURATIONS.md).

ACROBAT–ER was configured for 20 epochs, with 10 constant-learning-rate epochs
followed by 10 decay epochs. Training stopped during epoch 16 and was resumed
from a saved checkpoint. The trainer restarted that epoch's data traversal,
replaying part of epoch 16. This checkpoint therefore does not represent an
uninterrupted 20-epoch run. There is no ACROBAT PGLStain training preset in this
release; the inference configuration does not replace that missing recipe.

## Prediction

Run these commands from the release root after installation:

```bash
puzzlestain-predict checkpoints/mist_er_full MIST ER pglstain \
  /path/to/valA --gt-dir /path/to/valB \
  --cache-root ./results --device cuda:0

puzzlestain-predict checkpoints/acrobat_er_full ACROBAT ER pglstain \
  /path/to/valA --gt-dir /path/to/valB \
  --cache-root ./results --device cuda:0
```

Replace the source and reference directories with the corresponding dataset's
held-out patches. For CPU inference, replace `--device cuda:0` with
`--device cpu`.

The default prediction mode directly resizes images to the checkpoint's
512 × 512 canvas and runs the generator in evaluation mode. Keep these defaults
for the released inference protocol; do not add `--train-mode`. Other physical
resolution or sliding-window modes define a different preprocessing protocol.

Outputs are organized under `results/MIST/ER/pglstain/` and
`results/ACROBAT/ER/pglstain/`, with `fake/`, `source/`, and `gt/` subdirectories.
For example:

```bash
puzzlestain-eval image-quality dataset=MIST stain=ER model=pglstain \
  cache_root=./results

puzzlestain-eval image-quality dataset=ACROBAT stain=ER model=pglstain \
  cache_root=./results
```

See [VALIDATION.md](../VALIDATION.md) for the checks actually performed on this
release. Dataset images and original split manifests are not included; access
to weights alone does not establish an exact reproduction of published results.
