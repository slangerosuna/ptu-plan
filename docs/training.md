# Training on a rented GPU

Use a Linux GPU instance with CUDA-enabled PyTorch and the current stable Rust
toolchain. The models total 79,045 parameters; one modest GPU is sufficient.
Data generation and certificate auditing run on the CPU before the paid GPU
job. The prepared package contains symbolic observations and verified labels;
the GPU host does not need PANDA, the HDDL parser, IPC checkouts, or `../train`.

The WL pass is prepared separately under `runs/hilg-wl/`, preserving the original
trained models and benchmark reports in `runs/full/`. It contains **7,803 audited
rollouts from 3,156 problems in 34 domain groups**, versus 2,568 rollouts from
1,082 problems previously. Sources include 593 legacy problems (75 eligible
verified optimal PANDA problems) and 2,563 backward solutions, including 570
retained from the previous pass. All 3,156 collection jobs succeeded. Generation
attempted 4,096 solutions; rejections and quotas are recorded in `preparation.json`.
Backward solutions are feasible examples and are not labelled optimal.

| Evidence | Training | Validation | Test |
| --- | ---: | ---: | ---: |
| Domain groups | 24 | 5 | 5 |
| Audited rollouts | 5,334 | 1,399 | 1,070 |
| Merge/value examples | 5,019 | 1,324 | 931 |
| Interface examples | 5,926 | 1,469 | 1,345 |
| Latent transitions | 5,559 | 1,331 | 1,276 |

The exported manifest is authoritative after another preparation pass. See
[HILG/WL architecture](hilg-wl.md) for the paper adaptations, compatibility, and
limitations. A modest single GPU is sufficient; allow roughly 16 GB host RAM
for loading the dense portable records. No CUDA-dependent code runs during
preparation or native planner inference.

## Ready-to-transfer package

From the repository on the preparation machine:

```bash
python3 scripts/package_training.py
```

Copy `runs/hilg-wl-training.tar.gz` and its `.sha256` file to the GPU machine. After
selecting a PyTorch image or installing a CUDA wheel with the
[official installation selector](https://pytorch.org/get-started/locally/):

```bash
sha256sum -c hilg-wl-training.tar.gz.sha256
tar -xzf hilg-wl-training.tar.gz
cd ptu-plan-training
./scripts/train_gpu.sh
```

The launcher checks CUDA, builds the native coordinator, and trains every
learned component. Set `PTU_TRAIN_PYTHON=/path/to/venv/bin/python` if PyTorch is
installed in a separate environment. If using the full repository directly,
the equivalent command is:

```bash
cargo run --release --bin train -- fit --run-dir runs/hilg-wl --device cuda \
  --model-dir runs/hilg-wl/gpu-models
```

## Models and defaults

| Component | Parameters | Objective |
| --- | ---: | --- |
| HILG/WL projection + merge/value heads | 73,154 | Pairwise ranking and censored Huber effort regression |
| Interface predictor | 643 | Weighted Unknown/False/True cross entropy |
| Latent transition model | 5,248 | Embedding MSE against the selected frozen graph encoder |

The WL projection has width 32 and three dense layers. Training uses float32,
batches of 64, AdamW with learning rate `0.001`, weight decay `0.0001`, and
gradient clipping at 5. Cosine decay reaches roughly one tenth of the initial
rate. The ceiling is 100 epochs per component, with validation patience 12
after at least 20 epochs. Domains are sampled uniformly, then examples are
sampled within domains. Interface class weights are `[0.25, 1, 1]`.

The merge/value model and interface predictor train first. The latent model
then trains against the selected merge encoder, held fixed throughout that
phase. This keeps its embedding targets consistent. Symbolic partitioning,
exact solvers, POCL reconciliation, certificate checks, and conflict-core cuts
have no trainable parameters. They provide features, labels, and validated
search behavior around the learned models.

The repository also contains a completed CPU reference run in
`runs/hilg-wl/models/`. Use the separate `gpu-models` directory above for a fresh
GPU fit. The transfer archive contains prepared inputs and source, without the
CPU optimizer states, so its launcher starts a fresh fit.

## Outputs, reports, and resume

Outputs are in `runs/hilg-wl/models/`:

- `merge-policy.json`, `interface-policy.json`, and `latent-policy.json` load in
  the native Rust planner.
- `training-report.json` records loss by split, ranking accuracy, effort error,
  interface confusion matrices, and latent error against identity/pair-mean
  embeddings. Lower test loss alone does not establish a planner speedup.
- `*-state.pt` files contain the latest optimizer state and the best validation
  state. Rerun the same command to resume an interrupted job. Resume with these
  files; JSON optimizer exports are for native checkpoint compatibility.
- `bundle.json` is published after native checkpoint and inference checks pass.

Model selection uses validation domains. Test domains are reported after
selection and never enter optimizer updates. Every synthetic sibling of a
domain remains in that domain's split. The report and manifest record exact
source IDs and checksums. Initial forward passes, losses, and gradients are
checked against native float64 references; final predictions are checked in
Rust after float32 training.

Use a new `--model-dir` when changing data or hyperparameters. To resume a
particular package after another preparation job has published newer data,
pass `--package ID`. A running fit pins its package, so separate preparation
can continue without changing its training examples. Newly generated data is
included in the next fit; it is not streamed into an existing optimizer run.

## Preparing another pass

```bash
cargo run --release --bin train -- prepare --workers 4
```

Defaults select up to 1,024 multi-root legacy solutions, prioritizing verified
optimal PANDA records within each domain, and attempt 4,096 new backward
solutions. Templates span the available domain hashes; quotas may remain
unfilled because regression, bounds, and deduplication reject attempts.
Generation uses the configured two-to-eight-root range and runs domains in
parallel. The collector explores up to six merge levels and twelve candidates
per state. Full traces retain scoped local witnesses for interface labels. Each oracle
action has a 350 ms master budget and 100 ms local budget; timeouts remain
censored or unknown evidence. Successful jobs are immutable and reused.
Check `preparation.json` for actual counts and generation shortfalls.

`train all --device cuda` combines preparation and fitting on one machine.
To change architecture, regenerate the portable export with
`train export --hidden 32 --layers 3` and use a new model directory. A short
CPU check uses `train fit --device cpu --epochs 2 --model-dir runs/hilg-wl/check`.

## Planner-return fine-tuning

After supervised fitting, use `train rl --device cuda` to optimize the WL
encoder and merge/value policy against actual planner outcomes with dynamic
repartitioning. This requires the matching source store. See
[RL and escalation](repartition-rl.md) for the objective, held-out selection,
latent-model refresh, resume, and the `--include-rl` transfer archive.
