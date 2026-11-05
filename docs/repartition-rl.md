# Dynamic repartitioning and planner-return fine-tuning

## Shared-budget escalation

Enable escalation around sequential planning:

```bash
cargo run --release -- tests/fixtures/interleaving.json \
  --sequential --micro-max-size 1 --max-boundary-atoms 0 \
  --repartition --analysis-json --master-trace
```

`--repartition` starts with the selected spectral/greedy/learned partitions.
Boundary-limit events prioritize legal contractions touching the blocked block.
Other scoped failures trigger structural coarsening. Unknown local searches can
split their block into singleton roots once; if further coarsening is unavailable,
a larger local time slice can retry the same partition. Canonical member sets
prevent split/merge cycles. Merges retain the normal adjacency, size, and ordering
checks. If no legal contraction remains, a whole-network restart can bridge
disconnected blocks, provided all roots fit the cluster size limit. This explicit
fallback can recover cross-root interleavings; `--no-monolithic-fallback` disables it.

`--master-time-limit-ms`, `--max-local-calls`, and `--max-master-nodes` are shared
across attempts, including conflict-core probes. Defaults allow six attempts,
starting at `--repartition-stage-ms 350` per attempt. When time is the binding
limit and structural repairs are exhausted, stage and local time slices double,
up to `--repartition-max-stage-ms 4000`, within the shared master budget. Hard
expansion limits do not trigger time-only retries. Grounding and the
initial learned partition selection remain outside the master time budget.
Limits are cooperative; certificate checking and bookkeeping can add overhead.

Each attempt starts fresh proof caches. No-goods are not transferred between
partitions. Successful certificates undergo whole-problem replay. Failure to
find a repair returns `BudgetExhausted`, rather than global HTN infeasibility.
JSON includes `repartition.attempts`, their scoped statuses, actual budgets,
blocked partitions, and aggregate `work`. Full traces are nested per attempt.
The ordinary `sequential` result retains the final attempt's scoped evidence.

## On-policy RL

For background generation on a rented host, use `./scripts/train_gpu_online.sh`.
The [online workflow](online-training.md) reserves CPU cores, publishes certified
Train-only batches, and feeds them into RL between epochs.

Start from a supervised model bundle and its matching immutable source store:

```bash
cargo run --release --bin train -- rl \
  --run-dir runs/hilg-wl --device cuda \
  --model-dir runs/hilg-wl/rl-models
```

Use `--python /path/to/venv/bin/python` for a separate PyTorch environment,
`--device cpu` for local training, and `--parent-models PATH` to select another
pretrained bundle. Original models are preserved. The source store is required:
RL invokes the actual planner rather than replaying the supervised labels.

The complete WL projection and shared merge policy/value heads receive gradients
from sampled legal merge/STOP trajectories. The native `rl-environment` process
executes those choices, then runs escalation and certificate replay. Its signed
terminal reward is +1 for replay-verified success and -1 for an unsuccessful
budgeted run, minus `0.02 * log(1 + expanded + master_nodes + local_calls)` and
`0.05` per retry. This optimizes solve coverage and search work; the work penalty
is not a calibrated wall-time estimate. An unsuccessful run is not an
infeasibility training label.

Self-critical REINFORCE subtracts a detached greedy same-problem baseline.
Decision log probabilities sum over the trajectory. Entropy weight 0.01 and KL
weight 0.02 against the frozen starting actor limit premature collapse. The value
head fits observed log-work with Huber loss weighted 0.1; failed-search targets
are lower bounds. Scores use the same policy-minus-0.1-value rule as deployment.
Every sampled behavior distribution is checked against PyTorch before updating.

Defaults use AdamW at `1e-4`, decay `1e-4`, gradient clipping 1, twenty epochs,
32 problem draws per epoch, batches of eight, and two samples per draw. Sampling
is uniform over training domains, then problems. The environment selects up to
256 training and 32 validation problems. Each rollout shares 1,000 ms and 64
local calls across six attempts, with 350 ms initial stages and 100 ms initial
local slices. Stage growth is capped by the total rollout time. The solver
environment runs on CPU; PyTorch updates support CPU/CUDA.

Only sampled Train trajectories update the actor. Validation uses deterministic
greedy evaluation; model selection first compares domain mean solve rate, then
domain mean reward. The initial actor is eligible, so an unhelpful run publishes
its starting policy. Test domains are unavailable to the RL environment. Repeated
wall-time cutoffs can still introduce measurement noise.

The interface predictor stays frozen. After actor selection, the default workflow
refits the latent transition model against that exact frozen encoder for up to
50 epochs and publishes a native-validated three-model bundle. This prevents
using latent predictions from an incompatible pre-RL encoder. `--no-refresh-latent`
produces merge-only output with no bundle or latent checkpoint; disable latent
lookahead when deploying that output.

## Outputs, resume, and transfer

`rl-report.json` records source IDs, parent/interface/engine digests, rewards,
selection, and trajectory checksums. `episodes/*.json` retains behavior weights,
observations, sampled actions, baseline outcomes, work, and successful certificates.
`rl-state.pt` resumes at epoch boundaries, including Adam state and sampling RNG.
Rerun the same command; increasing `--epochs` extends the ceiling. Changed data,
reward settings, source code, or parent weights require a new model directory.
Resume through the RL state, rather than the supervised trainer's optimizer path.
Latent optimizer states live under `latent-fit/<encoder-digest>/`; extending RL
can select a new encoder and refit its transition model without reusing stale
latent targets or discarding the previous encoder's resumable fit.

The supervised transfer archive alone lacks the online planning environment's
source problems. Build an RL archive that includes the checksum-verified store
and pretrained JSON bundle:

```bash
python3 scripts/package_training.py --run-dir runs/hilg-wl \
  --include-rl --output runs/hilg-wl-rl-training.tar.gz
```

On a CUDA PyTorch host, extract it, enter `ptu-plan-training`, and run
`./scripts/train_gpu_online.sh` for concurrent generation and RL, or the `train
rl --device cuda` command above for fixed-data RL. This archive does not include RL optimizer
states; copy an existing RL output separately when resuming an experiment.

## Evaluation

`benchmark-models` accepts `policy-escalating-interface` and
`beam-escalating-interface`. They share the control variants' total budgets;
`--escalation-stage-ms 100` controls their stage cap. Reports retain every
attempt, aggregate all search counters, and show the final executed partitions.
Use the same held-out cohort and repetitions when comparing pretraining and RL.
Training reward alone does not establish a planner improvement.
