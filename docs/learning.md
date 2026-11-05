# Merge policy, value model, and experiment data

## Collect counterfactual outcomes

Build with `cargo build --offline --bins`. A small parser-free collection is:

```bash
target/debug/collect-rollouts tests/fixtures/solver.json \
  tests/fixtures/interleaving.json --store target/rollouts --full-trace
target/debug/dataset-store --store target/rollouts audit
```

To mix existing optimal PANDA and backward-generated problems, add
`--dataset ../train --limit 20`. Solution records are read through a fixed,
checksum-verified manifest snapshot. Source problems are copied into the
experiment store; existing solution files remain immutable.

At each greedy trajectory level, collection evaluates candidate merges and STOP
using the sequential master. `--candidates` caps candidates by structural score;
STOP is always retained. `--levels` caps the number of visited states. The record
retains omitted-candidate counts, budgets, partitions, observations, outcomes,
grounding size, phase timings, and master events. All evaluations start fresh
with the same configured master/local limits; wall-clock scheduling can still
change a budget outcome. Collection follows the greedy trajectory, so it does
not cover every partition or the learned policy's state distribution.

`--master-time-ms`, `--local-time-ms`, `--max-local-calls`, and
`--max-boundary-atoms` control evaluation cost. Grounding failures persist as
`CollectionFailure` objects; other problem errors are reported on stderr.
`--full-trace` retains every local request and successful oracle certificate.
Summary mode omits those per-call payloads but retains contextual no-goods.

## Store and audit

`src/store.rs` stores schema-versioned envelopes under
`objects/<hash-prefix>/<SHA256>.json`, indexed by an atomic `index.json` snapshot.
IDs cover metadata and payload bytes. Publication uses a cooperative writer lock,
fsync, and rename; readers hold stable index snapshots. `stats` reports counts,
groups, size, and snapshot hash. `repair` adopts verified unindexed objects after
an interrupted publication and refuses corrupted indexed objects. It does not
reconstruct a malformed existing index.

Audit verifies every object's checksum and metadata. Rollout auditing rebuilds
features and legal partitions from the source problem, checks event pairing and
proof scopes, replays positive local/global certificates, and verifies that local
segments compose the reported plan. Model auditing checks dimensions, finite
weights/optimizer state, and split isolation. Failed-search exhaustion remains
trusted solver evidence; audit does not rerun an entire failed search.
Both trainers require the same semantic checks before accepting examples. A
snapshot-scoped auditor reuses source compilation and verified identical core
queries; caches include compilation caps and proof budgets in their scope and
are released before model fitting. Full traces also bind local networks, budgets,
and interface contexts to their recorded partition.

## Features and model

`src/learning/` implements a CPU relational graph encoder, scalar reverse-mode
autodifferentiation, Adam, supervised training, and constrained inference. There
is no GPU or Burn runtime requirement. All encoder and head parameters train.

Feature schema 1 has 12 cluster features, 12 pair features, six global features,
and nine search-budget/context features. The quotient retains six graph views
with separate incoming/outgoing channels. Cluster features cover size, task
kind, possible reads/writes, boundary families, recursion, method branching,
ordering, and unordered structure. These are lifted structural summaries;
concrete worlds, object bindings, and goal literals are not model inputs.

The encoder uses 1–3 message-passing layers, hidden width 1–32, and mean pooling.
Shared action heads consume symmetric pair embeddings, graph context, pair
features, and a STOP indicator. Inference maximizes
`policy_logit - value_weight * predicted_log_effort`, with deterministic tie
breaking. `--merge-value-weight` sets the weight. Adjacency, maximum cluster
size, and acyclic ordering are enforced before selection; structural minimum
score is a greedy stopping rule and does not mask learned candidates.

## Labels, splits, and checkpoints

Successful actions outrank outcomes that did not solve within the evaluation
budget. Among successes, lower symbolic effort wins, then shorter action count.
No preference is manufactured between two unsolved outcomes or from an oracle
error. The value target is `ln(1 + local_expanded + w * master_nodes)`; `w` is
the library's `master_effort_weight`, default one. Feasible outcomes provide
observed costs. Master budget exhaustion supplies a lower bound only when no
local search was abandoned and no boundary-cap branch was skipped. Other
unsolved outcomes have no value target. This predicts search effort, not
optimal remaining plan cost; PANDA's optimal labels stay in the solution dataset.

Training combines pairwise softplus ranking and Huber value loss; censored
targets penalize underestimation only. A seeded domain-hash split assigns
80%/10%/10% to train/validation/test. Groups are hashes of the serialized domain
AST, not IPC family names: semantically equivalent rewritten domains may have
different groups. All problems and synthetic siblings with the same domain hash
stay together. Empty training splits are errors; validation/test examples never
substitute for training data. Reports expose unlabeled exclusions and split sizes.

```bash
target/debug/train-merge-model --store ../train/rollouts \
  --hidden 4 --layers 2 --epochs 100 --learning-rate 0.01 \
  --output target/merge-policy.json
target/debug/train-merge-model --store ../train/rollouts \
  --hidden 4 --layers 2 --resume target/merge-policy.json \
  --epochs 25 --output target/merge-policy-resumed.json
```

Resume preserves optimizer moments and epoch-dependent shuffle state. Match
model shape, initialization seed, and split seed; other options may deliberately
change. Exact continuation also requires unchanged training examples and optimizer
options. Checkpoints retain the input snapshot hash and selected record/group IDs.
Validation and test metrics are reported after training; neither controls updates.

## Initial pilot and limits

The first mixed pilot collected 29 rollout states from 14 source problems into
`../train/rollouts`; adding one checkpoint and report produced 45 objects. Twelve
states had no usable supervision. The labeled split contained 11 training,
three validation, and three test states. A width-four, two-layer model trained
for 100 epochs reduced training loss from 4.89403 to 0.01314. Training preference
accuracy was 4/4, validation 1/1, and test 1/4. This small pilot shows the pipeline
works and exposes weak held-out performance; it establishes no speed advantage.

The exported checkpoint is `target/merge-policy.json`. Learned inference produced
replay-valid two-action and four-action plans on the independent-task and
interleaving fixtures respectively. These are smoke checks, not a benchmark.
Features remain structural, search and graph representations remain dense, and
training rollouts still use contiguous-block search; the separate
[POCL master](interleaving.md) can interleave partitions. Broader domain coverage, evaluation
against greedy search, learned-trajectory collection, and sparse/batched neural
execution remain future work. The learned model contributes no proof rules.
