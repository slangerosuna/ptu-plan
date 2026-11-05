# Interface proposals, beam agglomeration, and Benders cuts

## Interface predictor

`src/learning/interface.rs` provides a shared grounded-atom MLP with a tanh
hidden layer and a three-class softmax: **unknown, false, true**. Unknown means
the outgoing interface leaves the atom unspecified; it does not assert false
in the concrete witness world.

Sixteen features describe incoming truth, possible local reads/adds/deletes,
future reads, global-goal membership/polarity, arity, network size, and footprint
sizes. Names and object identities are not model features. Reachability includes
all local refinement alternatives regardless of their current guards. The
predictor represents individual fluent probabilities, not their joint logical
consistency.
Inference constructs features only for boundary atoms while preserving the full
footprint counts, so its probabilities match the corresponding full-input rows.
It uses a values-only evaluation path; training accumulates independent atom
gradients without retaining a derivative graph for the entire interface.

The master projects predictions onto its exact grounded boundary. Unknown
predictions retain incoming truth for the first candidate. A bitwise XOR
permutation then enumerates every other valuation exactly once. The predictor
changes search order; it cannot establish feasibility, discard a valuation, or
create a no-good. The final block always checks the original goal.

```bash
target/debug/collect-rollouts --dataset ../train --store ../train/rollouts \
  --min-roots 2 --limit 24 --full-trace --conflict-core-checks 4
target/debug/train-interface-model --store ../train/rollouts \
  --hidden 4 --epochs 150 --learning-rate 0.01 --output target/interface-policy.json
target/debug/ptu-plan tests/fixtures/solver.json --sequential --micro-max-size 1 \
  --interface-model target/interface-policy.json --conflict-core-checks 8 \
  --analysis-json --master-trace
```

Training uses weighted categorical cross-entropy, reverse-mode gradients, and
Adam. `--unknown-weight` defaults to 0.25. Complete-plan examples come from
non-final segments of replay-valid master solutions. Full traces additionally
supply locally feasible interfaces from successful oracle calls, even when
downstream search fails or exhausts its budget. Every included witness replays;
core-minimization probes and final-block goals are excluded. Identical examples
within a rollout are deduplicated, preferring complete-plan evidence.

Local evidence guarantees execution of that block under its concrete input,
not compatibility with the remaining blocks. Reports count local and
complete-plan examples separately. `--global-only` trains only the latter.
Old traces without explicit interface context still supply complete-plan
examples. No failed/unknown query becomes a positive interface label.

Domain-hash splitting uses the same seeded 80/10/10 scheme as merge training.
There is no substitution of validation/test records into training. Checkpoints
retain model/feature versions, Adam moments, completed epochs, source record IDs,
groups, split seed, options, and input snapshot hash. Resume preserves deterministic
epoch shuffling; identical continuation requires identical examples and options.
Planner JSON includes the interface model hash and proposal timing/counts.

## Receding-horizon learned beam agglomeration

`src/learning/beam.rs` uses the existing relational merge model. At every beam
node it rebuilds the observation and legal candidate mask from the actual
symbolic quotient. By default each expansion re-encodes its graph; an optional
`--latent-model` instead predicts node embeddings inside the horizon. Each
expansion keeps up to `--beam-branching` merge actions and always offers STOP.

A branch score is cumulative normalized policy log-probability minus
`--merge-value-weight * predicted_log_effort` for its latest action. The latest
value estimates effort after that action; values are not accumulated as costs.
In the default exact-encoding mode, equivalent partitions are deduplicated by sorted original-node membership,
separately for terminal and expandable states. Stable first-action and partition
ordering break score ties.
Expandable duplicates retain the highest prefix log-probability, since a shared
future suffix replaces the predecessor's value estimate. Completed duplicates
retain the best final score. Predictions are cached within a search call using
ordered cluster IDs and membership under its fixed graph/model/budgets; this
preserves action positions. The cache is capped at 128 entries and 16 MiB of
accounted key/prediction payloads.

Search retains `--beam-width` branches for up to `--beam-depth` levels, commits
the best branch's first action, and searches again from that exact state. Thus
depth two can still produce a trajectory with many merges. Size, adjacency,
coverage, and ordering-DAG checks remain mandatory. STOP and the target cluster
count terminate search; `--beam-max-evaluations` bounds total model evaluations
across rounds and returns a valid current partition on exhaustion.

`merge_policy.beam` reports evaluations, generated/deduplicated branches, rounds,
prediction-cache hits, truncation, quotient construction, total feature/inference
time, and elapsed time. Selected-action timing
alone does not account for discarded branches; use these totals for overhead
comparisons. Beam search is a heuristic over partitions, not a planning proof.

`--latent-model target/latent-policy.json` enables the frozen-encoder transition
head with `--merge-beam` and depth 1–4. The first node of every real round is
encoded exactly. Imagined nodes use predicted embeddings with the existing
policy/value decoder; symbolic quotient construction and legal action features
remain exact. Each committed move discards imagined state. Because different
paths can predict different embeddings for one partition, this mode disables
symbolic transposition deduplication and prediction caching. Width and evaluation
caps still apply. Telemetry adds `exact_encodings`, `latent_transitions`, and the
transition model hash. See [latent transitions and causal certificates](latent-and-causal.md)
for training, scope, and pilot results.

## Verified outgoing conflict-core cuts

`src/conflicts.rs` represents a logic-based Benders cut. If the local oracle
proves `network, incoming world, outgoing conjunction` infeasible, deletion
minimization tries dropping each outgoing literal. A literal is removed only
after the same oracle proves the weaker conjunction infeasible. A feasible or
budget-exhausted probe leaves the last proven conjunction unchanged.
For wider contracts, a first empty-goal probe can establish an empty core in one
call. The final block's fixed goal is retained without deletion probes, but its
incoming-scoped cut can prune the same failure in another incoming world.

`--conflict-core-checks` caps deletion probes per primary failure. All probes
consume the existing master time/call limits and contribute to expansion,
generation, oracle-status, and core-check counters. No probe can silently extend
those budgets. A capped pass need not produce a subset-minimal core.

For example, a verified core `p` excludes both `p AND q` and `p AND NOT q`.
An empty core excludes every outgoing conjunction for that local network and
incoming scope. Cuts retain initial-network index, partition ID, the complete
source world, incoming assumptions, source goal, exactness/decomposition bound,
proof status, source call, verification call, and number of removed literals.
They are local to one master run; they do not transfer to another problem or
partitioning.

Incoming assumptions fix the truth value of every reachable action/method guard
atom and every retained core goal atom. All decomposition alternatives contribute,
including methods whose guards are currently false. With unconditional Boolean
effects, agreement on these atoms is preserved by identical local decisions;
irrelevant incoming facts cannot change plan existence. Goal atoms must remain
even when no guard reads them, since a plan can preserve their initial values.
This structural projection requires no additional oracle probes. It is
conservative: it does not try deleting genuinely referenced incoming atoms.

Matching requires agreement with that projection, identical proof bounds, and
outgoing conjunction containment. Legacy cuts without an `incoming` field retain
exact-world matching. Disjunctive/nonliteral goals retain exact no-goods.
Depth-limited cores never prune an unbounded search; bounded hits preserve the
master's bounded status. There is no learned hard cut or global HTN infeasibility
claim: exhaustive failure applies to the chosen contiguous partitioning.

An index groups active cuts by initial network and partition. A cut retires weaker
cuts only when its incoming assumptions and outgoing literals subsume theirs
under the same proof bound. All original proofs remain in the result archive.
`core_active`, `core_subsumed`, and `core_redundant` describe this lifecycle;
`core_generalized_hits` counts matches in a different complete source world.
On a hit, binary enumeration jumps over the contiguous forbidden cube without
changing the interface predictor's XOR ordering. `core_assignments_skipped`
counts all excluded assignments, including the visited hit, and saturates at
`usize::MAX`; `interface_candidates` counts only visited valuations.

Full traces distinguish normal solves from minimization calls, retain interface
training context, and link `ConflictCoreVerified` events to proof-producing
calls. `ConflictCoreApplied` events identify the published proof and skipped
ordinal range; full traces also retain incoming state, goal, and boundary.
Auditing checks matching and cube jumps, recomputes the canonical dependency
projection, and independently reruns each retained core with its recorded local
budget. Failure or resource exhaustion
during re-verification rejects the audit rather than accepting an unverified cut.
The snapshot auditor reuses compiled sources and previously verified identical
proof queries, including incoming assumptions, compilation cap, and local budget in the cache
scope. Both trainers use this auditor. Full request checks bind recorded networks,
limits, and interface context to the source problem and proposed partition.

## Paired conflict-core pilot

`benchmark-cores` loads each source problem once, creates singleton initial-root
partitions, and runs identical master limits with cores off and on. It alternates
execution order, uses no learned models, and independently re-proves every
generated cut. The JSON report retains the input snapshot, limits, per-problem
results, proof counts, preparation failures, and aggregate work counters.

```sh
cargo run --offline --bin benchmark-cores -- --store ../train/rollouts \
  --limit 29 --core-checks 4 --master-time-ms 500 --local-time-ms 100 \
  --max-local-calls 64 --max-boundary-atoms 12 \
  --output target/benders-benchmark.json
```

One local pass over the 29 stored problems independently verified 66 cuts:

| Measure | Cores off | Cores on |
| --- | ---: | ---: |
| Feasible | 4 | 4 |
| Sequentially infeasible | 3 | 14 |
| Budget exhausted | 22 | 11 |
| Local calls, including probes | 917 | 194 |
| Visited interface assignments | 930 | 165 |

Cuts used 88 probes and produced 57 hits, including 18 across different complete
incoming worlds, skipping 1,414 assignments. No paired results contradicted each
other. The extra completed decisions concern partition-restricted infeasibility,
not additional feasible plans or global unsatisfiability. This small, single-pass,
time-capped pilot supports reduced search work on this sample; it does not
establish a general runtime improvement. The report is a local artifact under
`target/`, not newly published training data.

## Current artifacts and limits

The experiment store at `../train/rollouts` currently contains 161 immutable
objects: 29 source problems, 124 rollout states, and eight model/report objects.
The two multi-root collection passes attempted 96 states; content addressing
deduplicated an identical record. These legacy exact-world evaluations predate
the incoming projection and cube skips. Across stored evaluations, 534 cores produced
1,240 pruning hits and required 2,090 deletion probes. These are instrumentation
counts, including repeated problems, not independent benchmark improvements.

`target/interface-policy.json` is the selected width-four, 150-epoch checkpoint.
Its training split has 15 interfaces from three domain groups: one complete-plan
interface and 14 local witnesses. Test has 12 complete-plan interfaces from one
domain group; validation is empty. Training atom accuracy is 274/276 (99.3%),
including 6/7 false and 27/28 true labels; test is 165/165. The test set has no
false labels and 144 unknown labels, so aggregate accuracy provides limited
evidence. Another 113 stored rollout states supplied no usable interfaces.

The fixture CLI checks cover guided exhaustive search and combined beam/interface/
core execution. They establish integration and replay validity. There is no
demonstrated planning-speed advantage. Features remain coarse, neural execution
is scalar CPU code, and training rollouts still search contiguous blocks. The
separate [POCL master](interleaving.md) supports interleaving; its search costs are
not yet training targets. Merge value
context encodes budgets but not the active interface model or core-probe setting;
cost predictions can shift when those controls change. Broader balanced data and
paired baseline comparisons are still required.
