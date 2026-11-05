# Latent transitions and causal local certificates

## Frozen-encoder transition training

`learning::latent::LatentModel` predicts the node embeddings produced after one
legal merge. The existing `MergeModel` now exposes `encode` and `predict_encoded`;
using true embeddings reproduces the original policy/value predictions exactly.
The policy/value encoder remains frozen during transition training.

`train-latent-model` reads an immutable dataset snapshot and uses the common
semantic auditor before constructing examples. For every retained merge action,
it performs the actual symbolic contraction, re-encodes the true child quotient,
and regresses toward those node embeddings. Targets include surviving nodes,
whose relational contexts also change after a merge. STOP has no transition.
Examples need no feasible planning outcome: symbolic merges provide exact targets
even when the local oracle exhausts its planning budget.

The shared residual MLP receives each node's previous embedding (the mean of the
merged pair for the fresh parent), pair and global embeddings, action features,
true child structural features, search context, and a merged-node indicator.
It predicts a bounded embedding per node. Mean squared error averages over nodes
and hidden coordinates per transition; training accumulates one node's gradient
at a time. Domain-hash splits isolate all siblings. Neither validation nor test
examples update weights.

```sh
cargo run --offline --bin train-latent-model -- --store target/rollouts \
  --merge-model target/merge-policy.json --epochs 50 \
  --output target/latent-policy.json
# Continue with Adam moments and epoch-dependent deterministic shuffling.
target/debug/train-latent-model --store target/rollouts \
  --merge-model target/merge-policy.json --resume target/latent-policy.json \
  --epochs 25 --output target/latent-policy-resumed.json
```

Checkpoints include the frozen encoder, its exact hash, transition weights, Adam
state, options, epochs, snapshot hash, and split record/group IDs. Loading checks
model compatibility and split overlap. Exact continuation requires identical
examples and optimizer settings. When given a pretrained merge checkpoint, the
CLI requires its split seed; raw model files have no training-split provenance,
so callers must ensure the encoder has not seen held-out domains. The library
training API accepts an explicitly supplied frozen model under the same condition.
`dataset-store audit` validates stored latent checkpoints. JSON uses exact float
round-tripping so model fingerprints survive reloads.

## Shallow latent lookahead

Add `--latent-model target/latent-policy.json` to `--merge-beam`, selecting
`--beam-depth 1..4`. Each real round encodes the current quotient exactly.
Inside the horizon, transitions predict embeddings and the original policy/value
decoder ranks actions. Only the first selected action is committed; the next
round starts from a fresh true encoding.

This is a hybrid implementation: structural quotient construction, legal merge
masks, and action features remain symbolic at every node. It avoids repeated GNN
passes, not all symbolic work. Imagined embeddings can differ between histories
that reach the same partition, so latent mode retains separate paths instead of
using the exact mode's transposition rule or prediction cache. Beam width,
branching, STOP, and the total evaluation cap still bound search. No prediction
establishes feasibility or supplies a hard conflict cut.

Beam statistics record exact encodings, latent transitions, model hash, expanded
node evaluations, generated branches, truncation, quotient construction time,
feature/inference time, and
total elapsed time. A transition and decoder together count as one node
valuation, just as an exact encoder and decoder do in the original beam.

## Measuring multi-step drift

`evaluate-latent` audits its source records, then runs bounded open-loop merge
paths without updating parameters or publishing dataset records:

```sh
cargo run --offline --bin evaluate-latent -- --store target/latent-pilot \
  --latent-model target/latent-policy.json --depth 4 --max-records 124 \
  --paths-per-record 8 --max-paths 512 --output target/latent-drift-review.json
```

Paths start from canonical audited counterfactual actions and continue using the
highest exact structural merge score, independently of learned dynamics and
held-out targets. Each depth compares predictions to the true encoder and to a
mean-pair baseline that retains surviving embeddings. Both predictors receive
the same actions; neither is re-encoded inside the path. Reports retain snapshot
and checkpoint hashes, evaluated record/group IDs, split assignment, limits,
path/sample counts, and truncation.

Metrics include mean node MSE, action-value MAE, and policy argmax agreement with
the exact encoder. Agreement excludes states with only STOP and includes an
explicit decision count; it measures fidelity to the existing policy, not good
planning choices. Missing depths or splits have null metrics. Depth cohorts can
differ as paths finish, so raw depth averages do not isolate compounding error
on a fixed sample population. Overlapping source states and domain groups are
not independent trials.

## Replay-derived causal contracts

`solver::causal::causal_contract(problem, request, witness, partition)` builds a
`CausalContract` from a replay-valid local witness. `validate_causal_contract`
independently replays and reconstructs every field. Successful sequential master
segments include these contracts automatically; older records without `causal`
remain readable. The rollout auditor verifies contracts alongside local and
global witnesses.

Each certificate contains:

- Action and method-refinement events with their selected guard requirements.
- Positive and negative causal links from inputs or the most recent writer to
  guard consumers and the output boundary.
- Imports identifying external support at a particular consumer event; effect
  exports and sufficient output-goal guarantees.
- Imported frame literals preserved through the output boundary.
- An internal partial order retaining chosen hierarchy constraints, lifted HTN
  precedence, causal support, and protection against opposite effects.
- Explicit threat resolutions before the producer or after the consumer.

Disjunctions select a sufficient true branch; nested negation is handled in
both polarities. Guards precede their event's own effects. Add wins over delete,
as in the exact solver. Method guards are checked at refinement time and are
not discarded when constructing support. Unknown, unlisted input facts remain
unconstrained. Exported guarantees concern the local output boundary; they do
not assert persistence after another partition executes.

Event indices identify the fixed chosen decomposition. Reordering refinements
requires remapping dynamically allocated occurrence IDs before executable replay.
`certify_causal_linearization(problem, source_request, witness, contract, order,
incoming_state)` performs that remapping, rejects missing/duplicate events and
ordering violations, checks imported assumptions, and certifies the resulting
execution and output guarantees. It accepts different concrete incoming worlds
when they satisfy those imports. The source request and witness remain the
evidence from which the original contract was derived. The retained witness
provides a valid linearization; small-model tests also replay every permitted
alternative through this API. The [POCL interleaving master](interleaving.md)
can now reconcile supplied local certificates or enumerate other method choices.
External imports require support and cross-partition threat resolution; global
acceptance requires whole-problem symbolic replay. The scope is the solver's
deterministic Boolean fragment, with no numeric
or temporal resource guarantees.

Master counters expose certificate count, links, imports, threat resolutions,
and extraction time. New certificates use schema version two: hierarchy entry
events and exit frontiers encode precedence without expanding every subtree's
descendants; an effect index finds only potential clobberers of each support
literal. Version-one certificates still validate using their original expanded
ordering representation. Reachability-equivalence tests cover both formats; a
96-level chain needs 96 ordering edges instead of 4,656. Replay still copies
concrete states, and intrinsically dense partial orders or many real threats can
remain expensive. Global reconciliation is provided separately by the POCL master.

## Local pilot

The pilot copied the 161-object experiment store to `target/latent-pilot` and
added one checkpoint/report there. The original `../train/rollouts` remains at
161 objects and passes its compatibility audit. The copied 163-object store also
passes auditing.

The width-four frozen encoder trained transitions for 50 epochs. The split has
115/23/23 transitions from 19/3/4 domain groups (60/11/14 contributing rollout
records). Training node MSE decreased from 0.13114 to 0.006365; validation MSE is
0.008333 and test MSE is 0.17859. Weak held-out error prevents a claim of useful
planning guidance.

The drift review audited all 124 rollouts and evaluated 161 paths without
truncation. Held-out test node MSE was 0.17859 versus baseline 0.14624 at depth
one (23 samples), and 0.45238 versus 0.14265 at depth two (16 samples). There
were six depth-three samples and no depth-four samples. Thus the current model
underperforms the simple baseline on those first two held-out cohorts; neither
the evaluator nor compact certificates fix that modeling weakness. The local
report is `target/latent-drift-review.json`.

`target/latent-guidance-result.json` records a combined four-action fixture run:
three true encodings, 17 imagined transitions, and a replay-valid complete plan.
Its one local causal contract contains eight events, 20 support links, 12 imports,
and four threat resolutions. It merges all roots, so this checks integration,
not interface prediction quality or interleaving. No runtime advantage over the
existing exact-encoding beam has been established.
