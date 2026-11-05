# ptu-plan

Partition To Unordered combines HDDL input, static HTN analysis, typed interaction
graphs, spectral micro-partitions, an exact local progression solver, and
validated synthetic/real training-data collection. Greedy agglomeration and a
sequential partial-state master compose and replay exact local witnesses.

## HDDL parser

The input adapter in `src/parsing/` is copied from `../pspr-planner`. It invokes
[koala-planner/HDDL-Parser](https://github.com/koala-planner/HDDL-Parser)'s
`hddl_analyzer` executable and decodes its JSON AST, using the same subprocess
interface as PSPR rather than linking the parser as a Rust dependency.

Build the external parser separately with `cargo build --release --bin
hddl_analyzer` in its checkout. Select the resulting executable using
`--parser PATH`, the `HDDL_PARSER` environment variable, or `hddl_analyzer` on
`PATH` (in that order).

```bash
cargo run -- domain.hddl p01.hddl --parser /path/to/hddl_analyzer
cargo run -- domain.hddl problems/
cargo run -- domain.hddl
cargo run -- program.json
```

Without explicit problems, the loader discovers sibling `.hddl` files, excluding
the domain and files named `domain.hddl`. A whole-program JSON export loads
without an installed parser and cannot be combined with separate problem files.
By default the CLI prints domain and problem summaries. Add `--solve` to search.

## Training all learned models

The training coordinator selects multi-root PANDA solutions, generates backward
synthetic problems, collects full-trace oracle data, audits each rollout, and
exports a portable package. The default pass uses up to 1,024 existing problems
and attempts 4,096 additional solutions across the available domain
templates. Failed or duplicate synthetic attempts are reported; generation
does not promise to reach the quota. Completed rollout jobs are reused if the
command is rerun. Data and checkpoints go under `runs/hilg-wl/`.

```bash
cargo run --release --bin train -- prepare \
  --run-dir runs/hilg-wl --problems 1024 --synthetic 4096 --workers 4
```

Copy `runs/hilg-wl/prepared/` and the repository to the rented Linux GPU machine.
Install a CUDA-enabled PyTorch wheel matching that machine's driver with the
[official PyTorch selector](https://pytorch.org/get-started/locally/), and build
the Rust binary there. Then run one job for all three learned components:

```bash
cargo run --release --bin train -- fit \
  --run-dir runs/hilg-wl --device cuda --python python3 \
  --model-dir runs/hilg-wl/gpu-models
```

`python3 scripts/package_training.py` creates a transfer archive containing the
source, launcher, and selected prepared data. See [GPU training instructions](docs/training.md)
for setup, transfer, resume, and output details.

The default encoder uses HILG with three rounds of Weisfeiler–Leman subtree
features and nonlinear merge/value heads. See [architecture and compatibility](docs/hilg-wl.md).

The job checks its prepared-data checksums and native float64 prediction and
gradient references before training. It fits the shared merge policy/value
model, interface predictor, then latent transition model using the final frozen
merge encoder. It selects checkpoints on domain-held-out validation data,
reports the test split, saves resumable per-model states, and validates exported
Rust checkpoints before publishing `runs/hilg-wl/models/bundle.json`. The default
models have 73,154, 643, and 5,248 parameters. Training uses float32, AdamW at
`1e-3`, batches of 64, a 100-epoch ceiling, and early stopping after 12 stale
validation epochs (with a 20-epoch minimum). It samples domains uniformly to
reduce domination by larger domains. One modest GPU is enough; multi-GPU
training is not configured.

The native Rust planner currently uses the merge and interface checkpoints;
the latent checkpoint is available for beam lookahead. Example planner flags:
`--agglomerate --merge-model runs/hilg-wl/models/merge-policy.json` or
`--sequential --interface-model runs/hilg-wl/models/interface-policy.json`.
Use `--device cpu --epochs 2 --model-dir runs/hilg-wl/smoke-models` for a short
local end-to-end check. `requirements-training.txt` lists the Python dependency.

To compare trained components against structural greedy merging and exact
graph re-encoding, run `cargo run --release --bin benchmark-models -- --store runs/hilg-wl/data
--models runs/hilg-wl/models --output runs/hilg-wl/benchmarks/heldout-350ms.json`, then
`python3 scripts/summarize_benchmark.py runs/hilg-wl/benchmarks/heldout-350ms.json`.
See [benchmark protocol and ablations](docs/benchmarking.md) for paired solver
budgets, held-out selection, repeat/resume behavior, and interpretation.

## Dynamic repartitioning and RL fine-tuning

`--sequential --repartition` repairs failed partition choices within shared
master time/call/node limits. Boundary cutoffs favor coarsening; unknown local
searches can split a block or increase its local time slice. Each attempt keeps
its own scoped proofs, and every successful plan is replayed.

```bash
cargo run --release --bin train -- rl --run-dir runs/hilg-wl \
  --device cuda --model-dir runs/hilg-wl/rl-models
```

RL samples legal merge/STOP decisions and trains the WL encoder and policy/value
heads from actual planner returns using a greedy same-instance baseline. The
interface model stays frozen; the latent model is refitted against the selected
actor before the bundle is published. Use `--device cpu` for a local run.
See [escalation, RL objectives, budgets, resume, and GPU transfer](docs/repartition-rl.md).

On a rented host with spare CPU cores, run `./scripts/train_gpu_online.sh` to
generate substantially more certified training problems in the background and
feed completed batches into RL between epochs. See [concurrent generation and
training](docs/online-training.md) for CPU limits, output paths, and resume.

## Static analysis and interaction graph

```bash
cargo run -- tests/fixtures/interaction.json --analyze
cargo run -- domain.hddl p01.hddl --analysis-json > analysis.jsonl
```

`--analyze` validates task references, arities, method heads, subtask labels,
and ordering DAGs, then reports structural features and graph view counts.
`--analysis-json` emits one JSON record containing `analysis` and `graph` for
each domain/problem pair (or just the domain when no problems are found).
Output IDs are deterministic for a fixed AST order, not persistent identifiers
across domain edits. Analysis errors produce a nonzero exit status.

- `src/analysis/` resolves task/method IDs, preserves task arguments and network
  constraints, computes ordering closure, detects recursion SCCs and syntactic
  tail recursion, and propagates possible reads/effects to a fixed point.
- `src/interaction/` builds distinct task-schema, method, and occurrence nodes.
  Parallel edges retain ordering, possible support, possible threats, hierarchy,
  and shared-fluent views. Peer interactions stay within each network; alternative
  methods remain separate. The resource edge type is reserved for future work.

The complexity signature considers all syntactically reachable methods from the
initial network; domain-only analysis uses every task as a root. A problem with
no initial task network has no task roots. Unordered classification requires no
precedence in any reachable network, including method bodies. Schema parameters
remain lifted even when the initial task arguments are ground.

Library entry points:

```rust
use ptu_plan::analysis::{NetworkId, analyze_program};
use ptu_plan::interaction::{InteractionGraph, ViewWeights};

// Given a loaded Program named `program`:
let analysis = analyze_program(program)?;
let graph = InteractionGraph::from_analysis(&analysis);
let roots = graph.network_nodes(NetworkId::Initial);
let affinity = graph.symmetric_affinity(&roots, &ViewWeights::default())?;
```

The affinity matrix is a separate symmetric projection: each selected edge adds
its view weight in both directions, parallel edges sum, and the diagonal stays
zero. Directed typed edges remain available for later reasoning and learning.

This scaffold uses predicate-family summaries that ignore argument unification
and union alternative decompositions. They are possible interactions, not
guaranteed support, executable plans, or evidence for pruning. Formula features
and numeric functions produce warnings rather than being silently treated as
exact semantics. This static analysis does not perform type checking, binding
validation, or initial-state/goal reasoning; the exact solver separately checks
and compiles its supported input fragment. Resource inference and formal
complexity bounds remain future work. Ordering closure and affinity matrices are
dense; graph construction compares peer pairs.

## Exact local solver

```bash
cargo run -- tests/fixtures/solver.json --solve
cargo run -- domain.hddl p01.hddl --solve --max-decomposition-depth 16 \
  --max-expansions 100000 --time-limit-ms 5000 --analysis-json > solution.jsonl
```

`src/solver/` compiles finite typed objects, task/method bindings, and Boolean
formulas directly from the parser AST. Supported conditions include negation,
conjunction, disjunction, implication, equality, and finite quantifiers. Effects
support conjunctions of add/delete literals and universal quantification.
Effects are simultaneous, with additions taking precedence over deletions.
Numeric functions, conditional/probabilistic effects, domain axioms, XOR, and
other effect formulas return explicit errors rather than approximate solutions.

Breadth-first progression explores enabled primitive executions and method
refinements. Method preconditions hold at refinement time. Replacing a compound
task inherits its ordering constraints; unordered roots can have interleaved
descendants. A missing initial network is empty, and initial-network parameters
are existentially grounded. Every feasible certificate records refinements and
actions and passes symbolic replay, including the final goal and state.

Results distinguish `Feasible`, `Infeasible`, `InfeasibleWithinBound`, and
`BudgetExhausted`. An exhausted depth bound is not an unrestricted infeasibility
proof. Unbounded recursive search can exhaust its time/node budget; exhaustive
failure is reported only after the symbolic frontier is empty. There is no
plan-cost optimality claim. `--max-ground-instances` bounds grounding work;
exceeding it returns an error. `--max-generated` bounds search generation.
The CLI and PANDA importer use `ground_reachable_program`: method heads are
unified with concrete initial-reachable tasks before free parameters are
enumerated. Every decomposition alternative is retained, including initially
false guards; this does not prune on the initial world. `ground_program` retains
the original full Cartesian grounding API. Both support multiple type parents
and reject cycles. Broad free-variable methods and quantifiers can still exceed
the grounding budget; search state and graph representations remain dense.
The CLI returns nonzero for unsupported input or any non-feasible solver result;
JSON mode still emits a solver result record for search failures.

For local partitions, `GroundNetwork::induced(&root_indices)` retains precedence
paths through omitted roots. Pass that network, a complete concrete state, and a
terminal `Condition` to `solve_local`. `BoundaryState::condition()` expresses
sparse positive/negative terminal requirements; unspecified facts stay free.
Incoming partial states require master-level completion before solving.
`validate_certificate` and `validate_problem_certificate` replay local and whole
problem witnesses. Certificates are tied to the same compiled model and request.

## Multi-view spectral micropartitioner

```bash
cargo run -- tests/fixtures/solver.json --micropartition --micro-max-size 1
cargo run -- domain.hddl p01.hddl --micropartition --micro-min-size 4 \
  --micro-max-size 16 --view-weights 1,2,3,1,1,2 --analysis-json
```

`src/partition/` combines the selected graph views into the existing symmetric
affinity, separates connected components and isolates, then recursively bisects
oversized components using the symmetric normalized Laplacian, a Fiedler-vector
sweep, and a normalized-cut score. The maximum cluster size is enforced; the
minimum is a split preference and can be relaxed for small/disconnected groups.
Output includes stable cluster IDs, cut weight, normalized cut, and split
diagnostics. View weight order is **ordering, support, threat, resource,
hierarchy, shared-state**. Default weights are one; default maximum size is eight.

Use `micropartition_network` for initial or method peer occurrences, or
`spectral_partition` for a precomputed affinity. The CLI selects initial task
occurrences, so a problem with an initial task network is required. Typed
directed edges remain intact; the partition never removes ordering constraints.
Dense eigendecomposition limits scalability. Repeated-eigenvalue bases can vary
across numerical platforms despite canonical node ordering and sign handling.

`--solve --micropartition` returns both results while solving the whole problem.
Use `--sequential` to reconcile local partition plans instead.

## Greedy agglomeration and sequential master

```bash
# Start with single-root spectral seeds, merge, and compose verified local plans.
cargo run -- tests/fixtures/solver.json --micro-max-size 1 \
  --agglomerate --target-clusters 2 --sequential --analysis-json
cargo run -- domain.hddl p01.hddl --sequential --agglomerate \
  --merge-max-size 16 --merge-min-score 0 --max-boundary-atoms 12 \
  --max-local-calls 1000 --master-time-limit-ms 30000
```

`src/agglomeration.rs` greedily merges adjacent quotient clusters. The score is
weighted typed coupling plus reduction in boundary predicate families, minus
size, lost fully-unordered root mass, and recursive/nonrecursive mixing penalties.
Library options expose every coefficient; `--view-weights` applies to both
spectral and merge coupling. Stable cluster IDs break score ties. A merge stops
at the target, size limit, or score threshold; disconnected groups stay separate.
Directed typed quotient edges and the full merge history remain in JSON output.
Ordering cycles introduced by spectral seed contraction are repaired by merging
their SCCs, subject to the hard size cap. Later cycle-producing merges are rejected.
Structural scores guide search and do not prove solvability or complexity bounds.
Ordering closure is computed once per merge level; candidate cycle checks use
that closure and interface scores reuse cluster features and family counts.

`src/master.rs` implements a deterministic constraint DFS over topological block
orders and true/false assignments to relevant grounded boundary atoms. It keeps
every possible future action guard, method guard, and final-goal read; facts a
block cannot write are carried through its concrete input state. Boundary
contracts leave unlisted facts unknown. The local oracle receives a complete
witness state and an outgoing condition; the last block checks the actual goal.
Failure downstream can revise earlier contracts or block orders. Only proven
local infeasibility creates a no-good, scoped to the initial grounding, partition,
exact incoming world, outgoing condition, and decomposition bound. Budget
exhaustion never creates a cut. Successful local certificates are translated to
global occurrence IDs and replayed against the original problem before success.
Equivalent failed suffixes are memoized by remaining partitions and the complete
world, separately for each initial grounding. Only exhaustive unbounded failures
enter that cache; uncertain or depth-limited failures never do.

`--sequential` uses spectral seeds unless `--agglomerate` is also selected;
it conflicts with monolithic `--solve`. Search limits include `--max-master-nodes`,
`--max-local-calls`, `--max-boundary-atoms`, and `--master-time-limit-ms`;
existing solver expansion, generation, time, and depth limits apply per oracle
call, clipped to remaining master time. JSON includes contracts, local witnesses,
cuts, statistics, and the complete certificate. Non-success returns nonzero.

Successful segments also include replay-derived causal contracts: guard imports,
effect exports, positive/negative support links, preservation requirements,
threat resolutions, and an internal event partial order. Auditing reconstructs
these claims from the witness. They provide local contract evidence; global
acceptance still requires replay of the composed plan.

The sequential mode requires contiguous partitions and does not optimize total plan
cost. `SequentiallyInfeasible` concerns only the supplied partitions;
interleaving may still solve the problem. `NotSequentiallyComposable` identifies
an ordering cycle in their quotient. A depth cutoff remains
`InfeasibleWithinBound`, and incomplete enumeration remains `BudgetExhausted`.
Merging all roots recovers the monolithic search space when size limits allow.
Interface enumeration is exponential; graph and agglomeration routines are dense.

## Cross-partition POCL interleaving

```sh
cargo run --offline --bin ptu-plan -- tests/fixtures/interleaving.json \
  --interleave --micro-max-size 1 --analysis-json
```

The POCL master links event-local guard requirements to initial facts or action
effects across partitions, resolves threats by promotion/demotion, and preserves
original HTN ordering. The signal-exchange fixture returns a four-action plan
that contiguous singleton blocks cannot realize. Method guards and negative
requirements are included; every accepted plan passes global HTN replay.

The library can also reconcile supplied local certificates. Exhaustion of that
fixed bundle is explicitly distinguished from searching all decompositions.
Use `--max-master-nodes`, `--max-generated`, `--master-time-limit-ms`,
`--max-pocl-events`, and `--max-pocl-frontier` to bound search. See
[POCL interfaces, proof scope, and limits](docs/interleaving.md).

## Backward and PANDA training data

```bash
# Build the external PANDA tools locally (requires network access once).
JOBS=4 bash scripts/install_panda.sh

# Generate an additive mix of 128 synthetic samples and up to 12 real plans.
cargo run --bin generate-training -- --output ../train

# Audit without needing PANDA or the original benchmark checkout.
cargo run --bin generate-training -- --output ../train --audit-only
```

The default real families are `Blocksworld-GTOHP`, `AssemblyHierarchical`, and
`Towers` from `../hddl/test_data/ipc`, visited round-robin. Repeat `--domain NAME`
to choose others. `--per-domain`, `--real-limit`, `--search-seconds`,
`--stage-seconds`, `--memory-bytes`, and `--max-ground-instances` bound collection.
Unsupported, unverified, or time-limited cases are recorded as failures and
excluded from optimal labels. Reruns retain previous records and add new ones;
the manifest can therefore contain more real examples than a single run's limit.
Use `--domain-file PATH --problem PATH` for explicit pairs, including per-instance
domains and `.pddl` problem filenames.

`src/training/backward.rs` samples grounded abstract roots, method refinements,
and a topological execution schedule before choosing a world state. It samples
a terminal literal, regresses requirements through simultaneous add/delete
effects and method/action preconditions, rejects contradictions, and completes
the initial state using the template's otherwise unchanged facts. Every result
is re-grounded and replayed as a new problem. Regression currently accepts
conjunctive literal conditions; richer Boolean solver conditions are rejected by
the generator. No implicit physical invariants beyond the domain's encoded
conditions are inferred. Library options permit optional irrelevant-fact
randomization. SplitMix64 seeds make sampling reproducible; records deduplicate
the symbolic program and action plan while ignoring cosmetic problem names.
Collection streams each validated sample to disk and retains a partial batch
when attempts run out. `generate` remains a strict quota API;
`generate_to` exposes incremental acceptance and a partial-progress report.
The collector uses initial-reachable grounding for synthetic templates by
default, avoiding even the attempted full Cartesian product. Use
`--backward-full-grounding` to sample the full task pool with a reachable fallback
if its grounding budget is exhausted. The library's default retains that broader
full-pool behavior; `BackwardOptions::grounding_mode` selects either strategy.

Synthetic-only collection can use a whole-program JSON template without PANDA,
an HDDL parser, or the IPC checkout:

```bash
cargo run --bin generate-training -- --template tests/fixtures/solver.json \
  --synthetic-count 4 --real-limit 0 --output target/example-data
```

`src/training/panda.rs` uses [PANDA's parser, grounder, and engine](https://github.com/panda-planner-dev/pandaPIengine).
The engine configuration is `-p -H zero -g action -w 1 -S 42`, with the normal
late goal test and cost-aware visited list. This is uniform-cost search over
primitive action cost, rather than the default heuristic configuration. Import
is restricted to implicit unit-cost Boolean HDDL. The pipeline converts the
engine's compiled plan back to original HDDL, verifies it with `pandaPIparser
-v`, checks completion/configuration/cost evidence, and recovers an independent
exact HTN certificate with `solve_with_plan`. That recovery verifies feasibility;
optimality comes from the configured PANDA run, not the local breadth-first
solver. Converted action count must equal PANDA's reported cost.

Each sample JSON contains its source program, decomposition decisions, action
sequence, state trace, regressed boundaries, complexity features, typed graph,
and micro-partitions. PANDA records additionally retain original HDDL text and
checksums, binary hashes, source revisions, arguments, search logs, the converted
hierarchical plan, and verification output. Synthetic suffix costs are feasible
upper bounds; only PANDA-backed residual-HTN suffix costs are labeled exact.
They are not exact values for every state satisfying a partial boundary.

`manifest.json` lists record checksums, costs, source labels, and domain-level
`group_id` values. Keep an entire group in one train/validation/test split to
avoid leaking a domain or its synthetic siblings. `collection-report.json`
records run options, counts, and failures. New records are published atomically
without overwriting existing samples. Auditing checks checksums, exact replay,
traces, boundary labels, cached analysis/graphs, and optimality evidence.
New records identify their `grounding_mode` so grounded method IDs replay under
the same strategy; missing mode means the original full grounding. Auditing also
checks cached micropartitions (with tolerance for floating-point metrics) and
rebuilds a missing manifest only after all
discovered records validate. Older records remain readable without rewriting them.

For review before writing a sibling directory, stage and publish explicitly:

```bash
cargo run --bin generate-training -- --output target/training-staging
python3 scripts/publish_training.py --source target/training-staging \
  --destination ../train
```

The publication helper audits both locations and refuses conflicting files.
The PANDA runner requires GNU `timeout` and `prlimit`; the helper uses Python 3.
The installer builds tools under ignored `target/panda`; set `PANDA_ROOT` for a
different install location and pass that location using `--panda-root`.

## Background collection across IPC 2023

```bash
git clone --depth 1 https://github.com/ipc2023-htn/ipc2023-domains.git target/ipc2023-domains
cargo build --bin generate-training
python3 scripts/collect_ipc2023.py --inventory-only
python3 scripts/collect_ipc2023.py --detach --output ../train
```

The collector inventories the [official IPC 2023 corpus](https://github.com/ipc2023-htn/ipc2023-domains),
including total-order and partial-order families, per-instance domain pairs,
Lamps PDDL files, the Ultralight type fix, and SharpSAT problems generated by its
supplied CNF encoder. The checked-out release contains 34 track/domain families
and 1,044 problems: 1,044 PANDA jobs plus 136 backward jobs, one per distinct
domain model. The first problem per model supplies the synthetic template.
Families are visited round-robin; this attempts every family without promising
that every feature or instance is supported by the verifier or generator.

By default two low-priority workers run concurrently, each with a 1 GiB address
space limit, 180-second job deadline, and 60-second PANDA search limit. Each
backward job requests 64 samples with at most 4,000 generation attempts. Worker
output is isolated and independently audited before a single publisher appends
verified records to `../train`. Existing samples remain immutable. A dataset lock
prevents multiple collectors from publishing simultaneously. Failed, timed-out,
or unsupported jobs are recorded and collection proceeds.

`--detach` snapshots the Rust worker and Python supervisor, starts a separate
session, and checks startup readiness; lock failures are reported to the caller.
Subsequent builds cannot replace its running worker. Inspect
`target/ipc2023-collection/launcher.json` for the PID and
`target/ipc2023-collection/progress.json` for active jobs, completed results,
dataset counts, and per-job logs. Progress is also mirrored to
`../train/ipc2023-progress.json`; `collector.log` records results as they finish.
The original `collection-report.json` describes its individual generator run;
the progress file describes this combined background run.

To stop a run, send TERM to its collector PID (for PID 12345:
`kill -TERM 12345`). It stops dispatching, terminates active job process groups,
and checkpoints interrupted jobs for resumption. Restart with the same collection
options and `--work` directory to skip completed jobs and retry interrupted ones.
Add `--retry-failed` to retry failed, partial, or audit-rejected jobs while keeping
successes; attempt history and configuration hashes are retained. Changing the
search-time or sampling settings changes job identities and can rerun successes.
Publication fsyncs records and metadata, and recovers a crash between linking a
record and committing its manifest when the newly audited checksum matches.
GNU `timeout`, `prlimit`,
`nice`, Python 3, the HDDL parser, and installed PANDA tools are required.

## Instrumentation, experiment store, and learned merges

```bash
cargo run --bin collect-rollouts -- --dataset ../train \
  --store ../train/rollouts --limit 20 --levels 3 --candidates 4 --full-trace
cargo run --bin dataset-store -- --store ../train/rollouts audit
cargo run --bin train-merge-model -- --store ../train/rollouts \
  --output target/merge-policy.json --epochs 50
cargo run -- tests/fixtures/interleaving.json --agglomerate --micro-max-size 1 \
  --merge-model target/merge-policy.json --sequential --analysis-json \
  --master-trace > target/learned-result.json
```

The experiment store is separate from the solution manifest: it contains immutable
source problems, counterfactual rollouts, model checkpoints, and training reports.
`collect-rollouts` also accepts whole-program JSON files as positional inputs.
It visits domain groups round-robin, prioritizes PANDA records within each group,
and evaluates legal candidate merges plus STOP with identical configured budgets.
Each state is published immediately; omitted candidates and failed oracle calls
are recorded. `dataset-store stats` reports its contents; `repair` recovers
committed objects missing from the index without replacing corrupt records.

The standalone CPU `train-merge-model` command retains the schema-1 relational
encoder. The coordinated `train` workflow above defaults to HILG/WL features
with policy and search-effort value heads. Training uses domain-group splits, pairwise preferences, and Huber
value loss. Budget failures with abandoned local searches receive no exact cost
label. Checkpoints include Adam state and split provenance; `--resume PATH`
continues training for the requested additional epochs. Learned merges obey the
same size, adjacency, and ordering constraints as greedy merges. The model can
choose STOP; it supplies no symbolic pruning rules or infeasibility proofs.

`--master-trace` requires `--sequential --analysis-json` and includes full local
requests and witnesses in `master_events`. Search statistics track guard
rejections, ordering blocks, frontier/pending peaks, decomposition depth, oracle
statuses, and master boundary sizes. Policy output records predictions, selected
actions, model hash, and feature/inference timings.

See [the learning workflow](docs/learning.md) for label semantics, schema details,
reproducibility limits, and the initial mixed-data pilot results.

## Interface prediction, learned beam search, and conflict cores

```bash
# Multi-root problems supply interfaces between blocks; full traces also retain
# locally feasible prefixes when the complete master search runs out of budget.
cargo run --bin collect-rollouts -- --dataset ../train --store ../train/rollouts \
  --min-roots 2 --limit 24 --levels 2 --candidates 3 --full-trace \
  --conflict-core-checks 4
cargo run --bin train-interface-model -- --store ../train/rollouts \
  --hidden 4 --epochs 150 --learning-rate 0.01 --output target/interface-policy.json
cargo run -- tests/fixtures/interleaving.json --sequential --agglomerate \
  --micro-max-size 1 --merge-model target/merge-policy.json --merge-beam \
  --beam-width 8 --beam-depth 3 --interface-model target/interface-policy.json \
  --conflict-core-checks 8 --analysis-json --master-trace
```

The interface predictor classifies grounded fluents as true, false, or unknown.
Its partial proposal chooses the first boundary valuation; exhaustive fallback
retains every valuation. Training reports distinguish complete-plan interfaces
from replay-valid local witnesses that may lack a feasible continuation.
`--global-only` excludes local-only evidence; `--resume` continues a checkpoint.

`--merge-beam` searches actual legal quotient contractions, re-encodes every node,
deduplicates equivalent partitions, and replans after committing one move.
`--beam-branching` caps merge children, while always retaining STOP;
`--beam-max-evaluations` caps total model evaluations across all rounds.

`--conflict-core-checks N` enables deletion-based outgoing conflict cores. Each
removed literal requires another proven infeasible oracle result. The resulting
cut rejects stronger outgoing conjunctions for the same partition, initial
grounding, and exactness bound when incoming worlds agree on all reachable
action/method guard reads and retained goal atoms. Indexed cuts retire subsumed
entries and skip forbidden ranges of boundary assignments. Final blocks can
reuse incoming cuts without extra probes. Unknown results supply no proof;
probe costs count against the master budgets and search statistics.

Run a paired comparison on stored source problems with independent cut checks:

```sh
cargo run --offline --bin benchmark-cores -- --store ../train/rollouts \
  --limit 29 --output target/benders-benchmark.json
```

See [search guidance](docs/search-guidance.md) for proof scope, trace/audit
semantics, model features, and current data limitations.

## Latent transitions and causal local certificates

Train a frozen-encoder transition head on audited symbolic merge pairs:

```sh
cargo run --offline --bin train-latent-model -- --store target/rollouts \
  --merge-model target/merge-policy.json --epochs 50 \
  --output target/latent-policy.json
target/debug/ptu-plan tests/fixtures/solver.json --sequential --agglomerate \
  --micro-max-size 1 --merge-model target/merge-policy.json --merge-beam \
  --beam-depth 3 --latent-model target/latent-policy.json --analysis-json
```

Latent lookahead predicts embeddings within a depth 1–4 horizon and re-encodes
after each committed merge. Legal contractions remain symbolic. The checkpoint
must match the merge encoder exactly. See [implementation and limits](docs/latent-and-causal.md)
for causal certificate APIs, training provenance, and pilot measurements.
Use `evaluate-latent --store target/rollouts --latent-model
target/latent-policy.json --depth 4` to compare open-loop prediction error with a
mean-pair baseline on audited merge trajectories. Reports separate domain splits
and identify missing depths and trivial policy decisions.

## Development

Use a Rust 2024 toolchain. `cargo build` builds the CLI, `cargo test` runs parser,
analysis, solver, spectral, agglomeration, master, learning, store, and CLI tests.
`python3 -m unittest discover -s tests -p 'test_*.py'` checks corpus discovery,
bounded supervision, publication, and resumption with isolated fake workers.
`cargo fmt -- --check` checks formatting
and `cargo clippy --all-targets -- -D warnings` checks lints. The
SCIP (`good_lp`), neural (`burn`), and parallel (`rayon`) dependencies are optional
Cargo features reserved for future integrations. Default builds do not compile
their stacks or download bundled SCIP; enabling `--features good_lp` requires
its native build prerequisites.

See `deep-research-report.md` for the proposed planner architecture.
See [the implementation review](docs/implementation-review.md) for observed
shortfalls, implemented fixes, verification, and remaining limits.
