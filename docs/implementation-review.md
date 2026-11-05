# Implementation review

The first IPC 2023 collection pass attempted 1,180 jobs. Of these, 1,042 had
nonzero exits; the mixed dataset contained 1,273 verified records, including
records from earlier runs. Grounding exhaustion was the dominant failure.
Attempting every family did not establish support or successful collection for
every family.

## Shortfalls addressed

| Shortfall | Change |
| --- | --- |
| Full Cartesian grounding enumerated unrelated task instances and method head arguments. | Planner and PANDA import now compile the initial-reachable hierarchy and bind method heads before enumerating free parameters. All alternatives remain available regardless of their initial guard values. |
| Grounded method IDs depended on an implicit compilation strategy. | New training records identify their grounding mode; legacy records default to full grounding. All 1,273 existing records passed a compatibility audit. |
| The type compiler rejected UM-Translog's multiple inheritance and case-only domain-name differences. | Types now have multiple parents with cycle detection; domain-name matching ignores ASCII case. Other mismatched domain names remain errors. |
| A backward quota failure discarded valid samples accumulated in memory. | Collection publishes each validated sample immediately. The incremental API returns partial progress; the strict quota API remains available. The collector defaults to reachable synthetic templates. |
| Candidate merges cloned partitions, rebuilt quotient closure, and repeatedly traversed descendants. | Closure is computed once per merge level; candidate cycle checks use cached reachability, and interface scores reuse cluster features and family counts. |
| The master explored equivalent failed suffixes through many block permutations. | Exhaustively disproven suffixes are cached by remaining blocks and complete world, separately per initial grounding. Unknown and depth-limited failures never enter that cache. |
| A crash between copying a sample and updating the manifest prevented resumption. | Publication fsyncs records and metadata and can adopt an untracked immutable record when a newly audited source has the identical checksum. |
| Detached startup could report success after a lock failure, and stopping could leave nested planners running. | Startup readiness is checked before replacing launcher metadata. Worker groups are supervised and terminated on timeout or stop; interrupted jobs resume automatically. Both executable and supervisor are snapshotted. |
| Completed failures required a fresh work directory to retry. | `--retry-failed` preserves successes and retains prior attempts and configuration hashes. |
| Auditing omitted cached micropartitions and updated the manifest before semantic validation. | Auditing checks memberships and metadata exactly, compares floating metrics with tolerance, and commits a rebuilt manifest only after validation. |
| Default builds pulled in unused native solver and neural stacks. | `good_lp`, `burn`, and `rayon` are optional Cargo features. |
| Solver work and intermediate oracle decisions were difficult to inspect. | Search/master counters and optional full request/certificate events expose frontier growth, guards, boundaries, cuts, and outcomes. Planner JSON can include the complete master trace. |
| Solution records did not capture counterfactual merge decisions or checkpoint provenance. | A separate content-addressed experiment store publishes immutable problems, rollouts, models, and reports with atomic indexes, writer locking, checksums, and orphan recovery. |
| Greedy scores supplied no trainable merge or STOP decision. | A CPU relational encoder with policy/value heads now trains from bounded candidate/STOP outcomes and runs through the same symbolic merge constraints. |
| Timeouts could become misleading cheap values or leak related examples across splits. | Only successful observed costs and clean master-prefix lower bounds supervise values; abandoned local searches and skipped boundary branches have no value label. Source-domain hashes enforce group splits. |
| Learned merge inference considered only one step. | Receding-horizon beam search re-encodes actual legal contractions, deduplicates equivalent partitions, retains STOP, and bounds total model evaluations. |
| Exhaustive boundary enumeration had no learned ordering. | A trained true/false/unknown atom predictor proposes the first valuation while a bijective exhaustive fallback retains all candidates. Complete-plan and local-witness training evidence have separate counts. |
| Exact no-goods could not exclude related outgoing contracts. | Oracle-verified outgoing conflict cores prune conjunction supersets under the same proof scope. Extraction shares master budgets and tries empty cores first. |
| Conflict cores required identical complete incoming worlds and still visited every boundary valuation. | Canonical guard/goal projections permit reuse across irrelevant fact changes; indexed subsumption retains only active strong cuts while preserving the proof archive. Binary cube jumps skip forbidden valuation ranges, including under learned ordering. |
| Final-block failures could not contribute reusable cores. | The fixed goal produces an incoming-scoped cut without deletion probes, including when the source proof consumes the remaining oracle budget. |
| Cut applications lacked trace provenance and paired evaluation. | Full traces check matching and cube jumps; audits recompute incoming dependencies and key proof caches by scope. The paired benchmark alternates execution order and independently verifies every cut. |
| Merge training trusted cached features that the dataset auditor would reject. | Both trainers now use the same snapshot-scoped semantic auditor before accepting labels. Source metadata, reconstructed observations, partitions, and positive witnesses must agree. |
| Full oracle traces could attribute a different network or larger budget to a valid partition. | Audit checks induced networks, initial-network/partition provenance, budget bounds, interface goals/future reads, and the requests behind core proof events. |
| Beam transpositions compared temporary value estimates even though the next expansion replaces them. | Expandable duplicates retain the highest prefix log-probability; completed duplicates retain the best final score. A shared-suffix regression check covers the distinction. |
| Beam rounds repeated identical model evaluations and copied large ancestor traces. | Predictions are memoized using ordered cluster IDs/membership under fixed graph/model/budgets, with 128-entry and 16 MiB payload limits. Branches share first-decision traces. |
| Inference allocated derivative graphs and evaluation metrics computed unused gradients. | A values-only tape now serves inference and loss evaluation. Interface proposals evaluate boundary atoms only, retaining identical features and probabilities. |
| Interface training retained a derivative graph for every atom at once. | Independent row gradients accumulate into a shared parameter gradient, bounding tape size independently of interface width. |
| Checkpoints could claim disjoint splits while listing the same record twice, or inconsistent model/options metadata. | Validation rejects record overlap, model-shape disagreement, and invalid optimizer/training options before training or checkpoint use. |
| Auditing repeatedly grounded the same sources and re-proved identical cores. | Snapshot-local caches reuse compiled sources and independently verified proofs keyed by source, compilation cap, network roots, world, core, proof scope, and local budget. |
| Every imagined beam node required another graph-encoder pass. | A separately trained frozen-encoder transition predicts node embeddings inside a depth 1–4 beam, with exact legality and action features. Every committed move re-encodes the true quotient. Imagined histories bypass symbolic transposition dominance and caching. |
| Successful local witnesses omitted causal support and threat obligations. | Replayed local segments now include positive/negative guard imports, effect exports, preservation, support links, threat resolutions, and an HTN-compatible event partial order. Audit reconstructs all claims; legacy segments remain readable. |
| Floating model weights could change by one bit across JSON reloads. | Exact JSON float round-tripping preserves encoder fingerprints and deterministic latent resume. |
| Causal hierarchy extraction copied every ancestor's descendants and scanned every event for every support link. | Version-two certificates use entry/exit frontiers and per-atom effect indexes. Legacy version-one certificates retain canonical validation; sparse and expanded orderings have identical tested reachability. |
| Alternative causal orderings required occurrence-remapping code found only in tests. | A public replay API validates event coverage, order and imported assumptions, remaps refinements, and returns a symbolically certified local plan in the supplied concrete world. |
| One-step latent MSE hid open-loop drift, and STOP-only states inflated apparent policy agreement. | A bounded audited evaluator reports per-depth errors against a mean-pair baseline, separates domain splits, excludes trivial policy decisions, and marks missing samples explicitly. The pilot exposes worse held-out error at depths one and two. |
| Beam feature timing omitted quotient construction for candidate children and committed moves. | A separate aggregate quotient-time counter accounts for initial, imagined-child, and committed quotient construction. |
| A surviving duplicated writer descriptor could prolong a completed dataset write's lock. | The writer guard explicitly unlocks before closing. A regression test retains a duplicate handle and verifies both immediate release and isolation of the next writer's lock. |
| Contiguous block execution excluded otherwise valid partition interleavings. | A POCL master now enumerates structural local plans, connects event-local support across partitions, resolves threats, and independently verifies and globally replays successful plans. |
| Local assumptions could only be interpreted as input-world facts. | POCL treats imports at their consumer events, including method guards and successive opposite requirements within one partition. A supplied-certificate adapter preserves local method/order evidence while distinguishing bundle-scoped failure from global exhaustion. |

PANDA timeout/unsolved statuses are also checked before conversion and cost
extraction, so a timeout is reported directly instead of as a missing action cost.

## Remaining limits

- Reachability grounding still eagerly expands free method parameters and finite
  quantifiers. Large reachable hierarchies continue to exceed the grounding cap;
  relational/static-condition pruning or lazy grounding remains future work.
- Sequential training rollouts still search contiguous blocks. The separate POCL
  master interleaves partitions but jointly enumerates structural choices, uses
  dense ordering matrices, and does not yet supply training rollouts, cost
  optimization, or reusable POCL conflict-core cuts.
- Spectral routines, ordering closure, and graph construction remain dense.
  Agglomeration still recomputes per-level features rather than maintaining a
  fully incremental sparse quotient.
- Backward regression accepts conjunctive literals. Numeric/conditional effects
  and domain axioms remain unsupported. Reachable-template sampling also limits
  the synthetic root pool; full-pool sampling is an explicit option.
- Some official files name a different domain, beyond letter case. They remain
  rejected rather than being silently rewritten. PANDA search and exact replay
  can independently exhaust their resource limits.
- The first learned model fits a small training set but scores only 1/4 on
  held-out preference comparisons. Inputs remain lifted structural summaries,
  collection follows greedy trajectories, and there is no demonstrated planning
  speed improvement. See [the learning workflow](learning.md) for pilot metrics.
- Merge value inputs encode search budgets but not interface-model identity or
  core-probe settings. Mixing those search configurations can confound cost
  targets. Domain hashes also do not identify semantically equivalent rewritten
  domains. Balanced data, configuration-aware costs, and broader evaluation remain
  open work.
- Semantic audit caches retain prepared sources for one snapshot. This reduces
  repeated work at the cost of memory; very large collections may need bounded
  source caching or source-grouped streaming.

## Verification

`cargo test --offline` passes 141 Rust tests, including small-model comparisons
between full/reachable/partitioned search, exhaustive DAG contraction checks,
all-parameter gradient checks, deterministic optimizer resume, domain isolation,
store crash recovery, full-trace auditing, and the collect/train/inference CLIs.
New checks compare core-enabled search on 48 small choice models, independently
re-verify cuts, preserve depth-bound scope and exhaustive fallback under poor
predictions, distinguish local interface evidence, and check beam transpositions,
receding-horizon search, and inference caps.
Review regressions also cover failed-request provenance, semantic validation
during merge training, snapshot/grounding-cap isolation, cached proof reuse,
boundary-only prediction equivalence, streaming gradient means, inference loss
equivalence, and checkpoint metadata validation.
Incoming-core checks exhaust all small matching worlds, reject omitted guard and
goal/frame dependencies, preserve legacy exact-world records, reject forged
scope cache hits and application traces, and check all five-bit forbidden cubes
under every XOR order. Wide-boundary tests cover 65,536 assignments with two
oracle calls and saturating counters across astronomical cubes. The benchmark
CLI is exercised on feasible and infeasible fixtures.
Latent checks cover every transition parameter's finite-difference gradient,
frozen-target fitting, exact encoder/decoder equivalence, deterministic resume,
checkpoint identity, capped receding-horizon inference, and both new CLI paths.
Causal checks replay every permitted linearization on small fixtures, including
refinement occurrence remapping, method guards, negative support, frame facts,
disjunctions, imports/exports for interleaving, and rejection of forged contracts.
Review checks cover compact/legacy order reachability, deep hierarchy edge counts,
the production causal replay API, surviving writer handles, multi-step drift
baselines and sample masks, source auditing, path limits, and reproducible
evaluation artifacts without dataset mutation.
POCL checks compare 128 finite models against monolithic solving, verify closure
and cycle detection across 4-node edge subsets, recover noncontiguous plans,
exercise promotion/demotion and changing imports, preserve original HTN order,
validate supplied local evidence, reject altered certificates, and distinguish
global, supplied-bundle, depth-bound, and resource-limited failure scopes.
The Python suite passes 11 supervision tests covering crash recovery, retries,
startup locks, nested-process cancellation, and interrupted-job resumption.
Formatting and Clippy checks pass. The background retry pass has finished all
1,180 jobs; the solution manifest now contains 1,495 records. The separate mixed
learning store contains 161 audited objects, including 124 rollouts, and selected
merge/interface checkpoints are available under `target/`. The interface model
has only 15 training interfaces and one held-out domain group; see
[search guidance](search-guidance.md) for scoped metrics and remaining limits. Inspect
`target/ipc2023-collection/progress.json` for per-job outcomes.

The stricter audit accepted all 161 existing objects: 29 source preparations,
238 independent core verifications, and 296 reused proofs. On the four-item
integration fixture, beam inference used ten model evaluations plus three cache
hits rather than 13 evaluations, while retaining a replay-valid four-action plan.
These are work counters and integration checks, not a planning-speed benchmark.
The separate 29-problem paired core pilot verifies all 66 generated cuts and
reduces local calls from 917 to 194, including 88 probes. Feasible coverage stays
at four problems; partition-level infeasibility decisions increase from three
to 14. See [search guidance](search-guidance.md) for limits and reproduction.
