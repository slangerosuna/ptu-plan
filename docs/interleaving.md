# Cross-partition POCL interleaving

## Run the master

```sh
cargo run --offline --bin ptu-plan -- tests/fixtures/interleaving.json \
  --interleave --micro-max-size 1 --analysis-json
```

The `exchange-signals` fixture requires events from two partitions to interleave.
The master returns the four actions `make-p`, `make-q`, `need-q`, `need-p`, with
two cross-partition support links and a globally replayed HTN witness. The
sequential master exhausts the same singleton partitions without a plan.

`--interleave`, `--sequential`, and `--solve` are separate planning modes.
Spectral seeds determine root ownership unless agglomeration is requested.
Partitions must cover every root exactly once; arbitrary unique partition IDs
are supported. The POCL master preserves original root and method precedence;
it does not require an acyclic quotient of contiguous partition blocks.

## Event-local causal reconciliation

`interleaving::solve_interleaved` first enumerates grounded method choices,
retaining each root's partition ownership throughout its descendants. These
local hierarchies are structural proposals, not feasibility certificates.
Method alternatives remain available even when their guards are initially false.

For each complete proposed decomposition, the master builds event entry/exit
frontiers that preserve hierarchy and HTN precedence. Every primitive action and
method refinement has its own guard obligation. The global goal belongs to a
separate finish boundary. Compound refinement guards are evaluated before their
children, at their actual refinement events.

POCL search then:

1. Expands conjunctions, disjunctions, and nested negations into sufficient guard
   branches without eagerly enumerating a full disjunctive normal form.
2. Chooses support from the actual initial world or a matching action effect.
   Negative literals can use initial absence or an effective delete effect.
3. Orders a producer before its consumer and rejects ordering cycles.
4. Detects opposite effects that could occur inside protected support intervals.
   It branches on ordering the threat before the producer or after the consumer.
5. Remaps decomposition occurrence IDs for a topological event order and replays
   every refinement and action against the original complete problem.

Imports are required at consumer events, not necessarily at the global start.
One partition can require `p` and later `NOT p` while another supplies the
intermediate writes. An action's guard precedes its own effects; add wins when
an action both adds and deletes an atom. Initial and finish boundaries cannot
be moved to evade threats. If one supporter fails, other supporters and
refinement choices remain available through backtracking.

## Existing local certificates

`interleaving::reconcile_local_certificates(problem, initial_network, sources,
options)` accepts independently solved `master::LocalSegment` values. Their
complete source worlds may contain assumed imports. The adapter checks each
induced network, boundary annotation, local witness, and optional causal
contract; legacy segments without a causal contract are reconstructed.

The selected grounded methods and certified internal ordering remain fixed.
The master re-establishes event-local guard support from actual global producers;
it never copies assumed source facts into the global initial state. Local output
facts do not automatically become final global requirements. Successful evidence
retains the source segments and is independently validated against them.
This adapter fixes decomposition and internal ordering; the original local
requested goals are not separate boundary requirements in the reconciled search.
Support is established from actual event effects for consumers and the global
goal. Use it as a source of structural evidence rather than a promise to preserve
every unused local export throughout global execution.

An exhausted supplied bundle reports `scope: SuppliedLocalCertificates`. Other
local decompositions can still solve the problem. Automatic structural search
reports `scope: AllDecompositions`. Both APIs return the explicit options,
exactness mode, proof status, certificate if any, and work counters.

## Certificate verification

`validate_interleaving_certificate(problem, certificate)` independently checks:

- Root coverage, task/partition ownership, grounded method heads, complete child
  allocation, decomposition depth, and original hierarchy/order constraints.
- Actual producer effects, sufficient positive/negative guard and goal witnesses,
  protected causal links, and absence of unresolved threats.
- Consistency with supplied local certificates when present.
- Event coverage and precedence of the selected linearization, occurrence-ID
  remapping, exact decision/action lists, final state, and whole-problem replay.

A successful certificate includes partition-owned steps, source evidence when
applicable, explicit ordering edges, causal links, guard/goal witnesses, the event
linearization, and the final `PlanCertificate`. It is tied to the same compiled
problem, as are existing exact-solver certificates. Learned scores supply no hard
constraints in this master.

## Bounds and status

Relevant CLI limits are:

| Flag | Meaning |
| --- | --- |
| `--max-master-nodes` | Total popped structural and causal search states |
| `--max-generated` | Total queued states across both phases |
| `--master-time-limit-ms` | Master search time, including proposal preparation |
| `--max-pocl-events` | Maximum events in a proposed decomposition; default 256 |
| `--max-pocl-frontier` | Combined pending structural/causal states; default 1,024 |
| `--max-decomposition-depth` | Optional exact refinement-depth bound |
| `--max-ground-instances` | Separate initial-reachable compilation cap |

The separate local-progression flags `--max-expansions`, `--time-limit-ms`, and
`--max-local-calls` do not control POCL search. Time is checked between states and
branch generation; final certificate verification finishes before acceptance.
Ordering uses dense matrices; the explicit event cap cannot exceed 4,096.

`Feasible` always requires a valid complete witness. For `AllDecompositions`,
`Infeasible` means all initial groundings, structural choices, guard branches,
supporters, and threat resolutions were exhausted with no cutoff. Depth-pruned
exhaustion is `InfeasibleWithinBound`; any dropped resource-limited branch makes
unsuccessful search `BudgetExhausted`. For `SuppliedLocalCertificates`, exhaustion
applies only to that supplied bundle. No timeout or heuristic failure creates a
permanent conflict cut.

Counters distinguish structural/causal states, local refinements, complete
proposals, support/condition/threat choices, cycle rejections, unsupported
conditions, depth/resource cutoffs, peak frontier, and replay checks. Choice
counters include unsuccessful branches; count links in the returned certificate
when measuring the final plan's cross-partition support.

## Validation and limits

Tests compare POCL and monolithic search on 128 finite Boolean models and compare
incremental ordering closure with independent closure across all 4-node directed
edge subsets. Additional cases cover both threat resolutions, changing imports,
method guards, empty refinements, add/delete conflicts, cyclic block quotients,
source-certificate adaptation and rejection, resource/depth scope, tampered
certificates, and CLI success/failure output.

A local smoke pass over the 29 existing source problems used 2,000 states,
10,000 generated states, 64 events, 64 pending states, and 250 ms per search.
It returned six replay-verified plans and 23 resource-limited results, with no
preparation/process failures. These are mixed, overlapping source problems,
not independent IPC coverage. The report and per-problem artifacts are under
`target/pocl-pilot/`; the signal-exchange proof is also saved as
`target/pocl-interleaving-result.json`. Different caps from previous sequential
pilots prevent a performance comparison. No optimality or speed claim follows.

This first implementation enumerates structural choices jointly and uses dense
ordering closure with cloned DFS states. It does not yet provide distributed
local proposal streams, cost optimization, numeric/temporal resource semantics,
POCL training rollouts, learned supporter ranking, or reusable POCL conflict-core
cuts. Existing training rollouts still use the sequential master. Unbounded
recursive HTNs can exhaust resources; the master does not promise termination
or completeness over the unrestricted formalism.
