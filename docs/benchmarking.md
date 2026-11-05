# Benchmarking trained planner components

`benchmark-models` compares frozen checkpoints using source problems from the
merge checkpoint's held-out test domains. It verifies store/model snapshot
identity and split provenance before running. No weights or thresholds are
changed during evaluation.

## Run and resume

```bash
cargo run --release --bin benchmark-models -- \
  --store runs/full/data --models runs/full/models \
  --output runs/full/benchmarks/heldout-350ms.json --repeats 3
python3 scripts/summarize_benchmark.py runs/full/benchmarks/heldout-350ms.json
```

The current cohort contains 151 unique source problems across five held-out
domain hashes. A positive `--limit` selects problems round-robin across domains;
the default zero includes all held-out sources. Each case is committed atomically
to the report. Add `--resume` with the same arguments after interruption.
Changed data, weights, or options require a new output file.

Eight variants isolate the components:

| Variant | Partition selection | Interface guidance |
| --- | --- | --- |
| singleton | Singleton spectral seeds | None |
| greedy | Structural greedy agglomeration | None |
| greedy-interface | Structural greedy agglomeration | Learned |
| policy | Single-step merge/value policy | None |
| policy-interface | Single-step merge/value policy | Learned |
| beam-exact | Learned beam with exact graph re-encoding | None |
| beam-exact-interface | Learned beam with exact graph re-encoding | Learned |
| beam-latent-interface | Learned beam with latent transitions | Learned |

All variants use the same sequential master, local oracle, graph weights,
cluster constraints, and conflict-core probe limit. Interface-only pairs must
produce identical partitions. Contradictory feasibility results for identical
partitions fail the benchmark. Every positive result receives independent
certificate replay outside the measured planning time.

## Budgets and measurement

The primary configuration matches collection: 350 ms master time, 100 ms per
local call, 64 local calls, 12 boundary atoms, and two conflict-core probes.
Micro-partitions start at size one. Learned scoring uses value weight 0.1; beam
width/depth/branching are 8/3/8 with at most 10,000 evaluations.

Native Rust CPU inference runs serially. Variant order rotates per problem and
alternates direction across repetitions. One problem warms all variants before
recorded runs. Shared parsing, analysis, graph construction, singleton spectral
partitioning, and grounding are excluded from planning time and reported
separately. Planning time includes partition selection, feature building,
inference, master search, and normal internal certificate checks.

The summary takes the median repetition for each problem, then averages across
problems. Solve counts require success in every repetition; unstable outcomes
and errors remain visible. PAR2 scores successful runs by observed planning
time and unsuccessful runs by twice the master time limit plus partition time.
Raw status counts distinguish bounded/unknown search and partition-level
sequential infeasibility. Neither implies global problem infeasibility.

Paired reports count gained/lost solves, identical partitions, local effort,
and runtime on commonly solved problems. Runtime ratios exceed one when the
candidate is faster. Domain averages give each domain equal weight; bootstrap
intervals resample domains and paired problems within them. Five domain groups
provide limited evidence, and common-solved runtime ratios must be read
alongside solve-rate changes. This mixed synthetic/legacy cohort supports no
claim of full IPC coverage.

Treat `greedy-interface` as a timing control when greedy produces a single
block: if its interface proposal count is zero, no predictor ran. Small timing
differences in that comparison reflect measurement variation rather than model
benefit. The bootstrap conditions on each problem's measured median; it does
not capture every source of hardware and scheduling variation.

A second budget checks sensitivity:

```bash
cargo run --release --bin benchmark-models -- \
  --master-time-ms 2000 --local-time-ms 500 --max-local-calls 256 \
  --output runs/full/benchmarks/heldout-2000ms.json --repeats 3
python3 scripts/summarize_benchmark.py runs/full/benchmarks/heldout-2000ms.json
```

The larger budgets also change the policy's search-context features, matching
normal planner behavior. Within each configuration, all variants see the same
budgets. Compare configurations separately; the second is not a pure oracle-only
ablation. The boundary cap and expansion/generation limits remain fixed.

## Artifacts

The raw JSON retains model hashes, source IDs, configurations, machine details,
partitions, per-run timing, statuses, replay results, and search statistics.
The summarizer produces `.summary.json`, `.csv`, and `.md` beside it. Small
timing differences and individual submillisecond ratios should not drive a
default configuration change without repetition on larger problems.

## Compare a new dataset on the original cohort

Pass `--cohort runs/full/benchmarks/heldout-350ms.json` when benchmarking the WL
models against a larger store. This selects the exact original problem payloads,
preserves their order, and rejects missing or ambiguous matches, duplicates,
incomplete reports, or domains outside the new model's test split. Collector
options can change provenance IDs; `problem_sha256` fingerprints match identical
programs across stores, and `cohort_problem_id` records the original ID. Legacy
reports without fingerprints require their original `options.store` to remain
available for checksum-verified reconstruction. The report records the
cohort file checksum; resume rejects a changed cohort. Use the same budgets,
variants, and repeat counts for comparisons. Keep the split seed fixed.

## Dynamic escalation variants

Add `policy-escalating-interface` or `beam-escalating-interface` to `--variants`
to evaluate repartitioning under the same total master time, local-call, and
master-node budgets. `--escalation-stage-ms` defaults to 100. The report retains
per-attempt budgets/statuses and aggregates all work; partition comparisons use
the final attempted partition. See [the escalation protocol](repartition-rl.md).
