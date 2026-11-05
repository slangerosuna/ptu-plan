#!/usr/bin/env python3
"""Summarize paired planner ablations with domain-clustered uncertainty."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = q*(len(values)-1)
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high]-values[low])*(pos-low)


def solved(run):
    return run["status"] == "Feasible" and run["replay_verified"] and not run["error"]


def interval(groups, transform=lambda x: x, iterations=2000):
    """Resample domains, then paired problems within each selected domain."""
    if not groups:
        return None
    rng = random.Random(42)
    domains = sorted(groups)
    scores = []
    for _ in range(iterations):
        sample = []
        for _ in domains:
            values = groups[rng.choice(domains)]
            sample.append(statistics.mean(rng.choice(values) for _ in values))
        scores.append(transform(statistics.mean(sample)))
    return [percentile(scores, 0.025), percentile(scores, 0.975)]


def summarize(report):
    variants = report["options"]["variants"]
    repeats = report["options"]["repeats"]
    cases = report["cases"]
    if not report["complete"] or len(cases) != report["source_problems"] or len({c['problem_id'] for c in cases}) != len(cases):
        raise ValueError("benchmark is incomplete or has duplicate cases")
    data = {v: {} for v in variants}
    for case in cases:
        for variant in variants:
            runs = [r for r in case["runs"] if r["variant"] == variant]
            if not case["error"] and (len(runs) != repeats or {r['repetition'] for r in runs} != set(range(repeats))):
                raise ValueError("missing or duplicate paired repetitions")
            if any(r["status"] == "Feasible" and not r["replay_verified"] and not r["error"] for r in runs):
                raise ValueError("unverified solution")
            if not runs:
                record = {"all_solved": False, "any_solved": False, "error": True, "runs": [], "group": case["group_id"]}
            else:
                record = {
                    "all_solved": all(map(solved, runs)), "any_solved": any(map(solved, runs)),
                    "error": any(r["error"] for r in runs), "runs": runs, "group": case["group_id"],
                    "planning_ms": statistics.median(r["planning_us"]/1000 for r in runs),
                    "partition_ms": statistics.median(r["partition_us"]/1000 for r in runs),
                    "master_ms": statistics.median(r["master_us"]/1000 for r in runs),
                    "par2_ms": statistics.median(r["planning_us"]/1000 if solved(r) else r["partition_us"]/1000+2*report["master"]["max_time_ms"] for r in runs),
                    "local_calls": statistics.median(r["statistics"]["local_calls"] for r in runs),
                    "local_expanded": statistics.median(r["statistics"]["local_expanded"] for r in runs),
                    "interface_candidates": statistics.median(r["statistics"]["interface_candidates"] for r in runs),
                    "interface_proposals": statistics.median(r["statistics"]["interface_proposals"] for r in runs),
                    "interface_inference_ms": statistics.median(r["statistics"]["interface_inference_us"]/1000 for r in runs),
                    "partitions": statistics.median(len(r["partitions"]) for r in runs),
                }
            data[variant][case["problem_id"]] = record
    groups = sorted({c['group_id'] for c in cases})
    domain_names = {c['group_id']: c['domain'] for c in cases}
    totals = {}
    for variant in variants:
        records = list(data[variant].values())
        measured = [r for r in records if r["runs"]]
        statuses = {}
        for r in records:
            for run in r["runs"]:
                name = "Error" if run["error"] else run["status"]
                statuses[name] = statuses.get(name, 0)+1
        by_domain = {}
        for group in groups:
            subset = [r for r in records if r["group"] == group]
            by_domain[group] = {"domain": domain_names[group], "cases": len(subset), "solved_all_repeats": sum(r['all_solved'] for r in subset), "solved_any_repeat": sum(r['any_solved'] for r in subset)}
        totals[variant] = {
            "cases": len(records), "solved_all_repeats": sum(r["all_solved"] for r in records),
            "solved_any_repeat": sum(r["any_solved"] for r in records),
            "unstable_solve_cases": sum(r["any_solved"] != r["all_solved"] for r in records),
            "errors": sum(r["error"] for r in records), "statuses_across_repeats": statuses,
            "macro_domain_solve_rate": statistics.mean(g['solved_all_repeats']/g['cases'] for g in by_domain.values()),
            "domains": by_domain,
        }
        for name, predicate in {
            "boundary_cutoff_cases": lambda r:r['statistics'].get('boundary_cutoffs', 0)>0,
            "local_unknown_cases": lambda r:r['statistics'].get('local_unknown', 0)>0,
            "local_call_limit_cases": lambda r:r['statistics']['local_calls']>=report['master'].get('max_local_calls', math.inf),
            "observed_master_deadline_cases": lambda r:r['master_us']>=report['master']['max_time_ms']*1000,
        }.items():
            totals[variant][name] = sum(any(not solved(run) and predicate(run) for run in r['runs']) for r in records)
        for metric in ("planning_ms", "partition_ms", "master_ms", "par2_ms", "local_calls", "local_expanded", "interface_candidates", "interface_proposals", "interface_inference_ms", "partitions"):
            values = [r[metric] for r in measured]
            totals[variant]["mean_"+metric] = statistics.mean(values) if values else None
            totals[variant]["median_"+metric] = statistics.median(values) if values else None
        totals[variant]["p95_planning_ms"] = percentile([r["planning_ms"] for r in measured], 0.95)
        totals[variant]["mean_beam_evaluations"] = statistics.mean((run.get("policy") or {}).get("beam", {}).get("evaluations", 0) if (run.get("policy") or {}).get("beam") else 0 for r in measured for run in r['runs']) if measured else None
        totals[variant]["mean_latent_transitions"] = statistics.mean((run.get("policy") or {}).get("beam", {}).get("latent_transitions", 0) if (run.get("policy") or {}).get("beam") else 0 for r in measured for run in r['runs']) if measured else None
    pairs = []
    comparisons = [("singleton", "greedy"), ("greedy", "policy"), ("greedy", "greedy-interface"), ("policy", "policy-interface"), ("policy-interface", "beam-exact-interface"), ("beam-exact", "beam-exact-interface"), ("beam-exact-interface", "beam-latent-interface"), ("greedy", "beam-latent-interface"), ("policy-interface", "policy-escalating-interface"), ("beam-latent-interface", "beam-escalating-interface"), ("greedy", "policy-escalating-interface")]
    for baseline, candidate in comparisons:
        if baseline not in data or candidate not in data:
            continue
        gained = lost = common = matches = unstable = 0
        solve_groups, log_groups, par2_groups = {}, {}, {}
        logs = []
        for case in cases:
            a, b = data[baseline][case['problem_id']], data[candidate][case['problem_id']]
            gained += b['all_solved'] and not a['all_solved']
            lost += a['all_solved'] and not b['all_solved']
            unstable += (a['all_solved'] != a['any_solved']) or (b['all_solved'] != b['any_solved'])
            solve_groups.setdefault(case['group_id'], []).append(int(b['all_solved'])-int(a['all_solved']))
            if a['runs'] and b['runs']:
                matches += all(x['partitions'] == y['partitions'] for x, y in zip(sorted(a['runs'], key=lambda r:r['repetition']), sorted(b['runs'], key=lambda r:r['repetition'])))
                par2_groups.setdefault(case['group_id'], []).append(b['par2_ms']-a['par2_ms'])
            if a['all_solved'] and b['all_solved']:
                common += 1
                ratio = math.log(max(a['planning_ms'], 0.001)/max(b['planning_ms'], 0.001))
                logs.append(ratio)
                log_groups.setdefault(case['group_id'], []).append(ratio)
        pairs.append({"baseline": baseline, "candidate": candidate, "gained_solves": gained, "lost_solves": lost, "unstable_cases": unstable, "common_solved": common, "same_partitions": matches,
                      "macro_domain_solve_delta_pp": 100*statistics.mean(statistics.mean(v) for v in solve_groups.values()),
                      "macro_domain_solve_delta_pp_ci95": interval(solve_groups, lambda x:100*x),
                      "common_solved_domains": len(log_groups), "common_solved_geomean_speedup": math.exp(statistics.mean(logs)) if logs else None,
                      "common_solved_macro_domain_speedup": math.exp(statistics.mean(statistics.mean(v) for v in log_groups.values())) if log_groups else None,
                      "common_solved_macro_domain_speedup_ci95": interval(log_groups, math.exp),
                      "macro_domain_par2_delta_ms": statistics.mean(statistics.mean(v) for v in par2_groups.values()) if par2_groups else None,
                      "macro_domain_par2_delta_ms_ci95": interval(par2_groups)})
    return {"schema_version": 1, "cases": len(cases), "domain_groups": len(groups), "repeats": repeats, "preparation_errors": sum(c['error'] is not None for c in cases), "mean_shared_preparation_ms": statistics.mean(c['preparation_us']/1000 for c in cases), "variants": totals, "comparisons": pairs}


def markdown(summary, report, source):
    n = summary["cases"]
    lines = ["# Held-out planner benchmark", "", f"Source: `{source}`. {n} problems in {summary['domain_groups']} held-out domain groups, {summary['repeats']} repetitions per variant. Native Rust CPU inference; variants execute serially with rotated and alternating run order. All feasible results are independently replay-validated.", "", "## Results", "", "Solve counts require success in every repetition. Durations and effort first take the median repetition for each problem, then the mean over problems. Errors and unstable solves are reported separately in the JSON summary.", "", "| Variant | Solved | Domain mean solve rate | Planning ms | Partition/inference ms | Local calls | Expanded | PAR2 ms |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in summary['variants'].items():
        def number(value, digits=3):
            return f"{value:.{digits}f}" if value is not None else "n/a"
        values = [name, f"{r['solved_all_repeats']}/{n}", f"{100*r['macro_domain_solve_rate']:.1f}%", number(r['mean_planning_ms']), number(r['mean_partition_ms']), number(r['mean_local_calls'], 2), number(r['mean_local_expanded'], 2), number(r['mean_par2_ms'])]
        lines.append("| "+" | ".join(values)+" |")
    lines += ["", "## Paired component differences", "", "Gained/lost solves compare the same problems. Speedup is baseline time divided by candidate time on problems consistently solved by both; greater than one means faster. The interval resamples domains and paired problems within domains. These intervals and domain averages are exploratory with so few held-out groups. They condition on each problem's measured median and do not capture every source of timing variation.", "", "| Baseline → candidate | Gained / lost solves | Same partitions | Common solved | Domain mean speedup [95% interval] |", "| --- | ---: | ---: | ---: | ---: |"]
    for p in summary['comparisons']:
        interval_value = p['common_solved_macro_domain_speedup_ci95']
        speed = f"{p['common_solved_macro_domain_speedup']:.2f}× [{interval_value[0]:.2f}, {interval_value[1]:.2f}]" if interval_value else "n/a"
        lines.append(f"| {p['baseline']} → {p['candidate']} | {p['gained_solves']} / {p['lost_solves']} | {p['same_partitions']}/{n} | {p['common_solved']} | {speed} |")
    lines += ["", "## Domain coverage", "", "| Domain (hash prefix) | Problems | " + " | ".join(summary['variants'])+" |", "| --- | ---: | "+" | ".join("---:" for _ in summary['variants'])+" |"]
    first = next(iter(summary['variants'].values()))
    for group, r in first['domains'].items():
        lines.append(f"| {r['domain']} ({group[:8]}) | {r['cases']} | "+" | ".join(str(v['domains'][group]['solved_all_repeats']) for v in summary['variants'].values())+" |")
    lines += ["", "## Effort and failure diagnostics", "", "Counts below are problems with at least one unsuccessful repetition exhibiting the indicated condition. Columns can overlap; a budget-exhausted status alone does not identify the binding limit.", "", "| Variant | Boundary cap | Local unknown | Local call cap | Observed master deadline | Mean interface proposals |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in summary['variants'].items():
        lines.append(f"| {name} | {r['boundary_cutoff_cases']} | {r['local_unknown_cases']} | {r['local_call_limit_cases']} | {r['observed_master_deadline_cases']} | {number(r['mean_interface_proposals'], 2)} |")
    m = report['master']
    lines += ["", "## Protocol and interpretation", "", f"- Source snapshot: `{report['snapshot_id']}`. Model file hashes and the machine description are in the raw report.", f"- The same master budget ({m['max_time_ms']} ms), local budget ({m['local_budget']['max_time_ms']} ms), call limit ({m['max_local_calls']}), boundary cap ({m['max_boundary_atoms']}), and conflict-core probe limit ({report['options']['core_checks']}) apply to every variant.", "- Micro-partitions start at size one. Structural and learned mergers share cluster constraints; learned variants may select STOP. Beam variants share width, depth, branching, and evaluation limits.", "- Shared analysis, graph construction, singleton spectral partitioning, and grounding happen once per problem and are excluded from planning times. Their mean cost is "+f"{summary['mean_shared_preparation_ms']:.3f} ms. Model files are loaded and validated before timing. One case warms all variants before recorded runs.", "- Planning time includes partition selection, features, model inference, master search, and its normal certificate checks. An additional independent replay check runs outside the timer.", "- PAR2 uses observed planning time for solved runs; failures receive twice the master time limit plus observed partition time. Timeout/budget exhaustion and partition-level sequential infeasibility remain distinct in raw status counts.", "- Escalating variants share the same total master time/call/node limits across retries. Their final partitions and complete attempt statistics are retained; individual local time slices may increase up to the escalation stage cap.", "- A sequentially infeasible partition does not establish global problem infeasibility. Different partition choices can legitimately change solve outcomes.", "- The cohort mixes synthetic and legacy problems from training-held-out domains. It is not a full IPC performance claim. Runtime ratios on common-solved cases do not account for solve-rate differences; read both together.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    report = json.loads(raw)
    summary = summarize(report)
    summary['raw_sha256'] = hashlib.sha256(raw).hexdigest()
    args.report.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    args.report.with_suffix('.md').write_text(markdown(summary, report, args.report))
    with args.report.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('problem_id','group_id','domain','roots','variant','repeat','status','error','planning_us','partition_us','master_us','partitions','local_calls','local_expanded','interface_candidates','plan_actions','replay_verified'))
        for case in report['cases']:
            for r in case['runs']:
                writer.writerow((case['problem_id'],case['group_id'],case['domain'],case['roots'],r['variant'],r['repetition'],r['status'],r['error'],r['planning_us'],r['partition_us'],r['master_us'],len(r['partitions']),r['statistics']['local_calls'],r['statistics']['local_expanded'],r['statistics']['interface_candidates'],r['plan_actions'],r['replay_verified']))
    print(args.report.with_suffix('.md'))
    for name, value in summary['variants'].items():
        print(name, f"{value['solved_all_repeats']}/{summary['cases']} solved", f"{value['mean_planning_ms']:.3f} ms", 'errors', value['errors'])


if __name__ == '__main__':
    main()
