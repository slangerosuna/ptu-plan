import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("benchmark_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_benchmark.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def run(variant, repeat, feasible, time):
    return {"variant": variant, "repetition": repeat, "status": "Feasible" if feasible else "BudgetExhausted", "replay_verified": feasible, "error": None,
            "planning_us": time, "partition_us": 10, "master_us": time-10, "partitions": [[0, 1]],
            "statistics": {"local_calls": 1, "local_expanded": 4, "interface_candidates": 1, "interface_proposals": 0, "interface_inference_us": 0}}


def fixture():
    cases = []
    for identifier, group, baseline, candidate in [("a", "large", [True, True], [True, True]), ("b", "large", [False, False], [True, True]), ("c", "small", [True, True], [True, False])]:
        runs = [run("greedy", i, value, 100) for i, value in enumerate(baseline)] + [run("policy", i, value, 50) for i, value in enumerate(candidate)]
        cases.append({"problem_id": identifier, "group_id": group, "domain": group, "preparation_us": 20, "error": None, "runs": runs})
    return {"complete": True, "source_problems": 3, "options": {"variants": ["greedy", "policy"], "repeats": 2}, "master": {"max_time_ms": 350}, "cases": cases}


class BenchmarkSummaryTests(unittest.TestCase):
    def test_paired_effects_do_not_hide_unstable_solves_or_domain_imbalance(self):
        result = summary.summarize(fixture())
        self.assertEqual(result["variants"]["policy"]["solved_all_repeats"], 2)
        self.assertEqual(result["variants"]["policy"]["solved_any_repeat"], 3)
        self.assertEqual(result["variants"]["policy"]["unstable_solve_cases"], 1)
        self.assertEqual(result["variants"]["greedy"]["macro_domain_solve_rate"], 0.75)
        pair = result["comparisons"][0]
        self.assertEqual((pair["gained_solves"], pair["lost_solves"], pair["common_solved"]), (1, 1, 1))
        self.assertEqual(pair["macro_domain_solve_delta_pp"], -25)
        self.assertAlmostEqual(pair["common_solved_geomean_speedup"], 2)
        self.assertEqual(pair["same_partitions"], 3)
        # Failed runs receive the fixed budget penalty, rather than their short duration.
        self.assertGreater(result["variants"]["greedy"]["mean_par2_ms"], 200)

    def test_rejects_incomplete_duplicate_and_unverified_samples(self):
        for mutation in (lambda r:r.update(complete=False), lambda r:r["cases"][0]["runs"].pop(), lambda r:r["cases"][0]["runs"][0].update(replay_verified=False), lambda r:r["cases"][1].update(problem_id="a")):
            report = copy.deepcopy(fixture())
            mutation(report)
            with self.assertRaises(ValueError):
                summary.summarize(report)


if __name__ == "__main__":
    unittest.main()
