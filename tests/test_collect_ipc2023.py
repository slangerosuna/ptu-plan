"""Supervisor tests use isolated fake workers; Rust tests validate sample semantics."""

import hashlib
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import threading
import time
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location(
    "collector", Path(__file__).resolve().parents[1] / "scripts/collect_ipc2023.py"
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "corpus"
        self.work = self.root / "work"
        self.work.mkdir()
        for track in ("total-order", "partial-order"):
            (self.corpus / track).mkdir(parents=True)

    def family(self, track, name, filenames):
        family = self.corpus / track / name
        family.mkdir()
        for name, contents in filenames.items():
            (family / name).write_text(contents)
        return family

    def options(self, **overrides):
        worker = self.root / "worker"
        worker.write_text("""#!/usr/bin/env python3
import hashlib, json, pathlib, sys, time
args = sys.argv[1:]
if '--audit-only' in args:
    sys.exit(0)
output = pathlib.Path(args[args.index('--output') + 1]); output.mkdir(parents=True, exist_ok=True)
problem = args[args.index('--problem') + 1]
if 'slow' in problem: time.sleep(30)
if 'fail' in problem: sys.exit(2)
data = b'{"sample":"fixture"}'; name = 'fixture.json'
(output / name).write_bytes(data)
(output / 'manifest.json').write_text(json.dumps([{'file':name, 'sha256':hashlib.sha256(data).hexdigest()}]))
(output / 'collection-report.json').write_text(json.dumps({'failures':[]}))
""")
        worker.chmod(0o755)
        values = dict(corpus=self.corpus, work=self.work, output=self.root / "dataset",
                      worker=worker, panda_root=self.root / "panda", workers=2, seed=2023,
                      synthetic_per_domain=2, search_seconds=1, job_seconds=2,
                      memory_bytes=268_435_456, max_ground_instances=200,
                      inventory_only=False, detach=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_discovery_pairs_pddl_per_instance_domains_and_ultralight_fix(self):
        self.family("total-order", "Lamps", {"domain.hddl": "(define (domain lamps))",
                    "01.pddl": "; (define (problem comment))\n(define (problem lamp))"})
        self.family("partial-order", "PCP", {"p01.hddl": "(define\n(problem p01))",
                    "p01-domain.hddl": "(define (domain pcp))"})
        self.family("partial-order", "Ultralight-Cockpit", {"p01.hddl": "(define (problem p01))",
                    "domain.hddl": "(define (domain original))", "UL_domain-type-fix.hddl": "(define (domain fixed))"})
        families, errors = collector.discover(self.corpus, self.work)
        self.assertFalse(errors)
        self.assertEqual([len(pairs) for _, pairs in families], [1, 1, 1])
        self.assertEqual(families[-1][1][0][0].name, "UL_domain-type-fix.hddl")
        self.assertEqual(families[1][1][0][0].name, "p01-domain.hddl")

    def test_sharpsat_uses_official_encoder_and_missing_pairs_are_reported(self):
        family = self.family("total-order", "SharpSAT", {
            "domain.hddl": "(define (domain sat))",
            "cnf2hddl.py": "print('(define (problem encoded))')"})
        (family / "cnfs").mkdir()
        (family / "cnfs/tiny.cnf").write_text("p cnf 1 1\n1 0\n")
        self.family("partial-order", "Missing", {"p01.hddl": "(define (problem p01))"})
        families, errors = collector.discover(self.corpus, self.work)
        self.assertEqual(len(families[0][1]), 1)
        self.assertIn("encoded", families[0][1][0][1].read_text())
        self.assertTrue(any("No paired domain" in error for error in errors))

    def test_round_robin_and_one_backward_job_per_domain_model(self):
        a = self.family("total-order", "A", {"domain.hddl": "(define (domain a))",
            "p1.hddl": "(define (problem p1))", "p2.hddl": "(define (problem p2))"})
        b = self.family("partial-order", "B", {"domain.hddl": "(define (domain b))",
            "p1.hddl": "(define (problem p1))"})
        options = self.options()
        families, _ = collector.discover(self.corpus, self.work)
        jobs = collector.jobs_for(families, options)
        self.assertEqual([j["family"] for j in jobs], ["total-order/A"] * 2 + ["partial-order/B"] * 2 + ["total-order/A"])
        self.assertEqual(sum(j["kind"] == "backward" for j in jobs), 2)
        self.assertEqual(jobs, collector.jobs_for(families, options))
        (a / "domain.hddl").write_text("(define (domain changed))")
        self.assertNotEqual(jobs[0]["id"], collector.jobs_for(families, options)[0]["id"])
        self.assertTrue(b.exists())

    def test_import_verifies_checksums_and_preserves_existing_record(self):
        output = self.root / "output"; output.mkdir()
        source = self.root / "source"; source.mkdir()
        content = b'{}'; (source / "sample.json").write_bytes(content)
        records = [{"file": "sample.json", "sha256": hashlib.sha256(content).hexdigest()}]
        manifest = {}
        result = dict(data=str(source), records=records)
        self.assertEqual(collector.import_records(result, output, manifest), 1)
        self.assertEqual(collector.import_records(result, output, manifest), 0)
        (source / "sample.json").write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, "checksum"):
            collector.import_records(result, output, manifest)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            collector.import_records(dict(data=str(source), records=[{"file":"../escape"}]), output, manifest)

    def test_timeout_is_bounded_and_supervisor_resumes_without_reimport(self):
        family = self.family("total-order", "A", {"domain.hddl":"(define (domain a))",
            "p01.hddl":"(define (problem p01))"})
        options = self.options()
        timeout_options = SimpleNamespace(**{**vars(options), "job_seconds":0.1})
        failed = collector.run_job(dict(id="timeout", kind="panda", family="A",
            domain=str(family / "domain.hddl"), problem="slow.hddl"), timeout_options)
        self.assertEqual(failed["returncode"], 124)
        self.assertFalse(failed["records"])
        collector.collect(options)
        state = json.loads((self.work / "progress.json").read_text())
        self.assertEqual(state["status"], "finished")
        self.assertEqual(len(state["completed"]), 2)
        self.assertEqual(state["dataset_records"], 1)
        self.assertTrue(all(r["audit_returncode"] == 0 for r in state["completed"].values()))
        collector.collect(options)
        resumed = json.loads((self.work / "progress.json").read_text())
        self.assertEqual(resumed["completed"], state["completed"])
        self.assertEqual(resumed["dataset_records"], 1)

    def test_failed_audit_never_publishes_and_dataset_lock_excludes_other_runs(self):
        options = self.options()
        options.worker.write_text(options.worker.read_text().replace("sys.exit(0)", "sys.exit(1)"))
        result = collector.run_job(dict(id="audit-fail", kind="panda", family="A",
            domain="domain.hddl", problem="problem.hddl"), options)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["audit_returncode"], 1)
        self.assertEqual(result["records"], [])
        options.output.mkdir()
        with (options.output / ".ipc2023-collector.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                collector.collect(options)

    def test_publication_recovers_a_crash_between_link_and_manifest(self):
        destination = self.root / "destination"; destination.mkdir()
        source = self.root / "source"; source.mkdir()
        content = b'{"committed":true}'
        for path in [source / "sample.json", destination / "sample.json"]:
            path.write_bytes(content)
        entry = {"file":"sample.json", "sha256":hashlib.sha256(content).hexdigest()}
        manifest = {}
        self.assertEqual(collector.import_records(dict(data=str(source),records=[entry]),destination,manifest),0)
        self.assertEqual(json.loads((destination / "manifest.json").read_text()),[entry])
        (destination / "sample.json").write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,"changed"):
            collector.import_records(dict(data=str(source),records=[entry]),destination,manifest)

    def test_retry_failed_preserves_successes_and_records_attempt_history(self):
        family = self.family("total-order","A",{"domain.hddl":"(define (domain a))",
            "p01.hddl":"(define (problem p01))", "p02-fail.hddl":"(define (problem p02))"})
        options = self.options()
        collector.collect(options)
        before=json.loads((self.work / "progress.json").read_text())
        failed=[k for k,r in before["completed"].items() if collector.unsuccessful(r)]
        self.assertEqual(len(failed),1)
        options.worker.write_text(options.worker.read_text().replace("if 'fail' in problem: sys.exit(2)","# fixed worker"))
        options.retry_failed=True
        collector.collect(options)
        after=json.loads((self.work / "progress.json").read_text())
        self.assertEqual(after["scheduled_this_run"],1)
        self.assertEqual(after["completed_this_run"],1)
        self.assertEqual(after["completed"][failed[0]]["returncode"],0)
        self.assertEqual(after["history"][failed[0]][0]["returncode"],2)
        for key in set(before["completed"]) - set(failed):
            self.assertEqual(after["completed"][key],before["completed"][key])
        # An interrupted retry must resume even though an older failed attempt
        # still has a completed entry and --retry-failed is no longer supplied.
        after["completed"][failed[0]]["returncode"]=2
        after["interrupted"]={failed[0]:{"interrupted":True}}
        (self.work / "progress.json").write_text(json.dumps(after))
        options.retry_failed=False
        collector.collect(options)
        resumed=json.loads((self.work / "progress.json").read_text())
        self.assertEqual(resumed["scheduled_this_run"],1)
        self.assertFalse(resumed["interrupted"])

    def test_supervision_stops_nested_processes_on_timeout_or_request(self):
        for mode in ["timeout", "stop"]:
            sentinel=self.root / (mode+".sentinel")
            script=self.root / (mode+".py")
            script.write_text("import subprocess, sys, time\nsubprocess.Popen([sys.executable,'-c',"
                +repr("import pathlib,time; time.sleep(0.4); pathlib.Path("+repr(str(sentinel))+").touch()")
                +"])\ntime.sleep(30)\n")
            stop=threading.Event()
            timer=None
            if mode=="stop":
                timer=threading.Timer(0.1,stop.set); timer.start()
            command=["timeout","--foreground","--kill-after=0.1s","0.1s" if mode=="timeout" else "30s","python3",str(script)]
            with (self.root / (mode+".log")).open("wb") as log:
                code=collector.supervised(command,log,stop)
            if timer:timer.join()
            self.assertNotEqual(code,0)
            time.sleep(0.45)
            self.assertFalse(sentinel.exists(),"nested process escaped supervision")

    def test_detached_launch_reports_a_lock_failure(self):
        self.family("total-order","A",{"domain.hddl":"(define (domain a))","p01.hddl":"(define (problem p01))"})
        options=self.options()
        script=Path(collector.__file__).resolve()
        launcher=self.work / "launcher.json"
        launcher.write_text('{"pid":12345}')
        with (self.work / ".collector.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            result=subprocess.run(["python3",str(script),"--detach","--corpus",str(self.corpus),"--work",str(self.work),
                "--worker",str(options.worker),"--output",str(options.output)],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertIn("failed to start",result.stderr)
        self.assertEqual(json.loads(launcher.read_text()),{"pid":12345})

    def test_term_checkpoints_active_jobs_for_automatic_resume(self):
        self.family("total-order","A",{"domain.hddl":"(define (domain a))","p01-slow.hddl":"(define (problem p01))"})
        options=self.options()
        with (self.root / "supervisor.log").open("wb") as log:
            process=subprocess.Popen(["python3",str(Path(collector.__file__).resolve()),"--corpus",str(self.corpus),
                "--work",str(self.work),"--output",str(options.output),"--worker",str(options.worker),
                "--synthetic-per-domain",str(options.synthetic_per_domain),"--search-seconds",str(options.search_seconds)],stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+5
                while not (self.work / "progress.json").exists():
                    if time.monotonic()>deadline:self.fail("collector did not become ready")
                    time.sleep(0.02)
                process.terminate()
                self.assertEqual(process.wait(timeout=5),0)
            finally:
                if process.poll() is None:
                    process.kill();process.wait()
        state=json.loads((self.work / "progress.json").read_text())
        self.assertEqual(state["status"],"stopped")
        self.assertFalse(state["completed"])
        self.assertTrue(state["interrupted"])
        options.worker.write_text(options.worker.read_text().replace("if 'slow' in problem: time.sleep(30)","# fixed"))
        collector.collect(options)
        resumed=json.loads((self.work / "progress.json").read_text())
        self.assertEqual(resumed["status"],"finished")
        self.assertFalse(resumed["interrupted"])


if __name__ == "__main__":
    unittest.main()
