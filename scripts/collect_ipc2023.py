#!/usr/bin/env python3
"""Resumable, bounded background collection of the official IPC 2023 corpus."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as output:
        output.write(json.dumps(value, indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def supervised(command, log, stop=None):
    """One process group per job; nested timeouts must use --foreground."""
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while True:
        try:
            code = process.wait(timeout=0.1)
            # --foreground timeouts signal the direct child. Reap any nested
            # stage processes in the same owned group after the wrapper exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return code
        except subprocess.TimeoutExpired:
            if stop is None or not stop.is_set():
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.returncode


def discover(corpus, work):
    """Only competition-level files; exclude `other` and unused generators."""
    families = []
    errors = []
    for track in ("total-order", "partial-order"):
        for family in sorted((corpus / track).iterdir()):
            if not family.is_dir():
                continue
            files = sorted(p for p in family.iterdir() if p.suffix.lower() in (".hddl", ".pddl"))
            problems = []
            for path in files:
                text = re.sub(r";[^\n]*", "", path.read_text())
                if re.search(r"\(define\s*\(problem\s", text, re.I):
                    problems.append(path)
            if family.name == "SharpSAT":
                generated = work / "generated-sharpsat"
                generated.mkdir(parents=True, exist_ok=True)
                for cnf in sorted((family / "cnfs").glob("*.cnf")):
                    target = generated / ("p-" + cnf.stem + ".hddl")
                    if not target.exists():
                        temporary = target.with_suffix(".tmp")
                        with temporary.open("wb") as output:
                            result = subprocess.run(["timeout", "30s", "python3", str(family / "cnf2hddl.py"), str(cnf)], stdout=output, stderr=subprocess.PIPE)
                        if result.returncode:
                            temporary.unlink(missing_ok=True)
                            errors.append(f"SharpSAT encoder failed: {cnf.name}: {result.stderr.decode(errors='replace')}")
                            continue
                        temporary.replace(target)
                    problems.append(target)
            pairs = []
            for problem in problems:
                if family.name == "Ultralight-Cockpit":
                    domain = family / "UL_domain-type-fix.hddl"
                elif (family / "domain.hddl").exists():
                    domain = family / "domain.hddl"
                else:
                    domain = family / (problem.stem + "-domain.hddl")
                if not domain.exists():
                    errors.append(f"No paired domain for {problem}")
                    continue
                pairs.append((domain, problem))
            if not pairs:
                errors.append(f"No competition problems found for {track}/{family.name}")
            families.append((f"{track}/{family.name}", pairs))
    return families, errors


def jobs_for(families, options):
    jobs = []
    seen_domains = set()
    # Round-robin across families before moving on to their larger instances.
    for index in range(max((len(pairs) for _, pairs in families), default=0)):
        for family, pairs in families:
            if index >= len(pairs):
                continue
            domain, problem = pairs[index]
            kinds = ["panda"]
            key = str(domain)
            if key not in seen_domains:
                seen_domains.add(key)
                kinds.insert(0, "backward")
            for kind in kinds:
                identity = [kind, str(domain), sha(domain), str(problem), sha(problem), options.seed, options.synthetic_per_domain, options.search_seconds]
                job_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
                jobs.append(dict(id=job_id, kind=kind, family=family, domain=str(domain), problem=str(problem)))
    return jobs


def run_job(job, options, stop=None):
    work = options.work / "jobs" / job["id"]
    work.mkdir(parents=True, exist_ok=True)
    output = work / "data"
    args = [str(options.worker), "--domain-file", job["domain"], "--problem", job["problem"], "--per-domain", "1",
            "--output", str(output), "--panda-root", str(options.panda_root), "--work", str(work / "panda"),
            "--seed", str(options.seed), "--max-ground-instances", str(options.max_ground_instances),
            "--search-seconds", str(options.search_seconds), "--stage-seconds", "30", "--memory-bytes", str(options.memory_bytes),
            "--max-attempts", "4000"]
    if job["kind"] == "backward":
        args += ["--synthetic-count", str(options.synthetic_per_domain), "--real-limit", "0"]
    else:
        args += ["--synthetic-count", "0", "--real-limit", "1"]
    command = ["nice", "-n", "10", "timeout", "--foreground", "--signal=TERM", "--kill-after=3s", f"{options.job_seconds}s",
               "prlimit", f"--as={options.memory_bytes}", "--"] + args
    started = time.time()
    with (work / "worker.log").open("wb") as log:
        returncode = supervised(command, log, stop)
    tail = (work / "worker.log").read_text(errors="replace")[-3000:]
    manifest = output / "manifest.json"
    entries = []
    audit_returncode = None
    interrupted = stop is not None and stop.is_set()
    # A killed worker can have committed a sample before creating its manifest.
    if not interrupted and (manifest.exists() or any(output.glob("backward-*.json")) or any(output.glob("panda-optimal-*.json"))):
        # Even an interrupted worker may have committed records. Audit them
        # before import, never infer validity from a file's presence alone.
        with (work / "audit.log").open("wb") as log:
            audit_returncode = supervised(["timeout", "--foreground", "--kill-after=3s", "60s", "prlimit", f"--as={options.memory_bytes}", "--", str(options.worker),
                                    "--output", str(output), "--max-ground-instances", str(options.max_ground_instances), "--audit-only"], log, stop)
        if audit_returncode == 0:
            entries = json.loads(manifest.read_text())
    interrupted = stop is not None and stop.is_set()
    report = output / "collection-report.json"
    failures = json.loads(report.read_text()).get("failures", []) if report.exists() else []
    return dict(job=job, returncode=returncode, interrupted=interrupted, seconds=round(time.time() - started, 3),
                records=entries, verified_records=len(entries), audit_returncode=audit_returncode,
                failures=failures, data=str(output), log=str(work / "worker.log"), tail=tail)


def import_records(result, destination, manifest):
    added = 0
    for entry in result["records"]:
        name = entry["file"]
        if Path(name).name != name or not name.endswith(".json") or name.startswith(".") or name in ("manifest.json", "collection-report.json", "ipc2023-progress.json"):
            raise ValueError("unsafe worker filename")
        source = Path(result["data"]) / name
        if sha(source) != entry["sha256"]:
            raise ValueError(f"worker checksum mismatch: {source}")
        target = destination / name
        if source.is_symlink() or target.is_symlink():
            raise ValueError("record paths must not be symlinks")
        if target.exists():
            # Same semantic ID can have a different cosmetic problem name or
            # provenance. Preserve the first immutable validated record.
            if name not in manifest and sha(target) == entry["sha256"]:
                # Recover a crash after linking the immutable record but before
                # committing the manifest, using this newly audited source.
                manifest[name] = entry
            if name not in manifest or sha(target) != manifest[name]["sha256"]:
                raise ValueError(f"untracked or changed existing record: {target}")
            continue
        if name in manifest and manifest[name]["sha256"] != entry["sha256"]:
            raise ValueError(f"cannot restore missing record with different content: {target}")
        temporary = destination / ("." + name + f".{os.getpid()}.tmp")
        shutil.copyfile(source, temporary)
        with temporary.open("rb") as copied:
            os.fsync(copied.fileno())
        try:
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        manifest[name] = entry
        added += 1
    atomic_json(destination / "manifest.json", [manifest[name] for name in sorted(manifest)])
    return added


def collect(options):
    options.work.mkdir(parents=True, exist_ok=True)
    with (options.work / ".collector.lock").open("a") as work_lock:
        fcntl.flock(work_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if options.inventory_only:
            return inventory_and_collect(options)
        options.output.mkdir(parents=True, exist_ok=True)
        with (options.output / ".ipc2023-collector.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inventory_and_collect(options)


def inventory_and_collect(options):
    families, errors = discover(options.corpus, options.work)
    jobs = jobs_for(families, options)
    inventory = dict(corpus=str(options.corpus), families={family: len(pairs) for family, pairs in families},
                     jobs=len(jobs), panda_jobs=sum(job["kind"] == "panda" for job in jobs),
                     backward_jobs=sum(job["kind"] == "backward" for job in jobs), discovery_errors=errors)
    revision = subprocess.run(["git", "-C", str(options.corpus), "rev-parse", "HEAD"], capture_output=True, text=True)
    inventory["corpus_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    atomic_json(options.work / "inventory.json", inventory)
    print(json.dumps(inventory), flush=True)
    if options.inventory_only:
        return
    collect_locked(options, inventory, jobs)


def unsuccessful(result):
    return bool(result.get("error") or result.get("returncode") or result.get("failures")
                or result.get("verified_records") == 0
                or (result.get("audit_returncode") is not None and result["audit_returncode"] != 0))


def collect_locked(options, inventory, jobs):
    progress_path = options.work / "progress.json"
    state = json.loads(progress_path.read_text()) if progress_path.exists() else {"completed": {}}
    if "configuration" in state:
        previous_run = {"configuration": state["configuration"], "worker_sha256": state["worker_sha256"]}
        key = hashlib.sha256(json.dumps(previous_run, sort_keys=True).encode()).hexdigest()
        state.setdefault("runs", {})[key] = previous_run
        for result in state["completed"].values():
            result.setdefault("configuration_sha256", key)
    state.update(pid=os.getpid(), status="running", inventory=inventory, worker_sha256=sha(options.worker),
                 completed_this_run=0,
                 configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()})
    run = {"configuration": state["configuration"], "worker_sha256": state["worker_sha256"]}
    run_key = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    state.setdefault("runs", {})[run_key] = run
    previous = json.loads((options.output / "manifest.json").read_text()) if (options.output / "manifest.json").exists() else []
    manifest = {entry["file"]: entry for entry in previous}
    queue = [job for job in jobs if job["id"] not in state["completed"]
             or job["id"] in state.get("interrupted", {})
             or (getattr(options, "retry_failed", False) and unsuccessful(state["completed"][job["id"]]))]
    state["scheduled_this_run"] = len(queue)
    futures = {}
    stop = threading.Event()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
    def checkpoint():
        state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["active"] = list(futures.values())
        state["pending"] = len(queue)
        state["dataset_records"] = len(manifest)
        atomic_json(progress_path, state)
        atomic_json(options.output / "ipc2023-progress.json", state)
    try:
        with ThreadPoolExecutor(max_workers=options.workers) as executor:
            while (queue and not stop.is_set()) or futures:
                while queue and not stop.is_set() and len(futures) < options.workers:
                    job = queue.pop(0)
                    futures[executor.submit(run_job, job, options, stop)] = job
                checkpoint()
                finished, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = futures.pop(future)
                    try:
                        result = future.result()
                        result["added"] = import_records(result, options.output, manifest)
                        result.pop("records")
                    except Exception as error:
                        result = dict(job=job, error=str(error), added=0)
                    result["configuration_sha256"] = run_key
                    if result.get("interrupted"):
                        state.setdefault("interrupted", {})[job["id"]] = result
                    else:
                        previous = state["completed"].get(job["id"])
                        if previous is not None:
                            state.setdefault("history", {}).setdefault(job["id"], []).append(previous)
                        state["completed"][job["id"]] = result
                        state.get("interrupted", {}).pop(job["id"], None)
                        state["completed_this_run"] += 1
                    print(json.dumps(result), flush=True)
    finally:
        for sig, previous in old_handlers.items():
            signal.signal(sig, previous)
    state["status"] = "stopped" if stop.is_set() else "finished"
    checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("target/ipc2023-domains"))
    parser.add_argument("--output", type=Path, default=Path("../train"))
    parser.add_argument("--work", type=Path, default=Path("target/ipc2023-collection"))
    parser.add_argument("--worker", type=Path, default=Path("target/debug/generate-training"))
    parser.add_argument("--panda-root", type=Path, default=Path("target/panda"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--synthetic-per-domain", type=int, default=64)
    parser.add_argument("--search-seconds", type=int, default=60)
    parser.add_argument("--job-seconds", type=int, default=180)
    parser.add_argument("--memory-bytes", type=int, default=1_073_741_824)
    parser.add_argument("--max-ground-instances", type=int, default=200_000)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed, partial, or audit-rejected jobs while retaining completed successes")
    options = parser.parse_args()
    for field in ("corpus", "output", "work", "worker", "panda_root"):
        setattr(options, field, getattr(options, field).resolve())
    if min(options.workers, options.job_seconds, options.search_seconds, options.memory_bytes,
           options.max_ground_instances, options.synthetic_per_domain) <= 0:
        parser.error("worker, time, memory, grounding, and sample limits must be positive")
    if options.detach:
        options.work.mkdir(parents=True, exist_ok=True)
        # Snapshot the executable so later Rust builds cannot change this run.
        snapshot = options.work / ("generate-training-" + sha(options.worker)[:16])
        if not snapshot.exists():
            shutil.copy2(options.worker, snapshot)
        supervisor = options.work / ("collector-" + sha(Path(__file__))[:16] + ".py")
        if not supervisor.exists():
            shutil.copy2(Path(__file__), supervisor)
        command = ["python3", str(supervisor)]
        for key, value in vars(options).items():
            if key in ("detach", "inventory_only"):
                continue
            if isinstance(value, bool):
                if value:
                    command += ["--" + key.replace("_", "-")]
                continue
            command += ["--" + key.replace("_", "-"), str(snapshot if key == "worker" else value)]
        with (options.work / "collector.log").open("ab", buffering=0) as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        deadline = time.monotonic() + 10
        ready = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                if process.returncode != 0:
                    parser.exit(1, f"Collector failed to start (exit {process.returncode}); see {options.work / 'collector.log'}\n")
            progress = options.work / "progress.json"
            if progress.exists() and json.loads(progress.read_text()).get("pid") == process.pid:
                ready = True
                break
            time.sleep(0.05)
        atomic_json(options.work / "launcher.json", dict(pid=process.pid, command=command, log=str(options.work / "collector.log")))
        print(f"Collector PID {process.pid} ({'ready' if ready else 'starting'}); progress: {options.work / 'progress.json'}")
        return
    collect(options)


if __name__ == "__main__":
    main()
