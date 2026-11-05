#!/usr/bin/env python3
"""Generate certified problems on reserved CPUs while GPU RL consumes commits."""
import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from online_data import OnlinePool, digest

STOP = False


def stop_signal(signum, frame):
    global STOP
    STOP = True


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def terminate(process, group=True):
    if process.poll() is None:
        try:
            if group:
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if group:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        except ProcessLookupError:
            pass


def catalog(args, parent):
    """Prefer small grounding templates; rotate several instances per domain."""
    store = args.run_dir / "data"
    index = json.loads((store / "index.json").read_text())
    groups = collections.defaultdict(list)
    for entry in index["entries"]:
        if entry["metadata"]["kind"] != "Problem" or entry["metadata"]["group_id"] not in parent["training_groups"]:
            continue
        identity = entry["id"]
        path = store / "objects" / identity[:2] / (identity + ".json")
        if digest(path) != identity:
            raise ValueError("source template checksum mismatch")
        program = json.loads(path.read_bytes())["payload"]["program"]
        problem = program.get("problem") or {}
        roots = len(((problem.get("init_tn") or {}).get("tn") or {}).get("subtasks", []))
        if roots >= 2:
            groups[entry["metadata"]["group_id"]].append((len(problem.get("objects", [])), roots, identity))
    if not groups:
        raise ValueError("no Train-domain generation templates available")
    return {g: [row[2] for row in sorted(rows)[:args.templates_per_domain]] for g, rows in sorted(groups.items())}


def plan(args, templates):
    jobs = []
    for round_index in range(args.rounds):
        for group, choices in templates.items():
            template = choices[round_index % len(choices)]
            seed = int(hashlib.sha256(f"{args.seed}:{group}:{round_index}".encode()).hexdigest()[:16], 16)
            identity = hashlib.sha256(f"{template}:{seed}:{args.samples_per_job}:{args.max_roots}".encode()).hexdigest()
            jobs.append(dict(id=identity, template=template, seed=seed, group=group,
                             roots=min(args.max_roots, (4, 8, 12)[round_index % 3])))
    return jobs


def directory_bytes(root):
    total=0
    for path in root.rglob("*"):
        if path.name.startswith(".") or path.suffix==".tmp":
            continue
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:
            pass  # A worker may have just atomically renamed a temporary file.
    return total


def collect(args, parent):
    pool = OnlinePool(args.online_dir, parent, digest(args.parent_models / "merge-policy.json"))
    jobs = plan(args, catalog(args, parent))
    signature = hashlib.sha256(json.dumps(dict(jobs=jobs, parent=pool.parent_sha256,
        native=digest(args.native), max_depth=args.max_depth,
        max_ground_instances=args.max_ground_instances, samples=args.samples_per_job),sort_keys=True).encode()).hexdigest()
    state_path = args.online_dir / "generation-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(signature=signature,jobs={},started=time.time())
    if state["signature"] != signature:
        raise ValueError("generation plan/source changed; use a new --online-dir")
    state.update(complete=False, stopped=None, planned_jobs=len(jobs), requested_problems=len(jobs)*args.samples_per_job)
    waiting = collections.deque(j for j in jobs if j["id"] not in state["jobs"] or state["jobs"][j["id"]]["status"] != "complete")
    running = {}
    started = time.monotonic()
    next_disk_check = 0
    reason = None
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (args.worker_memory_mb*1024**2,)*2)
        os.nice(10)
    try:
        while waiting or running:
            if STOP:
                reason = "signal"
                break
            if time.monotonic()-started >= args.generation_seconds:
                reason = "time_limit"
                break
            if time.monotonic() >= next_disk_check:
                size = directory_bytes(args.online_dir)
                state["bytes"] = size
                next_disk_check = time.monotonic()+30
                if size >= args.max_gb*1024**3:
                    reason = "disk_limit"
                    break
            while waiting and len(running) < args.workers:
                job = waiting.popleft()
                path = args.online_dir / "logs" / (job["id"]+".log")
                log = path.open("w")
                command = [str(args.native),"generate-online","--store",str(args.run_dir/"data"),
                    "--checkpoint",str(args.parent_models/"merge-policy.json"),"--output",str(args.online_dir),
                    "--template-id",job["template"],"--seed",str(job["seed"]),"--count",str(args.samples_per_job),
                    "--max-roots",str(job["roots"]),"--max-depth",str(args.max_depth),
                    "--max-ground-instances",str(args.max_ground_instances)]
                child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                         preexec_fn=limits)
                running[child.pid] = (child, job, log, time.monotonic())
            for pid,(child,job,log,begin) in list(running.items()):
                timeout = time.monotonic()-begin > args.job_seconds
                if timeout:
                    terminate(child,group=False)
                code = child.poll()
                if code is None:
                    continue
                log.close()
                state["jobs"][job["id"]] = dict(status="complete" if code==0 else "failed",
                    returncode=code,timed_out=timeout,group=job["group"],seconds=time.monotonic()-begin)
                del running[pid]
                references = pool.refresh()
                state.update(committed_problems=len(references),committed_batches=len(pool.manifests))
                save(state_path,state)
                print(f"generation {len(state['jobs'])}/{len(jobs)} jobs: {len(references)} unique certified problems",flush=True)
            time.sleep(0.2)
        references = pool.refresh()
        state.update(complete=not waiting and not running,stopped=reason,committed_problems=len(references),
                     committed_batches=len(pool.manifests),updated=time.time(),bytes=directory_bytes(args.online_dir),
                     completed_jobs=sum(v["status"]=="complete" for v in state["jobs"].values()),
                     failed_jobs=sum(v["status"]!="complete" for v in state["jobs"].values()))
        save(state_path,state)
    finally:
        for child,job,log,begin in running.values():
            terminate(child,group=False)
            log.close()
    return 0


def available_memory():
    memory=os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    meminfo=Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):memory=int(line.split()[1])*1024
    for limit_path,current_path in (("/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes","/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            limit=Path(limit_path).read_text().strip()
            if limit!="max":memory=min(memory,max(0,int(limit)-int(Path(current_path).read_text())))
        except (FileNotFoundError,ValueError,PermissionError):pass
    return memory


def cpu_layout(reserve, workers, worker_memory_mb=2048):
    available = sorted(os.sched_getaffinity(0))
    reserve = reserve or min(8,max(1,len(available)//3))
    training = available[:min(reserve,len(available))]
    generation = available[len(training):] or available
    count = workers or max(1,min(16,len(generation),int(available_memory()*0.6)//(worker_memory_mb*1024**2)))
    return training,generation,count


def supervise(args, parent):
    # Fail before starting CPU collection if the selected Torch device is absent.
    probe = "import torch; assert " + ("torch.cuda.is_available(), 'CUDA unavailable'" if args.device=="cuda" else "True")
    subprocess.run([sys.executable,"-c",probe],check=True)
    train_cpus,generation_cpus,args.workers = cpu_layout(args.reserve_cpus,args.workers,args.worker_memory_mb)
    generation_log = (args.online_dir/"generation.log").open("a")
    training_log = (args.online_dir/"training.log").open("a")
    collector = trainer = None
    try:
        # Apply affinity in the new Python process, before it launches native workers.
        command = [sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],"--collect-only",
                   "--workers",str(args.workers),"--affinity",",".join(map(str,generation_cpus))]
        collector = subprocess.Popen(command,stdout=generation_log,stderr=subprocess.STDOUT,start_new_session=True)
        pool = OnlinePool(args.online_dir,parent,digest(args.parent_models/"merge-policy.json"))
        begin = time.monotonic()
        while len(pool.refresh()) < args.warmup_problems and time.monotonic()-begin < args.warmup_seconds:
            if STOP or collector.poll() is not None:
                break
            time.sleep(1)
        if STOP:
            return 130
        if not pool.samples:
            raise RuntimeError("no certified online problems committed; inspect generation.log")
        command = [str(args.native),"rl","--run-dir",str(args.run_dir),"--parent-models",str(args.parent_models),
                   "--model-dir",str(args.model_dir),"--python",sys.executable,"--device",args.device,
                   "--online-dir",str(args.online_dir),"--online-fraction",str(args.online_fraction)]
        for name in ("epochs","episodes_per_epoch","batch_size","samples","train_problems","validation_problems",
                     "patience","total_ms","stage_ms","local_ms","local_calls","latent_epochs"):
            command += ["--"+name.replace("_","-"),str(getattr(args,name))]
        trainer = subprocess.Popen(command,stdout=training_log,stderr=subprocess.STDOUT,
                                   start_new_session=True,preexec_fn=lambda:os.sched_setaffinity(0,train_cpus))
        print(f"Training CPUs {train_cpus}; generation CPUs {generation_cpus}, {args.workers} workers; {len(pool.samples)} warmup problems",flush=True)
        next_update = 0
        while trainer.poll() is None or collector.poll() is None:
            if STOP:
                return 130
            if trainer.poll() not in (None,0):
                raise RuntimeError("training failed; inspect training.log")
            if collector.poll() not in (None,0):
                raise RuntimeError("collector failed; inspect generation.log")
            if trainer.poll()==0 and args.stop_generation_with_training:
                terminate(collector)
                break
            if time.monotonic()>=next_update:
                path=args.online_dir/"generation-state.json"
                state=json.loads(path.read_text()) if path.exists() else {}
                print(f"certified problems={state.get('committed_problems',len(pool.samples))}; training={'running' if trainer.poll() is None else 'complete'}; generation={'running' if collector.poll() is None else 'complete'}",flush=True)
                next_update=time.monotonic()+10
            time.sleep(0.5)
        print(f"Training complete: {args.model_dir}; generation state: {args.online_dir/'generation-state.json'}",flush=True)
        return 0
    finally:
        for child in (trainer,collector):
            if child:
                terminate(child)
        generation_log.close()
        training_log.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,default=Path("runs/hilg-wl"))
    p.add_argument("--parent-models",type=Path)
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--online-dir",type=Path)
    p.add_argument("--native",type=Path,default=Path("target/release/train"))
    p.add_argument("--device",choices=("cpu","cuda","auto"),default="cuda")
    p.add_argument("--collect-only",action="store_true")
    p.add_argument("--affinity",help=argparse.SUPPRESS)
    p.add_argument("--stop-generation-with-training",action="store_true")
    for name,default in (("workers",0),("reserve-cpus",0),("rounds",16),("samples-per-job",64),
        ("templates-per-domain",3),("max-roots",12),("max-depth",8),("max-ground-instances",200000),
        ("worker-memory-mb",2048),("job-seconds",180),("generation-seconds",7200),("seed",20261108),
        ("warmup-problems",256),("warmup-seconds",120),("epochs",80),("episodes-per-epoch",128),
        ("batch-size",8),("samples",2),("train-problems",1024),("validation-problems",64),
        ("patience",10),("total-ms",1000),("stage-ms",350),("local-ms",100),("local-calls",64),("latent-epochs",50)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--online-fraction",type=float,default=0.75)
    p.add_argument("--max-gb",type=float,default=20)
    args=p.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,int) and not isinstance(value,bool) and value<(0 if name in ("workers","reserve_cpus","seed") else 1):
            p.error(f"invalid {name}")
    if not 0<=args.online_fraction<=1 or not math.isfinite(args.max_gb) or args.max_gb<=0 or not 2<=args.max_roots<=32 or args.samples_per_job>1024 or args.batch_size*args.samples>128 or args.workers>64:
        p.error("invalid data/worker/batch limits")
    args.run_dir=args.run_dir.resolve()
    args.native=args.native.resolve()
    args.parent_models=(args.parent_models or args.run_dir/"models").resolve()
    args.model_dir=(args.model_dir or args.run_dir/"online-rl-models").resolve()
    args.online_dir=(args.online_dir or args.run_dir/"online").resolve()
    if args.online_dir==args.run_dir or args.online_dir.is_relative_to(args.run_dir/"data") or args.model_dir==args.parent_models:
        p.error("generation and output paths must preserve the immutable parent")
    args.online_dir.mkdir(parents=True,exist_ok=True)
    (args.online_dir/"logs").mkdir(exist_ok=True)
    if args.affinity:
        os.sched_setaffinity(0,[int(v) for v in args.affinity.split(",")])
    if args.collect_only and not args.workers:
        args.workers=cpu_layout(args.reserve_cpus,0,args.worker_memory_mb)[2]
    parent=json.loads((args.parent_models/"merge-policy.json").read_text())
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop_signal)
    lock_name="collector.lock" if args.collect_only else "supervisor.lock"
    with (args.online_dir/lock_name).open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return collect(args,parent) if args.collect_only else supervise(args,parent)


if __name__=="__main__":
    raise SystemExit(main())
