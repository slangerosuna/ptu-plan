#!/usr/bin/env python3
"""On-policy self-critical REINFORCE through replay-verified planner outcomes."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import torch
from torch.nn import functional as F
import train_all as sl
from online_data import OnlinePool, draw_problems


class Environment:
    def __init__(self, args):
        command = [str(args.native.resolve()), "rl-environment", "--store", str(args.run_dir / "data"),
                   "--checkpoint", str(args.parent_models / "merge-policy.json")]
        interface = args.parent_models / "interface-policy.json"
        if interface.exists():
            command += ["--interface", str(interface)]
        for name in ("train_problems", "validation_problems", "total_ms", "stage_ms", "local_ms", "local_calls"):
            command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ready = self.read()

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"native RL environment exited ({self.process.poll()}); see its stderr")
        return json.loads(line)

    def collect(self, model, split, indices, seed, sample):
        fresh = [i if isinstance(i, dict) else None for i in indices]
        if any(fresh) and (split != "Train" or self.pool is None):
            raise ValueError("online inputs are restricted to Train")
        payload = dict(model=model, split=split,
                       indices=[0 if f else i for i, f in zip(indices, fresh)], seed=seed, sample=sample)
        if any(fresh):
            cache = {f["id"]: self.pool.read(f) for f in fresh if f}
            payload["online"] = [cache[f["id"]] if f else None for f in fresh]
        self.process.stdin.write(json.dumps(payload, allow_nan=False)+"\n")
        self.process.stdin.flush()
        episodes = self.read()
        if len(episodes) != len(indices):
            raise ValueError("incomplete planner rollout batch")
        sources = self.ready["train" if split == "Train" else "validation"]
        for index, reference, episode in zip(indices, fresh, episodes):
            source = reference or sources[index]
            if episode["problem_id"] != source["id"] or episode["group"] != source["group"] or episode["split"] != split or episode["sampled"] != sample or episode.get("online", False) != bool(reference):
                raise ValueError("planner rollout provenance mismatch")
            if episode["solved"] != episode["replay_verified"] or (episode["solved"] and episode["certificate"] is None):
                raise ValueError("planner success lacks replay evidence")
        return episodes

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        try:
            code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            return
        if code:
            raise RuntimeError(f"planner environment failed: {code}")


def model_json(model, template):
    result = copy.deepcopy(template)
    result["parameters"] = model.theta.detach().cpu().tolist()
    return result


def policy_loss(actor, anchor, episodes, baselines, value_weight, entropy_weight, kl_weight):
    """Sum decision log probabilities per episode: terminal-return REINFORCE.

    The greedy same-instance baseline is detached and action-independent. Work
    regression is censored on failed searches, never an exact infeasibility label.
    """
    observations = [d["observation"] for e in episodes for d in e["decisions"]]
    if not observations:
        raise ValueError("RL batch contains no policy decisions")
    logits, values, offsets = actor(observations)
    with torch.no_grad():
        anchor_logits, anchor_values, _ = anchor(observations)
    total = actor.theta.sum()*0
    position = 0
    metrics = dict(policy=0.0, entropy=0.0, kl=0.0, value=0.0)
    for episode, baseline in zip(episodes, baselines):
        if episode["split"] != "Train" or not episode["sampled"]:
            raise ValueError("only sampled Train trajectories may update the actor")
        if episode["problem_id"] != baseline["problem_id"] or baseline["sampled"]:
            raise ValueError("self-critical baseline does not match its sampled problem")
        advantage = episode["reward"]-baseline["reward"]
        log_probability = actor.theta.sum()*0
        auxiliary = actor.theta.sum()*0
        count = len(episode["decisions"])
        for decision, behavior in zip(episode["decisions"], episode["probabilities"]):
            start, stop = offsets[position:position+2]
            score = logits[start:stop]-value_weight*values[start:stop]
            logp = score.log_softmax(0)
            p = logp.exp()
            if not torch.allclose(p.detach(), actor.tensor(behavior), atol=1e-4, rtol=1e-4):
                raise ValueError("on-policy behavior/native Torch probability mismatch")
            with torch.no_grad():
                logq = (anchor_logits[start:stop]-value_weight*anchor_values[start:stop]).log_softmax(0)
            action = decision["action"]
            log_probability = log_probability+logp[action]
            entropy = -(p*logp).sum()
            kl = (p*(logp-logq)).sum()
            residual = values[start+action]-episode["log_cost"]
            if not episode["solved"]:
                residual = residual.clamp_max(0)
            value_loss = F.huber_loss(residual, residual.new_zeros(()))
            auxiliary = auxiliary+(-entropy_weight*entropy+kl_weight*kl+0.1*value_loss)/max(1,count)
            for name, term in (("entropy", entropy), ("kl", kl), ("value", value_loss)):
                metrics[name] += float(term.detach())/len(observations)
            position += 1
        policy = -advantage*log_probability
        total = total+(policy+auxiliary)/len(episodes)
        metrics["policy"] += float(policy.detach())/len(episodes)
    if position != len(observations):
        raise ValueError("trajectory/action counts do not align")
    return total, metrics


def evaluate(env, actor, template, seed):
    payload = model_json(actor, template)
    episodes = []
    for start in range(0, len(env.ready["validation"]), 8):
        episodes += env.collect(payload, "Validation", list(range(start, min(start+8, len(env.ready["validation"])))), seed, False)
    groups = {}
    for episode in episodes:
        groups.setdefault(episode["group"], []).append(episode)
    score = tuple(sum(sum(e[key] for e in g)/len(g) for g in groups.values())/len(groups) for key in ("solved", "reward"))
    return score, dict(domain_solve_rate=score[0], domain_reward=score[1], solved=sum(e["solved"] for e in episodes), problems=len(episodes), mean_reward=sum(e["reward"] for e in episodes)/len(episodes))


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def refresh_bundle(args, actor, checkpoint, device, signature):
    parent_report = sl.load(args.parent_models/"training-report.json")
    package = args.run_dir/"prepared"/parent_report["package"]
    manifest = sl.load(package/"manifest.json")
    if manifest["snapshot_id"] != checkpoint["snapshot_id"]:
        raise ValueError("latent refresh package snapshot differs from RL source")
    files = dict(zip(("merge-template.json", "interface-template.json", "latent-template.json"), manifest["templates_sha256"]))
    files.update({"records.jsonl": manifest["records_sha256"], "references.json": manifest["references_sha256"]})
    for name, expected in files.items():
        if sl.digest(package/name) != expected:
            raise ValueError(f"latent refresh package checksum mismatch: {name}")
    with (package/"records.jsonl").open() as handle:
        records = [json.loads(line) for line in handle]
    latent_template = sl.load(package/"latent-template.json")
    latent = sl.Latent(latent_template["model"], device=device)
    actor.graph_cache.clear()
    transitions = sl.prepare_latent(actor, records, 64)
    actor.graph_cache.clear()
    fit = copy.copy(args)
    fit.epochs = args.latent_epochs
    fit.batch_size = 64
    fit.learning_rate = 0.001
    fit.patience = 12
    encoder_signature = hashlib.sha256(json.dumps(checkpoint["model"], sort_keys=True).encode()).hexdigest()
    # An extended RL run may select a new encoder. Its latent targets and
    # optimizer state belong to that encoder, while the actor resumes normally.
    fit.model_dir = args.model_dir/"latent-fit"/encoder_signature
    fit.model_dir.mkdir(parents=True, exist_ok=True)
    latent_checkpoint, report = sl.train_phase("latent", latent, transitions, latent_template, fit, signature+encoder_signature)
    latent_checkpoint["encoder"] = checkpoint["model"]
    sl.save(args.model_dir/"latent-policy.json", latent_checkpoint)
    interface_checkpoint = sl.load(args.parent_models/"interface-policy.json")
    sl.save(args.model_dir/"interface-policy.json", interface_checkpoint)
    interface = sl.Interface(interface_checkpoint["model"], device=device)
    parity = []
    with torch.no_grad():
        for reference in sl.load(package/"references.json"):
            r = reference["record"]
            logits, values, _ = actor([r["observation"]])
            check = dict(record=r["id"], prediction=dict(logits=logits.cpu().tolist(), log_effort=values.cpu().tolist()))
            if r["interfaces"]:
                rows = interface.tensor([a["features"] for a in r["interfaces"][0]["input"]["atoms"]])
                check["interface"] = interface(rows).softmax(-1).cpu().tolist()
            if r["transitions"]:
                t = r["transitions"][0]
                hidden = actor.encode([r["observation"]])[0][0, :len(r["observation"]["nodes"])]
                inputs = sl.transition_inputs(r["observation"], hidden, t["action"], t["after"])
                check["latent"] = latent(inputs).cpu().tolist()
            parity.append(check)
    sl.save(args.model_dir/"parity.json", parity)
    sl.save(args.model_dir/"training-report.json", dict(schema_version=1, package=manifest["package"], rl=sl.load(args.model_dir/"rl-report.json"), latent=report, interface="frozen parent checkpoint"))
    subprocess.run([str(args.native.resolve()), "finalize", "--run-dir", str(args.run_dir), "--model-dir", str(args.model_dir)], check=True)


def run(args, device):
    started = time.monotonic()
    parent_path = args.parent_models/"merge-policy.json"
    template = sl.load(parent_path)
    if args.parent_models.resolve() == args.model_dir.resolve():
        raise ValueError("RL output must differ from the parent directory")
    if not args.no_refresh_latent:
        parent_report = sl.load(args.parent_models/"training-report.json")
        sl.load(args.run_dir/"prepared"/parent_report["package"]/"manifest.json")
        sl.load(args.parent_models/"interface-policy.json")
    actor = sl.Merge(template["model"], device=device)
    anchor = sl.Merge(template["model"], device=device)
    anchor.theta.requires_grad_(False)
    optimizer = torch.optim.AdamW([actor.theta], lr=args.learning_rate, weight_decay=1e-4)
    env = Environment(args)
    env.pool = OnlinePool(args.online_dir, template, sl.digest(parent_path)) if args.online_dir else None
    try:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("epochs", "device", "native")}
        signature = hashlib.sha256(json.dumps(dict(version=1, config=config, environment=env.ready, backend_sha256=sl.digest(Path(__file__)), model_backend_sha256=sl.digest(Path(sl.__file__)),online_backend_sha256=sl.digest(Path(__file__).with_name("online_data.py"))), sort_keys=True).encode()).hexdigest()
        state_path = args.model_dir/"rl-state.pt"
        rng = random.Random(args.seed)
        if state_path.exists():
            state = torch.load(state_path, map_location=device, weights_only=True)
            if state["signature"] != signature:
                raise ValueError("RL source/config changed; use a new model directory")
            with torch.no_grad():
                actor.theta.copy_(state["theta"])
            optimizer.load_state_dict(state["optimizer"])
            rng.setstate(state["rng"])
        else:
            if (args.model_dir/"merge-policy.json").exists():
                raise ValueError("output already contains a policy without RL resume state")
            score, validation = evaluate(env, actor, template["model"], args.seed)
            state = dict(signature=signature, epoch=0, best_epoch=0, best_score=score, best_theta=actor.theta.detach().clone(), best_optimizer=None, bad_epochs=0, history=[], initial_validation=validation, trajectory_files=[])
        groups = {}
        for i, source in enumerate(env.ready["train"]):
            groups.setdefault(source["group"], []).append(i)
        if env.pool:
            env.pool.refresh(state.get("online_manifests", {}))
        parity_observations = []
        for epoch in range(state["epoch"], args.epochs):
            references = env.pool.refresh() if env.pool else []
            if len(references) > state.get("online_available",0):
                state["bad_epochs"] = 0
            if state["epoch"] >= min(5,args.epochs) and state["bad_epochs"] >= args.patience:
                break
            indices = draw_problems(rng, groups, references, args.episodes_per_epoch, args.online_fraction)
            print(f"RL epoch {epoch+1}: {len(references)} committed online problems available",flush=True)
            training = []
            for start in range(0,len(indices),args.batch_size):
                batch = indices[start:start+args.batch_size]
                current = model_json(actor, template["model"])
                seed = rng.getrandbits(63)
                baselines = env.collect(current, "Train", batch, seed, False)
                sampled = env.collect(current, "Train", [i for i in batch for _ in range(args.samples)], seed+len(batch), True)
                baseline_rows = [b for b in baselines for _ in range(args.samples)]
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = policy_loss(actor, anchor, sampled, baseline_rows, env.ready["options"]["value_weight"], args.entropy, args.kl)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite RL objective")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([actor.theta],1.0,error_if_nonfinite=True)
                optimizer.step()
                if not parity_observations:
                    parity_observations = [d["observation"] for e in sampled for d in e["decisions"]][:2]
                token = uuid.uuid4().hex
                path = args.model_dir/"episodes"/(token+".json")
                path.parent.mkdir(exist_ok=True)
                sl.save(path, dict(schema_version=1, snapshot_id=env.ready["snapshot_id"], epoch=epoch+1, batch=start//args.batch_size, behavior_model=current, sampled=sampled, baselines=baselines, loss=float(loss.detach()), metrics=metrics))
                state["trajectory_files"].append(dict(file=str(path.relative_to(args.model_dir)),sha256=sl.digest(path)))
                training += sampled
                actor.graph_cache.clear()
                anchor.graph_cache.clear()
                print(f"RL epoch {epoch+1} batch {start//args.batch_size+1}: loss={float(loss.detach()):.5f}",flush=True)
            score, validation = evaluate(env, actor, template["model"], args.seed)
            actor.graph_cache.clear()
            row = dict(epoch=epoch+1, training_reward=sum(e["reward"] for e in training)/len(training), validation=validation,
                       online_available=len(references), online_episodes=sum(e.get("online",False) for e in training))
            state["history"].append(row)
            state["epoch"] = epoch+1
            if score > tuple(state["best_score"]):
                state.update(best_score=score, best_epoch=epoch+1, best_theta=actor.theta.detach().clone(),best_optimizer=copy.deepcopy(optimizer.state_dict()),bad_epochs=0)
            else:
                state["bad_epochs"] += 1
            state.update(theta=actor.theta.detach().clone(),optimizer=optimizer.state_dict(),rng=rng.getstate(),
                         online_manifests=env.pool.manifests.copy() if env.pool else {},online_available=len(references))
            save_state(state_path,state)
            print(f"RL {epoch+1}: validation solved={validation['solved']}/{validation['problems']} reward={validation['domain_reward']:.5f}, selected={state['best_epoch']}",flush=True)
        with torch.no_grad():
            actor.theta.copy_(state["best_theta"])
        if not parity_observations:
            parity_observations = [d["observation"] for e in env.collect(model_json(actor,template["model"]),"Validation",[0],args.seed,False) for d in e["decisions"]][:2]
        checkpoint = copy.deepcopy(template)
        checkpoint["model"] = model_json(actor,template["model"])
        if state["best_optimizer"] is not None:
            opt = next(iter(state["best_optimizer"]["state"].values()))
            checkpoint["optimizer"] = dict(step=int(opt["step"]),first=opt["exp_avg"].cpu().tolist(),second=opt["exp_avg_sq"].cpu().tolist())
            checkpoint["completed_epochs"] += state["best_epoch"]
            checkpoint["options"].update(learning_rate=args.learning_rate,gradient_clip=1.0,
                batch_size=args.batch_size*args.samples,epochs=checkpoint["completed_epochs"],value_loss_weight=0.1)
        checks = []
        with torch.no_grad():
            for obs in parity_observations:
                logits, values, _ = actor([obs])
                checks.append(dict(observation=obs,prediction=dict(logits=logits.cpu().tolist(),log_effort=values.cpu().tolist())))
        (args.model_dir/"bundle.json").unlink(missing_ok=True)
        sl.save(args.model_dir/"merge-policy.json",checkpoint)
        sl.save(args.model_dir/"rl-parity.json",checks)
        subprocess.run([str(args.native.resolve()),"rl-check","--parent",str(parent_path),"--checkpoint",str(args.model_dir/"merge-policy.json"),"--parity",str(args.model_dir/"rl-parity.json")],check=True)
        score, validation = evaluate(env,actor,template["model"],args.seed)
        sl.save(args.model_dir/"rl-report.json",dict(schema_version=1,algorithm="self-critical REINFORCE",signature=signature,device=device,parent_sha256=sl.digest(parent_path),snapshot_id=env.ready["snapshot_id"],config=config,sources=env.ready,initial_validation=state["initial_validation"],validation=validation,best_epoch=state["best_epoch"],completed_epochs=state["epoch"],history=state["history"],trajectories=state["trajectory_files"],online_manifests=state.get("online_manifests",{}),interface="frozen",latent="disabled" if args.no_refresh_latent else "refit after actor selection",elapsed_seconds=time.monotonic()-started))
    finally:
        env.close()
    if args.no_refresh_latent:
        # A previous output's transition model must never be mistaken for one
        # trained against this actor. The independent parent bundle is untouched.
        for name in ("latent-policy.json","interface-policy.json","training-report.json"):
            (args.model_dir/name).unlink(missing_ok=True)
    else:
        refresh_bundle(args,actor,checkpoint,device,signature)
    print(f"Validated RL output: {args.model_dir}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir","parent-models","model-dir","native"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    for name,default in (("epochs",20),("episodes-per-epoch",32),("batch-size",8),("samples",2),("train-problems",256),("validation-problems",32),("patience",5),("seed",43),("total-ms",1000),("stage-ms",350),("local-ms",100),("local-calls",64),("latent-epochs",50)):
        p.add_argument("--"+name,type=int,default=default)
    for name,default in (("learning-rate",0.0001),("entropy",0.01),("kl",0.02)):
        p.add_argument("--"+name,type=float,default=default)
    p.add_argument("--no-refresh-latent",action="store_true")
    p.add_argument("--online-dir",type=Path)
    p.add_argument("--online-fraction",type=float,default=0.75)
    args=p.parse_args()
    if not math.isfinite(args.online_fraction) or not 0<=args.online_fraction<=1:
        p.error("online-fraction must be in [0,1]")
    if any(getattr(args,n)<=0 for n in ("epochs","episodes_per_epoch","batch_size","samples","train_problems","validation_problems","patience","total_ms","stage_ms","local_ms","local_calls","latent_epochs")) or args.batch_size*args.samples>128 or args.batch_size>128 or not 0<=args.seed<2**64:
        p.error("counts/budgets must be positive; batch_size*samples <=128")
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or any(not math.isfinite(v) or v<0 for v in (args.entropy,args.kl)):
        p.error("invalid optimizer/regularization settings")
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    device="cuda" if args.device=="cuda" or (args.device=="auto" and torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    args.model_dir.mkdir(parents=True,exist_ok=True)
    with (args.model_dir/"rl-training.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args,device)


if __name__=="__main__":
    main()
