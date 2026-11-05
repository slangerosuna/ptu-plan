#!/usr/bin/env python3
"""Batched training with native Rust layouts, audited inputs, and resumable phases."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class FlatModel(nn.Module):
    def __init__(self, model, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(model["parameters"], dtype=dtype, device=device))
        self.h = model.get("hidden", model.get("config", {}).get("hidden"))

    def tensor(self, value, dtype=None):
        return torch.as_tensor(value, dtype=dtype or self.theta.dtype, device=self.theta.device)


class Merge(FlatModel):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.layers = model["config"]["layers"]
        self.architecture = model["schema_version"]
        if self.architecture not in (1, 2):
            raise ValueError("unsupported merge architecture")
        self.graph_cache = {}

    def encode_wl(self, observations):
        h, p = self.h, self.theta
        rows, counts = [], []
        for obs in observations:
            graph = obs.get("hilg")
            if graph is None or graph.get("wl", {}).get("schema_version") != 1:
                raise ValueError("WL checkpoint requires HILG subtree features")
            cached = self.graph_cache.get(id(obs))
            if cached is None:
                features = [wl + node["features"] for wl,node in zip(graph["wl"]["clusters"],obs["nodes"])]
                cached = (obs,torch.tensor(features,dtype=p.dtype).reshape(-1,2060))
                self.graph_cache[id(obs)] = cached
            rows.append(cached[1]); counts.append(len(obs["nodes"]))
        hidden = torch.cat(rows).to(p.device)
        offset, width = 0, 2060
        for _ in range(self.layers):
            matrix = p[offset:offset+h*width].view(h,width); offset += h*width
            bias = p[offset:offset+h]; offset += h
            hidden = torch.tanh(F.linear(hidden,matrix,bias))
            width = h
        maximum = max(1,max(counts))
        rows = [F.pad(row,(0,0,0,maximum-len(row))) for row in hidden.split(counts)]
        hidden = torch.stack(rows)
        return hidden,hidden.sum(1)/self.tensor([max(1,n) for n in counts]).unsqueeze(1),offset

    def encode(self, observations):
        if self.architecture == 2:
            return self.encode_wl(observations)
        h, p = self.h, self.theta
        count = len(observations)
        width = max(1, max(len(o["nodes"]) for o in observations))
        # Collate on the CPU, then transfer whole batches. Per-edge CUDA writes
        # otherwise launch thousands of tiny kernels before the graph encoder.
        x = torch.zeros(count, width, 12, dtype=p.dtype)
        mask = torch.zeros(count, width, 1, dtype=p.dtype)
        adjacency = torch.zeros(count, 12, width, width, dtype=p.dtype)
        edges, weights = [], []
        for b, obs in enumerate(observations):
            n = len(obs["nodes"])
            if n:
                x[b, :n] = torch.tensor([node["features"] for node in obs["nodes"]], dtype=p.dtype)
                mask[b, :n] = 1
            for edge in obs["relations"]:
                edges.append((b, edge["channel"], edge["target"], edge["source"]))
                weights.append(edge["weight"])
        if edges:
            indices = torch.tensor(edges, dtype=torch.long).T
            adjacency.index_put_(tuple(indices), torch.tensor(weights, dtype=p.dtype), accumulate=True)
        adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1e-30)
        x, mask, adjacency = (v.to(p.device) for v in (x, mask, adjacency))
        hidden = torch.tanh(F.linear(x, p[:12*h].view(h, 12), p[12*h:13*h])) * mask
        offset = 13*h
        for _ in range(self.layers):
            own = p[offset:offset+h*h].view(h, h)
            offset += h*h
            relation = p[offset:offset+12*h*h].view(12, h, h)
            offset += 12*h*h
            bias = p[offset:offset+h]
            offset += h
            messages = torch.einsum("brts,bsk,rdk->btd", adjacency, hidden, relation)
            hidden = torch.tanh(F.linear(hidden, own, bias) + messages) * mask
        pool = hidden.sum(1) / mask.sum(1).clamp_min(1)
        return hidden, pool, offset

    def forward(self, observations):
        hidden, pool, offset = self.encode(observations)
        batch, left, right, stopping, features, offsets = [], [], [], [], [], [0]
        for b, obs in enumerate(observations):
            for action in obs["actions"]:
                pair = action["pair"]
                batch.append(b)
                left.append(pair[0] if pair else 0)
                right.append(pair[1] if pair else 0)
                stopping.append(pair is None)
                features.append(action["features"] + obs["global"] + obs["context"] + [float(pair is None)])
            offsets.append(len(batch))
        batch, left, right = [self.tensor(v, torch.long) for v in (batch, left, right)]
        stop = self.tensor(stopping, torch.bool).unsqueeze(1)
        a, b, pooled = hidden[batch, left], hidden[batch, right], pool[batch]
        first = torch.where(stop, pooled, (a+b)*0.5)
        delta = torch.where(stop, torch.zeros_like(a), (a-b).abs())
        product = torch.where(stop, pooled.square(), a*b)
        encoded = torch.cat((first, delta, product, pooled, self.tensor(features)), dim=1)
        width = encoded.shape[1]
        if self.architecture == 2:
            decoder = self.theta[offset:offset+self.h*(width+1)].view(self.h,width+1)
            offset += self.h*(width+1)
            encoded = torch.tanh(F.linear(encoded,decoder[:,:-1],decoder[:,-1]))
            width = self.h
        policy = self.theta[offset:offset+width+1]
        value = self.theta[offset+width+1:]
        return encoded @ policy[:-1] + policy[-1], F.softplus(encoded @ value[:-1] + value[-1], threshold=100), offsets

    def loss(self, records):
        logits, values, offsets = self([r["observation"] for r in records])
        preferred, other, rank_weight, actions, targets, censored, value_weight = [], [], [], [], [], [], []
        for record, start in zip(records, offsets):
            labels = record["labels"]
            for a, b in labels["preferences"]:
                preferred.append(start+a)
                other.append(start+b)
                rank_weight.append(1 / (len(records)*len(labels["preferences"])))
            for target in labels["values"]:
                actions.append(start+target["action"])
                targets.append(target["log_effort"])
                censored.append(target["lower_bound"])
                value_weight.append(1 / (len(records)*len(labels["values"])))
        loss = self.theta.sum()*0
        if preferred:
            difference = logits[self.tensor(other, torch.long)] - logits[self.tensor(preferred, torch.long)]
            loss = loss + (F.softplus(difference, threshold=100)*self.tensor(rank_weight)).sum()
        if actions:
            residual = values[self.tensor(actions, torch.long)] - self.tensor(targets)
            residual = torch.where(self.tensor(censored, torch.bool), residual.clamp_max(0), residual)
            loss = loss + (F.huber_loss(residual, torch.zeros_like(residual), reduction="none") * self.tensor(value_weight)).sum()
        return loss


class Interface(FlatModel):
    def forward(self, rows):
        first = self.theta[:self.h*17].view(self.h, 17)
        last = self.theta[self.h*17:].view(3, self.h+1)
        return F.linear(torch.tanh(F.linear(rows, first[:, :16], first[:, 16])), last[:, :self.h], last[:, self.h])

    def loss(self, examples):
        rows, targets, weights = [], [], []
        for example in examples:
            atoms = example["input"]["atoms"]
            rows.extend(a["features"] for a in atoms)
            targets.extend(example["targets"])
            weights.extend([1/(len(examples)*len(atoms))]*len(atoms))
        target = self.tensor(targets, torch.long)
        ce = F.cross_entropy(self(self.tensor(rows)), target, reduction="none")
        return (ce * self.tensor([0.25, 1.0, 1.0])[target] * self.tensor(weights)).sum()


class Latent(FlatModel):
    def forward(self, rows):
        width = rows.shape[1]
        first = self.theta[:self.h*(width+1)].view(self.h, width+1)
        last = self.theta[self.h*(width+1):].view(self.h, self.h+1)
        intermediate = torch.tanh(F.linear(rows, first[:, :width], first[:, width]))
        return torch.tanh(rows[:, :self.h]+F.linear(intermediate, last[:, :self.h], last[:, self.h]))

    def loss(self, examples):
        rows = torch.cat([e["inputs"] for e in examples]).to(self.theta)
        target = torch.cat([e["target"] for e in examples]).to(self.theta)
        weights = self.tensor([1/(len(examples)*len(e["inputs"])*self.h) for e in examples for _ in e["inputs"]])
        return ((self(rows)-target).square().sum(1)*weights).sum()


def transition_inputs(before, hidden, action, after):
    a, b = before["actions"][action]["pair"]
    pair = (hidden[a]+hidden[b])*0.5
    pool = hidden.mean(0)
    positions = {n["id"]: i for i, n in enumerate(before["nodes"])}
    rows = []
    for node in after["nodes"]:
        index = positions.get(node["id"])
        own = pair if index is None else hidden[index]
        tail = before["actions"][action]["features"] + node["features"] + before["context"] + [float(index is None)]
        rows.append(torch.cat((own, pair, pool, hidden.new_tensor(tail))))
    return torch.stack(rows)


def assert_close(actual, expected, name, tolerance=1e-8):
    actual = torch.as_tensor(actual, dtype=torch.float64, device="cpu")
    expected = torch.as_tensor(expected, dtype=torch.float64, device="cpu")
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        raise ValueError(f"native parity failed: {name}; shape {actual.shape} vs {expected.shape}")


def reference_parity(templates, references):
    if not all(any(key in reference for reference in references) for key in ("merge_loss_gradient", "interface_loss_gradient", "latent_loss_gradient")):
        raise ValueError("prepared package must include native reference gradients for every model")
    models = [Merge(templates[0]["model"], dtype=torch.float64), Interface(templates[1]["model"], dtype=torch.float64), Latent(templates[2]["model"], dtype=torch.float64)]
    for reference in references:
        r = reference["record"]
        logits, values, _ = models[0]([r["observation"]])
        assert_close(logits.detach(), reference["prediction"]["logits"], "merge logits")
        assert_close(values.detach(), reference["prediction"]["log_effort"], "merge values")
        tasks = [(models[0], [r], reference["merge_loss_gradient"], "merge")]
        if "interface_loss_gradient" in reference:
            e = r["interfaces"][0]
            probabilities = models[1](models[1].tensor([a["features"] for a in e["input"]["atoms"]])).softmax(-1)
            assert_close(probabilities.detach(), reference["interface_prediction"], "interface probabilities")
            tasks.append((models[1], [e], reference["interface_loss_gradient"], "interface"))
        if "latent_loss_gradient" in reference:
            t = r["transitions"][0]
            hidden = models[0].encode([r["observation"]])[0][0, :len(r["observation"]["nodes"])]
            inputs = transition_inputs(r["observation"], hidden.detach(), t["action"], t["after"])
            assert_close(inputs, reference["latent_inputs"], "latent input")
            tasks.append((models[2], [{"inputs": inputs, "target": torch.tensor(reference["latent_target"], dtype=torch.float64)}], reference["latent_loss_gradient"], "latent"))
        for model, examples, (expected_loss, expected_gradient), name in tasks:
            model.zero_grad()
            loss = model.loss(examples)
            loss.backward()
            assert_close(loss.detach(), expected_loss, name+" loss")
            assert_close(model.theta.grad, expected_gradient, name+" gradient")
    print("Native float64 forward/loss/gradient parity passed for all three models", flush=True)


def evaluate(model, examples, batch_size):
    if not examples:
        return None
    with torch.no_grad():
        return sum(float(model.loss(examples[i:i+batch_size]))*len(examples[i:i+batch_size]) for i in range(0, len(examples), batch_size)) / len(examples)


def diagnostics(name, model, examples, batch_size):
    """Report interpretable held-out scores as well as the optimized loss."""
    pairs = correct = censored = exact = 0
    absolute_error = predicted_error = identity_error = 0.0
    confusion = [[0]*3 for _ in range(3)]
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start+batch_size]
            if name == "merge":
                logits, values, offsets = model([r["observation"] for r in batch])
                logits, values = logits.cpu().tolist(), values.cpu().tolist()
                for record, offset in zip(batch, offsets):
                    for a, b in record["labels"]["preferences"]:
                        pairs += 1
                        correct += logits[offset+a] > logits[offset+b]
                    for target in record["labels"]["values"]:
                        if target["lower_bound"]:
                            censored += 1
                        else:
                            exact += 1
                            absolute_error += abs(values[offset+target["action"]]-target["log_effort"])
            elif name == "interface":
                rows = [a["features"] for e in batch for a in e["input"]["atoms"]]
                targets = [c for e in batch for c in e["targets"]]
                predicted = model(model.tensor(rows)).argmax(-1).cpu().tolist()
                for target, prediction in zip(targets, predicted):
                    confusion[target][prediction] += 1
            else:
                lengths = [len(e["inputs"]) for e in batch]
                inputs = torch.cat([e["inputs"] for e in batch]).to(model.theta)
                targets = torch.cat([e["target"] for e in batch]).to(model.theta)
                learned = (model(inputs)-targets).square().mean(-1).cpu()
                identity = (inputs[:, :model.h]-targets).square().mean(-1).cpu()
                position = 0
                for length in lengths:
                    predicted_error += float(learned[position:position+length].mean())
                    identity_error += float(identity[position:position+length].mean())
                    position += length
    if name == "merge":
        return {"preferences": pairs, "pair_accuracy": correct/pairs if pairs else None, "exact_values": exact, "exact_log_effort_mae": absolute_error/exact if exact else None, "censored_values": censored}
    if name == "interface":
        total = sum(map(sum, confusion))
        recalls = [confusion[c][c]/sum(confusion[c]) if sum(confusion[c]) else None for c in range(3)]
        return {"class_order": ["Unknown", "False", "True"], "confusion_actual_predicted": confusion, "accuracy": sum(confusion[c][c] for c in range(3))/total if total else None, "class_recall": recalls}
    return {"transitions": len(examples), "node_mse": predicted_error/len(examples), "identity_pair_mean_node_mse": identity_error/len(examples)}


def train_phase(name, model, examples, template, args, signature):
    splits = [[e for e in examples if e["split"] == split] for split in ("Train", "Validation", "Test")]
    if not all(splits):
        raise ValueError(f"{name} requires nonempty train, validation and test splits; got {list(map(len, splits))}")
    groups = {}
    for e in splits[0]:
        groups.setdefault(e["group"], []).append(e)
    group_names = sorted(groups)
    optimizer = torch.optim.AdamW([model.theta], lr=args.learning_rate, weight_decay=1e-4)
    state_path = args.model_dir / (name+"-state.pt")
    state = None
    if state_path.exists():
        state = torch.load(state_path, weights_only=True, map_location=model.theta.device)
        if state["signature"] != signature:
            raise ValueError(f"{state_path}: data/config changed; use a new --model-dir")
        with torch.no_grad():
            model.theta.copy_(state["theta"])
        optimizer.load_state_dict(state["optimizer"])
        print(f"{name}: resuming after epoch {state['epoch']}", flush=True)
    else:
        state = {"signature": signature, "epoch": 0, "best_epoch": 0, "best_score": math.inf, "bad_epochs": 0, "history": [], "initial_training": evaluate(model, splits[0], args.batch_size)}
    minimum_epochs = min(20, args.epochs)
    for epoch in range(state["epoch"], args.epochs):
        if state["epoch"] >= minimum_epochs and state["bad_epochs"] >= args.patience:
            break
        lr = args.learning_rate*(0.1 + 0.9*0.5*(1+math.cos(math.pi*epoch/args.epochs)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        rng = random.Random(template["options"]["seed"]+epoch)
        sampled = [rng.choice(groups[rng.choice(group_names)]) for _ in splits[0]]
        for start in range(0, len(sampled), args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(sampled[start:start+args.batch_size])
            if not torch.isfinite(loss):
                raise ValueError(f"{name}: nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.theta], 5.0, error_if_nonfinite=True)
            optimizer.step()
        training = evaluate(model, splits[0], args.batch_size)
        validation = evaluate(model, splits[1], args.batch_size)
        state["epoch"] = epoch+1
        state["history"].append({"epoch": epoch+1, "training": training, "validation": validation, "learning_rate": lr})
        if validation < state["best_score"]-1e-6:
            state.update(best_score=validation, best_epoch=epoch+1, bad_epochs=0, best_theta=model.theta.detach().clone(), best_optimizer=copy.deepcopy(optimizer.state_dict()))
        else:
            state["bad_epochs"] += 1
        state["theta"] = model.theta.detach().clone()
        state["optimizer"] = optimizer.state_dict()
        temporary = state_path.with_suffix(".pt.tmp")
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(state_path)
        print(f"{name} {epoch+1}/{args.epochs}: train={training:.6f} validation={validation:.6f} best={state['best_epoch']}", flush=True)
    with torch.no_grad():
        model.theta.copy_(state["best_theta"])
    checkpoint = copy.deepcopy(template)
    checkpoint["model"]["parameters"] = model.theta.detach().cpu().tolist()
    opt = next(iter(state["best_optimizer"]["state"].values()))
    checkpoint["optimizer"] = {"step": int(opt["step"]), "first": opt["exp_avg"].cpu().tolist(), "second": opt["exp_avg_sq"].cpu().tolist()}
    checkpoint["completed_epochs"] = state["best_epoch"]
    checkpoint["options"].update(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    report = {"parameters": model.theta.numel(), "examples": list(map(len, splits)), "groups": [len({e['group'] for e in s}) for s in splits], "initial_training": state["initial_training"], "best_epoch": state["best_epoch"], "completed_epochs": state["epoch"], "history": state["history"], "training": evaluate(model, splits[0], args.batch_size), "validation": evaluate(model, splits[1], args.batch_size), "test": evaluate(model, splits[2], args.batch_size), "diagnostics": {split: diagnostics(name, model, data, args.batch_size) for split, data in zip(("training", "validation", "test"), splits)}}
    return checkpoint, report


def prepare_latent(merge, records, batch_size):
    examples = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start+batch_size]
            hidden = merge.encode([r["observation"] for r in batch])[0].cpu()
            transitions = [(b, r, t) for b, r in enumerate(batch) for t in r["transitions"]]
            for offset in range(0, len(transitions), batch_size):
                chunk = transitions[offset:offset+batch_size]
                targets = merge.encode([t["after"] for _, _, t in chunk])[0].cpu()
                for index, (b, r, t) in enumerate(chunk):
                    n = len(t["after"]["nodes"])
                    inputs = transition_inputs(r["observation"], hidden[b, :len(r["observation"]["nodes"])], t["action"], t["after"])
                    examples.append({"group": r["group"], "split": r["split"], "inputs": inputs.cpu(), "target": targets[index, :n].cpu()})
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--package", help="Pin an exported package ID, including when resuming after more data was prepared")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.cpu_threads) <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("counts and learning rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; install the appropriate CUDA-enabled PyTorch wheel")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.model_dir.mkdir(parents=True, exist_ok=True)
    # One owner per model directory, released by the OS on interruption.
    import fcntl
    with (args.model_dir / "training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, device)


def run(args, device):
    started = time.monotonic()
    manifest_path = args.run_dir / "prepared" / args.package / "manifest.json" if args.package else args.run_dir / "prepared/current.json"
    manifest = load(manifest_path)
    if manifest["schema_version"] != 1 or (args.package and manifest["package"] != args.package):
        raise ValueError("unsupported or mismatched prepared package")
    package = args.run_dir / "prepared" / manifest["package"]
    files = {"records.jsonl": manifest["records_sha256"], "references.json": manifest["references_sha256"]}
    names = ("merge-template.json", "interface-template.json", "latent-template.json")
    files.update(zip(names, manifest["templates_sha256"]))
    for name, expected in files.items():
        if digest(package/name) != expected:
            raise ValueError(f"prepared package checksum mismatch: {name}")
    templates = [load(package/name) for name in names]
    references = load(package/"references.json")
    reference_parity(templates, references)
    with (package/"records.jsonl").open() as handle:
        records = [json.loads(line) for line in handle]
    signature = hashlib.sha256(json.dumps({"package": manifest["package"], "config": [args.epochs, args.batch_size, args.learning_rate, args.patience], "backend_version": 3}, sort_keys=True).encode()).hexdigest()
    print(f"Device={device}; parameters={manifest['parameters']}; split counts={manifest['splits']}", flush=True)
    merge = Merge(templates[0]["model"], device=device)
    interface = Interface(templates[1]["model"], device=device)
    latent = Latent(templates[2]["model"], device=device)
    merge_records = [r for r in records if r["labels"]["preferences"] or r["labels"]["values"]]
    interfaces = [dict(e, group=r["group"], split=r["split"]) for r in records for e in r["interfaces"]]
    # Validate all heads before spending time on the first phase.
    for name, data in (("merge", merge_records), ("interface", interfaces), ("latent", [r for r in records if r["transitions"]])):
        if any(not any(e["split"] == split for e in data) for split in ("Train", "Validation", "Test")):
            raise ValueError(f"{name} has an empty domain holdout split; prepare more data")
    merge_checkpoint, merge_report = train_phase("merge", merge, merge_records, templates[0], args, signature)
    interface_checkpoint, interface_report = train_phase("interface", interface, interfaces, templates[1], args, signature)
    # Frozen final encoder makes targets consistent throughout latent training.
    transitions = prepare_latent(merge, records, args.batch_size)
    encoder_signature = hashlib.sha256(json.dumps(merge_checkpoint["model"], sort_keys=True).encode()).hexdigest()
    latent_checkpoint, latent_report = train_phase("latent", latent, transitions, templates[2], args, signature+encoder_signature)
    latent_checkpoint["encoder"] = merge_checkpoint["model"]
    (args.model_dir/"bundle.json").unlink(missing_ok=True)
    for name, checkpoint in zip(("merge-policy.json", "interface-policy.json", "latent-policy.json"), (merge_checkpoint, interface_checkpoint, latent_checkpoint)):
        save(args.model_dir/name, checkpoint)
    parity = []
    with torch.no_grad():
        for reference in references:
            r = reference["record"]
            logits, values, _ = merge([r["observation"]])
            result = {"record": r["id"], "prediction": {"logits": logits.cpu().tolist(), "log_effort": values.cpu().tolist()}}
            if r["interfaces"]:
                rows = interface.tensor([a["features"] for a in r["interfaces"][0]["input"]["atoms"]])
                result["interface"] = interface(rows).softmax(-1).cpu().tolist()
            if r["transitions"]:
                t = r["transitions"][0]
                hidden = merge.encode([r["observation"]])[0][0, :len(r["observation"]["nodes"])]
                inputs = transition_inputs(r["observation"], hidden, t["action"], t["after"])
                result["latent"] = latent(inputs).cpu().tolist()
            parity.append(result)
    save(args.model_dir/"parity.json", parity)
    save(args.model_dir/"training-report.json", {"schema_version": 1, "package": manifest["package"], "device": device, "torch_version": str(torch.__version__), "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "weight_decay": 1e-4, "gradient_clip": 5, "patience": args.patience, "minimum_epochs": min(20, args.epochs), "sampler": "uniform domain, uniform example", "precision": "float32"}, "merge": merge_report, "interface": interface_report, "latent": latent_report, "elapsed_seconds": time.monotonic()-started})
    subprocess.run([str(args.native.resolve()), "finalize", "--run-dir", str(args.run_dir), "--model-dir", str(args.model_dir)], check=True)
    print(f"Validated native checkpoints and bundle: {args.model_dir}", flush=True)


if __name__ == "__main__":
    main()
