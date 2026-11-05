#!/usr/bin/env python3
"""Create a self-contained source and prepared-data archive for a rented GPU."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/hilg-wl"))
    parser.add_argument("--output", type=Path, default=Path("runs/hilg-wl-training.tar.gz"))
    parser.add_argument("--include-rl", action="store_true", help="Include the immutable source store and pretrained JSON models for on-policy RL")
    parser.add_argument("--models-dir", type=Path, help="Pretrained parent models for --include-rl; defaults to RUN_DIR/models")
    args = parser.parse_args()
    if args.models_dir and not args.include_rl:
        parser.error("--models-dir requires --include-rl")
    root = Path(__file__).resolve().parents[1]
    run = (root / args.run_dir).resolve()
    manifest = json.loads((run / "prepared/current.json").read_text())
    package = run / "prepared" / manifest["package"]
    if manifest["schema_version"] != 1 or json.loads((package / "manifest.json").read_text()) != manifest:
        raise ValueError("prepared manifest/pointer mismatch")
    files = {"records.jsonl": manifest["records_sha256"], "references.json": manifest["references_sha256"]}
    files.update(zip(("merge-template.json", "interface-template.json", "latent-template.json"), manifest["templates_sha256"]))
    for name, expected in files.items():
        if digest(package / name) != expected:
            raise ValueError(f"prepared-data checksum mismatch: {name}")
    rl_objects = []
    rl_models = []
    if args.include_rl:
        index_path = run / "data/index.json"
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
        fingerprint = hashlib.sha256(json.dumps(index,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        if fingerprint != manifest["snapshot_id"]:
            raise ValueError("RL store snapshot differs from the prepared package")
        models = (root/args.models_dir).resolve() if args.models_dir else run/"models"
        for name in ("merge-policy.json","interface-policy.json","latent-policy.json","training-report.json","bundle.json","parity.json"):
            path = models/name
            payload = path.read_bytes()
            value = json.loads(payload)
            if name.endswith("-policy.json") and value["snapshot_id"] != fingerprint:
                raise ValueError(f"RL parent snapshot mismatch: {name}")
            if name == "training-report.json" and value["package"] != manifest["package"]:
                raise ValueError("RL parent package does not match the selected prepared data")
            rl_models.append((payload,name))
        model_bytes={name:payload for payload,name in rl_models}
        bundle=json.loads(model_bytes["bundle.json"])
        if bundle["snapshot_id"]!=fingerprint or bundle["package"]!=manifest["package"]:
            raise ValueError("RL parent bundle provenance mismatch")
        for name,expected in bundle["files"].items():
            if Path(name).name!=name or name not in model_bytes or hashlib.sha256(model_bytes[name]).hexdigest()!=expected:
                raise ValueError(f"RL parent bundle checksum mismatch: {name}")
        for entry in index["entries"]:
            name = entry["id"]
            if len(name)!=64 or any(c not in "0123456789abcdef" for c in name):
                raise ValueError("unsafe store object ID")
            path=run/"data/objects"/name[:2]/(name+".json")
            if path.is_symlink() or path.stat().st_size!=entry["bytes"] or digest(path)!=name:
                raise ValueError(f"RL store checksum mismatch: {name}")
            rl_objects.append((path,name))
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    prefix = Path("ptu-plan-training")
    with tarfile.open(temporary, "w:gz") as archive:
        for name in ("Cargo.toml", "Cargo.lock", "README.md", "requirements-training.txt", "src", "scripts/train_all.py", "scripts/finetune_rl.py", "scripts/online_data.py", "scripts/train_online.py", "scripts/train_gpu_online.sh", "scripts/train_gpu.sh", "docs/training.md", "docs/hilg-wl.md", "docs/repartition-rl.md", "docs/online-training.md"):
            archive.add(root / name, arcname=str(prefix / name))
        for name in (*files, "manifest.json"):
            archive.add(package / name, arcname=str(prefix / "runs/hilg-wl/prepared" / manifest["package"] / name))
        payload = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo(str(prefix / "runs/hilg-wl/prepared/current.json"))
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        if args.include_rl:
            info=tarfile.TarInfo(str(prefix/"runs/hilg-wl/data/index.json"))
            info.size=len(index_bytes)
            archive.addfile(info,io.BytesIO(index_bytes))
            for path,name in rl_objects:
                archive.add(path,arcname=str(prefix/"runs/hilg-wl/data/objects"/name[:2]/(name+".json")))
            for payload,name in rl_models:
                info=tarfile.TarInfo(str(prefix/"runs/hilg-wl/models"/name))
                info.size=len(payload)
                archive.addfile(info,io.BytesIO(payload))
        for name in ("preparation.json", "source-summary.json"):
            if (run / name).is_file():
                archive.add(run / name, arcname=str(prefix / "runs/hilg-wl" / name))
    temporary.replace(output)
    checksum = digest(output)
    Path(str(output) + ".sha256").write_text(f"{checksum}  {output.name}\n")
    print(f"{output} ({output.stat().st_size:,} bytes)")
    print(f"Package: {manifest['package']}")
    print(f"SHA-256: {checksum}")
    if args.include_rl:
        print("On the GPU host: extract the archive, cd ptu-plan-training, then ./scripts/train_gpu_online.sh")
    else:
        print("On the GPU host: tar -xzf hilg-wl-training.tar.gz && cd ptu-plan-training && ./scripts/train_gpu.sh")


if __name__ == "__main__":
    main()
