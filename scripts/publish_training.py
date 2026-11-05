#!/usr/bin/env python3
"""Audit and publish a prepared dataset without replacing existing files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(source, destination, auditor):
    if destination.is_symlink():
        raise ValueError("destination must be a directory, not a symlink")
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    subprocess.run([str(auditor), "--output", str(source), "--audit-only"], check=True)
    entries = json.loads((source / "manifest.json").read_text())
    names = []
    for entry in entries:
        name = entry["file"]
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("unsafe manifest filename")
        if digest(source / name) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {name}")
        names.append(name)
    names += ["manifest.json", "collection-report.json", "README.md"]
    # Check every collision before creating any records in the destination.
    for name in names:
        existing = destination / name
        if existing.is_symlink():
            raise ValueError(f"destination file is a symlink: {name}")
        if existing.exists() and digest(existing) != digest(source / name):
            raise ValueError(f"refusing to overwrite different existing file: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    added = 0
    for name in names:
        target = destination / name
        if target.exists():
            continue
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".publish-", delete=False) as out:
            temporary = Path(out.name)
            out.write((source / name).read_bytes())
            out.flush()
            os.fsync(out.fileno())
        try:
            os.link(temporary, target)
            added += 1
        except FileExistsError:
            if digest(target) != digest(source / name):
                raise ValueError(f"concurrent conflicting write: {name}")
        finally:
            temporary.unlink()
    subprocess.run([str(auditor), "--output", str(destination), "--audit-only"], check=True)
    print(f"Published {len(entries)} audited records to {destination} ({added} files added)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path("../train"))
    parser.add_argument("--auditor", type=Path, default=Path(__file__).resolve().parents[1] / "target/debug/generate-training")
    options = parser.parse_args()
    publish(options.source, options.destination, options.auditor.resolve(strict=True))


if __name__ == "__main__":
    main()
