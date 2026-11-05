"""Read immutable, committed backward batches without changing the base store."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OnlinePool:
    def __init__(self, directory, parent, parent_sha256):
        self.directory = Path(directory)
        self.parent_sha256 = parent_sha256
        self.snapshot_id = parent["snapshot_id"]
        self.groups = set(parent["training_groups"])
        self.samples = {}
        self.instances = set()
        self.manifests = {}

    def object_path(self, reference):
        identity = reference["id"]
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("invalid online object identity")
        expected = f"objects/{identity[:2]}/{identity}.json"
        if reference["file"] != expected:
            raise ValueError("invalid online object path")
        path = self.directory / expected
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("online objects must remain inside the collection directory")
        return path

    def read(self, reference):
        raw = self.object_path(reference).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["id"]:
            raise ValueError("online object checksum mismatch")
        record = json.loads(raw)
        if record["schema_version"] != 1 or record["group"] != reference["group"] or record["group"] not in self.groups:
            raise ValueError("online object is outside the parent's Train domains")
        return dict(id=reference["id"], json=raw.decode())

    def refresh(self, pinned=None):
        # A resume may discover new commits, but previously observed commits
        # must still exist byte-for-byte. Incomplete jobs have no ready marker.
        for name, expected in (pinned or self.manifests).items():
            path = self.directory / name
            if digest(path) != expected:
                raise ValueError("previous online manifest changed")
        for path in sorted((self.directory / "ready").glob("*.json")):
            name = str(path.relative_to(self.directory))
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if name in self.manifests:
                if checksum != self.manifests[name]:
                    raise ValueError("committed online manifest changed")
                continue
            batch = json.loads(raw)
            if (batch["schema_version"] != 1 or batch["parent_sha256"] != self.parent_sha256
                    or batch["snapshot_id"] != self.snapshot_id):
                raise ValueError("online batch belongs to a different parent/snapshot")
            for reference in batch["samples"]:
                if reference["group"] not in self.groups:
                    raise ValueError("online batch contains a held-out domain")
                record = json.loads(self.read(reference)["json"])
                instance = hashlib.sha256(json.dumps(record["program"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
                if instance not in self.instances:
                    self.instances.add(instance)
                    self.samples[reference["id"]] = reference
            self.manifests[name] = checksum
        return [self.samples[k] for k in sorted(self.samples)]


def draw_problems(rng, base_groups, references, count, fraction):
    fresh = {}
    for reference in references:
        fresh.setdefault(reference["group"], []).append(reference)
    domains = sorted(set(base_groups) | (set(fresh) if fraction > 0 else set()))
    result = []
    for _ in range(count):
        group = rng.choice(domains)
        if group in fresh and (group not in base_groups or rng.random() < fraction):
            result.append(rng.choice(fresh[group]))
        else:
            result.append(rng.choice(base_groups[group]))
    return result
