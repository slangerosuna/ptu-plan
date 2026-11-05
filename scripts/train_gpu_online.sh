#!/usr/bin/env bash
set -euo pipefail
training_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$training_root"
training_python="${PTU_TRAIN_PYTHON:-python3}"
cargo build --release --bin train
exec "$training_python" scripts/train_online.py "$@"
