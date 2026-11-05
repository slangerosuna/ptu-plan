#!/usr/bin/env bash
set -euo pipefail

training_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$training_root"
training_python="${PTU_TRAIN_PYTHON:-python3}"

"$training_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Use a GPU instance with CUDA-enabled PyTorch; see docs/training.md.")
print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}", flush=True)
PY

if ! command -v cargo >/dev/null 2>&1; then
    echo "Install the current stable Rust toolchain (https://rustup.rs), then rerun this command." >&2
    exit 2
fi
cargo build --release --bin train
exec target/release/train fit --run-dir runs/hilg-wl --device cuda --python "$training_python" "$@"
