# Background generation on the rented host

Extract the RL transfer archive, enter `ptu-plan-training`, and install stable
Rust and CUDA-enabled PyTorch. Start generation and training together:

```bash
./scripts/train_gpu_online.sh
# With a separate Python environment:
PTU_TRAIN_PYTHON=/path/to/venv/bin/python ./scripts/train_gpu_online.sh
```

This builds the native trainer and launches `scripts/train_online.py`. It reserves
up to eight CPU cores (roughly one third of those available) for RL, and uses the
remaining cores for up to sixteen generation workers. Automatic concurrency also
accounts for available RAM. Workers run at lower CPU priority, with a 2 GiB address
space limit and 180-second job deadline. The default generation session is bounded
by two hours and a 20 GiB soft output limit; active jobs can overshoot the disk
limit before the next check. All child processes stop on interruption.

## Data and training

The default plan requests 64 samples per job, sixteen rounds per training domain:
24,576 samples for the current 24 training domain groups. Jobs rotate up to three
small template instances per domain, vary root limits across 4/8/12, and use new
deterministic seeds. Unsupported regression formulas, duplicate instances and
resource limits can reduce the actual yield. Progress reports **unique programs**,
deduplicating across certificates and batches. These are certified feasible
backward examples, without a claim of optimality. PANDA is not required.

Only the pretrained parent's training domains are eligible. A batch becomes visible
after its immutable ready marker is published. RL imports new batches between
epochs, samples domains uniformly, and chooses fresh problems with probability
0.75 within domains where they are available. Native code checks content hashes,
domain membership, and certificate replay before using a fresh problem. The base
store and prepared package remain unchanged. Validation stays fixed; test domains
are excluded. Newly available examples reset the early-stopping patience counter.

Training starts after 256 unique examples arrive, or after 120 seconds if some
examples are ready. The combined launcher uses an 80-epoch ceiling, 128 problem
draws per epoch, two trajectories per draw, batch eight, and AdamW at 1e-4 with
the existing entropy/KL regularization. The model remains small; native planner
rollouts use CPU while model updates can use CUDA.

Fresh examples train the WL encoder and policy/value heads through actual planner
returns. The interface model remains frozen; the latent model is refitted against
the selected actor using the pinned supervised package. Full synthetic witnesses
are retained for subsequent supervised oracle collection; this workflow does not
silently add interface labels or claim all three heads receive online RL updates.

## Outputs and resume

Defaults write:

- `runs/hilg-wl/online/`: committed objects, batch manifests, resumable generation
  state, per-job logs and `jobs/*/solutions/` with complete synthetic witnesses.
- `runs/hilg-wl/online/training.log` and `generation.log`: detailed progress.
- `runs/hilg-wl/online-rl-models/`: fine-tuned checkpoints and the validated bundle.

```bash
tail -f runs/hilg-wl/online/training.log
cat runs/hilg-wl/online/generation-state.json
./scripts/train_gpu_online.sh --workers 12 --reserve-cpus 8 \
  --rounds 32 --samples-per-job 64 --generation-seconds 14400 --max-gb 40
```

Choose the generation plan before starting. Rerun the same command to resume;
completed generation jobs are reused, failed/interrupted jobs retry, and RL
resumes at epoch boundaries. Worker counts and resource deadlines can change;
changing the generation plan, parent, or code requires a new online directory.
Previously observed manifests are pinned in RL state: new commits may arrive,
but modifying or removing old commits fails validation. Per-epoch availability
and the number of fresh training episodes are recorded in `rl-report.json`;
batch journals pin the actual sampled problems. Collection arrival time can
change future sampling even with the same seed.

Collection continues to its quota/resource limit if training finishes early.
Use `--stop-generation-with-training` to stop it with training. A later run may
use all collected problems with `train rl --online-dir runs/hilg-wl/online` and
the same parent; choose a new model directory when changing RL settings. The
supervised checkpoint's store provenance remains pinned to its original labels;
online training evidence is recorded separately in the RL report and journals.

For a CPU smoke test, use `--device cpu --rounds 1 --samples-per-job 2 --workers 2
--warmup-problems 2 --epochs 1 --episodes-per-epoch 4 --latent-epochs 1` and separate
`--online-dir`/`--model-dir` paths.
