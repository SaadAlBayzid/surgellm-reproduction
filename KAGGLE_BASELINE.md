# Two-T4 baseline execution

Copy the existing project, including the unchanged `data/frozen/v1/` bundle, into `/kaggle/working/surgellm-reproduction`. Select Kaggle GPU T4 x2. Use the Python/CUDA environment specified in `environment-gpu.yml`; the CPU environment used locally is not that environment. Activate that environment before the commands below. The notebook needs Internet access on the first smoke run to cache the pinned Hugging Face checkpoint/tokenizer. No dataset download or split-generation command is needed.

Run from `/kaggle/working/surgellm-reproduction`. Both output directories must be new. These are prepared commands, not commands executed in the local CPU session.

## 1. Tiny CUDA smoke

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.train_four_task --config configs/baseline_roberta_four_task.json --mode smoke --seed 0 --output results/kaggle-t4x2-smoke
```

This is the real shared encoder/four-head baseline. It uses 16 examples/rank, two ranks, accumulation 2, FP16, length 128, and one tiny training epoch: 64 frozen training rows per task as an in-memory view, eight global microbatches, four optimizer updates. Both scheduling policies are supported. Validation covers all 2,291 frozen validation rows; best checkpoint restoration precedes all 2,398 test rows. Thus the optimization is tiny, but evaluation deliberately remains complete. Treat smoke test-set metrics as diagnostic only; do not tune settings against them. Smoke checkpoints/results cannot be used as initialization for the full run.

The smoke must update all four heads, complete every planned optimizer update without an AMP skip, restore the saved checkpoint exactly, and complete validation/test successfully. Only a successful two-T4 FP16 run writes `results/kaggle-t4x2-smoke/cuda_smoke_pass.json`. CPU fixture tests cannot issue this certificate.

## 2. Full seed-0 run — prepared only

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.train_four_task --config configs/baseline_roberta_four_task.json --mode full --seed 0 --cuda-certificate results/kaggle-t4x2-smoke/cuda_smoke_pass.json --output results/baseline-roberta-seed0
```

The command rejects a missing/stale certificate before model loading. The certificate records SHA-256 identities for all source/scripts/tests/config and project data files, environment setup files, resolved configuration, and locked dataset manifests, plus Git HEAD when available. This directory currently has no Git metadata, which is recorded explicitly. Full execution compares an environment SHA-256 covering software, NVIDIA driver, GPU details, world size and precision. Any covered file addition/removal/change or Git HEAD/environment change requires a new smoke; generated results/caches and Markdown documentation are excluded. `full_run_enabled=false` records the default blocked state; changing that boolean cannot bypass the gate. No automatic full run follows a smoke run. Full training starts fresh from the pinned pretrained encoder and seed-specific head initialization.

Before explicitly starting the full command, obtain a smoke-based estimate:

```bash
python -m scripts.estimate_four_task_runtime --config configs/baseline_roberta_four_task.json --smoke results/kaggle-t4x2-smoke
```

To select proportional scheduling, edit only `training.task_schedule` to `proportional` in a separate config copy. Keep every data path/count/manifest unchanged. The trainer derives scheduler steps from the selected plan and logs the resolved values; a matching new CUDA smoke is required for that config. Legacy descriptive `task_sampling`/`epoch_definition` and scheduler counts in the original config describe the default balanced schedule; `training.task_schedule` and the emitted `resolved` section control actual scheduling.

Outputs include resolved config, software/device details, dataset manifest hashes, per-epoch global batch plans, metrics/timings, best model plus optimizer/scheduler/scaler states, test metrics, and peak allocated VRAM per rank. Resume is intentionally not offered: stored optimizer state supports inspection, but a deterministic mid-run resume protocol is not implemented. Use a fresh output directory for reruns.
