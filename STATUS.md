# Pre-GPU status

READY_FOR_KAGGLE: YES — ready to attempt the tiny CUDA smoke, not cleared for full training.

## Implemented

One pinned shared `roberta-base` encoder and four task-specific 768→384→2 GELU classification heads. No SURGELLM components. Configurable proportional/task-balanced scheduling, deterministic task/rank batch plans, gradient accumulation, AdamW exclusions, optimizer-step warmup/linear decay, unscale-before-clipping, FP16 CUDA/FP32 CPU, strict-improvement early stopping, complete per-task evaluation, aggregate split ROC-AUC (null with a reason for one-class splits), best checkpoint restoration before test, and per-epoch metrics are implemented. CPU fixture smoke and two-process Gloo tests exercise the trainer without pretrained model training.

Default batch is 16 per GPU × 2 GPUs × 2 accumulation = 64 global examples per update. DDP synchronizes every backward because active heads change; no untested no-sync shortcut is used. The default balanced full epoch has 864 microbatches/rank, 432 optimizer updates, and 27,648 example exposures. Three epochs give 1,296 planned updates and 78 warmup updates. The proportional option derives its own counts (440 microbatches, 220 updates/epoch). Cycling/padding are logged as training exposures and do not alter split membership.

## Verification

The original 30 CPU tests passed. The added certificate invalidation regression first failed against the six-file fingerprint; it now checks other code/config additions and changes as well as resolved-config/data identity. **All 31 tests passed**, including the original 30. Final suite evidence: `results/pre-gpu-unit-tests.log`. Tests include all heads, shared encoder gradients from every task, routing, deterministic scheduling, combined-batch equivalence, actual two-process CPU DDP with changing heads, complete evaluation, single-class ROC-AUC, decay exclusions, stopping ties/reset, checkpoint save/load, end-to-end tiny fixture smoke, and certificate rejection.

Frozen integrity/preflight evidence: `results/baseline-preflight-pre-gpu.json`; static command/batch audit: `results/pre-gpu-verification.json`. Full-run gate remains BLOCKED without a valid matching CUDA smoke certificate. No CUDA execution, full training, or real seed experiment was run locally. Synthetic fixture metrics are not pretrained model results.

## Frozen data and existing assumptions — unchanged

Frozen train/validation/test counts remain D1=6,899/767/872; D2=1,400/300/300; D3=2,214/474/476; D4=3,500/750/750. Every locked manifest and split file is verified before execution. No raw files, split manifests, hashes, sampling rules, or provenance decisions were edited.

All prior limitations remain in `DATASET_PROVENANCE.md` and `REPRODUCTION_DECISIONS.md`: D1 split/cap ordering; D2 source/count/context/quota discrepancy; D3 proportional cap versus the paper's impossible balanced draw without replacement; recovered same-parent D4 subset rather than authors' exact draw. Baseline interpretation, head initialization/encoder dropout, global64 versus prose32, task scheduling/epoch definition, decay exclusions, warmup rounding, tokenizer defaults, full-validation aggregation versus unknown QuickVal, and fixed data seed remain documented assumptions. No author behavior was invented or performance tuned.

## Certificate and full-run gate

A successful smoke certificate records code/config/environment-file SHA-256 values, resolved configuration, frozen manifest hashes, Git HEAD when available, and an environment SHA-256 covering software versions, driver, GPU details, precision and world size. This directory currently has no Git checkout: that absence is explicit, and content hashes are authoritative. A later Git checkout/HEAD change requires a new smoke. Source/scripts/tests/config and project data-file additions, changes or removals invalidate the content fingerprint; changed frozen split bytes fail integrity checks even before the gate. Generated outputs, caches and Markdown documentation are outside the fingerprint so normal logging does not invalidate a run.

Full execution additionally compares the current environment hash. A missing/stale/malformed certificate or non-two-GPU runtime fails closed. Do not hand-edit the certificate or use CPU fixture reports as CUDA evidence. `full_run_enabled=false` remains the default state; manually changing it cannot bypass the certificate requirement.

## Exact GPU prerequisites and next steps

1. Copy the project with its unchanged frozen bundle to `/kaggle/working/surgellm-reproduction`; use that directory as the working directory. Do not run data preparation/freezing scripts.
2. Select two NVIDIA T4 GPUs, each 16 GB, visible to the same process environment. Use a compatible NVIDIA driver and the activated Python 3.10 / PyTorch 2.1.2 / CUDA 11.8 environment from `environment-gpu.yml`, including its pinned tokenizer, Transformers and sklearn packages. NCCL and torchrun must work. This target environment has not been installed/validated locally; local Torch is 2.6.0+cpu.
3. Provide a writable Hugging Face cache and Internet access to fetch the pinned encoder/tokenizer, or populate the exact pinned cache beforehand. Allow several GB of writable disk for model download, best checkpoints and logs. Both output directories below must be new.
4. Run only the tiny smoke command first. Inspect both ranks, FP16, 64 effective batch, four successful optimizer updates, complete evaluation, checkpoint restore, and recorded peak allocated bytes for each GPU. The smoke has eight microbatches and one tiny training epoch, but intentionally evaluates every validation/test row.

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.train_four_task --config configs/baseline_roberta_four_task.json --mode smoke --seed 0 --output results/kaggle-t4x2-smoke
```

5. Inspect `results/kaggle-t4x2-smoke/cuda_smoke_pass.json` and run `python -m scripts.estimate_four_task_runtime --config configs/baseline_roberta_four_task.json --smoke results/kaggle-t4x2-smoke`. Full execution remains a separate explicit action after a successful smoke. The prepared command is:

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.train_four_task --config configs/baseline_roberta_four_task.json --mode full --seed 0 --cuda-certificate results/kaggle-t4x2-smoke/cuda_smoke_pass.json --output results/baseline-roberta-seed0
```

See `KAGGLE_BASELINE.md` for interpretation and rerun instructions. A failed/OOM smoke must be diagnosed before any full run; do not silently lower batch size or change training semantics. Mid-run resume is not implemented; interrupted jobs require a fresh run directory.

## Expected VRAM and runtime

VRAM is not measured on CUDA yet. Approximately 125 million parameters require about 1.9 GiB/rank for FP32 parameters, gradients and two Adam moments alone; DDP buckets, autocast copies, activations, temporaries and CUDA context add more. A conservative planning allowance is roughly 6–10 GiB per T4 at batch16/length128, not a verified peak or fit guarantee. The smoke records peak PyTorch allocated VRAM per rank; this excludes allocator reserve and driver/context memory.

No defensible measured T4 runtime is available from the CPU fixture tests. Conditional training-only estimates for 27,648 exposures/epoch are:

| Assumed combined throughput | Per epoch | Three epochs |
|---|---:|---:|
| 50 examples/s | 9.22 min | 27.65 min |
| 100 examples/s | 4.61 min | 13.82 min |
| 200 examples/s | 2.30 min | 6.91 min |

These rates are scenarios, not measured T4 performance. Add complete validation/checkpointing each epoch, final test, tokenization and initial download. The post-smoke estimator extrapolates the actual eight-microbatch timing; first-step allocation/warmup can make that estimate conservative. Early stopping can shorten the three-epoch budget.

Remaining prerequisites before the real seed-0 run: provision the target two-T4 environment, pass the real FP16 smoke without skipped updates, inspect measured VRAM/timing, and explicitly launch the separate full command. No remaining local implementation blocker is known; CUDA/NCCL/AMP behavior remains execution-unvalidated.
