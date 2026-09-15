# CUDA preparation (full experiment blocked)

Update: frozen data and the four-head trainer are implemented. See STATUS.md and KAGGLE_BASELINE.md for the authoritative two-T4 commands and certificate gate. The single-head asset smoke instructions below remain debugging-only. CUDA execution is still unvalidated locally.

Create an isolated environment on the target CUDA machine:

```sh
conda env create -f environment-gpu.yml
conda activate surgellm-gpu
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The file selects Python 3.10, PyTorch 2.1.2/CUDA 11.8, Transformers 4.35.2 and Accelerate 0.24.1: patch-level choices within the paper's stated minor versions, not an author lockfile. Python/CUDA versions were not specified by the paper. Use a compatible NVIDIA driver on the target machine. See [PyTorch archived installation combinations](https://pytorch.org/get-started/previous-versions/) and [AMP guidance](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html). Environment file has not been installed or tested on a GPU in this session.

Set `HF_HOME` to a writable cache. Copy the existing `work/roberta-assets` directory from the development workspace, or fetch and convert it using the README's asset scripts. `--assets` below must name that directory (containing `model/`, `train.tsv`, `dev.tsv`, `provenance.json`).

```sh
python -m scripts.smoke_cuda --assets /path/to/roberta-assets --output results/cuda-tiny.json
```

This performs at most two microbatch-1 optimizer calls at length 128 with real pretrained RoBERTa, autocast FP16, GradScaler, unscale-before-clipping and evaluation. It logs peak allocated/reserved VRAM. Without CUDA it writes **SKIPPED**, not PASSED. This is a debugging check, not a four-task experiment or dataset benchmark. It is not automatically scheduled to run in the future; rerun the command on the CUDA machine.

The SST-2 debug pipeline is separately runnable with CPU FP32 or CUDA FP16:

```sh
python -m scripts.train_roberta_sst2 --config configs/debug_sst2_roberta.json --smoke --device auto --precision auto --assets /path/to/roberta-assets --seed 0 --output results/debug-sst2-auto-seed0
```

`auto` uses CUDA/FP16 only if `torch.cuda.is_available()`; otherwise CPU/FP32. Explicit unavailable CUDA and CPU/FP16 raise errors. CPU smoke support remains intact. The config's stored CPU default preserves the original debug behavior unless these flags are supplied. AMP overflow skips optimizer and scheduler updates rather than advancing the scheduler on a skipped step. No DDP/four-task execution has been implemented: it depends on unresolved batch and task-sampling semantics.

## Seed-0 scientific preflight

```sh
python -m scripts.preflight_baseline --config configs/baseline_roberta_four_task.json --seed 0
```

This exact command is intentionally **preflight only** and returns exit code 2. It now verifies frozen four-task hashes/counts and lists documented assumptions and implementation blockers. The four-head model/trainer and distributed integration are still absent. Do not convert this into a training command by merely flipping `full_run_enabled`. Implement and validate the complete four-task path against FINAL_REPRO_CONFIG.md after reviewing its best-effort decisions. No full experiment is automatically launched by environment creation or any readiness check.

## VRAM estimate

For roughly 125M parameters plus four small heads, FP32 master/model weights + gradients + two Adam moment buffers require approximately `125M * (4+4+8) = 2.0 GB` (about 1.86 GiB) before activations, attention, FP16 cast caches, temporary optimizer buffers, CUDA context and allocator reservation. AMP does not halve FP32 optimizer state. Some optimizer implementations temporarily allocate another parameter-sized buffer. DDP replicates model/optimizer state per GPU; two 16-GB GPUs do not become one 32-GB pool.

Planning estimate, **not measured**: allow roughly 6-10 GiB per GPU at sequence length 128 and per-device batch 16; 16 GiB per device is a sensible reference capacity because the paper reports two 16-GB T4s. Exact peak depends on attention implementation, DDP buckets, optimizer and batch policy. Microbatch 1 tiny smoke may fit roughly 3-4 GiB but is not guaranteed on a display-attached 4-GB card. Do not silently lower the scientific batch size to fit hardware; gradient accumulation and task sampling must be decided explicitly.

Current hardware observation: GTX 1650 Ti, 4,096 MiB, driver 516.91 (`nvidia-smi` reports CUDA 11.7 compatibility); installed test environment is PyTorch 2.6.0+cpu. A physical NVIDIA GPU is present but CUDA is unavailable to this Python runtime. The CUDA smoke probe records that limitation; no driver/environment replacement was performed.
