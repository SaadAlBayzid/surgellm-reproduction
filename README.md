# SURGELLM reproduction

Current state: **four-task baseline trainer implemented; CUDA smoke required before full training**. Frozen data and provenance are unchanged. Start with [STATUS.md](STATUS.md) and [KAGGLE_BASELINE.md](KAGGLE_BASELINE.md) for pre-GPU checks and exact commands. The SST-2-only pipeline remains a separate debugging configuration.

The experimental specification is in [REPRODUCTION_PLAN.md](REPRODUCTION_PLAN.md). The source is the user-uploaded [paper](paper/2026.trustnlp-main.47.pdf).

Project directories: `paper/`, `src/`, `scripts/`, `configs/`, `tests/`, and `results/`.

The paper contains conflicting experimental details. The plan preserves them as `AMBIGUOUS`; it does not claim that exact reproduction is currently possible.

The SST-2 data layer provides TSV loading, explicit split alternatives, source-ID preservation, input hashes, overlap reporting and tests. The single-task classifier is retained for debugging; the four-task baseline is now implemented separately. No surgical features, prefixes, gates, or IWN are included.

Run tests from this folder: `python -m unittest discover -s tests -v`.

Prepare user-supplied labeled GLUE TSV files:

```sh
python -m scripts.prepare_sst2 --train /path/train.tsv --dev /path/dev.tsv --policy cap_then_holdout --seed 0 --output results/sst2-seed0
```

The example explicitly selects one interpretation; it is not an author-verified default. Alternatives are `holdout_then_cap` and `capped_70_15_15`. See `src/sst2.py` for rounding rules. All are engineering interpretations of an ambiguous paper. The manifest marks `exact_author_split` false. Source text remains unchanged, invalid/empty records raise errors, and duplicate text across splits is reported without silently deleting it.

## Plain RoBERTa SST-2 baseline

`configs/roberta_sst2.json` holds all training hyperparameters, smoke sizes, and an explicit `temporary_choices` register. Paper settings retained: LR 2e-5, AdamW beta/epsilon/weight decay, max length 128, 6% warmup, linear decay, clipping 1.0, and patience 2. Three epochs are recorded for the future baseline; the runnable entry point **requires `--smoke` and only runs one epoch**. Full-experiment execution is intentionally unavailable at this stage.

This is Hugging Face `roberta-base` with its standard binary sequence-classification head and full encoder fine-tuning. The standard head is an explicitly documented temporary choice because the paper's task-specific head description does not unambiguously specify the plain baseline. This standalone SST-2 smoke result is not a reproduction of the paper's four-task Table 3 row.

The current smoke runtime is CPU/FP32 with microbatch 4 and accumulation 8 (global 32), using Python 3.9, PyTorch 2.6.0+cpu, and Transformers 4.35.2. These runtime deviations from the original two T4 GPUs/FP16/PyTorch 2.1 are recorded. Dependencies are listed in `requirements-baseline.txt`. A local environment was prepared under the parent workspace's `work/roberta39/`; it inherits existing CPU PyTorch, with other pinned packages isolated in that environment.

From this project root, with the baseline environment active and `HF_HOME` set to a writable cache outside deliverables:

```sh
python -m scripts.fetch_roberta_assets --output /path/to/work/roberta-assets
python -m scripts.convert_sst2_assets --assets /path/to/work/roberta-assets
python -m scripts.train_roberta_sst2 --smoke --assets /path/to/work/roberta-assets --seed 0 --output results/roberta-smoke-seed0
python -m unittest discover -s tests -v
```

Assets come from Hugging Face `roberta-base` and `stanfordnlp/sst2`. Downloading source data does not run the full experiment. The loader uses 128 sampled training rows and 24 internal-validation rows for smoke; official GLUE validation is reserved, not evaluated. Model and dataset commit IDs and hashes are saved in `asset_provenance.json`; exact selected IDs and split choice are in `data_manifest.json`.

Each new output directory contains `resolved_config.json`, `runtime.json`, `training_log.jsonl`, `validation_metrics.json`, `smoke_verification.json`, and `best_model.pt`. Logs record every microbatch loss/LR and each optimizer step's gradient norm. Validation computes example-weighted loss, accuracy and fixed-two-class macro-F1. Smoke verification checks tokenization, shapes, finite outputs, backward, and changed classifier **and encoder** weights. The checkpoint is a state dictionary for the configured Hugging Face model, not a complete resume-state bundle.

Unit tests use a tiny random RoBERTa only to check evaluation weighting without downloads; the smoke test separately loads the actual pretrained 125M-parameter checkpoint. Never interpret smoke accuracy/F1 as experimental performance.

Completed run: [seed-0 smoke summary](results/roberta-smoke-seed0/SUMMARY.md). One epoch on 128/24 examples passed every requested check; validation loss 0.688422, accuracy 0.541667, macro-F1 0.351351. All seven unit tests passed. The full experiment has not been run.
