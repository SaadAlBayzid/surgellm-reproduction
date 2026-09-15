# Central reproduction configuration (implemented; CUDA gate pending)

Machine-readable authority: `configs/baseline_roberta_four_task.json`. Dataset authority: `data/frozen/v1/experiment.lock.json`. This is the exact **best-effort target** implemented by the baseline trainer, not an exact reconstruction of unknown author code. Assumptions are marked below and in the JSON. Only CPU fixture tests have run; full training remains gated.

| Setting | Target value | Evidence / status |
|---|---|---|
| Tasks | D1 SST-2, D2 binary HotPotQA difficulty, D3 human/AI LLM-7, D4 human/LLM recovered HumLLM | User decisions; provenance verified as documented |
| Encoder | Shared roberta-base; revision e2da8e2f811d1448a5b465c236feacd80ffbac7b | Revision is assumption, original author checkpoint unknown |
| Architecture specification | Four binary heads, 768->384->2 GELU, first-token input; no task embedding or surgical components | Paper-derived interpretation; implemented in the baseline trainer |
| Head dropout / initialization | .1 before first linear, .05 before second; normal(0,.02) weights, zero biases | Dropout from section3.7; initialization assumption |
| Encoder dropout | Hidden .1, attention .1 | Checkpoint defaults; assumption |
| Training seeds | 0,1,2 | Paper; none launched |
| Data seed | Fixed0 for every model/training seed; use frozen membership | Explicit reuse requirement; differs from possibly seed-specific author splits |
| Optimizer | AdamW; LR2e-5; betas(.9,.999); epsilon1e-8 | Section5, Table13 |
| Weight decay | .01 except bias and LayerNorm.weight | Scalar from paper; exclusions assumption |
| Max epochs | 3 | Baseline Table13; not5 |
| Runtime | Two CUDA devices, FP16 autocast/gradient scaling; native torch.distributed/DDP | Paper reference; GPU machine required and CUDA execution not yet validated |
| Batch/accumulation | 16 examples per rank,2 accumulation steps,2 ranks =>64 global | **Assumption:** Table13 precedence over prose effective32 |
| Task sampling | Task-homogeneous batches, same task on each rank; D1-D4 round-robin; shuffle each task and cycle smaller tasks; deterministically pad tails | **Assumption**, not author code |
| Epoch definition | 216 task rounds: ceil(largest training set6,899 / global microbatch32);4 task batches per round | **Assumption** based on balanced tasks |
| Update count | 864 microbatches/rank/epoch;432 optimizer updates/epoch;1,296 over3 epochs | Derived from specified sampler; AMP-skipped steps may reduce actual updates |
| Schedule | Linear warmup then linear decay;ceil(.06*1296)=78 warmup updates; advance only after real optimizer updates | .06/linear from paper; rounding/skip semantics assumption |
| Gradient clipping | Global norm1.0 after AMP unscale | Section5 |
| Max lengths | D1-D4 all128 including special tokens; D2 context first capped300 words | Table13/section4.2 |
| Tokenizer | Fast; original case/text, right padding to128/right truncation; add_prefix_space=False; no new prefix vocabulary | **Assumption**; selected checkpoint revision fixed |
| Early stopping | Evaluate every epoch; strictly greater mean task macro-F1; patience2; restore best before reserved tests | Rule from AppendixK; full-val/unweighted-task reduction assumption |
| Metrics | Example-weighted loss per task; accuracy; macro-F1/precision/recall over fixed labels[0,1], zero_division0; ROC-AUC using P(class1); mean task scores | Metrics from paper; edge-case/aggregation conventions assumed |
| Test protocol | Best validation checkpoint, evaluate each frozen test partition; report per-seed and mean/sample-SD across3 seeds | Best checkpoint from paper; sample-SD convention assumption |
| Software | Python3.10, Torch2.1.2/CUDA11.8, Transformers4.35.2, Accelerate.24.1, sklearn1.3.2 | `environment-gpu.yml`; patch/runtime assumptions |

Frozen train sizes: D1=6,899, D2=1,400, D3=2,214, D4=3,500. Validation=767/300/474/750; test=872/300/476/750. The pooled training size14,013 does not set epoch length under the assumed equal-task cycling policy. Smaller-task repeated exposures belong to the training sampler, not newly generated split membership. Never rebuild data splits per backbone/variant/seed.

The four-head model, both batch scheduling policies, DDP training loop, complete-split metrics and best-checkpoint restoration are implemented. Actual scheduler totals are derived from the selected batch plan and saved under `resolved`; tabulated counts above describe the default task-balanced configuration. CPU tests include two-process DDP and a tiny end-to-end fixture smoke. Mid-run resume is not offered.

The full-run gate requires matching two-T4 FP16 smoke evidence. Prepared commands and current prerequisites are in `KAGGLE_BASELINE.md` and `STATUS.md`. Existing dataset/protocol assumptions are unchanged; implementation does not resolve unknown author behavior.
