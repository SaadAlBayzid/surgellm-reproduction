# CPU runtime estimate and baseline configuration audit

No training was started for this audit. Inputs: `roberta-smoke-seed0/smoke_verification.json`, `data_manifest.json`, `runtime.json`; `configs/roberta_sst2.json`; `scripts/train_roberta_sst2.py`; `REPRODUCTION_PLAN.md`.

## Measured timing and estimate

The smoke timer recorded 105.048006773 seconds for 128 training examples (32 microbatches of 4), 24 validation examples, and best-checkpoint saving. It starts after model loading, tokenization and optimizer setup. Separate training, evaluation and checkpoint timers were not recorded, and microbatch logs lack timestamps. Therefore exact training-only throughput cannot be recovered.

Amortized throughput, using training work divided by the entire timed interval:

- 1.21849 training examples/second.
- 0.304623 training microbatches/second (batch size 4).
- 0.0380778 optimizer steps/second (4 optimizer calls; the initial warmup call has zero LR).

Selected `cap_then_holdout` policy: cap 7,666 source training rows, hold out ceil(10%) = 767, leaving **6,899 training examples**. Official validation's 872 examples remain the reserved test set. The saved manifest confirms these counts.

Full epoch: ceil(6899/4) = 1,725 training microbatches, ceil(1725/8) = 216 optimizer calls, and ceil(767/4) = 192 validation batches. A direct microbatch-scaled estimate is:

`105.048006773 * (1725 / 32) = 5662.744 seconds = 94.379 minutes per epoch`.

| Scenario | Estimated time | Valid under current maximum? |
|---|---:|---|
| One full epoch | 1 h 34 min | Yes |
| Maximum budget, 3 epochs | 4 h 43 min | Yes |
| Stop at end of epoch 3 | 4 h 43 min | Yes; coincides with budget exhaustion |
| Stop at end of epoch 4 | 6 h 18 min | No; hypothetical extended budget only |
| Stop at end of epoch 5 | 7 h 52 min | No; hypothetical extended budget only |

These are approximate, conservatively biased planning estimates, not measured full-epoch runtimes or statistical bounds. Scaling the whole smoke interval repeats checkpoint overhead and assumes 323 rather than 192 validation batches per full epoch, tending to overestimate at unchanged compute speed. CPU contention, memory pressure and sustained thermal throttling could increase actual runtime. Initial model/data loading and tokenization, and eventual reserved-test evaluation, are not separately included. No unsupported training/evaluation timing breakdown is inferred.

## Config audit against the plan

There is **no runnable resolved full-run config yet**. The config records `max_epochs=3`, but the entry point requires `--smoke`, subsamples data, calculates scheduler steps from `smoke_epochs`, and runs only `smoke_epochs=1`. This audit derives the prospective full-run settings without changing that guard or launching anything.

| Setting | Prospective seed-0 setting | Plan reference / verdict |
|---|---|---|
| Model | roberta-base, binary classifier | Backbone matches section 5; standalone SST-2 is user-selected scope |
| Seed | 0 | In paper seeds {0,1,2}, section 5 / Appendix I |
| Epoch maximum | 3 | Matches Baseline-RoBERTa Table 13; retain despite Appendix H's inconsistent five-epoch prose |
| AdamW | LR 2e-5; decay .01; betas .9/.999; epsilon 1e-8 | Matches section 5 and Table 13 |
| Sequence length | 128 including special tokens | Length matches Table 13; exact tokenization is a temporary choice |
| Schedule | Linear warmup .06 then decay | Matches section 5; chosen rounding yields 39 warmup steps of 648 total for 3 epochs |
| Clipping | Global gradient norm 1.0 | Matches section 5 |
| Early stopping | Strict validation macro-F1 improvement, patience 2, evaluated each epoch | Matches Appendix K; single-task full validation is a temporary interpretation of QuickVal |
| Microbatch / accumulation | 4 / 8; single device, effective 32 | Temporary choice matching section 5 global 32, not Table 13 per-GPU 16/2 on two GPUs (implied global 64) |
| Execution | CPU FP32, four threads, PyTorch 2.6.0+cpu, Transformers 4.35.2 | Documented deviation from two T4 GPUs, FP16, Accelerate and PyTorch 2.1 in Appendix I |

The last optimizer window contains 19 examples instead of 32 and is normalized by its actual example count. A real full-run path would need 648 total optimizer calls and 39 warmup calls rather than smoke's 4/1, best-checkpoint restoration before reserved-test evaluation, and exclusion of smoke subsampling. None of those full-run changes were made in this audit.

## Material unresolved comparisons

1. **Split/cap ambiguity:** current train=6,899 follows cap-before-holdout, not a final training size of 7,666. Section 4.1 and the generic 70/15/15 protocol conflict; exact author sampled IDs, rounding and RNG are unknown. Actual minority percentage is not forced to Table 2's 49.5%. [Plan section 1; paper sections 4.1-4.2, Table 2]
2. **Single-task versus paper setup:** this run deliberately excludes multi-task training. The paper's Table 3 SST-2 figure .929 +/- .004 is within its four-task evaluation, not an established identical single-task protocol. Seed 0 alone also cannot reproduce a three-seed mean/SD. [Plan sections 2, 4, 5]
3. **Classification head/dropout:** chosen standard Hugging Face head is 768->768->2 with tanh and dropout .1 at both sites. Paper section 3.7 describes 768->384->2 with GELU and .1/.05 dropout; applicability to plain baseline is unclear. Encoder defaults are retained. [Plan section 2]
4. **Batch semantics:** global 32 versus implied 64 is unresolved. Microbatch/accumulation also differs from Table 13. [Plan section 3]
5. **Numerics/software:** CPU FP32 and newer PyTorch, no Accelerate/DDP, deterministic CPU settings; author patch versions/backend settings remain unknown. [Plan sections 3, 7]
6. **Optimizer/scheduler details:** decay currently applies to biases and LayerNorm parameters as well; warmup uses ceiling; final short window uses actual count. Author parameter exclusions and rounding are unspecified. [Plan section 3]
7. **Tokenization/revisions:** raw casing preserved, fast tokenizer, right fixed-length padding/truncation, add_prefix_space=False; downloaded model/data hashes are pinned for our run but author revisions/settings are unknown. [Plan sections 1-2]
8. **Evaluation/checkpoint selection:** use the entire 767-row internal-validation set, argmax predictions, fixed classes [0,1], zero-division=0. Paper QuickVal coverage and aggregation are underspecified. Current metrics cover loss/accuracy/macro-F1 only, not every paper metric. A comparable final score must use the reserved 872 examples after best-checkpoint restoration; internal-validation F1 is not the Table 3 test score. [Plan section 4]

Learning-rate conflicts for FULL-ALBERT and IWN-BERT do not apply to this plain RoBERTa baseline. Surgical-feature and IWN ambiguities likewise do not apply to the requested scope.
