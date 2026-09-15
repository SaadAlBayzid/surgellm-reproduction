# Dataset provenance and frozen experiment v1

The data freeze is a reproducible **best-effort protocol**, not a claim to have recovered the authors' exact samples. Data seed 0 is fixed across all model variants and training seeds 0/1/2. No model-dependent sampling is permitted. `data/frozen/v1/experiment.lock.json` pins task manifests, which pin split bytes and source IDs. All loaders emit `text`, binary `label`, `task_id`, `source_id`, and zero-based `row_index`.

## Sources and preprocessing

| Task | Canonical source / exact local version | Raw counts | Used columns and mapping | Sampling / final count | Limitations |
|---|---|---|---|---|---|
| D1 SST-2 | [GLUE SST-2](https://huggingface.co/datasets/stanfordnlp/sst2), revision `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`; existing converted TSV files | Train 67,349: 29,780 negative / 37,569 positive. Official dev 872: 428 / 444. | `sentence` preserved verbatim; `label`: negative=0, positive=1. | Reuse historical debugging manifest exactly: 7,666 source-train cap -> 6,899 train / 767 internal val; official dev 872 test. | Paper section 4.1 and generic 70/15/15 conflict; cap order remains unknown. Our cap is 3,390/4,276, not Table 2's stated 49.5% minority. Debug setup unchanged. |
| D2 HotPotQA | [canonical project](https://hotpotqa.github.io/), [HF release](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab), `distractor/train` two parquet shards | 90,447 actual rows: easy=17,972, medium=56,814, hard=15,661; binary 17,972/72,475. | `id`, `question`, `context`, `supporting_facts`, `level`; easy=0, medium/hard=1. `[Q]question[CTX]context`; join sentences from gold-supporting paragraph titles in source order, omit titles, retain first 300 whitespace words. | SHA256-ranked quota 980 easy + 1,020 medium/hard; 2,000 total; stratified 70/15/15 -> 1,400/300/300. | Paper calls 90,564 rows validation; pinned source has 90,447 train. Initial 90,564 assertion failed and was explicitly replaced with source-card count. Supporting-paragraph assembly and quota matching Table 2's 49% are assumptions, not author code. No alternative task/dataset used. |
| D3 LLM-7 | [Carl McBride Ellis, LLM: 7 prompt training dataset](https://www.kaggle.com/datasets/carlmcbrideellis/llm-7-prompt-training-dataset?select=train_essays_7_prompts.csv); Kaggle API **datasetVersionNumber=1**, file `train_essays_7_prompts.csv` | **14,877: human 13,712; AI 1,165**. Human/AI ratio 11.77:1. | Exact columns `text`,`label`; human=0, AI=1; full raw text/punctuation retained. Interpretation explicitly selected by user. | Proportional largest-remainder cap without replacement: **2,916 human + 248 AI = 3,164**; 70/15/15 -> 2,214/474/476. | Paper Table 2 reports balanced 3,164, but this version has only 1,165 AI rows (<1,582 required). We do not fabricate/duplicate AI rows. Loader rejects changed raw total or class counts; frozen hash pins content, not counts alone. Later CSV versions excluded. |
| D4 HumLLM | [Zachary Grinberg, Human vs. LLM Text Corpus](https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus); supplied `ai_human_sample_10k.csv` recovered sample from parent 788,922 | Local **10,000: human 4,452; LLM 5,548**. Parent class counts and exact original version not independently recoverable here. | `text`,`label`; check `source`: Human -> 0, every other source -> 1. `prompt_id`,`text_length`,`word_count` ignored; labels validated against source. | SHA256-ranked selection **2,500 human + 2,500 LLM = 5,000** without replacement; train 1,750+1,750, val/test each 375+375. | User documents same-parent recovery. No parent-row IDs/version evidence exists locally; stable IDs pin recovered CSV row indices, not original parent IDs. This is **not the authors' unrecoverable exact 5,000-row draw**. Parent artifact unavailable per user. Paper 9.3:1 cannot be inferred from recovered sample; no reweighting to manufacture it. |

The D4 source-to-label mapping is also described by the [dataset creator](https://www.kaggle.com/starblasters8/discussion). The local `surgellm_reproduction.zip` found in Downloads is a separate reproduction package, not authenticated author code; it was not executed or merged.

## Frozen counts (class 0 / class 1)

| Task | Train | Validation | Test | All frozen rows |
|---|---|---|---|---:|
| D1 | 6,899 (3,051 / 3,848) | 767 (339 / 428) | 872 (428 / 444) | 8,538 |
| D2 | 1,400 (686 / 714) | 300 (147 / 153) | 300 (147 / 153) | 2,000 |
| D3 | 2,214 (2,040 / 174) | 474 (437 / 37) | 476 (439 / 37) | 3,164 |
| D4 | 3,500 (1,750 / 1,750) | 750 (375 / 375) | 750 (375 / 375) | 5,000 |
| Total | 14,013 | 2,291 | 2,398 | 18,702 |

Paper caps total 17,830; our frozen all-partition total is 18,702 because D1 includes its separate 872 official-dev test examples. This distinction is intentional and recorded, not hidden.

For D2-D4, the rank key is SHA256 of `seed|namespace|source_id`. Quotas use proportional largest remainder with label-order tie-break. Splits use floor(70% N), floor(15% N), remainder, with stratified quotas at each stage. D4 source IDs incorporate the raw CSV hash and row index; selecting rows again with the same input cannot change the draw. Existing bundles are byte-verified and cannot be overwritten by `freeze_data.py`. All future trainers must read the lock, not call sampling functions for each model/seed.

No empty text or duplicate IDs are accepted. Exact-text overlaps between splits within each task were checked and are zero. This does **not** prove absence of semantic duplicates, prompt-family overlap, or cross-corpus overlap; original group metadata and author deduplication rules are unavailable. No text cleaning/deduplication was silently applied.

Machine-readable raw statistics and hashes: `results/data_audit.json`. Raw D1 and D2 files are retained in the workspace `work/` paths listed below; D3/D4 raw CSVs are under project `data/raw/`. Frozen JSONL bundles are self-contained for copying the project to a GPU machine.


## SHA-256 inventory of local raw data artifacts

| Path relative to project | SHA-256 |
|---|---|
| ../../work/roberta-assets/train.tsv | 680bb74f87741f136dbfb7c5ea26319e7f099a9337c91c0959e277b8f637b008 |
| ../../work/roberta-assets/dev.tsv | ff5fbb5435d9ac6c7dc1772f4ec458a5d78a1ec30e4c59ef907d96efdd8669f6 |
| ../../work/data-downloads/hotpot/train-00000-of-00002.parquet | 76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68 |
| ../../work/data-downloads/hotpot/train-00001-of-00002.parquet | 713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f |
| data/raw/train_essays_7_prompts.csv | 4f03a2d745b28ccaf36a110dac06167f3193e8478b8279a00f95dac91a1ff8fe |
| data/raw/ai_human_sample_10k.csv | a00a8412eb7b4bcd2673212d91139689002aa6f7c72315d738f9a7cb4dcad0ef |
| ../../work/data-downloads/d3_v1.zip | b59efeaa4e2db09c4f3564ef3d2f06aa2246e3a268ed8c4439da37cd0d036e70 |
| ../../work/roberta-assets/data/train-00000-of-00001.parquet | c7921283b75a42e685f50edecb96798607ea0fcbfd0739ee8975f22c12d55f09 |
| ../../work/roberta-assets/data/validation-00000-of-00001.parquet | fb00fe008f6828f86ba2beda8415a4cf5da0c884f21c5f238c87131b5aa19529 |
