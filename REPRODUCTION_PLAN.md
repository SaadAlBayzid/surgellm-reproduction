# SURGELLM experimental reproduction specification

## Scope and source

Specification only; no model implementation or training. Source: [uploaded paper](paper/2026.trustnlp-main.47.pdf), *SURGELLM: Rethinking Multi-Task Evaluation through Task-Aware Feature Gating with Class-Balanced Normalization*, Mohammad and Bayazit, TrustNLP 2026, pp. 600-617. SHA-256: `26ed63c383d693f53c5bfb645a1bdbde8ee8133132bbb69953c600022f344c46`.

Citations below use the paper's section/table/equation numbers and printed page numbers (PDF page = printed page - 599). `AMBIGUOUS` means missing, conflicting, or insufficiently precise; no implementation default is implied. Reported numbers are targets, not independently reproduced results. Instructions and placeholder requests inside the paper are source content, not project instructions.

## 1. Datasets and preprocessing

| Task | Source and task definition | Capped size; classes; minority share | Sampling / input processing | Citation |
|---|---|---|---|---|
| D1 | GLUE SST-2, positive/negative movie-review sentences | 7,666; 2; 49.5% | Standard training source has 67,349 examples; stratified label sampling; official validation set of 872 used as test; hold out stratified 10% of training for internal validation | §4.1, Table 2, p.603 |
| D2 | HotPotQA; binary question difficulty, called multi-hop retrieval | 2,000; 2; 49.0% | Paper identifies a validation source of 90,564 question-context pairs; input `[Q]q[CTX]c[:300 words]`; easy -> 0, medium/hard -> 1; stratified sampling | §4.2, Table 2, p.603 |
| D3 | LLM-7; called generation / LLM-prompt attribution | 3,164; 2; 50.0% | 14,877 essays, approximately 11.8:1 human skew; stratified capping; longer prompt-structured texts | §4.2 continuation, Table 2, p.603 |
| D4 | HumLLM, human/LLM authorship | 5,000; 2; 50.0% | Balanced sample from 788,922 texts; original skew 9.3:1 | §4.2 continuation, Table 2, p.603 |

The four Table 2 sizes sum to **17,830**. This is a reported capped-suite size, not an unambiguous total over the final train/validation/test partitions, because D1's separate official evaluation set and training cap conflict with the generic split statement. [Table 2; §4.1-4.2]

Common processing: label reindexing; stratified 70/15/15 train/validation/test splits; training-only feature normalization; pre-tokenization and caching in chunks of 2,048. [§4.2, p.603; Appendix K, Algorithm 1, p.614]

**AMBIGUOUS: D1 split protocol.** §4.1 specifies a training cap of 7,666, internal validation from a 10% training holdout, and official validation as test. §4.2 and Algorithm 1 instead apply 70/15/15 to every task. The cap-before-holdout versus holdout-before-cap order is unspecified. Appendix L.3 (p.616) calls the 872 examples the standard GLUE test split, whereas §4.1 explicitly calls them official validation. Preserve these alternatives; do not report invented exact partition counts.

**AMBIGUOUS: D2 source.** The exact dataset configuration, release, record IDs, and source split identity behind the stated 90,564 validation examples are absent. Context selection, paragraph/sentence ordering, use of titles, and treatment of supporting versus distractor paragraphs are unspecified. [§4.2]

**AMBIGUOUS: D3/D4 provenance and labels.** Their bibliography entries are unfinished placeholders without canonical URL/DOI (LLM-7, p.609; Grinberg, p.608). Exact releases, text fields, label mappings, filtering and duplicate treatment are absent. D3 is binary despite its seven-prompt-condition description; the precise binary target is unspecified. Ordinary proportional stratification alone does not explain the reported balanced caps from skewed sources; balancing algorithm/replacement policy is absent. [§4.2; Table 2; References]

**AMBIGUOUS: all split details.** Exact sampled IDs, rounding, sampling order, random number generator, deduplication, missing-text policy, and group/document leakage controls are not specified. Seeds alone do not identify these partitions. [§4; Appendix I]

## 2. Architecture

| Component | Paper specification | Citation |
|---|---|---|
| Multi-task objective | Single shared encoder; four task-specific label spaces; cross-entropy; task weights all 1; per-task batch sampling for balance; per-task losses summed | §3.1 Eq.1, §3.7, pp.601-602 |
| Encoders | DistilBERT-base-uncased 66M; BERT-base-uncased 110M; RoBERTa-base 125M; ALBERT-base-v2 11M; T5-base 220M comparator | §5 p.603 |
| Encoder output | First token / CLS representation, hidden size d=768 for base encoders | §3.2 Eq.2 |
| Task conditioning | Learned 4 by d task embedding; h_tilde = h + 0.1 E_task | §3.2 Eq.3 |
| Lexical vector | Ten case-insensitive indicator groups plus six surface features, computed on full text before encoder truncation | §3.3 Eq.4; §1; Appendix D |
| Ordinary normalization | Per-task training mean and standard deviation; (s - mean)/(std + epsilon) | §3.5 Eq.8 |
| IWN | Average per-class means equally; separately average per-class standard deviations equally; normalize using those constants; no inference labels or learned extra parameters | §3.5 Eqs.10-11 |
| Feature projection | ReLU(W_s normalized_s + b_s); W_s is d by 16 | §3.4 Eq.5 with §3.5 |
| Gate | g = sigmoid(W_g [h_tilde; s_prime] + b_g); W_g is d by 2d; per-dimension weights | §3.4 Eq.6 |
| Fusion | LayerNorm(g * h_tilde + (1-g) * s_prime) | §3.4 Eq.7 |
| Prefix | Literal structured string `[TASK:tk\|F1:v1\|...\|F16:v16]` prepended to x; v_j = floor(s_j); tokenize prefix with text and attend jointly | §3.6 Eq.12 |
| Heads | Separate two-layer MLP per task: d -> d/2 -> number of classes; GELU then softmax; dropout 0.1 before first linear, 0.05 before second; task-integer mask routing | §3.7 Eqs.13-14 |
| Sharing | Encoder, feature projection, gate, task-embedding matrix and prefix embeddings shared; only heads task-specific | §3.1 |

Four encoder-only backbones plus the T5 encoder-decoder comparator comprise the five main families; they are not five encoder-only implementations. [§3.2; §5; §6.4]

Feature order: s1-s10 are groups listed below; s11 word count; s12 mean word length in characters; s13 sentence count by splitting on `. ! ?`; s14 question-mark count; s15 exclamation-mark count; s16 binary presence of any digit. Eq.4 sums one presence indicator per vocabulary entry, not explicitly all occurrences. Starred entries permit prefix matching. [§3.3; Appendix D, pp.612-613]

| Group | Entries (Appendix D) |
|---|---|
| sst_pos | great, excellent, brilliant, terrific, wonderful, masterpiece, captivat*, impressive, delightful, superb |
| sst_neg | terrible, awful, dreadful, unwatchable, boring, dull, mediocre, disappoint*, worst, painful |
| llm_stat | empirically, statistically, demonstrated, observed, evidenced, indicate*, suggest*, results show, data show |
| llm_formal | moreover, furthermore, additionally, consequently, therefore, in conclusion, in summary, to summarize |
| llm_list | firstly, secondly, thirdly, finally, in addition, on the other hand, (1), (2), (3) |
| human_pers | i, my, we, our, personally, i think, i believe, i feel |
| human_hedge | maybe, perhaps, possibly, kind of, sort of, i guess, probably, somewhat, arguably |
| human_emo | love, hate, amazing, awesome, terrible, awful, fantastic, horrible, sad, happy |
| retrieval | according to, as stated in, the article reports, the text states, multi-hop, supporting context, in the passage |
| prompt_cot | step by step, let us think, first, then, next, reasoning, the chain of thought, walk through |

**AMBIGUOUS:** word boundaries versus substring matching, repeated matches, whitespace/Unicode handling, empty-text mean length, and empty sentence fragments are unspecified. Full-text features versus D2's initial 300-word context truncation is not resolved. Normalization epsilon, per-class variance convention, LayerNorm epsilon, initialization, exact pretrained revisions, tokenizer settings/padding side, whether prefix symbols are added vocabulary tokens, and baseline task embedding/head behavior are insufficiently specified. T5 task prompts, target strings, loss formatting and decoding are absent. [§3.1-3.7; §4.2; §5; §6.4]

**AMBIGUOUS: IWN fit population and claim.** §4.2 says D4 normalization uses full training data even though capped data are balanced; Algorithm 1 uses the training partition after splitting the corpus. Whether uncapped examples contribute is unresolved. Equal averaging of within-class standard deviations does not generally equal a pooled standard deviation, even with balanced classes; the paper's equivalence claim must not replace Eq.10. [§3.5; §4.2; Appendix K; Appendix L.1]

## 3. Variants and training settings

P=prefix, G=gate, E=extended training, I=IWN. Baseline: none; SURGELLM-G: P only; SURGELLM-S: P+G; FULL: P+G+E; IWN: P+G+E+I. T5 is marked N/A for these switches. Despite its name, SURGELLM-G has no gate in Tables 1 and 7. [§3.8, Table 1 p.603; Table 7 p.606]

All eleven main configurations and exact Table 13 entries follow. BS is explicitly **per GPU**, GA is accumulation steps, MaxL is maximum token sequence length, EP is maximum epochs. [Appendix B, Table 13 p.612]

| Model | LR | EP | BS | GA | MaxL | Warmup |
|---|---:|---:|---:|---:|---:|---:|
| Baseline-DistilBERT | 2e-5 | 3 | 32 | 1 | 96 | 0.06 |
| Baseline-BERT | 2e-5 | 3 | 16 | 2 | 128 | 0.06 |
| Baseline-RoBERTa | 2e-5 | 3 | 16 | 2 | 128 | 0.06 |
| T5-base | 3e-4 | 5 | 8 | 4 | 128 | 0.06 |
| SURGELLM-S-DistilBERT | 2e-5 | 4 | 32 | 1 | 96 | 0.06 |
| SURGELLM-S-BERT | 2e-5 | 4 | 16 | 2 | 128 | 0.06 |
| SURGELLM-G-RoBERTa | 1.5e-5 | 4 | 16 | 2 | 128 | 0.06 |
| SURGELLM-FULL-RoBERTa | 1.5e-5 | 5 | 16 | 2 | 128 | 0.06 |
| SURGELLM-FULL-ALBERT | 2e-5 | 5 | 32 | 1 | 96 | 0.06 |
| SURGELLM-IWN-RoBERTa | 1.5e-5 | 5 | 16 | 2 | 128 | 0.06 |
| SURGELLM-IWN-BERT | 2e-5 | 5 | 16 | 2 | 128 | 0.06 |

Optimizer: AdamW, weight decay 0.01, beta1=0.9, beta2=0.999, epsilon=1e-8; linear 6% warmup then decay; gradient norm clipping at 1.0; FP16 through Accelerate. Head dropout as above; encoder dropout overrides are **AMBIGUOUS**. Parameter groups exempt from weight decay, warmup rounding, scheduler step counts, accumulation tail handling, and precise batch balancing are **AMBIGUOUS**. [§5; §3.7; Algorithm 1]

**AMBIGUOUS: batch size.** §5 states effective batch 32, while Table 13's per-GPU BS times GA is 32 on each GPU, implying 64 globally for two DDP GPUs. No reconciliation is supplied. [§5; Table 13; Algorithm 1]

**AMBIGUOUS: learning rates.** §5 groups FULL and IWN under 1.5e-5; Table 13 instead gives FULL-ALBERT and IWN-BERT 2e-5. Both claims are retained. [§5; Table 13]

**AMBIGUOUS: epochs and component isolation.** G has E disabled in Table 1 but receives 4 epochs versus the 3-epoch RoBERTa baseline; FULL changes training duration and gate together. §6.5 describes G stopping at epoch 4 and saving one epoch although its Table 13 maximum is 4. Appendix H describes a shared five-epoch budget inconsistent with baseline Table 13. [Tables 1, 6, 7, 13; §6.5; Appendix H]

Additional reported variants, outside the eleven-row main experiment: group-count sweep 0/5/10/15/20, curated/random-BNC/auto-extracted vocabulary, lexical-only/surface-only, ten leave-one-group-out settings, and XLM-R-base baseline/SURGELLM-G with auto vocabulary for French Allocine and German GermanSentiment (5,000 training cap, official test splits). Reduced groups use training chi-square rankings; expanded groups use thesaurus variants. Auto extraction uses class-conditional log odds with informative Dirichlet prior, top 50 per class in Appendix E, SBERT embeddings, k-means with 10 clusters. Prior parameters, SBERT checkpoint, clustering settings, exact random/thesaurus word lists and multilingual hyperparameters are **AMBIGUOUS**; §7.3 describes top 50 per task instead. [§7.2-7.3, Tables 8-11; Appendices E and J, Table 16]

## 4. Seeds, stopping and evaluation

Seeds **0, 1, 2** govern data splits, weight initialization, dropout masks and CUDA determinism. Exact deterministic backend flags are **AMBIGUOUS**. [§5; Appendix I, pp.613-614]

Evaluate validation macro-F1 after each epoch. Start best score at negative infinity and patience counter at zero. A strictly higher score replaces the best checkpoint and resets patience; otherwise increment the counter and stop at 2. Restore best weights before test evaluation. Equal scores count as no improvement. [Appendix K, Algorithm 1, lines 11-33; §5]

**AMBIGUOUS:** `QuickVal` data coverage, per-task averaging versus pooled validation macro-F1, distributed gather/deduplication, and NaN handling are unspecified. [Algorithm 1; §5]

Metrics: accuracy, macro-F1, precision, recall, ROC-AUC, task averages; report macro-F1 mean and SD over three seeds. Table 3 averages the four task F1 scores. Precision/recall averaging, AUC positive class, threshold/decision rule, SD convention and missing-class handling are **AMBIGUOUS**. [§5; Table 3; Table 5]

Statistical evaluation: Welch t-test with Benjamini-Hochberg FDR 0.05; §6.2 specifies 16 task-variant comparisons and calls tests paired Welch tests. Bootstrap 95% confidence intervals with 2,000 replicates, seed 0. **AMBIGUOUS:** paired t-test versus independent Welch test, complete comparison family, and bootstrap resampling unit/stratification/CI method. [§5; §6.2, Table 4; Appendix I]

## 5. Expected Table 3 results

Transcribed from Table 3, p.604; visually checked against PDF page 5. D1-D4 are macro-F1 mean +/- SD over seeds 0,1,2. Delta is relative to Baseline-RoBERTa; time is mean training seconds on two T4 GPUs. These are reported targets, with no guaranteed reproduction tolerance specified.

| Model | D1 | D2 | D3 | D4 | Avg | Delta | Seconds |
|---|---|---|---|---|---:|---:|---:|
| T5-base | .928 +/- .005 | .939 +/- .007 | .972 +/- .004 | .748 +/- .013 | .897 | -.007 | 412 |
| Baseline-DistilBERT | .901 +/- .006 | .940 +/- .008 | .955 +/- .006 | .749 +/- .012 | .886 | -.018 | 82 |
| Baseline-BERT | .918 +/- .004 | .934 +/- .007 | .963 +/- .005 | .760 +/- .011 | .894 | -.010 | 227 |
| Baseline-RoBERTa | .929 +/- .004 | .947 +/- .006 | .978 +/- .003 | .762 +/- .010 | .904 | -- | 233 |
| SURGELLM-S-DistilBERT | .911 +/- .007 | .961 +/- .006 | .925 +/- .009 | .681 +/- .013 | .870 | -.034 | 119 |
| SURGELLM-S-BERT | .926 +/- .005 | .939 +/- .007 | .965 +/- .004 | .748 +/- .011 | .894 | -.010 | 317 |
| SURGELLM-G-RoBERTa | .937 +/- .004 | .949 +/- .005 | .977 +/- .003 | .760 +/- .010 | .906 | +.002 | 327 |
| SURGELLM-FULL-RoBERTa | .932 +/- .005 | .950 +/- .006 | .961 +/- .005 | .711 +/- .012 | .889 | -.015 | 326 |
| SURGELLM-FULL-ALBERT | .918 +/- .006 | .961 +/- .005 | .957 +/- .005 | .708 +/- .013 | .886 | -.018 | 317 |
| SURGELLM-IWN-RoBERTa | .933 +/- .004 | .954 +/- .005 | .979 +/- .003 | .892 +/- .009 | .940 | +.036 | 332 |
| SURGELLM-IWN-BERT | .927 +/- .005 | .946 +/- .006 | .968 +/- .004 | .866 +/- .010 | .927 | +.023 | 322 |

G-RoBERTa and FULL-RoBERTa carry the early-stopping marker. Parameter counts are those listed in §2 above, repeated by backbone in Table 3; Table 3 does not add the approximately 1.2M surgical parameters described in Appendix H.

## 6. Expected Table 7 results

Transcribed from Table 7, p.606; visually checked against PDF page 7. Delta is relative to the same-backbone baseline, unlike Table 3. 1/0 denotes enabled/disabled. No ALBERT baseline row is reported.

| Model | P | G | E | I | D1 | D2 | D3 | D4 | Avg | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-RoBERTa | 0 | 0 | 0 | 0 | .929 | .947 | .978 | .762 | .904 | -- |
| SURGELLM-G-RoBERTa | 1 | 0 | 0 | 0 | .937 | .949 | .977 | .760 | .906 | +.002 |
| SURGELLM-FULL-RoBERTa | 1 | 1 | 1 | 0 | .932 | .950 | .961 | .711 | .889 | -.015 |
| SURGELLM-IWN-RoBERTa | 1 | 1 | 1 | 1 | .933 | .954 | .979 | .892 | .940 | +.036 |
| Baseline-BERT | 0 | 0 | 0 | 0 | .918 | .934 | .963 | .760 | .894 | -- |
| SURGELLM-S-BERT | 1 | 1 | 0 | 0 | .926 | .939 | .965 | .748 | .894 | +/-.000 |
| SURGELLM-IWN-BERT | 1 | 1 | 1 | 1 | .927 | .946 | .968 | .866 | .927 | +.033 |
| Baseline-DistilBERT | 0 | 0 | 0 | 0 | .901 | .940 | .955 | .749 | .886 | -- |
| SURGELLM-S-DistilBERT | 1 | 1 | 0 | 0 | .911 | .961 | .925 | .681 | .870 | -.016 |
| SURGELLM-FULL-ALBERT | 1 | 1 | 1 | 0 | .918 | .961 | .957 | .708 | .886 | -- |

**AMBIGUOUS: headline comparator.** The abstract calls +.036 the improvement over the strongest non-IWN baseline; Table 3 contains G-RoBERTa at .906, above Baseline-RoBERTa .904. The +.036 comparison is to Baseline-RoBERTa, not the highest non-IWN row. Preserve reported values. [Abstract; Table 3]

## 7. Software and hardware

| Item | Reported requirement | Citation |
|---|---|---|
| PyTorch | 2.1 | Appendix I p.614 |
| Hugging Face Transformers | 4.35 | Appendix I |
| Accelerate | 0.24; DDP and FP16 mixed precision | Appendix I; Algorithm 1 |
| scikit-learn | 1.3 | Appendix I |
| sentence-transformers | 2.2; SBERT used for automatic vocabulary extraction | Appendix I; Appendix E |
| GPUs | 2 x NVIDIA T4, 16 GB each | Appendix I; §5 |
| Compute | Approximately 38 GPU-hours for all main and ablation results | Appendix I |
| Author code location | Paper lists https://surgellm-iwn.github.io as project webpage; repository contents were not used to fill specification gaps | Appendix I |

**AMBIGUOUS:** Python, CUDA, cuDNN, driver, OS, package patch versions, datasets/tokenizers/numpy/scipy versions, CPU, RAM, storage, and exact environment lockfile are not supplied. Reported GPU setup is the original platform, not an established minimum requirement. [§5; Appendix I]

## 8. Review and bounded next step

Review completed: all requested categories are covered; Table 2 sums to 17,830; all eleven Table 3 variants match Table 13; Tables 3/7/13 were visually verified. Four encoder backbones plus T5, AdamW, seeds 0/1/2, macro-F1 selection with patience 2, and two 16-GB T4 GPUs match the headline setup. Contradictions remain explicitly unresolved rather than converted to defaults.

Next implementation scope requested by the user: **SST-2 loading and preprocessing with tests only; no model.** Because the paper does not identify one exact SST-2 split protocol, any implementation must require an explicit protocol choice and record it. A data layer can preserve source text/IDs, validate binary labels, support deterministic stratified sampling, prevent train/validation/test overlap, and write a provenance manifest without claiming an exact author split. These are proposed engineering checks, not additional paper claims.

Before claiming exact experimental reproduction, resolve D1 split/cap order, D2 source, D3/D4 provenance/labels, D4 normalization population, global batch size, conflicting learning rates, and unspecified tokenizer/feature semantics. Do not train models to approximate away these gaps.
