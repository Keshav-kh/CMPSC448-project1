---
title: "Who Wrote It? Identifying LLM Families from Their Responses"
subtitle: "CMPSC 448 Midterm Project, Fall 2026"
author:
  - "Team leader: Keshav Khandelwal"
  - "Team member: Viraj Mishra"
date: "October 11, 2026"
---

**Repository:** <https://github.com/Keshav-kh/CMPSC448-project1>. **Provenance:** every *result* number in this report (dataset counts, accuracies, F1, confidence intervals, statistics, timings) is inserted by `src/make_report.py` from `results/*.json` or `data/processed/data_stats.json`. Hyperparameters and the narrative are written by us. All {n_runs} neural runs share one model version (`{model_version}`) and one data file (md5 `{data_md5}`); `src/eval.py` refuses to aggregate otherwise.

# 1. Problem definition and dataset curation

**Task.** Given a response written by a large language model, predict which *LLM family* produced it. This is a 5-way text classification problem. We also ask whether the user prompt helps (RQ2), whether the learned "fingerprint" transfers across task types (RQ3), and which stylistic signals separate the families (RQ4).

**Source.** We use the public **UltraFeedback** dataset (`openbmb/UltraFeedback`, MIT licence on its dataset card; Cui et al., 2023). Each record has a prompt (`instruction`), the sub-dataset it came from (`source`: Evol-Instruct, ShareGPT, UltraChat, FLAN, TruthfulQA, FalseQA), and four completions, each tagged with the generating `model`. This gives the required (LLM_name, LLM_input, LLM_output) triples directly. The raw files contain {raw_completions} completions for {raw_prompts} unique prompts from 17 models. Because every prompt was answered by four randomly chosen models, **the same prompt appears with several different labels**, which matters for RQ2. We did not query any model or collect any data ourselves. We did not audit the prompts for personal information, though. Some of them come from ShareGPT, i.e. conversations real users shared, and a dataset licence does not guarantee they are free of personal or third-party content. We use the text only for aggregate statistics and classification, and we redistribute only the processed file that the public dataset already contains.

**Family grouping.** We grouped models by developer and kept only unambiguous ones:

| family | models |
|---|---|
| GPT (OpenAI) | gpt-3.5-turbo, gpt-4 |
| LLaMA-2 (Meta) | llama-2-7b-chat, llama-2-13b-chat, llama-2-70b-chat |
| Bard (Google, predecessor of Gemini) | bard |
| Falcon (TII) | falcon-40b-instruct |
| MPT (MosaicML) | mpt-30b-chat |

We dropped third-party LLaMA fine-tunes (Vicuna, WizardLM, UltraLM, Alpaca): they share LLaMA's base weights but were tuned largely on GPT-distilled data, so assigning them to either family would blur the labels. We also dropped StarChat (a code-only model) and Pythia-12B (only 834 completions).

**Cleaning and balancing.** Starting from {after_family_filter} completions of the kept models, we first removed **generator artifacts**. Literal end-of-sequence markers such as `</s>` appear almost only in Falcon output: {artifact_falcon} Falcon rows vs {artifact_other_rows} from all other families combined. Such a marker would identify the label without saying anything about style, so it is stripped. We then removed empty texts, exact duplicate responses and duplicate (prompt, model) pairs, and kept responses of 10–1000 words. {after_length_filter} rows remained, but they were very unbalanced (LLaMA-2 {llama_before} vs Bard {bard_before}). We randomly downsampled every family to 4,000 rows, giving **{n_total} examples** and {n_unique_prompts} unique prompts. Finally, we shuffled the rows so that neither files nor evaluation batches are ordered by family.

**Task label.** For RQ3 we derived a task label for every prompt. TruthfulQA and FalseQA prompts become *factual_qa*. For all others we apply keyword rules on the prompt, in order: code terms give *coding*; math terms or arithmetic give *math*; remaining FLAN prompts give *nlp_task*; writing verbs (write, essay, story, email, ...) give *writing*; everything else is *chat*. These labels are heuristic and noisy.

{task_table}

**Split.** We split 80/10/10 *by unique prompt* (seed 448), so a prompt and all its responses fall in a single split. The code asserts that no prompt crosses splits. After downsampling, {prompts_with_multiple_families} prompts are still answered by more than one family.

{data_table}

# 2. Models and training

**Tokenisation and truncation.** We use a case-preserving regex tokenizer (`\n|\w+|[^\w\s]`). Every punctuation mark is its own token and newlines become a `<nl>` token, so formatting stays visible to the models. The vocabulary is built from the training split only (minimum frequency 2, capped at {vocab} entries, including `<pad>`, `<unk>` and `<sep>`). Sequences are truncated to **256 tokens**. This truncation is substantial and differs strongly by family: {trunc_all}% of all responses are cut off.

{trunc_table}

**Padding invariance.** Batches are padded dynamically. A first version of both models was *not* padding-invariant, so a prediction could depend on the other examples in its batch (Section 5). The final models guarantee that a sequence's output depends only on its own tokens. The guarantee is checked by `tests/test_invariance.py`, which compares predictions made one-by-one, in a padded batch, in a reordered batch and with extra padding (maximum logit difference below 1e-5).

**CNN.** Embedding (128-d), then three parallel `Conv1d` layers with kernel sizes 3, 4 and 5 (128 filters each, ReLU), then max-over-time pooling, concatenation (384-d), dropout 0.5 and a linear layer to 5 classes. Only convolution windows that lie *entirely* inside the real tokens enter the max-pool. {cnn_params} parameters.

**RNN (BiLSTM).** Embedding (128-d), a bidirectional LSTM (64 hidden units per direction), then masked mean pooling and masked max pooling over time, concatenated (256-d), dropout 0.5 and a linear layer to 5 classes. The backward direction runs on each sequence reversed *within its own length*, so it never reads padding before real tokens. This is equivalent to packed sequences, but faster on our CPU: {bench_ours} s vs {bench_packed} s per batch of 64 × 256 tokens ({bench_speedup}× speed-up; `scripts/bench_lstm.py`). {rnn_params} parameters.

**Training.** Adam (learning rate 1e-3), cross-entropy loss, batch size 64, gradient clipping at 5. Training batches are bucketed by length. We train for at most 6 epochs with early stopping on validation accuracy (patience 2) and restore the best-validation weights before testing. Every *neural* configuration is run with seeds 1, 2 and 3, and we report mean ± standard deviation over seeds. Confidence intervals come from a **prompt-cluster bootstrap** (2,000 resamples of test prompts, not rows, because rows that share a prompt are not independent). The TF-IDF and feature baselines are single deterministic runs. The best validation epochs for output-only RQ1 runs were {cnn_best_epochs} (CNN) and {rnn_best_epochs} (BiLSTM). Several models peaked in the last allowed epoch, so they are probably under-trained.

**Hardware.** All {n_runs} neural runs were trained on CPU nodes of Penn State's ICDS Roar Collab cluster (one Slurm array task per run, 4 cores each; recorded devices: {devices}). The mean wall-clock time of one output-only run was {cnn_time} min (CNN) and {rnn_time} min (BiLSTM). Baselines and feature analyses ran on a laptop CPU.

**Baselines.** (a) TF-IDF over word unigrams and bigrams (case-preserving, punctuation kept, sublinear tf), followed by logistic regression (C = 4). (b) Gradient boosting on 34 hand-crafted style features (RQ4).

# 3. Results

## RQ1: Can we identify which LLM generated a response?

Output-only classification on the held-out test split (1,992 responses, about 400 per family):

{rq1_table}

{rq1_text}

Per-family test accuracy (recall), mean ± std over seeds:

{rq1_family_table}

![CNN, output only: confusion matrix summed over 3 seeds (rows normalised).](figures/cm_cnn_output_none.png){{width=48%}} ![BiLSTM, output only: confusion matrix summed over 3 seeds.](figures/cm_rnn_output_none.png){{width=48%}}

{rq1_family_text}

## RQ2: Does the user's prompt help identify the LLM?

The same code, vocabulary procedure and hyperparameters are used for all three setups. *Input + output* is the prompt (first 64 tokens), then `<sep>`, then the response, truncated to 256 tokens in total.

{rq2_table}

![RQ2: test accuracy by input setup (error bars: std over 3 seeds).](figures/rq2_setups.png){{width=75%}}

{rq2_text}

# 4. In-depth analysis

## RQ3: Do LLM fingerprints generalise across tasks?

**Setup.** For each task *T* we train the CNN (3 seeds) on the training split *without any task-T rows*, early-stop on validation rows not from *T*, and test on the task-*T* test rows. Removing a task also shrinks the training set, so we add a **size-matched control**: an in-domain CNN whose training set keeps all tasks but is randomly subsampled to exactly the same size. Both models are scored on the same task-*T* test rows, so the gap between them isolates the effect of never having seen task *T*. As a larger second sample we also report the held-out model's accuracy on *all* task-*T* rows (all splits), since it never saw any of them. For *writing* and *coding* we repeat everything with the BiLSTM.

{rq3_table}

![RQ3: CNN accuracy on each task when the task is held out vs the size-matched in-domain control (error bars: std over seeds).](figures/rq3_heldout.png){{width=75%}}

Per-family accuracy of the held-out CNN on all rows of the unseen task (mean of 3 seeds, %):

{rq3_family_table}

{rq3_text}

## RQ4: What characteristics distinguish the LLMs?

**Per-family statistics** (all 4,000 responses per family):

{rq4_stats_table}

{rq4_openings_table}

![Response length by family.](figures/rq4_length_box.png){{width=48%}} ![Markdown use by family.](figures/rq4_markdown.png){{width=48%}}

**Vocabulary.** We compared the 1,000 most frequent lower-cased words of each family on the training split. Overlap (Jaccard similarity) ranges from {jac_lo} to {jac_hi}.

**A classifier on hand-crafted features only.** We built 34 features in five groups: length, lexical, markdown, punctuation and opening phrase (`src/features.py`). Logistic regression on these features reaches {feat_lr}% test accuracy; histogram gradient boosting reaches **{feat_gb}%** (macro-F1 {feat_gb_f1}%). To avoid tuning our interpretation on the test set, the permutation importances and group ablations below are computed on the **validation** split (all-feature validation accuracy {feat_gb_val}%), and the test split is scored only once. The top permutation importances are {top_feats}.

![Permutation importance of the hand-crafted features (gradient boosting, validation split).](figures/rq4_importance.png){{width=60%}} ![Vocabulary overlap.](figures/rq4_vocab_jaccard.png){{width=38%}}

Retraining gradient boosting with one feature group removed, or with only one group (validation accuracy):

{rq4_group_table}

**Ablations on the neural models.** We re-train both neural models (3 seeds each) on transformed responses. *window64* is a control for *no_length*: it uses the same 64-token budget but keeps length information. The difference between the two therefore isolates length; the difference between *window64* and full text isolates "less text".

{rq4_abl_table}

{rq4_text}

# 5. Discussion and lessons learned

{lessons_text}

# References

- G. Cui et al. *UltraFeedback: Boosting Language Models with High-quality Feedback.* arXiv:2310.01377, 2023. Dataset: huggingface.co/datasets/openbmb/UltraFeedback (MIT licence).
- Y. Kim. *Convolutional Neural Networks for Sentence Classification.* EMNLP 2014.
- S. Hochreiter and J. Schmidhuber. *Long Short-Term Memory.* Neural Computation, 1997.
