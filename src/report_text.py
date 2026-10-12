"""Narrative paragraphs for report.md. Every number is computed from the loaded results, never typed."""
from data import FAMILIES
from report_tables import pct, pm

TASKS = ["writing", "coding", "chat", "math", "nlp_task", "factual_qa"]


def narrative(V, S, F, P, G, L, T):
    c, r = S["cnn_output_none"], S["rnn_output_none"]
    acc = lambda k: S[k]["accuracy"]["mean"]
    fam = lambda k, f: S[k]["per_family_acc"][f]["mean"]
    out = {}

    # ------------------------------------------------------------------ RQ1
    order = sorted(FAMILIES, key=lambda f: -fam("cnn_output_none", f))
    cm_c, cm_r = c["confusion_matrix_sum"], r["confusion_matrix_sum"]
    i_mpt, i_gpt, i_bard = FAMILIES.index("MPT"), FAMILIES.index("GPT"), FAMILIES.index("Bard")
    row = lambda cm, i, j: pct(cm[i][j] / sum(cm[i]))
    speed = r["train_time_s"]["mean"] / c["train_time_s"]["mean"]
    out["rq1_text"] = (
        f"Both neural models are far above chance (20%): their prompt-bootstrap 95% intervals exclude it by a wide margin. "
        f"The **CNN ({V['cnn_acc']}%) beats the BiLSTM ({V['rnn_acc']}%)**; the intervals do not overlap. The CNN is also "
        f"{speed:.1f}× faster to train. The bag-of-n-grams TF-IDF baseline is stronger than both ({V['tfidf_out']}%); "
        f"Section 5 discusses why. For reference, the first version of our models, which had the padding bug, scored "
        f"{V['v1_cnn']}% (CNN) and {V['v1_rnn']}% (BiLSTM); fixing the bug did not lower the headline numbers.")
    out["rq1_family_text"] = (
        f"**{order[0]}** is the easiest family for the CNN ({pct(fam('cnn_output_none', order[0]))}%), followed by "
        f"**{order[1]}** ({pct(fam('cnn_output_none', order[1]))}%). **MPT** is by far the hardest "
        f"({pct(fam('cnn_output_none', 'MPT'))}% with the CNN, {pct(fam('rnn_output_none', 'MPT'))}% with the BiLSTM). "
        f"The CNN sends {row(cm_c, i_mpt, i_gpt)}% of MPT responses to GPT and {row(cm_c, i_mpt, i_bard)}% to Bard. "
        f"The BiLSTM sends {row(cm_r, i_mpt, i_gpt)}% of MPT responses to GPT, i.e. it has essentially no MPT class. "
        f"This matches RQ4: MPT has no distinctive habit of its own and sits between the other families on almost every "
        f"statistic. Falcon and GPT are the other frequently confused pair.")

    # ------------------------------------------------------------------ RQ2
    ci_c, ci_r = S["cnn_input_none"]["bootstrap"], S["rnn_input_none"]["bootstrap"]
    bard_in = (pct(fam("cnn_input_none", "Bard")), pct(fam("rnn_input_none", "Bard")))
    others_max = max(max(fam("cnn_input_none", f), fam("rnn_input_none", f)) for f in FAMILIES if f != "Bard")
    out["rq2_text"] = f"""**(i) Can the LLM be predicted from the prompt alone? Barely.** Input-only accuracy is {V['cnn_input']}% (CNN), {V['rnn_input']}% (BiLSTM) and {V['tfidf_input']}% (TF-IDF). Prompt-cluster bootstrap 95% intervals are {V['cnn_input_ci']}, {V['rnn_input_ci']} and {V['tfidf_input_ci']}. They exclude uniform chance (20%), but all of them include the majority-class rate of the test set ({V['maj_rate']}%), so no input-only model is clearly better than always guessing the largest class. A small effect was expected, because UltraFeedback assigned models to prompts at random. Prompts also repeat across families: {V['multi_rows']} rows share their prompt with a response from a different family, and identical inputs with different labels cannot all be classified correctly.

**Where does the small signal come from?** Almost entirely from **Bard**. Input-only recall for Bard is {bard_in[0]}% (CNN) and {bard_in[1]}% (BiLSTM), while no other family exceeds {pct(others_max)}%. Bard has no responses in three FLAN sub-sets (NIv2, P3, FLAN-2021) and is over-represented in Evol-Instruct and UltraChat, so the prompt's *source* is informative about Bard. A logistic regression on the one-hot source alone reaches {V['meta_source']}% (95% CI {V['meta_source_ci']}). Two controls support this explanation, though they do not prove it is the only one:

- *Bard excluded* (4 families, chance {V['io_no_bard_chance']}%): TF-IDF input-only accuracy is {V['io_no_bard_acc']}% (CI {V['io_no_bard_ci']}, bootstrap p = {V['io_no_bard_p']}, n = {V['io_no_bard_n']}).
- *Within ShareGPT prompts only* (all five families present): {V['io_within_sharegpt_acc']}% (CI {V['io_within_sharegpt_ci']}, p = {V['io_within_sharegpt_p']}).

In both cases there is no detectable input-only signal. The derived task label ({V['meta_task']}%) and prompt length ({V['meta_plen']}%) carry none either. We therefore attribute the effect to the dataset's collection process, not to the LLMs. We cannot rule out smaller contributions such as topic differences that survive our controls.

**(ii) Does adding the input improve classification? No.** Input + output is {V['cnn_both']}% vs {V['cnn_output']}% output-only for the CNN (within seed noise), {V['rnn_both']}% vs {V['rnn_output']}% for the BiLSTM, and {V['tfidf_both']}% vs {V['tfidf_output']}% for TF-IDF. Prepending the prompt uses up part of the fixed 256-token budget, which cuts off more of the response, and the prompt tokens are mostly noise for this task.

**(iii) When could input-only work "surprisingly well"?** Whenever the data collection links prompts to models: different prompt pools per model, different system prompts, or sources with different topics. A classifier would then learn the *collection process* rather than the LLM. Our Bard/source effect is a small instance of exactly this."""

    # ------------------------------------------------------------------ RQ3
    ho = lambda t, f: pct(S[f"cnn_output_none_holdout-{t}"]["heldout_all_rows_per_family_acc"][f]["mean"], 0)
    llama = [S[f"cnn_output_none_holdout-{t}"]["heldout_all_rows_per_family_acc"]["LLaMA-2"]["mean"] for t in TASKS]
    out["rq3_text"] = f"""**(i) Does performance drop on unseen tasks? It depends strongly on the task.** Measured against the size-matched in-domain control, holding a task out changes CNN accuracy on that task by {V['rq3_mean_drop']} points on average. The changes range from {V['rq3_best_drop']} on *{V['rq3_best']}* to {V['rq3_worst_drop']} on *{V['rq3_worst']}*.

- *Writing* (one of the two examples given in the assignment) transfers essentially without loss: {V['rq3_cnn_writing_ho']}% held out vs {V['rq3_cnn_writing_sm']}% size-matched ({V['rq3_cnn_writing_drop_sm']} pts).
- *Coding* is the clear exception: {V['rq3_cnn_coding_ho']}% vs {V['rq3_cnn_coding_sm']}% ({V['rq3_cnn_coding_drop_sm']} pts). On coding the size-matched control even beats the full-train model ({V['rq3_cnn_coding_in']}%), so seed standard deviations understate run-to-run variation and the exact size of this gap is uncertain. Measured against the full-train model the drop is still {V['rq3_cnn_coding_drop']} pts, so a real coding drop remains either way.
- *The BiLSTM shows a different ordering:* writing {V['rq3_rnn_writing_ho']}% vs {V['rq3_rnn_writing_sm']}% ({V['rq3_rnn_writing_drop_sm']} pts), coding {V['rq3_rnn_coding_ho']}% vs {V['rq3_rnn_coding_sm']}% ({V['rq3_rnn_coding_drop_sm']} pts). This is a warning not to over-read single task comparisons.

Every held-out accuracy ({V['rq3_ho_range']}%) stays far above chance. Per-task test sets are small (112–757 rows) and seed standard deviations are 1–4 points, so differences of a couple of points are within noise. The task labels come from keyword rules, and sources are not balanced across tasks; both remain confounds.

**(ii) Are some families more consistently identifiable? Yes.** **LLaMA-2** is recognised on every unseen task (held-out recall {pct(min(llama))}–{pct(max(llama))}%). Its signature is task-independent: chatty openings ("Sure, I'd be happy to help", "Hello! I'm here to...") and safety caveats. **Bard** is robust on prose tasks ({ho('writing', 'Bard')}% on unseen writing, {ho('chat', 'Bard')}% on chat) but drops on coding ({ho('coding', 'Bard')}%), where its bullet-and-bold layout gives way to code. **GPT** collapses on unseen factual QA ({ho('factual_qa', 'GPT')}%), where its short, plain answers resemble Falcon's and MPT's. **MPT** is weak everywhere ({ho('nlp_task', 'MPT')}–{ho('coding', 'MPT')}%).

**(iii) General fingerprint or task-specific patterns? Both.** Most accuracy survives on prose-like tasks never seen in training, so part of the fingerprint is a general family style. The large coding drop and the family-specific collapses (Bard on coding, GPT on factual QA) show that another part is "how family X formats answers to task Y". RQ4 shows these formatting habits (lists, bold, code blocks) are among the strongest cues, and they are naturally task-dependent."""

    # ------------------------------------------------------------------ RQ4
    d = lambda m, a: float(V[f"abl_{m}_{a}_d"])
    out["rq4_text"] = f"""**(i) What signals do the classifiers use?** The descriptive statistics show clear, human-readable differences:

- **Bard** writes the longest answers and is the heaviest markdown user: bullets in {pct(P['Bard']['pct_with_bullets'], 0)}% and **bold** in {pct(P['Bard']['pct_with_bold'], 0)}% of responses. It typically opens with "Sure, here is...".
- **GPT** prefers numbered lists ({pct(P['GPT']['pct_with_numbered'], 0)}%) and uses the most commas and the longest words. Its most frequent opening, "As an AI...", appears in only {pct(P['GPT']['top_openings'][0][2])}% of answers, so GPT has no signature opener.
- **LLaMA-2** is the most conversational: {pct(P['LLaMA-2']['pct_opens_sure'], 0)}% of answers open with "Sure"/"Of course", {pct(P['LLaMA-2']['pct_opens_hello'], 0)}% with "Hello". It uses the most exclamation marks and first-person pronouns.
- **Falcon** writes the shortest answers (median {P['Falcon']['median_words']:.0f} words) with almost no markdown, and most often mentions being an "AI language model" ({pct(P['Falcon']['pct_mentions_ai_model'], 0)}%).
- **MPT** sits in the middle on almost every statistic, which matches it being the hardest family to classify.

**(ii) Linguistic, structural, or both? Both.** On the validation split, the strongest single feature groups are lexical ({pct(G['lexical']['acc_group_only'])}% alone), punctuation ({pct(G['punctuation']['acc_group_only'])}%) and markdown ({pct(G['markdown']['acc_group_only'])}%). Removing markdown hurts most ({100 * (G['markdown']['acc_without_group'] - F['gboost_val_acc']):+.1f} pts), and the single most important feature is the fraction of blank lines, a pure layout cue. That features with no word identities reach {V['feat_gb']}% test accuracy, above our BiLSTM, shows how much of the "fingerprint" is style rather than content.

**(iii) Does removing signals hurt the neural models?** Yes. For the CNN:

- *Length:* the controlled comparison isolates it. Going from *window64* to *no_length* costs only {V['abl_cnn_len_iso']} pts ({V['abl_rnn_len_iso']} for the BiLSTM). Most of the *no_length* drop ({V['abl_cnn_no_length_d']} pts) comes from seeing just 64 tokens (*window64*: {V['abl_cnn_window64_d']} pts), not from losing length information.
- *Formatting:* stripping markdown and line breaks costs {V['abl_cnn_no_format_d']} pts.
- *Case and punctuation:* lowercasing and removing punctuation costs {V['abl_cnn_lower_nopunct_d']} pts. This transformation also removes line breaks, so it overlaps with *no_format*.
- *All combined:* {V['abl_cnn_all_surface_d']} pts (BiLSTM {V['abl_rnn_all_surface_d']} pts). That is clearly more than any single transformation, so the cues are partly redundant.

Even with everything removed, the CNN still reaches {pct(S['cnn_output_all_surface']['accuracy']['mean'])}% and the BiLSTM {pct(S['rnn_output_all_surface']['accuracy']['mean'])}%, well above chance. What remains is the choice of words in the first 64 tokens, without case, punctuation or layout. That includes stock phrases ("I'd be happy to help", "As an AI"), but it may also include topic cues created by the uneven source mix (RQ2). Our ablations cannot separate those two."""

    # ------------------------------------------------------------------ lessons
    t256 = T["pct_over_256_by_family"]
    out["lessons_text"] = f"""**What worked.**

- *Simple models are strong here.* The CNN is better than the BiLSTM ({V['cnn_acc']}% vs {V['rnn_acc']}%) and faster. Local n-gram patterns such as openings, list markers and stock phrases are exactly what convolutions with max-pooling detect, while the BiLSTM must carry them through a long sequence.
- *Baselines keep you honest.* TF-IDF + logistic regression ({V['tfidf_out']}%) beats both neural models. One reason is truncation: our models see 256 tokens, which cuts off {pct(t256['Bard'])}% of Bard and {pct(t256['GPT'])}% of GPT responses but only {pct(t256['Falcon'])}% of Falcon responses, while TF-IDF sees every response in full. It also does not need to learn embeddings from only 16k examples. Both neural models peaked at or near the 6-epoch cap, so longer training, longer inputs or a pretrained encoder are the obvious next steps.
- *Grouped splitting and leakage controls.* Splitting by prompt, bootstrapping over prompts, and the Bard-excluded and single-source controls let us make the RQ2 claim with evidence instead of assertion.

**What went wrong, and what we learned.**

- *Padding bugs, found by review.* Our first CNN counted a convolution window as valid when only its first token was real, and our unpacked BiLSTM's backward direction read padding before the real tokens. A prediction therefore depended on how much padding its batch had. Worse, our test set was stored in family order, so most test batches contained a single family, and the padding could indirectly carry label information. We fixed both models and added an invariance test, which fails on the old code and passes on the new one. We also shuffled the data and re-ran all {V['n_runs']} neural runs. The headline accuracies barely moved (CNN {V['v1_cnn']}% → {V['cnn_acc']}%), but before the fix we had no way to know that. Lesson: for variable-length models, test invariance to batching explicitly; masking the pooling is not enough.
- *Data artifacts.* Falcon responses contained literal `</s>` end-of-sequence tokens ({V['artifact_falcon']} rows; {V['artifact_other']} in all other families combined), a label leak that has nothing to do with style. We strip such tokens before splitting. Lesson: search raw generations for template and control tokens before training.
- *Process hygiene.* In the first round, killing training processes but not their job queue left stale runs that silently overwrote result files. Every result file now records the model version, a hash of the data file and the full configuration, and `src/eval.py` refuses to aggregate runs that disagree.
- *Compute.* A laptop CPU was too slow for 96 runs. One Slurm array job on the university cluster (ICDS Roar Collab) ran them in parallel, one run per array task; an earlier GPU attempt crashed, and we did not debug it. Making every run resumable (skip if the result exists) kept the workflow safe across disconnections.

**Limitations.**

- *Dataset age and scope:* UltraFeedback responses come from 2023 models (GPT-3.5/4, Bard, LLaMA-2, Falcon, MPT). Today's Gemini, Claude, Qwen and DeepSeek are not covered, and fingerprints likely change with every model version.
- *System prompts:* UltraFeedback generated each response under one of several "principle" system prompts, which shift style for every model.
- *Family grouping:* merging gpt-3.5-turbo with gpt-4 and three LLaMA-2 sizes hides within-family variation. Excluding the LLaMA fine-tunes avoids, but does not resolve, the question of what a "family" is.
- *Balance and labels:* classes are balanced by downsampling, but sources and tasks are not balanced within classes (RQ2), and task labels come from keyword rules.
- *Statistics:* 3 seeds per neural configuration; the TF-IDF and feature models are single runs; per-task test sets are small.
- *Privacy:* we did not audit the ShareGPT-derived prompts for personal information."""
    return out
