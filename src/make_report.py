"""Build report.md from results/*.json (every number is read from those files), then render report.pdf.

python src/eval.py && python src/make_report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data import FAMILIES  # noqa: E402
from report_tables import bar_figure, load, md_table, pct, pm  # noqa: E402

TASKS = ["writing", "coding", "chat", "math", "nlp_task", "factual_qa"]
ABLATIONS = [("no_length", "remove length (fixed 64-token window, short texts tiled)"),
             ("no_format", "strip markdown + line breaks"),
             ("lower_nopunct", "lowercase + strip punctuation")]


def main():
    S = load("summary.json")
    D = json.load(open("data/processed/data_stats.json"))
    B = load("baseline_tfidf.json")
    L = load("input_leakage.json")
    F = load("rq4_features.json")
    V = {}  # values injected into the template

    # ------------------------------------------------------------------ data
    V["data_table"] = md_table(
        ["split", "total"] + FAMILIES,
        [[s, sum(D["split_family_counts"][s].values())] + [D["split_family_counts"][s][f] for f in FAMILIES]
         for s in ("train", "val", "test")])
    task_counts = {t: sum(D["task_family_counts"][t].values()) for t in D["task_family_counts"]}
    V["task_table"] = md_table(["task", "rows"] + FAMILIES,
                               [[t, task_counts[t]] + [D["task_family_counts"][t][f] for f in FAMILIES] for t in TASKS])
    for k in ("raw_completions", "raw_prompts", "after_family_filter", "after_length_filter", "n_total",
              "n_unique_prompts", "prompts_with_multiple_families"):
        V[k] = f"{D[k]:,}"
    V["bard_before"] = f"{D['per_family_before_balance']['Bard']:,}"
    V["llama_before"] = f"{D['per_family_before_balance']['LLaMA-2']:,}"

    # ------------------------------------------------------------------ training details
    c, r = S["cnn_output_none"], S["rnn_output_none"]
    V["cnn_params"], V["rnn_params"] = f"{c['n_params']:,}", f"{r['n_params']:,}"
    V["vocab"] = f"{c['vocab_size']:,}"
    V["cnn_time"] = f"{c['train_time_s']['mean'] / 60:.1f}"
    V["rnn_time"] = f"{r['train_time_s']['mean'] / 60:.1f}"
    V["cnn_best_epochs"] = ", ".join(map(str, c["best_epoch"]))
    V["rnn_best_epochs"] = ", ".join(map(str, r["best_epoch"]))

    # ------------------------------------------------------------------ RQ1
    V["rq1_table"] = md_table(
        ["model", "test accuracy (%)", "macro-F1 (%)", "seeds"],
        [["CNN (ours)", pm(c["accuracy"]), pm(c["macro_f1"]), 3],
         ["BiLSTM (ours)", pm(r["accuracy"]), pm(r["macro_f1"]), 3],
         ["TF-IDF + LogReg (baseline)", pct(B["output"]["accuracy"]), pct(B["output"]["macro_f1"]), "1 (deterministic)"],
         ["Hand-crafted features + GBoost (RQ4)", pct(F["gboost"]["accuracy"]), pct(F["gboost"]["macro_f1"]), "1"],
         ["Chance (uniform)", pct(0.2), "-", "-"]])
    V["rq1_family_table"] = md_table(
        ["family", "CNN", "BiLSTM", "TF-IDF"],
        [[f, pm(c["per_family_acc"][f]), pm(r["per_family_acc"][f]), pct(B["output"]["per_family_acc"][f])] for f in FAMILIES])
    for m, s in (("cnn", c), ("rnn", r)):
        V[f"{m}_acc"], V[f"{m}_f1"] = pm(s["accuracy"]), pm(s["macro_f1"])
        for f in FAMILIES:
            V[f"{m}_fam_{f}"] = pct(s["per_family_acc"][f]["mean"])
    V["tfidf_out"] = pct(B["output"]["accuracy"])
    cm = c["confusion_matrix_sum"]
    mpt = FAMILIES.index("MPT")
    V["mpt_to_gpt"] = pct(cm[mpt][FAMILIES.index("GPT")] / sum(cm[mpt]))
    V["mpt_to_bard"] = pct(cm[mpt][FAMILIES.index("Bard")] / sum(cm[mpt]))

    # ------------------------------------------------------------------ RQ2
    rows = []
    for setup, label in (("input", "input only (prompt)"), ("output", "output only (response)"), ("both", "input + output")):
        cs, rs = S[f"cnn_{setup}_none"], S[f"rnn_{setup}_none"]
        rows.append([label, pm(cs["accuracy"]), pm(rs["accuracy"]), pct(B[setup]["accuracy"]),
                     pm(cs["macro_f1"]), pm(rs["macro_f1"]), pct(B[setup]["macro_f1"])])
        V[f"cnn_{setup}"], V[f"rnn_{setup}"], V[f"tfidf_{setup}"] = pm(cs["accuracy"]), pm(rs["accuracy"]), pct(B[setup]["accuracy"])
    V["rq2_table"] = md_table(["setup", "CNN acc", "BiLSTM acc", "TF-IDF acc", "CNN F1", "BiLSTM F1", "TF-IDF F1"], rows)
    bar_figure("figures/rq2_setups.png", ["input only", "output only", "input+output"],
               {m: [S[f"{k}_{s}_none"]["accuracy"]["mean"] for s in ("input", "output", "both")] for m, k in (("CNN", "cnn"), ("BiLSTM", "rnn"))}
               | {"TF-IDF": [B[s]["accuracy"] for s in ("input", "output", "both")]},
               "test accuracy", "RQ2: which part of the interaction identifies the LLM?",
               errs={m: [S[f"{k}_{s}_none"]["accuracy"]["std"] for s in ("input", "output", "both")] for m, k in (("CNN", "cnn"), ("BiLSTM", "rnn"))} | {"TF-IDF": None},
               chance=0.2)
    V["maj_rate"] = pct(L["test_majority_rate"])
    V["p_uniform"] = f"{L['binomtest_vs_uniform_p']:.3f}"
    V["p_major"] = f"{L['binomtest_vs_majority_p']:.2f}"
    V["upper_bound"] = pct(L["test_input_only_upper_bound"])
    V["meta_source"] = pct(L["meta_source_acc"])
    V["meta_task"] = pct(L["meta_task_acc"])
    V["meta_plen"] = pct(L["meta_prompt_length_acc"])
    V["tfidf_input_bard"] = pct(B["input"]["per_family_acc"]["Bard"])
    V["multi_rows"] = f"{L['rows_whose_prompt_has_multi_family']:,}"
    V["plen"] = ", ".join(f"{f} {L['median_prompt_words_by_family'][f]:.0f}" for f in FAMILIES)

    # ------------------------------------------------------------------ RQ3
    rq3_rows, fam_rows = [], []
    for t in TASKS:
        for m in ("cnn", "rnn"):
            k = f"{m}_output_none_holdout-{t}"
            if k not in S:
                continue
            ind = S[f"{m}_output_none"]["per_task_acc"][t]
            ho = S[k]
            rq3_rows.append([t, m.upper() if m == "cnn" else "BiLSTM", ho["n_test"], pm(ind), pm(ho["accuracy"]),
                             f"{100 * (ho['accuracy']['mean'] - ind['mean']):+.1f}",
                             f"{pm(ho['heldout_all_rows_accuracy'])} (n={ho['n_heldout_all_rows']:,})"])
            V[f"rq3_{m}_{t}_in"], V[f"rq3_{m}_{t}_ho"] = pm(ind), pm(ho["accuracy"])
            V[f"rq3_{m}_{t}_drop"] = f"{100 * (ho['accuracy']['mean'] - ind['mean']):+.1f}"
            if m == "cnn":
                fam_rows.append([t] + [pct(ho["heldout_all_rows_per_family_acc"][f]["mean"]) for f in FAMILIES])
    V["rq3_table"] = md_table(["held-out task", "model", "n test", "in-domain acc", "held-out acc (same rows)", "Δ (pts)",
                               "held-out acc, all rows of task"], rq3_rows) if rq3_rows else "_RQ3 runs missing_"
    V["rq3_family_table"] = md_table(["held-out task"] + FAMILIES, fam_rows) if fam_rows else ""
    if fam_rows:
        cnn_rows = [x for x in rq3_rows if x[1] == "CNN"]
        bar_figure("figures/rq3_heldout.png", [x[0] for x in cnn_rows],
                   {"in-domain (CNN)": [S["cnn_output_none"]["per_task_acc"][x[0]]["mean"] for x in cnn_rows],
                    "task held out (CNN)": [S[f"cnn_output_none_holdout-{x[0]}"]["accuracy"]["mean"] for x in cnn_rows]},
                   "test accuracy", "RQ3: leave-one-task-out vs in-domain", chance=0.2)
        drops = {t: S[f"cnn_output_none_holdout-{t}"]["accuracy"]["mean"] - S["cnn_output_none"]["per_task_acc"][t]["mean"] for t in TASKS if f"cnn_output_none_holdout-{t}" in S}
        V["rq3_mean_drop"] = f"{100 * sum(drops.values()) / len(drops):+.1f}"
        V["rq3_worst"] = min(drops, key=drops.get)
        V["rq3_worst_drop"] = f"{100 * drops[V['rq3_worst']]:+.1f}"
        V["rq3_best"] = max(drops, key=drops.get)
        V["rq3_best_drop"] = f"{100 * drops[V['rq3_best']]:+.1f}"
        fam_mean = {f: sum(S[f"cnn_output_none_holdout-{t}"]["heldout_all_rows_per_family_acc"][f]["mean"] for t in drops) / len(drops) for f in FAMILIES}
        V["rq3_fam_mean"] = ", ".join(f"{f} {pct(v)}" for f, v in sorted(fam_mean.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------ RQ4
    P = F["per_family"]
    V["rq4_stats_table"] = md_table(
        ["statistic"] + FAMILIES,
        [["median length (words)"] + [f"{P[f]['median_words']:.0f}" for f in FAMILIES],
         ["type-token ratio, first 100 words"] + [f"{P[f]['ttr_first100']:.2f}" for f in FAMILIES],
         ["mean word length (chars)"] + [f"{P[f]['mean_word_len']:.2f}" for f in FAMILIES],
         ["% with bullet list"] + [pct(P[f]["pct_with_bullets"], 0) for f in FAMILIES],
         ["% with numbered list"] + [pct(P[f]["pct_with_numbered"], 0) for f in FAMILIES],
         ["% with **bold**"] + [pct(P[f]["pct_with_bold"], 0) for f in FAMILIES],
         ["% with code block"] + [pct(P[f]["pct_with_code_block"], 0) for f in FAMILIES],
         ["commas per 100 words"] + [f"{P[f]['commas_per_100w']:.1f}" for f in FAMILIES],
         ["'!' per 100 words"] + [f"{P[f]['exclam_per_100w']:.2f}" for f in FAMILIES],
         ["% opening with Sure/Certainly/Of course"] + [pct(P[f]["pct_opens_sure"], 0) for f in FAMILIES],
         ["% opening with Hello/Hi"] + [pct(P[f]["pct_opens_hello"], 0) for f in FAMILIES],
         ["% mentioning 'AI language model'"] + [pct(P[f]["pct_mentions_ai_model"], 0) for f in FAMILIES]])
    V["rq4_openings_table"] = md_table(
        ["family", "most frequent 3-word openings (share of family's responses)"],
        [[f, "; ".join(f"\"{o}\" ({pct(s)}%)" for o, _, s in P[f]["top_openings"][:3])] for f in FAMILIES])
    J = F["vocab_jaccard_top1000"]
    pairs = [(a, b, J[a][b]) for i, a in enumerate(FAMILIES) for b in FAMILIES[i + 1:]]
    lo, hi = min(pairs, key=lambda x: x[2]), max(pairs, key=lambda x: x[2])
    V["jac_lo"] = f"{lo[0]}–{lo[1]} ({lo[2]:.2f})"
    V["jac_hi"] = f"{hi[0]}–{hi[1]} ({hi[2]:.2f})"
    V["feat_lr"], V["feat_gb"] = pct(F["logreg"]["accuracy"]), pct(F["gboost"]["accuracy"])
    V["feat_gb_f1"] = pct(F["gboost"]["macro_f1"])
    imp = list(F["gboost_permutation_importance"].items())
    V["top_feats"] = ", ".join(f"`{k}` ({100 * v:.1f} pts)" for k, v in imp[:5])
    G = F["gboost_group_ablation"]
    V["rq4_group_table"] = md_table(
        ["feature group", "# features", "acc. with this group only", "acc. without this group", "drop vs all features (pts)"],
        [[g, len(F["feature_groups"][g]), pct(G[g]["acc_group_only"]), pct(G[g]["acc_without_group"]),
          f"{100 * (G[g]['acc_without_group'] - F['gboost']['accuracy']):+.1f}"] for g in F["feature_groups"]])
    abl_rows = []
    for m, name in (("cnn", "CNN"), ("rnn", "BiLSTM")):
        base = S[f"{m}_output_none"]["accuracy"]
        abl_rows.append([name, "none (full text)", pm(base), "-"])
        for a, desc in ABLATIONS:
            k = f"{m}_output_{a}"
            if k in S:
                d = S[k]["accuracy"]["mean"] - base["mean"]
                abl_rows.append([name, desc, pm(S[k]["accuracy"]), f"{100 * d:+.1f}"])
                V[f"abl_{m}_{a}"], V[f"abl_{m}_{a}_d"] = pm(S[k]["accuracy"]), f"{100 * d:+.1f}"
    V["rq4_abl_table"] = md_table(["model", "ablation", "test acc (%)", "Δ vs full (pts)"], abl_rows)

    for k in ("rq3_text", "rq4_text", "lessons_text"):
        V.setdefault(k, "_(pending)_")
    tmpl = open("report_template.md", encoding="utf-8").read()
    open("report.md", "w", encoding="utf-8").write(tmpl.format(**V))
    print("wrote report.md")

    import pypandoc
    pypandoc.convert_file(
        "report.md", "pdf", outputfile="report.pdf",
        extra_args=["--pdf-engine=xelatex", "-V", "geometry:margin=0.85in", "-V", "fontsize=10pt",
                    "-V", "mainfont=Calibri", "-V", "monofont=Consolas", "-V", "colorlinks=true",
                    "--resource-path=.", "-V", "tables=true"])
    print("wrote report.pdf")


if __name__ == "__main__":
    main()
