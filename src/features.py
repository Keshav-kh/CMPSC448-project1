"""RQ4: hand-crafted stylistic features, per-family statistics and a feature-only classifier.

python src/features.py -> results/rq4_features.json, figures/rq4_*.png
"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data import FAMILIES, load_dataset  # noqa: E402
from eval import compute_metrics  # noqa: E402

WORD_RE = re.compile(r"[A-Za-z']+")

FEATURE_GROUPS = {
    "length": ["n_words", "n_chars", "n_sentences", "n_lines"],
    "lexical": ["ttr_first100", "mean_word_len", "mean_sent_len", "frac_long_words", "frac_first_person", "frac_second_person", "frac_contractions"],
    "markdown": ["bullet_lines", "numbered_lines", "header_lines", "code_blocks", "bold", "inline_code", "frac_blank_lines"],
    "punctuation": ["commas_pw", "exclam_pw", "question_pw", "colon_pw", "semicolon_pw", "paren_pw", "dash_pw", "quote_pw", "non_ascii_pw"],
    "opening": ["opens_sure", "opens_hello", "opens_as_ai", "opens_sorry", "opens_yes_no", "opens_restate", "mentions_ai_model"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def featurize(text: str) -> dict:
    words = WORD_RE.findall(text)
    lw = [w.lower() for w in words]
    nw = max(len(words), 1)
    lines = text.split("\n")
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    first = text.strip()[:80].lower()
    return {
        "n_words": np.log1p(len(words)),
        "n_chars": np.log1p(len(text)),
        "n_sentences": np.log1p(len(sents)),
        "n_lines": np.log1p(len(lines)),
        "ttr_first100": len(set(lw[:100])) / max(len(lw[:100]), 1),
        "mean_word_len": np.mean([len(w) for w in words]) if words else 0.0,
        "mean_sent_len": len(words) / max(len(sents), 1),
        "frac_long_words": sum(len(w) >= 9 for w in words) / nw,
        "frac_first_person": sum(w in ("i", "me", "my", "mine", "i'm", "i've", "i'd", "i'll") for w in lw) / nw,
        "frac_second_person": sum(w in ("you", "your", "yours", "you're", "you've") for w in lw) / nw,
        "frac_contractions": sum("'" in w for w in words) / nw,
        "bullet_lines": sum(bool(re.match(r"\s*[-*+•]\s+", l)) for l in lines) / len(lines),
        "numbered_lines": sum(bool(re.match(r"\s*\d+[.)]\s+", l)) for l in lines) / len(lines),
        "header_lines": sum(bool(re.match(r"\s{0,3}#{1,6}\s", l)) for l in lines) / len(lines),
        "code_blocks": text.count("```") / 2,
        "bold": text.count("**") / 2,
        "inline_code": len(re.findall(r"(?<!`)`[^`\n]+`(?!`)", text)),
        "frac_blank_lines": sum(not l.strip() for l in lines) / len(lines),
        "commas_pw": text.count(",") / nw,
        "exclam_pw": text.count("!") / nw,
        "question_pw": text.count("?") / nw,
        "colon_pw": text.count(":") / nw,
        "semicolon_pw": text.count(";") / nw,
        "paren_pw": text.count("(") / nw,
        "dash_pw": len(re.findall(r"\s[-–—]\s|—", text)) / nw,
        "quote_pw": (text.count('"') + text.count("“")) / nw,
        "non_ascii_pw": sum(ord(c) > 127 for c in text) / nw,
        "opens_sure": float(bool(re.match(r"(sure|certainly|of course|absolutely)\b", first))),
        "opens_hello": float(bool(re.match(r"(hello|hi|hey|greetings)\b", first))),
        "opens_as_ai": float(bool(re.match(r"as an? (ai|artificial|language model|helpful)", first))),
        "opens_sorry": float(bool(re.match(r"(i'm sorry|i am sorry|sorry|i apologi[sz]e|unfortunately)", first))),
        "opens_yes_no": float(bool(re.match(r"(yes|no)\b", first))),
        "opens_restate": float(bool(re.match(r"(the |this |here )", first))),
        "mentions_ai_model": float(bool(re.search(r"\b(ai language model|as an ai|large language model)\b", text.lower()))),
    }


def opening(text, n=3):
    return " ".join(text.strip().split()[:n]).lower()


def main():
    df = load_dataset()
    feats = pd.DataFrame([featurize(t) for t in df.response])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    out = {}

    # ---------- per-family descriptive statistics (raw scale for interpretability)
    raw_words = df.response.str.split().str.len()
    desc = {}
    for fam in FAMILIES:
        g = df[df.family == fam]
        rw = raw_words[df.family == fam]
        resp = g.response
        desc[fam] = {
            "median_words": float(rw.median()), "mean_words": float(rw.mean()),
            "ttr_first100": float(g.ttr_first100.mean()),
            "mean_word_len": float(g.mean_word_len.mean()),
            "pct_with_bullets": float((g.bullet_lines > 0).mean()),
            "pct_with_numbered": float((g.numbered_lines > 0).mean()),
            "pct_with_headers": float((g.header_lines > 0).mean()),
            "pct_with_code_block": float((g.code_blocks > 0).mean()),
            "pct_with_bold": float((g.bold > 0).mean()),
            "commas_per_100w": float(100 * g.commas_pw.mean()),
            "exclam_per_100w": float(100 * g.exclam_pw.mean()),
            "pct_opens_sure": float(g.opens_sure.mean()),
            "pct_opens_hello": float(g.opens_hello.mean()),
            "pct_opens_as_ai": float(g.opens_as_ai.mean()),
            "pct_mentions_ai_model": float(g.mentions_ai_model.mean()),
            "frac_first_person": float(g.frac_first_person.mean()),
            "top_openings": [[o, int(c), float(c / len(g))] for o, c in Counter(opening(t) for t in resp).most_common(5)],
        }
    out["per_family"] = desc

    # ---------- vocabulary overlap: Jaccard of top-1000 lower-cased words (train split)
    tr_df = df[df.split == "train"]
    top = {}
    for fam in FAMILIES:
        cnt = Counter(w.lower() for t in tr_df[tr_df.family == fam].response for w in WORD_RE.findall(t))
        top[fam] = {w for w, _ in cnt.most_common(1000)}
    out["vocab_jaccard_top1000"] = {a: {b: len(top[a] & top[b]) / len(top[a] | top[b]) for b in FAMILIES} for a in FAMILIES}
    out["vocab_unique_top1000"] = {a: sorted(top[a] - set().union(*(top[b] for b in FAMILIES if b != a)))[:40] for a in FAMILIES}

    # ---------- feature-only classifiers
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    Xtr, Xte = tr[FEATURES].values, te[FEATURES].values
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)).fit(Xtr, tr.label)
    out["logreg"] = compute_metrics(te.label.values, lr.predict(Xte), tasks=te.task.values)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(Xtr, tr.label)
    out["gboost"] = compute_metrics(te.label.values, gb.predict(Xte), tasks=te.task.values)
    out["gboost_val_acc"] = float((gb.predict(va[FEATURES].values) == va.label.values).mean())
    out["analysis_split"] = "permutation importance and group ablations use the validation split; test is scored once"
    pi = permutation_importance(gb, va[FEATURES].values, va.label.values, n_repeats=5, random_state=0, scoring="accuracy")
    out["gboost_permutation_importance"] = dict(sorted(
        {f: float(m) for f, m in zip(FEATURES, pi.importances_mean)}.items(), key=lambda kv: -kv[1]))
    # group importance: drop a whole feature group and retrain
    grp = {}
    for gname, cols in FEATURE_GROUPS.items():
        keep = [f for f in FEATURES if f not in cols]
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(tr[keep].values, tr.label)
        grp[gname] = {"acc_without_group": float((m.predict(va[keep].values) == va.label.values).mean())}
        m2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(tr[cols].values, tr.label)
        grp[gname]["acc_group_only"] = float((m2.predict(va[cols].values) == va.label.values).mean())
    out["gboost_group_ablation"] = grp
    coefs = lr[-1].coef_
    out["logreg_top_coefs"] = {
        fam: sorted([[f, float(c)] for f, c in zip(FEATURES, coefs[i])], key=lambda x: -x[1])[:6] for i, fam in enumerate(FAMILIES)}
    out["feature_groups"] = FEATURE_GROUPS
    json.dump(out, open("results/rq4_features.json", "w"), indent=1)

    # ---------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.boxplot([raw_words[df.family == f] for f in FAMILIES], showfliers=False)
    ax.set_xticks(range(1, len(FAMILIES) + 1), FAMILIES)
    ax.set_ylabel("response length (words)")
    ax.set_title("Response length by family")
    fig.tight_layout(); fig.savefig("figures/rq4_length_box.png", dpi=150); plt.close(fig)

    keys = ["pct_with_bullets", "pct_with_numbered", "pct_with_headers", "pct_with_bold", "pct_with_code_block"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w = 0.16
    for i, fam in enumerate(FAMILIES):
        ax.bar(np.arange(len(keys)) + i * w, [desc[fam][k] for k in keys], w, label=fam)
    ax.set_xticks(np.arange(len(keys)) + 2 * w, [k.replace("pct_with_", "") for k in keys])
    ax.set_ylabel("fraction of responses")
    ax.set_title("Markdown use by family")
    ax.legend(fontsize=7, ncol=5)
    fig.tight_layout(); fig.savefig("figures/rq4_markdown.png", dpi=150); plt.close(fig)

    imp = list(out["gboost_permutation_importance"].items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.barh([k for k, _ in imp], [v for _, v in imp])
    ax.set_xlabel("drop in validation accuracy when permuted")
    ax.set_title("Gradient boosting: permutation importance")
    fig.tight_layout(); fig.savefig("figures/rq4_importance.png", dpi=150); plt.close(fig)

    J = np.array([[out["vocab_jaccard_top1000"][a][b] for b in FAMILIES] for a in FAMILIES])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(J, cmap="viridis", vmin=0.5, vmax=1)
    ax.set_xticks(range(5), FAMILIES, rotation=30); ax.set_yticks(range(5), FAMILIES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if J[i, j] < 0.8 else "black")
    ax.set_title("Top-1000 vocabulary Jaccard")
    fig.tight_layout(); fig.savefig("figures/rq4_vocab_jaccard.png", dpi=150); plt.close(fig)

    print("logreg acc", out["logreg"]["accuracy"], "gboost acc", out["gboost"]["accuracy"])
    print(json.dumps(out["gboost_group_ablation"], indent=1))
    print(list(out["gboost_permutation_importance"].items())[:10])


if __name__ == "__main__":
    main()
