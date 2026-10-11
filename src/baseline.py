"""TF-IDF + logistic regression baseline (all three RQ2 setups), input-only leakage checks,
and truncation statistics.

python src/baseline.py -> results/baseline_tfidf.json, results/input_leakage.json, results/truncation.json

Significance uses a prompt-cluster bootstrap: prompts (not rows) are resampled, because rows that
share a prompt are not independent.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(__file__))
from data import FAMILIES, build_tokens, load_dataset  # noqa: E402
from eval import cluster_bootstrap, compute_metrics  # noqa: E402

TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def text_for(df, setup):
    if setup == "output":
        return df.response.tolist()
    if setup == "input":
        return df.prompt.tolist()
    return (df.prompt + " <sep> " + df.response).tolist()


def tfidf_lr(tr, te, setup):
    vec = TfidfVectorizer(lowercase=False, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                          min_df=2, max_features=200000, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(text_for(tr, setup)), vec.transform(text_for(te, setup))
    clf = LogisticRegression(C=4.0, max_iter=2000)
    clf.fit(Xtr, tr.label)
    return clf.predict(Xte)


def main():
    df = load_dataset()
    tr, te = df[df.split == "train"], df[df.split == "test"]
    res = {}
    for setup in ("output", "input", "both"):
        p = tfidf_lr(tr, te, setup)
        res[setup] = compute_metrics(te.label.values, p, tasks=te.task.values)
        res[setup]["bootstrap"] = cluster_bootstrap(te.label.values, p, te.prompt_id.values)
        print(setup, round(res[setup]["accuracy"], 4), flush=True)
    res["note"] = "single deterministic run (no seeds); bootstrap CI resamples test prompts"
    json.dump(res, open("results/baseline_tfidf.json", "w"), indent=1)

    # ---------------- input-only leakage analysis
    leak = {"chance_uniform": 1 / len(FAMILIES)}
    counts = te.family.value_counts()
    leak["test_majority_class"] = counts.idxmax()
    leak["test_majority_rate"] = float(counts.max() / len(te))
    fam_per_prompt = df.groupby("prompt").family.nunique()
    leak["n_prompts"] = int(len(fam_per_prompt))
    leak["n_prompts_multi_family"] = int((fam_per_prompt > 1).sum())
    leak["rows_whose_prompt_has_multi_family"] = int(df.prompt.map(fam_per_prompt).gt(1).sum())
    best = te.groupby("prompt").family.agg(lambda s: s.value_counts().iloc[0]).sum()
    leak["test_input_only_upper_bound"] = float(best / len(te))
    b = res["input"]["bootstrap"]
    leak["tfidf_input_acc"] = res["input"]["accuracy"]
    leak["tfidf_input_ci95"] = [b["ci_low"], b["ci_high"]]
    leak["tfidf_input_p_vs_uniform"] = b["p_le_chance"]

    def meta_clf(cols_tr, cols_te, name):
        clf = LogisticRegression(max_iter=2000).fit(cols_tr, tr.label)
        p = clf.predict(cols_te)
        leak[f"meta_{name}_acc"] = float((p == te.label.values).mean())
        leak[f"meta_{name}_ci95"] = (lambda bb: [bb["ci_low"], bb["ci_high"]])(cluster_bootstrap(te.label.values, p, te.prompt_id.values))

    enc = OneHotEncoder(handle_unknown="ignore")
    meta_clf(enc.fit_transform(tr[["source"]]), enc.transform(te[["source"]]), "source")
    enc2 = OneHotEncoder(handle_unknown="ignore")
    meta_clf(enc2.fit_transform(tr[["task"]]), enc2.transform(te[["task"]]), "task")
    plen = lambda d: np.log1p(d.prompt.str.split().str.len().values)[:, None]
    meta_clf(plen(tr), plen(te), "prompt_length")

    # Controls: does input-only signal survive without Bard (the family with a skewed source mix),
    # and within a single source where every family is present?
    for name, mask_tr, mask_te, n_cls in (
        ("no_bard", tr.family != "Bard", te.family != "Bard", 4),
        ("within_sharegpt", tr.source == "sharegpt", te.source == "sharegpt", 5),
    ):
        p = tfidf_lr(tr[mask_tr], te[mask_te], "input")
        t = te[mask_te]
        bb = cluster_bootstrap(t.label.values, p, t.prompt_id.values, chance=1 / n_cls)
        maj = float(t.family.value_counts().max() / len(t))
        leak[f"input_only_{name}"] = {"acc": float((p == t.label.values).mean()), "ci95": [bb["ci_low"], bb["ci_high"]],
                                      "p_le_chance": bb["p_le_chance"], "chance": 1 / n_cls, "majority_rate": maj, "n_test": int(len(t))}
    leak["source_by_family_test"] = pd.crosstab(te.source, te.family).to_dict(orient="index")
    leak["median_prompt_words_by_family"] = df.groupby("family").prompt.apply(lambda s: float(s.str.split().str.len().median())).to_dict()
    json.dump(leak, open("results/input_leakage.json", "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in leak.items() if not isinstance(v, dict) or k.startswith("input_only")}, indent=1))

    # ---------------- truncation: how much of each response the 256-token models see
    ntok = pd.Series([len(t) for t in build_tokens(df, "output", max_len=10**9)], index=df.index)
    trunc = {"max_len": 256,
             "pct_over_256_all": float((ntok > 256).mean()),
             "pct_over_256_by_family": {f: float((ntok[df.family == f] > 256).mean()) for f in FAMILIES},
             "median_tokens_by_family": {f: float(ntok[df.family == f].median()) for f in FAMILIES},
             "pct_over_64_by_family": {f: float((ntok[df.family == f] > 64).mean()) for f in FAMILIES}}
    json.dump(trunc, open("results/truncation.json", "w"), indent=1)
    print(json.dumps(trunc, indent=1))


if __name__ == "__main__":
    main()
