"""TF-IDF + logistic regression baseline (all three RQ2 setups) and input-only leakage checks.

python src/baseline.py -> results/baseline_tfidf.json, results/input_leakage.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(__file__))
from data import FAMILIES, load_dataset  # noqa: E402
from eval import compute_metrics  # noqa: E402

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
    return compute_metrics(te.label.values, clf.predict(Xte), tasks=te.task.values)


def main():
    df = load_dataset()
    tr, te = df[df.split.isin(["train"])], df[df.split == "test"]
    res = {}
    for setup in ("output", "input", "both"):
        res[setup] = tfidf_lr(tr, te, setup)
        print(setup, round(res[setup]["accuracy"], 4), round(res[setup]["macro_f1"], 4), flush=True)
    json.dump(res, open("results/baseline_tfidf.json", "w"), indent=1)

    # ---------------- input-only leakage analysis
    leak = {"chance_uniform": 1 / len(FAMILIES)}
    counts = te.family.value_counts()
    leak["test_majority_class"] = counts.idxmax()
    leak["test_majority_rate"] = float(counts.max() / len(te))
    # prompts repeating across families
    fam_per_prompt = df.groupby("prompt").family.nunique()
    leak["n_prompts"] = int(len(fam_per_prompt))
    leak["n_prompts_multi_family"] = int((fam_per_prompt > 1).sum())
    leak["rows_whose_prompt_has_multi_family"] = int(df.prompt.map(fam_per_prompt).gt(1).sum())
    # Upper bound on any input-only classifier on the test set (identical prompt -> identical prediction)
    best = te.groupby("prompt").family.agg(lambda s: s.value_counts().iloc[0]).sum()
    leak["test_input_only_upper_bound"] = float(best / len(te))
    # Is TF-IDF input-only better than chance?
    acc = res["input"]["accuracy"]
    k = int(round(acc * len(te)))
    leak["tfidf_input_acc"] = acc
    leak["binomtest_vs_uniform_p"] = float(binomtest(k, len(te), 1 / len(FAMILIES), alternative="greater").pvalue)
    leak["binomtest_vs_majority_p"] = float(binomtest(k, len(te), leak["test_majority_rate"], alternative="greater").pvalue)

    # Which prompt-side signals leak? LR on single metadata features.
    def meta_clf(cols_tr, cols_te, name):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(cols_tr, tr.label)
        m = compute_metrics(te.label.values, clf.predict(cols_te))
        leak[f"meta_{name}_acc"] = m["accuracy"]
        leak[f"meta_{name}_macro_f1"] = m["macro_f1"]

    enc = OneHotEncoder(handle_unknown="ignore")
    meta_clf(enc.fit_transform(tr[["source"]]), enc.transform(te[["source"]]), "source")
    enc2 = OneHotEncoder(handle_unknown="ignore")
    meta_clf(enc2.fit_transform(tr[["task"]]), enc2.transform(te[["task"]]), "task")
    plen = lambda d: np.log1p(d.prompt.str.split().str.len().values)[:, None]
    meta_clf(plen(tr), plen(te), "prompt_length")
    leak["source_by_family_test"] = pd.crosstab(te.source, te.family).to_dict(orient="index")
    leak["median_prompt_words_by_family"] = df.groupby("family").prompt.apply(lambda s: float(s.str.split().str.len().median())).to_dict()
    json.dump(leak, open("results/input_leakage.json", "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in leak.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
