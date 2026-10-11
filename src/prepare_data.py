"""Build the (model, family, prompt, response, task) dataset from openbmb/UltraFeedback.

Usage: python src/prepare_data.py [--per_family 4000]
Expects the raw UltraFeedback *.jsonl files in data/raw/ (see README for download).
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

SEED = 448

# model name -> family. Models not listed are dropped (see DECISIONS.md).
FAMILY = {
    "gpt-3.5-turbo": "GPT",
    "gpt-4": "GPT",
    "llama-2-7b-chat": "LLaMA-2",
    "llama-2-13b-chat": "LLaMA-2",
    "llama-2-70b-chat": "LLaMA-2",
    "bard": "Bard",
    "falcon-40b-instruct": "Falcon",
    "mpt-30b-chat": "MPT",
}

CODE_RE = re.compile(
    r"```|\b(code|python|java|javascript|typescript|c\+\+|c#|sql|html|css|php|ruby|golang|rust|swift|kotlin|"
    r"function|algorithm|program|script|regex|api|json|bash|shell|compile|debug|class|array|latex|matlab|"
    r"react|node\.js|django|flask|pandas|numpy)\b",
    re.I,
)
MATH_RE = re.compile(
    r"\b(calculate|equation|solve|integral|derivative|probability|arithmetic|algebra|geometry|"
    r"how many|how much|sum of|product of|percentage|percent|square root|math|mathematical|"
    r"remainder|divided by|multiply|fraction)\b|\d+\s*[\+\-\*/\^=]\s*\d+",
    re.I,
)
WRITE_RE = re.compile(
    r"\b(write|essay|story|poem|article|blog|letter|compose|rewrite|paraphrase|draft|slogan|"
    r"script for|narrative|lyrics|speech|summar(y|ize|ise)|headline|tweet|email)\b",
    re.I,
)


# Generator/serialisation artifacts (e.g. Falcon's literal "</s>" end-of-sequence token) would
# give away the label without saying anything about writing style, so they are removed.
ARTIFACT_RE = re.compile(r"</?s>|<\|endoftext\|>|<\|im_(?:start|end)\|>|\[/?INST\]")


def assign_task(source: str, prompt: str) -> str:
    if source in ("truthful_qa", "false_qa"):
        return "factual_qa"
    head = prompt[:600]
    if CODE_RE.search(head):
        return "coding"
    if MATH_RE.search(head):
        return "math"
    if source.startswith("flan"):
        return "nlp_task"
    if WRITE_RE.search(head):
        return "writing"
    return "chat"


def load_raw(raw_dir: str) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for c in r["completions"]:
                    rows.append(
                        dict(
                            source=r["source"],
                            prompt=r["instruction"],
                            model=c["model"],
                            response=c["response"],
                        )
                    )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--per_family", type=int, default=4000)
    ap.add_argument("--min_words", type=int, default=10)
    ap.add_argument("--max_words", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stats = {}

    df = load_raw(args.raw_dir)
    stats["raw_completions"] = len(df)
    stats["raw_prompts"] = int(df.prompt.nunique())
    stats["raw_per_model"] = df.model.value_counts().to_dict()

    df = df[df.model.isin(FAMILY)].copy()
    df["family"] = df.model.map(FAMILY)
    stats["after_family_filter"] = len(df)

    df["prompt"] = df.prompt.fillna("").astype(str).str.strip()
    df["response"] = df.response.fillna("").astype(str)
    has_art = df.response.str.contains(ARTIFACT_RE)
    stats["artifact_rows_removed_tokens_by_family"] = df[has_art].family.value_counts().to_dict()
    df["response"] = df.response.str.replace(ARTIFACT_RE, "", regex=True).str.strip()
    df = df[(df.prompt != "") & (df.response != "")]
    stats["after_drop_empty"] = len(df)
    df = df.drop_duplicates(subset=["response"])
    stats["after_dedup_response"] = len(df)
    df = df.drop_duplicates(subset=["prompt", "model"])
    stats["after_dedup_prompt_model"] = len(df)

    nw = df.response.str.split().str.len()
    df = df[(nw >= args.min_words) & (nw <= args.max_words)]
    stats["after_length_filter"] = len(df)
    stats["per_family_before_balance"] = df.family.value_counts().to_dict()

    rng = np.random.RandomState(SEED)
    n = min(args.per_family, df.family.value_counts().min())
    df = (
        df.groupby("family", group_keys=False)
        .apply(lambda g: g.sample(n=n, random_state=rng))
        .reset_index(drop=True)
    )
    stats["per_family_after_balance"] = df.family.value_counts().to_dict()

    df["task"] = [assign_task(s, p) for s, p in zip(df.source, df.prompt)]

    # Grouped 80/10/10 split by prompt: a prompt never crosses splits.
    prompts = df.prompt.unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(prompts)
    n_p = len(prompts)
    tr, va = int(0.8 * n_p), int(0.9 * n_p)
    split_of = {p: "train" for p in prompts[:tr]}
    split_of.update({p: "val" for p in prompts[tr:va]})
    split_of.update({p: "test" for p in prompts[va:]})
    df["split"] = df.prompt.map(split_of)
    df["prompt_id"] = pd.factorize(df.prompt)[0]

    # Shuffle rows so files and evaluation batches are not ordered by family.
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df["prompt_id"] = pd.factorize(df.prompt)[0]
    df = df[["prompt_id", "source", "task", "model", "family", "prompt", "response", "split"]]
    df.to_parquet(os.path.join(args.out_dir, "dataset.parquet"), index=False)

    stats["n_total"] = len(df)
    stats["n_unique_prompts"] = int(df.prompt.nunique())
    stats["responses_per_prompt"] = df.groupby("prompt").size().value_counts().sort_index().to_dict()
    fams_per_prompt = df.groupby("prompt").family.nunique()
    stats["prompts_with_multiple_families"] = int((fams_per_prompt > 1).sum())
    stats["split_counts"] = df.split.value_counts().to_dict()
    stats["split_family_counts"] = pd.crosstab(df.split, df.family).to_dict(orient="index")
    stats["task_family_counts"] = pd.crosstab(df.task, df.family).to_dict(orient="index")
    stats["source_family_counts"] = pd.crosstab(df.source, df.family).to_dict(orient="index")
    for s in ("train", "val", "test"):
        others = set(df[df.split != s].prompt)
        assert not (set(df[df.split == s].prompt) & others), "prompt leakage across splits"
    stats = json.loads(json.dumps(stats, default=int))
    with open(os.path.join(args.out_dir, "data_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
