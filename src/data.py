"""Tokenisation, vocabulary, input setups and text ablations."""
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"
FAMILIES = ["Bard", "Falcon", "GPT", "LLaMA-2", "MPT"]
FAM2ID = {f: i for i, f in enumerate(FAMILIES)}

# Case-preserving regex tokenizer: newlines, words, and single punctuation chars.
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")


def tokenize(text: str):
    return [NL if t == "\n" else t for t in TOKEN_RE.findall(text)]


# ---------------------------------------------------------------- ablations
MD_PATTERNS = [
    (re.compile(r"```[^\n]*"), " "),                        # code fences (keep code body)
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),           # headers
    (re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M), ""),  # bullet / numbered list markers
    (re.compile(r"\*\*|__|\*|`"), ""),                     # bold / italics / inline code
    (re.compile(r"\s*\n\s*"), " "),                         # line structure
]
PUNCT_RE = re.compile(r"[^\w\s]")


def strip_markdown(text: str) -> str:
    for pat, rep in MD_PATTERNS:
        text = pat.sub(rep, text)
    return text


def lower_nopunct(text: str) -> str:
    return re.sub(r"\s+", " ", PUNCT_RE.sub(" ", text.lower())).strip()


def apply_text_ablation(text: str, ablation: str) -> str:
    if ablation == "no_format":
        return strip_markdown(text)
    if ablation in ("lower_nopunct", "all_surface"):
        return lower_nopunct(strip_markdown(text) if ablation == "all_surface" else text)
    return text


# ---------------------------------------------------------------- vocab
class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=30000):
        # Keep the most frequent tokens seen at least min_freq times; rarer ones become <unk>.
        cnt = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in cnt.most_common(max_size) if c >= min_freq]
        self.itos = [PAD, UNK, SEP] + [w for w in words if w not in (PAD, UNK, SEP)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in toks]


def build_tokens(df: pd.DataFrame, setup: str, ablation: str = "none", max_len=256, prompt_len=64):
    """Return a list of token lists for each row according to the input setup.

    setup: 'output' (response only), 'input' (prompt only), 'both' (prompt <sep> response).
    ablation 'no_length': fixed-length window of the response; short responses are tiled
    cyclically so neither padding nor truncation reveals the length.
    ablation 'window64': first 64 response tokens only (control: same budget, length kept).
    ablation 'all_surface': markdown stripped + lowercased + punctuation removed + no_length.
    """
    # RQ2 compares the three setups; RQ4 ablations only change the response text.
    out = []
    for p, r in zip(df.prompt, df.response):
        r = apply_text_ablation(r, ablation)
        rt = tokenize(r)
        if ablation in ("no_length", "all_surface"):
            L = 64  # tile short responses so neither padding nor truncation reveals length
            rt = (rt * (L // max(len(rt), 1) + 1))[:L] if rt else [UNK] * L
        elif ablation == "window64":
            rt = rt[:64]  # control for no_length: same 64-token budget, length still visible
        if setup == "output":
            toks = rt[:max_len]
        elif setup == "input":
            toks = tokenize(p)[:max_len]
        elif setup == "both":
            pt = tokenize(p)[:prompt_len]
            toks = (pt + [SEP] + rt)[:max_len]
        else:
            raise ValueError(setup)
        out.append(toks if toks else [UNK])
    return out


class TextDataset(Dataset):
    def __init__(self, token_lists, labels, vocab, max_len=256):
        self.x = [vocab.encode(t)[:max_len] for t in token_lists]
        self.y = list(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def collate(batch):
    # Pad every sequence in the batch with 0 (<pad>) up to the longest one. lens records the real
    # lengths so the models can ignore the padding.
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs])
    X = torch.zeros(len(xs), int(lens.max()), dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, : len(x)] = torch.tensor(x)
    return X, lens, torch.tensor(ys)


def load_dataset(path="data/processed/dataset.parquet"):
    df = pd.read_parquet(path)
    df["label"] = df.family.map(FAM2ID).astype(int)
    return df
