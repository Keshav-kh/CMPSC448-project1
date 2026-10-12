"""Helpers that turn results/*.json into markdown tables and extra figures for the report.

Every number that appears in report.md is produced here (or by make_report.py) from results/*.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import FAMILIES  # noqa: E402

R = "results"


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pm(s, d=1):
    """mean ± std (percent) from an eval.ms() dict."""
    if s["n_seeds"] > 1:
        return f"{100 * s['mean']:.{d}f} ± {100 * s['std']:.{d}f}"
    return f"{100 * s['mean']:.{d}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def bar_figure(path, labels, series, ylabel, title, errs=None, chance=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(labels))
    for i, (name, vals) in enumerate(series.items()):
        e = errs[name] if errs else None
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=e, capsize=3, label=name)
    if chance is not None:
        ax.axhline(chance, color="gray", ls="--", lw=1, label="chance")
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curve_figure(path, histories):
    """Training loss and validation accuracy per epoch; one colour per model, one line per seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, (name, runs) in enumerate(histories.items()):
        for j, h in enumerate(runs):
            ep = [e["epoch"] for e in h]
            kw = dict(color=f"C{i}", marker="o", ms=3, lw=1, label=name if j == 0 else None)
            ax_l.plot(ep, [e["train_loss"] for e in h], **kw)
            ax_a.plot(ep, [e["val_acc"] for e in h], **kw)
    ax_l.set_xlabel("epoch")
    ax_l.set_ylabel("training loss")
    ax_a.set_xlabel("epoch")
    ax_a.set_ylabel("validation accuracy")
    ax_a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
