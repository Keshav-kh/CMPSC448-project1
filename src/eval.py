"""Metrics, plus aggregation of results/*.json into summary tables and figures.

python src/eval.py   -> results/summary.json, results/tables.md, figures/*.png
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import FAMILIES  # noqa: E402


def compute_metrics(y_true, y_pred, tasks=None):
    labels = list(range(len(FAMILIES)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_family_acc": {f: float(cm[i, i] / max(cm[i].sum(), 1)) for i, f in enumerate(FAMILIES)},
        "per_family_f1": dict(zip(FAMILIES, map(float, f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)))),
        "n": int(len(y_true)),
    }
    if tasks is not None:
        tasks = np.asarray(tasks)
        m["per_task_acc"] = {t: float((y_pred[tasks == t] == y_true[tasks == t]).mean()) for t in sorted(set(tasks))}
        m["per_task_n"] = {t: int((tasks == t).sum()) for t in sorted(set(tasks))}
    return m


def cluster_bootstrap_correct(correct, groups, chance=0.2, n_boot=2000, seed=0):
    """Accuracy CI that resamples *prompts* (clusters) with replacement, not rows.

    correct: per-row correctness in [0, 1] (may be averaged over seeds); groups: prompt ids.
    p_le_chance: share of bootstrap replicates with accuracy <= chance (one-sided p-value).
    """
    correct = np.asarray(correct, dtype=float)
    g_ids, inv = np.unique(np.asarray(groups), return_inverse=True)
    c_sum = np.bincount(inv, weights=correct)
    n = np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(g_ids), size=(n_boot, len(g_ids)))
    accs = c_sum[idx].sum(1) / n[idx].sum(1)
    return {"acc": float(correct.mean()), "ci_low": float(np.percentile(accs, 2.5)),
            "ci_high": float(np.percentile(accs, 97.5)), "p_le_chance": float((accs <= chance).mean()),
            "n_boot": n_boot, "n_clusters": int(len(g_ids))}


def cluster_bootstrap(y_true, y_pred, groups, chance=0.2, n_boot=2000, seed=0):
    return cluster_bootstrap_correct(np.asarray(y_true) == np.asarray(y_pred), groups, chance, n_boot, seed)


EXPECTED = {"epochs": 6, "patience": 2, "batch_size": 64, "lr": 1e-3, "max_len": 256, "max_train": None}


def validate_runs(runs):
    """Refuse to aggregate runs that do not share code version, data file and training config."""
    errors = []
    versions = {r.get("model_version") for r in runs}
    datas = {r.get("data_md5") for r in runs}
    if len(versions) != 1 or None in versions:
        errors.append(f"model_version differs or missing: {versions}")
    if len(datas) != 1 or None in datas:
        errors.append(f"data_md5 differs or missing: {datas}")
    for r in runs:
        for k, v in EXPECTED.items():
            if r["args"].get(k) != v:
                errors.append(f"{r['name']}: {k}={r['args'].get(k)} (expected {v})")
        if "device" not in r:
            errors.append(f"{r['name']}: no device recorded")
    if errors:
        raise SystemExit("Result validation failed:\n  " + "\n  ".join(errors[:20]))
    return {"n_runs": len(runs), "model_version": versions.pop(), "data_md5": datas.pop(),
            "devices": sorted({r["device"] for r in runs}), "expected_args": EXPECTED}


def ms(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "n_seeds": len(xs), "values": [float(x) for x in xs]}


def fmt(d):
    return f"{100 * d['mean']:.1f} ± {100 * d['std']:.1f}" if d["n_seeds"] > 1 else f"{100 * d['mean']:.1f}"


def plot_cm(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(FAMILIES)), FAMILIES, rotation=30)
    ax.set_yticks(range(len(FAMILIES)), FAMILIES)
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    runs = [json.load(open(p)) for p in sorted(glob.glob("results/runs/*.json"))]
    validation = validate_runs(runs)
    print("validated", validation)
    groups = defaultdict(list)
    for r in runs:
        groups[r["name"].rsplit("_s", 1)[0]].append(r)

    summary = {}
    for key, rs in sorted(groups.items()):
        s = {
            "accuracy": ms([r["test"]["accuracy"] for r in rs]),
            "macro_f1": ms([r["test"]["macro_f1"] for r in rs]),
            "best_val_acc": ms([r["best_val_acc"] for r in rs]),
            "epochs_run": [r["epochs_run"] for r in rs],
            "best_epoch": [r["best_epoch"] for r in rs],
            "train_time_s": ms([r["train_time_s"] for r in rs]),
            "n_params": rs[0]["n_params"], "vocab_size": rs[0]["vocab_size"],
            "n_train": rs[0]["n_train"], "n_test": rs[0]["n_test"],
            "per_family_acc": {f: ms([r["test"]["per_family_acc"][f] for r in rs]) for f in FAMILIES},
            "confusion_matrix_sum": np.sum([r["test"]["confusion_matrix"] for r in rs], axis=0).tolist(),
        }
        # prompt-cluster bootstrap CI of the seed-averaged accuracy
        tp = rs[0]["test_predictions"]
        correct = np.mean([np.asarray(r["test_predictions"]["y_pred"]) == np.asarray(r["test_predictions"]["y_true"]) for r in rs], axis=0)
        s["bootstrap"] = cluster_bootstrap_correct(correct, tp["prompt_id"])
        s["devices"] = sorted({r["device"] for r in rs})
        if "per_task_acc" in rs[0]["test"]:
            s["per_task_acc"] = {t: ms([r["test"]["per_task_acc"][t] for r in rs]) for t in rs[0]["test"]["per_task_acc"]}
            s["per_task_n"] = rs[0]["test"]["per_task_n"]
        if "test_heldout_all_rows" in rs[0]:
            s["heldout_all_rows_accuracy"] = ms([r["test_heldout_all_rows"]["accuracy"] for r in rs])
            s["heldout_all_rows_macro_f1"] = ms([r["test_heldout_all_rows"]["macro_f1"] for r in rs])
            s["heldout_all_rows_per_family_acc"] = {
                f: ms([r["test_heldout_all_rows"]["per_family_acc"][f] for r in rs]) for f in FAMILIES}
            s["n_heldout_all_rows"] = rs[0]["n_test_heldout_all_rows"]
        summary[key] = s

    os.makedirs("figures", exist_ok=True)
    for key, s in summary.items():
        if key.endswith("_none") and key.split("_")[1] == "output":
            plot_cm(s["confusion_matrix_sum"], f"{key.split('_')[0].upper()} output-only (sum of seeds)", f"figures/cm_{key}.png")
    summary["_validation"] = validation
    json.dump(summary, open("results/summary.json", "w"), indent=1)
    summary.pop("_validation")

    lines = ["| run | test acc (%) | macro-F1 (%) | seeds |", "|---|---|---|---|"]
    for key, s in summary.items():
        lines.append(f"| {key} | {fmt(s['accuracy'])} | {fmt(s['macro_f1'])} | {s['accuracy']['n_seeds']} |")
    open("results/tables.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
