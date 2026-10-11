"""Train one CNN / BiLSTM run and write its metrics to results/<name>.json.

Examples
  python src/train.py --model cnn --setup output --seed 1
  python src/train.py --model rnn --setup both --seed 2
  python src/train.py --model cnn --setup output --holdout_task writing --seed 1   # RQ3
  python src/train.py --model cnn --setup output --ablation no_format --seed 1     # RQ4
  python src/train.py --model cnn --setup output --max_train 500 --epochs 1        # smoke test
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FAMILIES, TextDataset, Vocab, build_tokens, collate, load_dataset
from eval import compute_metrics
from models import build_model


class BucketBatchSampler:
    """Shuffle, sort by length within chunks of 50 batches, then shuffle the batches (less padding)."""

    def __init__(self, lengths, batch_size, seed):
        self.lengths, self.bs, self.rng = np.asarray(lengths), batch_size, np.random.RandomState(seed)

    def __iter__(self):
        idx = self.rng.permutation(len(self.lengths))
        chunk = self.bs * 50
        batches = []
        for i in range(0, len(idx), chunk):
            c = idx[i : i + chunk]
            c = c[np.argsort(self.lengths[c], kind="stable")]
            batches += [c[j : j + self.bs].tolist() for j in range(0, len(c), self.bs)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for X, lens, _ in loader:
        preds.append(model(X, lens).argmax(1))
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "rnn"], required=True)
    ap.add_argument("--setup", choices=["output", "input", "both"], default="output")
    ap.add_argument("--ablation", choices=["none", "no_length", "no_format", "lower_nopunct"], default="none")
    ap.add_argument("--holdout_task", default=None, help="RQ3: exclude this task from train/val, test on it")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_train", type=int, default=None, help="subsample train set (smoke tests)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out_dir", default="results/runs")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    name = args.name or "_".join(
        [args.model, args.setup, args.ablation]
        + ([f"holdout-{args.holdout_task}"] if args.holdout_task else [])
        + [f"s{args.seed}"]
    )

    df = load_dataset()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    if args.holdout_task:
        t = args.holdout_task
        te_all = df[df.task == t]           # every row of the unseen task (never trained on)
        te = te[te.task == t]               # same rows the in-domain models are scored on
        tr, va = tr[tr.task != t], va[va.task != t]
    if args.max_train:
        tr = tr.sample(n=min(args.max_train, len(tr)), random_state=args.seed)

    tok = lambda d: build_tokens(d, args.setup, args.ablation, args.max_len)
    tr_tok = tok(tr)
    vocab = Vocab(tr_tok)
    mk = lambda toks, d, shuffle: DataLoader(
        TextDataset(toks, d.label, vocab, args.max_len), batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate
    )
    train_ds = TextDataset(tr_tok, tr.label, vocab, args.max_len)
    train_loader = DataLoader(
        train_ds, batch_sampler=BucketBatchSampler([len(x) for x in train_ds.x], args.batch_size, args.seed),
        collate_fn=collate,
    )
    val_loader, test_loader = mk(tok(va), va, False), mk(tok(te), te, False)

    model = build_model(args.model, len(vocab), len(FAMILIES))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state, best_epoch, bad, history = -1.0, None, 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for X, lens, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(X, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        val_acc = float((predict(model, val_loader) == va.label.values).mean())
        history.append({"epoch": epoch, "train_loss": tot / n, "val_acc": val_acc, "time_s": time.time() - t0})
        print(f"[{name}] epoch {epoch} loss {tot / n:.4f} val_acc {val_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc, best_epoch, bad = val_acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    res = {
        "name": name, "args": vars(args), "n_params": n_params, "vocab_size": len(vocab),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "best_epoch": best_epoch, "epochs_run": len(history), "best_val_acc": best_acc,
        "train_time_s": time.time() - t0, "history": history,
    }
    p = predict(model, test_loader)
    res["test"] = compute_metrics(te.label.values, p, tasks=te.task.values)
    if args.holdout_task:
        all_loader = mk(tok(te_all), te_all, False)
        res["test_heldout_all_rows"] = compute_metrics(te_all.label.values, predict(model, all_loader))
        res["n_test_heldout_all_rows"] = len(te_all)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, name + ".json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{name}] TEST acc {res['test']['accuracy']:.4f} macroF1 {res['test']['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
