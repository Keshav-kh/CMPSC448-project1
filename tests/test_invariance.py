"""Predictions must not depend on padding, batch size, or batch neighbours.

python tests/test_invariance.py   (exits non-zero on failure)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import collate  # noqa: E402
from models import BiLSTM, TextCNN  # noqa: E402

TOL = 1e-5


def check(model_cls):
    torch.manual_seed(0)
    m = model_cls(200, 5).eval()
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(1, 200, (n,), generator=g).tolist() for n in (1, 2, 4, 6, 9, 30, 120, 256)]
    alone = torch.cat([m(*collate([(s, 0)])[:2]) for s in seqs])          # batch size 1, no padding
    together = m(*collate([(s, 0) for s in seqs])[:2])                    # one batch, heavy padding
    order = list(reversed(range(len(seqs))))
    shuffled = m(*collate([(seqs[i], 0) for i in order])[:2])[torch.argsort(torch.tensor(order))]
    X, lens, _ = collate([(s, 0) for s in seqs])
    extra = m(torch.cat([X, torch.zeros(len(seqs), 50, dtype=torch.long)], 1), lens)  # extra padding
    for name, other in (("batched", together), ("reordered", shuffled), ("extra padding", extra)):
        diff = float((alone - other).abs().max())
        status = "OK " if diff < TOL else "FAIL"
        print(f"{status} {model_cls.__name__:8s} {name:14s} max |logit diff| = {diff:.2e}")
        if diff >= TOL:
            return False
    return True


if __name__ == "__main__":
    ok = all([check(TextCNN), check(BiLSTM)])
    sys.exit(0 if ok else 1)
