"""CNN and BiLSTM text classifiers.

Both models are padding-invariant: the output for a sequence depends only on its own
`lens` real tokens, never on how much padding its batch adds (tested in tests/test_invariance.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_VERSION = "v2-padding-invariant"


class TextCNN(nn.Module):
    """Kim-style CNN: embedding -> parallel Conv1d (k=3,4,5) -> max-over-time -> dropout -> linear."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes)
        self.max_k = max(kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lens):
        if x.size(1) < self.max_k:
            x = F.pad(x, (0, self.max_k - x.size(1)))
        e = self.drop(self.emb(x)).transpose(1, 2)  # B x E x T
        pooled = []
        for conv in self.convs:
            k = conv.kernel_size[0]
            h = F.relu(conv(e))  # B x F x (T-k+1); window i covers tokens i..i+k-1
            # A window is valid only if it lies entirely inside the real tokens. Sequences shorter
            # than k keep exactly one window (start 0) over their tokens + zero pad embeddings,
            # which is identical for every batch because x is always padded to at least max_k.
            last_start = (torch.clamp(lens, min=k) - k).unsqueeze(1)  # B x 1
            starts = torch.arange(h.size(2), device=x.device).unsqueeze(0)
            valid = (starts <= last_start).unsqueeze(1)
            h = h.masked_fill(~valid, float("-inf"))
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


def reverse_within_length(t, lens):
    """Reverse each sequence's first lens[i] steps along dim 1; padding positions stay in place."""
    T = t.size(1)
    pos = torch.arange(T, device=t.device).unsqueeze(0)  # 1 x T
    idx = torch.where(pos < lens.unsqueeze(1), lens.unsqueeze(1) - 1 - pos, pos)  # B x T
    return t.gather(1, idx.unsqueeze(2).expand(-1, -1, t.size(2)))


class BiLSTM(nn.Module):
    """Embedding -> BiLSTM -> [masked mean ; masked max] over time -> dropout -> linear.

    The backward direction is a second LSTM run on each sequence reversed *within its own
    length*, so it never reads padding before real tokens (equivalent to packed sequences,
    but ~3x faster on our CPU). Right-padding only follows real tokens in the forward pass,
    and padded positions are excluded from both poolings.
    """

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=64, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fwd = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.bwd = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 * 2, n_classes)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        out_f, _ = self.fwd(e)
        out_b, _ = self.bwd(reverse_within_length(e, lens))
        out_b = reverse_within_length(out_b, lens)
        out = torch.cat([out_f, out_b], dim=2)
        mask = (torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]).unsqueeze(2)
        mean = (out * mask).sum(1) / lens[:, None].clamp(min=1)
        mx = out.masked_fill(~mask, float("-inf")).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))


def build_model(name, vocab_size, n_classes):
    if name == "cnn":
        return TextCNN(vocab_size, n_classes)
    if name == "rnn":
        return BiLSTM(vocab_size, n_classes)
    raise ValueError(name)
