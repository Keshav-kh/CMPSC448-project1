"""CNN and BiLSTM text classifiers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        mask = (x != 0)
        e = self.drop(self.emb(x)).transpose(1, 2)  # B x E x T
        pooled = []
        for conv, k in zip(self.convs, [c.kernel_size[0] for c in self.convs]):
            h = F.relu(conv(e))  # B x F x (T-k+1)
            # window valid if its first position is a real token
            valid = mask[:, : h.size(2)].unsqueeze(1)
            h = h.masked_fill(~valid, float("-inf"))
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Embedding -> BiLSTM -> [masked mean ; masked max] over time -> dropout -> linear.

    Sequences are not packed (packing was ~3x slower on our CPU); padded positions are
    excluded from both poolings via the mask, and batches are length-bucketed so padding is small.
    """

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=64, layers=1, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 * 2, n_classes)

    def forward(self, x, lens):
        out, _ = self.lstm(self.drop(self.emb(x)))
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
