"""Benchmark: packed vs length-reversed BiLSTM training step on CPU. Writes results/bench_lstm.json."""
import json, os, sys, time
import torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models import BiLSTM

class PackedBiLSTM(nn.Module):
    def __init__(self, V, C, emb=128, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(4 * hidden, C)
    def forward(self, x, lens):
        p = nn.utils.rnn.pack_padded_sequence(self.emb(x), lens, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(p)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        m = (torch.arange(x.size(1))[None] < lens[:, None]).unsqueeze(2)
        return self.fc(torch.cat([(out * m).sum(1) / lens[:, None], out.masked_fill(~m, -1e9).max(1).values], 1))

torch.set_num_threads(4); torch.manual_seed(0)
X = torch.randint(1, 30000, (64, 256)); lens = torch.randint(64, 257, (64,)); y = torch.randint(0, 5, (64,))
res = {"threads": 4, "batch": 64, "seq_len": 256, "hidden": 64, "torch": torch.__version__}
for name, m in (("packed_bilstm", PackedBiLSTM(30000, 5)), ("length_reversed_bilstm (ours)", BiLSTM(30000, 5))):
    opt = torch.optim.Adam(m.parameters()); ts = []
    for i in range(6):
        t = time.time(); opt.zero_grad(); nn.functional.cross_entropy(m(X, lens), y).backward(); opt.step(); ts.append(time.time() - t)
    res[name + "_s_per_batch"] = sum(ts[1:]) / 5
res["speedup"] = res["packed_bilstm_s_per_batch"] / res["length_reversed_bilstm (ours)_s_per_batch"]
json.dump(res, open("results/bench_lstm.json", "w"), indent=1); print(res)
