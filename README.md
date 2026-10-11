# Who Wrote It? Identifying LLM families from their responses (CMPSC 448)

CNN and BiLSTM classifiers (PyTorch) that predict which LLM family (GPT, LLaMA-2, Bard, Falcon, MPT)
wrote a response, built on the public **UltraFeedback** dataset. Report: [`report.pdf`](report.pdf)
(source: `report.md`). Data documentation: [`DATA.md`](DATA.md). Decision log: [`DECISIONS.md`](DECISIONS.md).

## Layout
```
src/prepare_data.py   download-free data builder: raw UltraFeedback JSONL -> data/processed/dataset.parquet
src/data.py           tokenizer, vocab, input setups (output / input / both), text ablations
src/models.py         TextCNN (k=3,4,5) and BiLSTM
src/train.py          one training run -> results/runs/<name>.json
src/eval.py           metrics + aggregation -> results/summary.json, results/tables.md, figures/cm_*.png
src/baseline.py       TF-IDF + logistic regression baseline, input-only leakage checks
src/features.py       RQ4 hand-crafted features, per-family stats, feature-only classifiers
src/make_report.py    builds report figures/tables from results/*.json and renders report.pdf
scripts/*.txt         exact command lists used for every experiment; scripts/run_parallel.sh runs them
```

## Setup
```
python -m pip install -r requirements.txt
```
Tested with Python 3.14, torch 2.9.1 (CPU only), scikit-learn 1.7.2, pandas 2.3.3.

## Reproduce
1. **Data** (≈940 MB download; the processed parquet is already committed, so this is optional):
   ```
   python -c "from huggingface_hub import hf_hub_download as d; [d('openbmb/UltraFeedback', f, repo_type='dataset', local_dir='data/raw') for f in ['evol_instruct.jsonl','false_qa.jsonl','flan.jsonl','sharegpt.jsonl','truthful_qa.jsonl','ultrachat.jsonl']]"
   python src/prepare_data.py
   ```
2. **Smoke test**: `python src/train.py --model cnn --setup output --max_train 400 --epochs 1 --out_dir results/smoke`
3. **RQ1 + RQ2** (2 models x 3 setups x 3 seeds): `bash scripts/run_parallel.sh scripts/rq12_cmds.txt 4`
4. **Baseline + input leakage**: `python src/baseline.py`
5. **RQ3** (leave-one-task-out): `bash scripts/run_parallel.sh scripts/rq3_cmds.txt 4`
6. **RQ4** features: `python src/features.py`; ablations: `bash scripts/run_parallel.sh scripts/rq4_cmds.txt 4`
7. **Aggregate + report**: `python src/eval.py && python src/make_report.py`

All runs use fixed seeds (1, 2, 3); data split seed is 448. Hardware: CPU only (16 logical cores), 4 runs in parallel x 4 threads.

## Resuming and running on a cluster (ICDS Roar Collab)
- `train.py` skips any run whose `results/runs/<name>.json` exists (`--force` to redo), so `bash scripts/resume_local.sh` resumes after an interruption.
- On Roar Collab: copy the repo, create a conda env (`module load anaconda; conda create -n whowrote python=3.11; source activate whowrote; pip install -r requirements.txt`), then submit one array task per command line:
  ```
  sed '/--gpus=1/d' scripts/icds_array.slurm > scripts/icds_array_cpu.slurm   # CPU version (used for our runs)
  sbatch --account=open --time=02:00:00 --array=1-24 scripts/icds_array_cpu.slurm scripts/rq3_cmds.txt
  sbatch --account=open --time=02:00:00 --array=1-18 scripts/icds_array_cpu.slurm scripts/rq4_cmds.txt
  ```
  Copy `results/runs/*.json` back and run `python src/eval.py && python src/make_report.py`.
