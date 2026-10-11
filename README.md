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
tests/test_invariance.py  padding / batch-size / order invariance test for both models
scripts/*.txt         exact command lists used for every experiment; scripts/run_parallel.sh runs them
```

## Setup
```
python -m pip install -r requirements.txt
```
Tested with Python 3.14, torch 2.9.1 (CPU only), scikit-learn 1.7.2, pandas 2.3.3.

## Reproduce
One entry point runs everything (after `pip install -r requirements.txt`):
```
bash scripts/run_all.sh 4
```
Steps it runs, which can also be run individually:
1. `python tests/test_invariance.py` checks that CNN/BiLSTM outputs do not depend on padding, batch size or order.
2. `python src/prepare_data.py` rebuilds `data/processed/dataset.parquet` from the raw UltraFeedback JSONL (download command in DATA.md; the processed file is already included).
3. `python src/baseline.py` runs the TF-IDF baseline, the input-only leakage checks (prompt-cluster bootstrap, no-Bard and within-ShareGPT controls) and the truncation statistics. `python src/features.py` runs the RQ4 feature analysis. `python scripts/bench_lstm.py` runs the BiLSTM speed benchmark.
4. `bash scripts/run_parallel.sh scripts/v2_all_cmds.txt 4` trains all 96 neural runs: RQ1/RQ2 2 models x 3 setups x 3 seeds; RQ3 held-out and size-matched controls; RQ4 5 ablations x 2 models x 3 seeds. Runs that already have a result JSON are skipped.
5. `python src/eval.py` validates that every run shares the same model version, data file hash and training configuration, then aggregates them (mean ± std over seeds, prompt-cluster bootstrap CIs).
6. `python src/make_report.py` renders `report.md` and `report.pdf`.

`results/archive_v1_padding_bug/` keeps the first round of results, produced by models with a padding bug (see report Section 5); it is not used by any table.

## Resuming and running on a cluster (ICDS Roar Collab)
- `train.py` skips any run whose `results/runs/<name>.json` exists (`--force` to redo), so `bash scripts/resume_local.sh` resumes after an interruption.
- On Roar Collab: copy the repo, create a conda env (`module load anaconda; conda create -n whowrote python=3.11; source activate whowrote; pip install -r requirements.txt`), then submit one array task per command line:
  ```
  sbatch --account=open --array=1-96 scripts/icds_array_cpu.slurm scripts/v2_all_cmds.txt
  ```
  Copy `results/runs/*.json` back and run `python src/eval.py && python src/make_report.py`.
