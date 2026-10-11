#!/usr/bin/env bash
# Canonical end-to-end reproduction (CPU). Usage: bash scripts/run_all.sh [parallel_jobs=4]
# Neural runs take several hours on a laptop CPU; on a Slurm cluster submit
# scripts/v2_all_cmds.txt with scripts/icds_array_cpu.slurm instead of step 4 (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/runs figures
python tests/test_invariance.py                       # 1. models must be padding/batch invariant
[ -f data/processed/dataset.parquet ] || python src/prepare_data.py   # 2. data (needs data/raw, see DATA.md)
python src/baseline.py                                # 3. TF-IDF baseline, leakage checks, truncation stats
python src/features.py                                #    RQ4 hand-crafted features
python scripts/bench_lstm.py                          #    packed vs length-reversed BiLSTM speed
bash scripts/run_parallel.sh scripts/v2_all_cmds.txt "${1:-4}"   # 4. all 96 CNN/BiLSTM runs (resumable)
python src/eval.py                                    # 5. validate + aggregate runs
python src/make_report.py                             # 6. report.md -> report.pdf
