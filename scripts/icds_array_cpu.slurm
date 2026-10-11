#!/bin/bash
# Run one line of a command file per Slurm array task on ICDS Roar Collab.
# Usage (from the repo root):
#   N=$(wc -l < scripts/rq12_cmds.txt)
#   sbatch --account=<YOUR_ALLOCATION> --array=1-$N scripts/icds_array.slurm scripts/rq12_cmds.txt
#SBATCH --job-name=whowrote
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm_%A_%a.out

set -euo pipefail
CMDFILE=${1:?usage: sbatch ... scripts/icds_array.slurm <cmdfile>}
module load anaconda
source activate whowrote          # conda env created once, see README "Running on ICDS"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results/runs
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CMDFILE")
echo "[$(date)] host=$(hostname) task=$SLURM_ARRAY_TASK_ID: $CMD"
nvidia-smi -L || true
eval "$CMD"
