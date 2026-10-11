#!/usr/bin/env bash
# Resume all experiments after an interruption: finished runs (result JSON exists) are skipped.
cd "$(dirname "$0")/.."
mkdir -p logs
cat scripts/rq12_cmds.txt scripts/rq4_cmds.txt scripts/rq3_cmds.txt > scripts/_all_cmds.txt
bash scripts/run_parallel.sh scripts/_all_cmds.txt "${1:-4}"
