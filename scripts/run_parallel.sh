#!/usr/bin/env bash
# Run each line of a command file, N at a time. Usage: bash scripts/run_parallel.sh cmds.txt 4
N=${2:-4}
xargs -P "$N" -I{} bash -c "{}" < "$1"
