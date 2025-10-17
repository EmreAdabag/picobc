#!/usr/bin/env bash

# Very basic sweep over number of demos
# Usage: ./run_train_sweep.sh [DATASET] [OUT_ROOT]
# Defaults: DATASET=expert.h5, OUT_ROOT=runs

set -e

DATASET=expert_multitask_20k.h5
OUT_ROOT=experiments

mkdir -p "$OUT_ROOT"

for N in 200 400 600 800 1000; do
  OUT_DIR="$OUT_ROOT/ndemos_${N}"
  mkdir -p "$OUT_DIR"
  echo "Running num_demos=$N -> $OUT_DIR"
  python3 train.py --data "$DATASET" --out "$OUT_DIR" --num_demos "$N" --steps 55000
done

