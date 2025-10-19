#!/usr/bin/env bash

set -e

DATASET=expert_1k.h5
OUT_ROOT=experimentsday2

mkdir -p "$OUT_ROOT"

for N in 200 1000; do
  OUT_DIR="$OUT_ROOT/ndemos_${N}"
  mkdir -p "$OUT_DIR"
  echo "Running num_demos=$N -> $OUT_DIR"
  python3 train.py --data "$DATASET" --out "$OUT_DIR" --num_demos "$N" --steps 55000
done

