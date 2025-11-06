#!/usr/bin/env bash
set -euo pipefail

# Launch 10 parallel collect_lerobot_data.py processes.
# Each process handles 10 object IDs and writes to its own root.
#
# Usage:
#   ./launch_collect_lerobot_data.sh [extra args passed to python]
# Example (override episodes per task):
#   ./launch_collect_lerobot_data.sh --episodes_per_task 50

PROCS=30
SHARD_SIZE=6

for i in $(seq 0 $((PROCS-1))); do
  obj_s=$(( i * SHARD_SIZE ))
  obj_e=$(( (i + 1) * SHARD_SIZE ))
  root="datasets/run_${i}"

  echo "Launching shard $i: obj_id_s=$obj_s obj_id_e=$obj_e root=$root"

  # Run each collector in the background; capture logs per shard
  python -u collect_lerobot_data.py \
    --obj_id_s "$obj_s" \
    --obj_id_e "$obj_e" \
    --root "$root" \
    --episodes_per_task 100 \
    --seed $i \
    "$@" \
    >"datasets/out_${i}.log" 2>&1 &
done

echo "All collect_lerobot_data jobs started."