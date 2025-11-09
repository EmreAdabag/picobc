#!/usr/bin/env bash
set -euo pipefail

PROCS=4

for i in $(seq 0 $((PROCS-1))); do
  goal_s=0
  # For the first object, end range one short to omit (obj_0, goal_{n-1})
  if [[ "$i" -eq 0 ]]; then
    goal_e=$((PROCS-1))
  else
    goal_e=$PROCS
  fi
  root="datasets/obj_${i}"

  echo "Launching shard $i: obj_id=$i goal_id_s=$goal_s goal_id_e=$goal_e root=$root"

  python -u collect_lerobot_data.py \
    --object_id "$i" \
    --goal_id_s "$goal_s" \
    --goal_id_e "$goal_e" \
    --root "$root" \
    --episodes_per_task 400 \
    --seed $i \
    "$@" \
    >"datasets/out_${i}.log" 2>&1 &
done

echo "All collect_lerobot_data jobs started."
