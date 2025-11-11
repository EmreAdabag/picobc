#!/usr/bin/env bash
set -euo pipefail

PROCS=8
EPS=400

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

  # first 4
  python -u collect_lerobot_data.py \
    --object_id "$i" \
    --goal_id_s "$goal_s" \
    --goal_id_e 4 \
    --root "${root}_p0" \
    --episodes_per_task $EPS \
    --seed $i \
    "$@" \
    >"datasets/out_${i}_p0.log" 2>&1 &

  # 4 to end
  python -u collect_lerobot_data.py \
    --object_id "$i" \
    --goal_id_s 4 \
    --goal_id_e "$goal_e" \
    --root "${root}_p1" \
    --episodes_per_task $EPS \
    --seed $i \
    "$@" \
    >"datasets/out_${i}_p1.log" 2>&1 &
done

echo "All collect_lerobot_data jobs started."
