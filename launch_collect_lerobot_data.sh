#!/usr/bin/env bash
set -euo pipefail

PROCS=4
EPS=400
GOFFSET=100

for i in $(seq 0 $((PROCS-1))); do
  goal_s=$((PROCS + GOFFSET))
  # For the first object, end range one short to omit (obj_0, goal_{n-1})
  if [[ "$i" -eq 0 ]]; then
    goal_e=$((goal_s + PROCS - 1))
  else
    goal_e=$((goal_s + PROCS))
  fi
  root="datasets/obj_${i}"

  echo "Launching shard $i: obj_id=$i goal_id_s=$goal_s goal_id_e=$goal_e root=$root"

  for j in $(seq $goal_s $((goal_e - 1))); do
    # first 1
    python -u collect_lerobot_data.py \
      --object_id "$i" \
      --goal_id_s "$j" \
      --goal_id_e "$((j + 1))" \
      --root "${root}_p${j}" \
      --episodes_per_task $EPS \
      --seed $i \
      "$@" \
      >"datasets/out_${i}_p${j}.log" 2>&1 &
  done
done

echo "All collect_lerobot_data jobs started."
