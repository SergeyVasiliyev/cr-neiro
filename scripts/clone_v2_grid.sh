#!/bin/bash
# The user's clone v2: examples from the 22 two-sided battles of 2026-11-09 (video43-45, the user on the red side, every
# tile allowed), clones from the all-tiles sx1 at three pulls to it, each checked against sx1 (temp 0.5, as live).
#   bash scripts/clone_v2_grid.sh            (log: raw/clone_v2_grid.log)
cd "$(dirname "$0")/.."
LOG=raw/clone_v2_grid.log
PY=.venv/Scripts/python.exe
P=""
for b in video43_b01 video43_b02 video43_b03 video43_b04 video43_b05 video43_b06 video43_b07 video43_b08 video43_b09 \
         video43_b10 video44_b01 video44_b02 video44_b05 video45_b01 video45_b02 video45_b03 video45_b04 video45_b05 \
         video45_b06 video45_b07 video45_b08 video45_b09; do
  P="$P ${b}_red"
done
echo "=== examples $(date)" > $LOG
$PY tools/human_examples.py --players $P --spots 0 --out raw/bc/user_v2.npz >> $LOG 2>&1
for KL in 0.05 0.2 0.5; do
  echo "=== clone kl $KL $(date)" >> $LOG
  $PY tools/clone_user.py --start runs/sx1/all_tiles.pt --data raw/bc/user_v2.npz --kl $KL --epochs 40 \
      --holdout-every 5 --out runs/clone/user_v2_kl$KL.pt 2>&1 | grep -v Warning >> $LOG
  $PY python/evaluate.py --model runs/clone/user_v2_kl$KL.pt --opponent runs/sx1/best.pt --games 300 \
      --model-temp 0.5 --opp-temp 0.5 --native-rhythm 2>&1 | grep -v Warning >> $LOG
done
echo "=== all done $(date)" >> $LOG
