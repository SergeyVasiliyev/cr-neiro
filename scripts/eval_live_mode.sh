#!/bin/bash
# Every 2 h: the latest snapshot of a run against the panel at temperature 0.5 on both sides (as live), one line per
# check in raw/<run>_temp05.txt. The trainer's own evaluations are greedy, which hides a policy that spreads its plays.
#   bash scripts/eval_live_mode.sh v10
RUN=${1:-v10}
cd "$(dirname "$0")/.."
OUT=raw/${RUN}_temp05.txt
# (only the log since the latest start: a run resumed after it ended once has "=== all done" further up)
running() { [ -f runs/$RUN/latest.pt ] && ! tail -n +$(grep -n "^run dir" runs/$RUN/train.log | tail -1 | cut -d: -f1) runs/$RUN/train.log | grep -q "=== all done"; }
while running; do
  cp runs/$RUN/latest.pt raw/${RUN}_eval_copy.pt
  STEP=$(grep -E "^it " runs/$RUN/train.log | tail -1 | awk '{print $4}')
  LINE="$(date +%H:%M) at ${STEP}:"
  for OPP in heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt; do
    EXTRA="--decision-ticks 4"
    case $OPP in runs/*) EXTRA="--native-rhythm --opp-temp 0.5";; esac
    S=$(.venv/Scripts/python.exe python/evaluate.py --model raw/${RUN}_eval_copy.pt --opponent $OPP --games 200 --envs 100 \
        --model-temp 0.5 $EXTRA 2>/dev/null | grep -oE "score [0-9.]+" | awk '{print $2}')
    LINE="$LINE $(basename $OPP .pt) $S"
  done
  echo "$LINE" | tee -a $OUT
  sleep 7200
done
