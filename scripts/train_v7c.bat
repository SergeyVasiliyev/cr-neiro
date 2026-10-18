@echo off
rem v7c: v7b (runs\v7b\final.pt) trained on, to get past its plateau (python\train_v7.py):
rem - decisions every 0.2 s instead of 0.3 (4 ticks; gammas and lambda rescaled to the same horizons in seconds);
rem - moves land 0.15-0.45 s after the decision (3-9 ticks, as measured live), the side waiting meanwhile;
rem - 15% endgame starts (late, towers at random HP): finishing a low tower, defending one, overtime;
rem - the hints stay faded (shaping 0.1, bank 0.3): the reward is the result;
rem - more entropy on which card where (1.0 -> 0.6, v7b ended at 0.6), a light pull to v7b at first;
rem - the pool is the strong ones: v7b and its 13M/19M snapshots, ppo_v6, ppo_v5d, and the two bots;
rem - same mild observation noise as v7b.
rem   scripts\train_v7c.bat [hours] [resume]    (resume: go on from runs\v7c\latest.pt; hours = time left)
rem Progress: runs\v7c\train.log (">>> eval" lines: greedy games vs the bots, ppo_v6 and v7b, and the overtime
rem finishing test). PC stays awake.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=12
set START=--init runs\v7b\final.pt
if "%~2"=="resume" set START=--resume runs\v7c\latest.pt
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\v7c\train.log -Until "=== all done" -Hours 16
".venv\Scripts\python.exe" python\train_v7.py --run v7c %START% --hours %HOURS% ^
  --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 --endgame 0.15 ^
  --anchor runs\v7b\final.pt --anchor-coef 0.02 0 --anchor-until 0.3 ^
  --w-shaping 0.1 0.1 --w-style 0.3 0.3 --ent-place 1.0 0.6 ^
  --pool heuristic pusher runs/ppo_v5d/latest.pt runs/ppo_v6/best.pt runs/v7b/step_13M.pt runs/v7b/step_19M.pt runs/v7b/final.pt ^
  --bench heuristic pusher runs/ppo_v6/best.pt runs/v7b/final.pt
