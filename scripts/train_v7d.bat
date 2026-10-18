@echo off
rem v7d: v7c's best point (runs\v7c\step_9M.pt, 22:40: beat v7b 63%) trained on with v7c's settings but no
rem endgame starts - the low-tower overtime starts made it play worse (v7b 63% -> 14% in 4 h) and it still
rem never tried finishing a tower (that needs forced tries, later).
rem   scripts\train_v7d.bat [hours]
rem Progress: runs\v7d\train.log (">>> eval" lines). PC stays awake.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\v7d\train.log -Until "=== all done" -Hours 12
".venv\Scripts\python.exe" python\train_v7.py --run v7d --resume runs\v7c\step_9M.pt --hours %HOURS% ^
  --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 --endgame 0 ^
  --anchor runs\v7b\final.pt --anchor-coef 0.02 0 --anchor-until 0.3 ^
  --w-shaping 0.1 0.1 --w-style 0.3 0.3 --ent-place 1.0 0.6 ^
  --pool heuristic pusher runs/ppo_v5d/latest.pt runs/ppo_v6/best.pt runs/v7b/step_13M.pt runs/v7b/step_19M.pt runs/v7b/final.pt ^
  --bench heuristic pusher runs/ppo_v6/best.pt runs/v7b/final.pt
