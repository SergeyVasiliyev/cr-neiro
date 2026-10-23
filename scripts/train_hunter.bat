@echo off
rem A hunter: the copy of the user (runs\clone\user_v1.pt, tools\clone_user.py) trained to beat the frozen top policy
rem (runs\v7e\top.pt) and nothing else, starting from the user's style so it finds the holes the user finds.
rem   scripts\train_hunter.bat [hours] [target] [run name]
rem Progress: runs\<run>\train.log; the result: runs\<run>\final.pt (">>> eval" lines: vs the target). PC stays awake.
rem Best started through scripts\run_detached.bat, so a closing Claude session does not take it down.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=3
set TARGET=%~2
if "%TARGET%"=="" set TARGET=runs/v7e/top.pt
set RUN=%~3
if "%RUN%"=="" set RUN=hunter_v2
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\%RUN%\train.log -Until "=== all done" -Hours 8
".venv\Scripts\python.exe" python\train_v7.py --run %RUN% --init runs\clone\user_v1.pt --hours %HOURS% ^
  --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 --leak-relative ^
  --lr 5e-5 --lr-range 1e-5 5e-5 --anchor runs\clone\user_v1.pt --anchor-coef 0.05 0 --anchor-until 0.4 ^
  --w-shaping 0.1 0.1 --w-style 0.3 0.3 --ent-place 0.8 0.5 ^
  --self-envs 4 --pool-envs 92 --fixed-share 0 --pool %TARGET% ^
  --eval-minutes 45 --eval-games 200 --final-games 400 --bench %TARGET% heuristic
