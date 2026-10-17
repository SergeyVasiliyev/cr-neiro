@echo off
rem v7b: ppo_v6's play distilled into the bigger three-critic net, then trained on with python\train_v7.py
rem (pulled towards ppo_v6 at first, a fixed share of games against the bots and earlier nets).
rem   scripts\train_v7b.bat [distill hours] [training hours]
rem Progress: runs\v7b_distill\train.log, then runs\v7b\train.log (">>> eval" lines). PC stays awake.
cd /d "%~dp0.."
set DISTILL=%~1
if "%DISTILL%"=="" set DISTILL=1.25
set TRAIN=%~2
if "%TRAIN%"=="" set TRAIN=6
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\v7b\train.log -Until "=== all done" -Hours 10
".venv\Scripts\python.exe" python\train_v7.py --run v7b_distill --distill runs\ppo_v6\best.pt --lr 5e-4 --hours %DISTILL% --eval-minutes 999 || exit /b 1
".venv\Scripts\python.exe" python\train_v7.py --run v7b --init runs\v7b_distill\final.pt --anchor runs\ppo_v6\best.pt --hours %TRAIN%
