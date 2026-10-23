@echo off
rem v9: a 2.2x bigger net (3.7M: 4 transformer layers of 224, a 64-channel grid, a 448 trunk) - v7e's top policy
rem distilled into it (1.5 h), then trained on in gated chunks (tools\train_gated.py) with a higher rate than for a
rem settled policy (up to 1e-4: a freshly distilled net grows at it, as v7b did); top, v7e's best, v7b, ppo_v6, v7d
rem and the bots in the pool / on the panel. Minibatches of 1024: 2048 nearly filled the 6 GB card and slowed it to a crawl.
rem   scripts\train_v9.bat [distill hours] [chunk hours in all]
rem Progress: runs\v9_distill\train.log, then raw\train_v9_log.txt (">>>" lines); the best: runs\v9\best.pt.
cd /d "%~dp0.."
set DISTILL=%~1
if "%DISTILL%"=="" set DISTILL=1.5
set HOURS=%~2
if "%HOURS%"=="" set HOURS=7
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\train_v9_log.txt -Until "=== gated training done" -Hours 14
".venv\Scripts\python.exe" python\train_v7.py --run v9_distill --distill runs\v7e\top.pt --size 224,4,7,64,448 ^
  --lr 5e-4 --minibatch 1024 --hours %DISTILL% --eval-minutes 999 --final-games 200 --bench heuristic runs/v7e/top.pt ^
  --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 --leak-relative || exit /b 1
".venv\Scripts\python.exe" tools\train_gated.py --name v9 --start runs\v9_distill\final.pt --hours %HOURS% ^
  --pool-add runs/v7e/best.pt runs/v7e/top.pt --panel-add runs/v7e/top.pt ^
  --chunk-extra "--minibatch 1024 --lr 1e-4 --lr-range 1e-5 1e-4 --leak-relative --ent-place 0.8 0.4"
