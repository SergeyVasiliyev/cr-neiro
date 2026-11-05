@echo off
rem Expert iteration from v10 (tools\search_iter.py) with a search that also looks where the policy would not:
rem two more candidates per search (a random card and spot each) and 5%% of the moments it is sure to wait searched
rem anyway, so plays it gives ~0 chance (a spell finishing a tower: the net 0/96, the search 39/48) can be learned.
rem   scripts\train_search_v10.bat [hours] [start checkpoint] [name]
rem Detached:  scripts\run_detached.bat train_search_v10.bat 9 runs/v10/final_cycle2.pt sx1
rem Progress: raw\train_<name>_log.txt (">>>" lines); the best: runs\<name>\best.pt.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=9
set START=%~2
if "%START%"=="" set START=runs/v10/final_cycle2.pt
set NAME=%~3
if "%NAME%"=="" set NAME=sx1
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\train_%NAME%_log.txt -Until "=== search iteration done" -Hours 14
".venv\Scripts\python.exe" tools\search_iter.py --name %NAME% --start %START% --hours %HOURS% --promote 0.52 --workers 3 --games 100 --top %START% --panel runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt "--games-extra=--explore 2 --think-anyway 0.05"
