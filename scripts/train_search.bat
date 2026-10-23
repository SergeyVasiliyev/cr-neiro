@echo off
rem Expert iteration (tools\search_iter.py): the best policy plays itself with lookahead (python\search.py), a copy
rem learns the moves the lookahead found (soft targets), and replaces the best only if it beats it (at temperature
rem 0.5, as live, 800 games) and holds up on the panel. Round after round, a better net making a better search.
rem   scripts\train_search.bat [hours] [start checkpoint] [name] [promote]
rem Detached (survives a closed session):  scripts\run_detached.bat train_search.bat 10 runs/si2/start.pt si2 0.52
rem Progress: raw\train_<name>_log.txt (">>>" lines); the best: runs\<name>\best.pt.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=10
set START=%~2
if "%START%"=="" set START=runs/v7e/top.pt
set NAME=%~3
if "%NAME%"=="" set NAME=si
set PROMOTE=%~4
if "%PROMOTE%"=="" set PROMOTE=0.53
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\train_%NAME%_log.txt -Until "=== search iteration done" -Hours 16
".venv\Scripts\python.exe" tools\search_iter.py --name %NAME% --start %START% --hours %HOURS% --promote %PROMOTE%
