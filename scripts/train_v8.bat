@echo off
rem v8: v7e's top policy (chunk 4) trained on in gated 1.5 h chunks (tools\train_gated.py), aimed at what the user
rem beats it with - the hints stay light (0.1: a troop's worth depends too much on the moment), the leak penalty only
rem for a leak the opponent does not match, less entropy at the end of a chunk; v7e's best and top in the pool and on
rem the panel.
rem   scripts\train_v8.bat [hours]
rem Progress: raw\train_v8_log.txt (">>>" lines); the best: runs\v8\best.pt. PC stays awake.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=7
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\train_v8_log.txt -Until "=== gated training done" -Hours 24
".venv\Scripts\python.exe" tools\train_gated.py --name v8 --start runs\v7e\top.pt --hours %HOURS% ^
  --pool-add runs/v7e/best.pt runs/v7e/top.pt --panel-add runs/v7e/best.pt ^
  --chunk-extra "--leak-relative --ent-place 0.8 0.4"
