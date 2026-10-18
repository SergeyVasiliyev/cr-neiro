@echo off
rem v7e: v7c's way of training (no endgame starts) in gated 1.5 h chunks from the best policy so far (v7c's 22:40
rem snapshot at first, ~60% vs v7b); a chunk replaces the best only if it beats it fairly (each at its own rhythm)
rem and is no worse on a panel (v7b, ppo_v6, v7d, pusher). See tools\train_gated.py.
rem   scripts\train_v7e.bat [hours]
rem Progress: raw\train_v7e_log.txt (">>>" lines); the best: runs\v7e\best.pt. PC stays awake.
rem Stop: close this window (best.pt stays the best so far); started again, it goes on from best.pt.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=20
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\train_v7e_log.txt -Until "=== all done" -Hours 24
".venv\Scripts\python.exe" tools\train_gated.py --name v7e --start runs\v7c\step_9M.pt --hours %HOURS%
