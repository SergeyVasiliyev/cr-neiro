@echo off
rem v7 from scratch (see python\train_v7.py): three critics, self-play on both sides, opponents picked by win rate,
rem entropy targets, move delay and observation noise. The schedule runs on the clock: pass the hours it has.
rem   scripts\train_v7.bat 7.5
rem Progress: runs\v7\train.log (">>> eval" lines: greedy games against the bots, ppo_v5d and ppo_v6). PC stays awake.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=7.5
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\v7\train.log -Until "=== all done" -Hours 12
".venv\Scripts\python.exe" python\train_v7.py --run v7 --hours %HOURS%
