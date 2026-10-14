@echo off
rem After recording two-sided battles: reads the new ones, labels them automatically, trains the next
rem detector from the current one and compares the two (several hours: fine overnight, PC stays awake).
rem Progress: raw\detector_round_log.txt (ends with "=== done"). Cut off? Run it again with --resume.
cd /d "%~dp0.."
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\detector_round_log.txt -Until "=== done" -Hours 14
".venv\Scripts\python.exe" tools\detector_round.py %*
pause
