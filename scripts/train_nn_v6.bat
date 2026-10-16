@echo off
rem Keeps training the best policy (ppo_v5d at first), mostly against itself, a little against the bots, in ~1 h
rem chunks; a chunk replaces the best one only if it beats it (see tools\train_more.py). PC stays awake.
rem Progress: raw\train_v6_log.txt (">>>" lines = results of each chunk); the best: runs\ppo_v6\best.pt.
rem Stop: close this window (best.pt stays the best so far). Started again, it goes on from best.pt.
cd /d "%~dp0.."
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\train_v6_log.txt -Until "=== all done" -Hours 13
".venv\Scripts\python.exe" tools\train_more.py --hours 12 %*
