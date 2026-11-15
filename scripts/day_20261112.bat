@echo off
rem The day of 2026-11-12: called by scripts\night_20261112.bat after h1 cycle 6's checks (a line appended to it while
rem it ran, the user's wish at 09:50: the next cycle right after cycle 6). Keeps cycle 6's final as
rem runs\h1\final_cycle6.pt, picks the better of it and cycle 6's start (tools\pick_start.py; the bot scripts then play
rem it), then h1 cycle 7 from it (8 h).
rem Log: raw\day_20261112.txt (ends with "=== day done"); cycle 7's results: raw\h1_cycle7_after.txt.
cd /d "%~dp0.."
set PY=".venv\Scripts\python.exe"
set LOG=raw\day_20261112.txt
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\h1_cycle7_after.txt -Until "=== all done" -Hours 11
set C6START=runs/h1/final_cycle5.pt
for /f "usebackq tokens=1" %%a in ("raw\h1_cycle6_start.txt") do set "C6START=%%a"
echo === pick %date% %time% (cycle 6's start: %C6START%) >> %LOG%
%PY% tools\pick_start.py --new runs/h1/final.pt --save-as runs/h1/final_cycle6.pt --old %C6START% ^
  --result raw/h1_cycle6_after.txt --out raw/h1_cycle7_start.txt --cycle 6 >> %LOG% 2>&1
set START=%C6START%
set PREV=%C6START%
for /f "usebackq tokens=1,2" %%a in ("raw\h1_cycle7_start.txt") do (
  set "START=%%a"
  set "PREV=%%b"
)
echo === h1 cycle 7 from %START% (the other: %PREV%) %date% %time% >> %LOG%
call scripts\train_h1_cycle7.bat 8 %START% %PREV%
echo === day done %date% %time% >> %LOG%
