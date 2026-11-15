@echo off
rem The night of 2026-11-12: called by scripts\train_h1_cycle5.bat after its checks (a line appended to it while it ran,
rem the user's wish: the next cycle right after cycle 5). Keeps cycle 5's final as runs\h1\final_cycle5.pt, picks the
rem better of it and cycle 4's (tools\pick_start.py; the bot scripts then play it), then h1 cycle 6 from it (8 h).
rem Log: raw\night_20261112.txt (ends with "=== night done"); cycle 6's results: raw\h1_cycle6_after.txt.
cd /d "%~dp0.."
set PY=".venv\Scripts\python.exe"
set LOG=raw\night_20261112.txt
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\h1_cycle6_after.txt -Until "=== all done" -Hours 11
echo === pick %date% %time% >> %LOG%
%PY% tools\pick_start.py --new runs/h1/final.pt --save-as runs/h1/final_cycle5.pt --old runs/h1/final_cycle4.pt ^
  --result raw/h1_cycle5_after.txt --out raw/h1_cycle6_start.txt --cycle 5 >> %LOG% 2>&1
set START=runs/h1/final_cycle4.pt
set PREV=runs/h1/final_cycle4.pt
for /f "usebackq tokens=1,2" %%a in ("raw\h1_cycle6_start.txt") do (
  set "START=%%a"
  set "PREV=%%b"
)
echo === h1 cycle 6 from %START% (the other: %PREV%) %date% %time% >> %LOG%
call scripts\train_h1_cycle6.bat 8 %START% %PREV%
echo === night done %date% %time% >> %LOG%
rem 2026-11-12 09:55, the user's wish: h1 cycle 7 right after cycle 6 (appended while this ran)
call scripts\day_20261112.bat
