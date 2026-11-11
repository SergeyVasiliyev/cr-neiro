@echo off
rem The checks h1 cycle 2 did not get (2026-11-10): the user stopped its training at 19:13, at 96.7%% of the schedule,
rem before scripts\train_h1_cycle2.bat reached them. Its last save (19:12, nearly annealed) is runs\h1\cycle2_stopped.pt.
rem At temp 0.5, native rhythm: against cycle 1's final (800 games), the panel (400 each), step_53M (the best greedy
rem evals of the run) against cycle 1's final, and the back giant check for both nets with the same seed.
rem Results in raw\h1_cycle2_after.txt (ends with "=== all done"). About an hour.
rem   scripts\check_h1_cycle2.bat
rem Detached:  scripts\run_detached.bat check_h1_cycle2.bat
cd /d "%~dp0.."
set NEW=runs/h1/cycle2_stopped.pt
set OLD=runs/h1/final_cycle1.pt
set OUT=raw\h1_cycle2_after.txt
set PY=".venv\Scripts\python.exe"
set EV=--envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 2
echo ### %date% %time% > %OUT%
echo === cycle 2 vs cycle 1 final, 800 games >> %OUT%
%PY% python\evaluate.py --model %NEW% --opponent %OLD% --games 800 %EV% >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt) do (
  echo === cycle 2 vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model %NEW% --opponent %%O --games 400 %EV% >> %OUT% 2>&1
)
echo === step_53M vs cycle 1 final, 800 games >> %OUT%
%PY% python\evaluate.py --model runs/h1/step_53M.pt --opponent %OLD% --games 800 %EV% >> %OUT% 2>&1
echo === back giant check: cycle 2 vs itself >> %OUT%
%PY% tools\back_giant_check.py --model %NEW% >> %OUT% 2>&1
echo === back giant check: cycle 1 final vs itself, the same seed >> %OUT%
%PY% tools\back_giant_check.py --model %OLD% >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
