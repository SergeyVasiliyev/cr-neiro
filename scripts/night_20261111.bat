@echo off
rem The night of 2026-11-11, after the user's live games against h1 (video46, video47): the user's clone v3 from every
rem two-sided battle recorded so far (video43-47, the user's own recordings: *_bNN_red), the clone v2 recipe (from the
rem all-tiles sx1, pull 0.05), checked against sx1; then h1 cycle 3 (scripts\train_h1_cycle3.bat, 8 h), which uses it.
rem Log: raw\night_20261111.txt (ends with "=== night done"); cycle 3's results: raw\h1_cycle3_after.txt.
rem Detached:  scripts\run_detached.bat night_20261111.bat
cd /d "%~dp0.."
setlocal EnableDelayedExpansion
set PY=".venv\Scripts\python.exe"
set LOG=raw\night_20261111.txt
rem first the perception of video46/47 and their cut into battles (raw\perceive_v46_47.log), at most an hour
for /L %%i in (1,1,60) do (
  findstr /C:"=== all done" raw\perceive_v46_47.log >nul 2>&1 && goto perceived
  timeout /t 60 /nobreak >nul
)
:perceived
set P=
for %%f in (raw\perception\cache\video*_b??_red.json) do set P=!P! %%~nf
echo === examples %date% %time%:!P! >> %LOG%
%PY% tools\human_examples.py --players !P! --spots 0 --out raw\bc\user_v3.npz >> %LOG% 2>&1
echo === clone v3 %date% %time% >> %LOG%
%PY% tools\clone_user.py --start runs/sx1/all_tiles.pt --data raw/bc/user_v3.npz --kl 0.05 --epochs 40 ^
  --holdout-every 5 --out runs/clone/user_v3_kl0.05.pt >> %LOG% 2>&1
echo === clone v3 vs sx1, 300 games >> %LOG%
%PY% python\evaluate.py --model runs/clone/user_v3_kl0.05.pt --opponent runs/sx1/best.pt --games 300 ^
  --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %LOG% 2>&1
echo === h1 cycle 3 %date% %time% >> %LOG%
call scripts\train_h1_cycle3.bat 8
echo === night done %date% %time% >> %LOG%
