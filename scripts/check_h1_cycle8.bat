@echo off
rem h1 cycle 8's post-run checks. The training finished on 2026-11-13 20:43 (runs\h1\final_cycle8.pt), but the job
rem was ended during its checks (2026-11-14 02:17, the PC went to sleep). The same checks as scripts\train_h1_cycle8.bat,
rem then tools\pick_start.py: the bot scripts switch to cycle 8 if it beats cycle 6 at temp 0.5.
rem   scripts\check_h1_cycle8.bat      (~30 min; raw\h1_cycle8_after.txt, ends with "=== all done")
cd /d "%~dp0.."
set NEW=runs/h1/final_cycle8.pt
set START=runs/h1/final_cycle6.pt
set PREV=runs/h1/final_cycle7.pt
set C5=runs/h1/final_cycle5.pt
set C4=runs/h1/final_cycle4.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set OUT=raw\h1_cycle8_after.txt
set PY=".venv\Scripts\python.exe"
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 2
echo ### %date% %time% (cycle 8, the long run from %START%, checked afterwards) > %OUT%
echo === cycle 8 final vs its start %START%, 800 games >> %OUT%
%PY% python\evaluate.py --model %NEW% --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %PREV% %C5% %C4% %CLONE%) do (
  echo === cycle 8 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model %NEW% --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 8 final >> %OUT%
%PY% tools\push_defence_check.py --model %NEW% --rollouts 4 >> %OUT% 2>&1
echo === spells on the sleeping king from the user's openings: cycle 8 final >> %OUT%
%PY% tools\king_wake_check.py --model %NEW% --back-starts --games 400 >> %OUT% 2>&1
echo === pick >> %OUT%
%PY% tools\pick_start.py --new %NEW% --save-as %NEW% --old %START% --result raw/h1_cycle8_after.txt --out raw/h1_cycle9_start.txt --cycle 8 --old-cycle 6 >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
