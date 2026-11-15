@echo off
rem h1 cycle 7's post-run checks. The chain was stopped on 2026-11-12 19:02 at the user's request (to play without lag):
rem the training itself had just finished and runs\h1\final.pt was kept as runs\h1\final_cycle7.pt. The same checks as
rem scripts\train_h1_cycle7.bat (its start's push and king checks are in raw\h1_cycle6_after.txt already), then
rem tools\pick_start.py: the bot scripts switch to cycle 7 if it beats cycle 6 at temp 0.5.
rem   scripts\check_h1_cycle7.bat      (~25 min; raw\h1_cycle7_after.txt, ends with "=== all done")
cd /d "%~dp0.."
set NEW=runs/h1/final_cycle7.pt
set START=runs/h1/final_cycle6.pt
set PREV=runs/h1/final_cycle5.pt
set C4=runs/h1/final_cycle4.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set OUT=raw\h1_cycle7_after.txt
set PY=".venv\Scripts\python.exe"
echo ### %date% %time% (cycle 7 from %START%, checked afterwards) > %OUT%
echo === cycle 7 final vs its start %START%, 800 games >> %OUT%
%PY% python\evaluate.py --model %NEW% --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %PREV% %C4% %CLONE%) do (
  echo === cycle 7 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model %NEW% --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 7 final >> %OUT%
%PY% tools\push_defence_check.py --model %NEW% --rollouts 4 >> %OUT% 2>&1
echo === spells on the sleeping king from the user's openings: cycle 7 final >> %OUT%
%PY% tools\king_wake_check.py --model %NEW% --back-starts --games 400 >> %OUT% 2>&1
echo === pick >> %OUT%
%PY% tools\pick_start.py --new %NEW% --save-as %NEW% --old %START% --result raw/h1_cycle7_after.txt --out raw/h1_cycle8_start.txt --cycle 7 >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
