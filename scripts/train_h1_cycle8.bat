@echo off
rem h1 cycle 8 (2026-11-12 evening, the user's idea): one long run instead of an 8-h cycle. Cycles 3-7 had only ~4.8 h
rem at the full rate before the anneal (mid-run evals always below the start; cycle 5 tied, cycle 7 lost to cycle 6
rem 42%%). Here ~20 h from cycle 6's final (the best: beats cycle 5 58%%, cycle 7 58%%): half-way through the schedule
rem as before, but --anneal 0.125, so ~15 h at the full rate and ~5 h of annealing. Recipe otherwise cycle 6/7's.
rem Don't stop it midway: a policy stopped at the full rate plays clearly worse.
rem Afterwards: the new final at temp 0.5 against cycle 6 (800 games) and the panel (400 each), the push wait and the
rem king spells, then tools\pick_start.py (the bot scripts switch to it if it beats cycle 6), in raw\h1_cycle8_after.txt
rem (ends with "=== all done").
rem   scripts\train_h1_cycle8.bat [hours]   (default 20)
rem Detached:  scripts\run_detached.bat train_h1_cycle8.bat 20
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=20
set START=runs/h1/final_cycle6.pt
set PREV=runs/h1/final_cycle7.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set C5=runs/h1/final_cycle5.pt
set C4=runs/h1/final_cycle4.pt
set C3=runs/h1/final_cycle3.pt
set C2=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set OUT=raw\h1_cycle8_after.txt
set PY=".venv\Scripts\python.exe"
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 24
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume %START% --progress 0.5 --anneal 0.125 ^
  --giant-starts 0.1 --giant-lead 2 4 --push-starts 0.2 --back-starts 0.15 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --anchor %CLONE% --anchor-coef 0.03 0.01 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --strong-weights 1 1 1 1 2 --strong-share 0.4 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %C2% %C3% %C4% %C5% %PREV% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --eval-minutes 120 --eval-games 150 --final-games 400
copy /y runs\h1\final.pt runs\h1\final_cycle8.pt >nul
set NEW=runs/h1/final_cycle8.pt
echo ### %date% %time% (cycle 8, the long run from %START%) > %OUT%
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
