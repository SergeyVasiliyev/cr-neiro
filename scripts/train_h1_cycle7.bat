@echo off
rem h1 cycle 7 (2026-11-12 day): from the better of cycle 6's final and cycle 5's (tools\pick_start.py picks it after
rem cycle 6's checks: scripts\day_20261112.bat), half-way through the schedule, cycle 6's recipe unchanged: the fixed
rem twin's push starts 20%%, back giant starts with blue 2-4 elixir ahead 10%%, the user's openings (--back-starts) 15%%.
rem Cycle 4 joins the pool and the panel (the strongest against v7e top so far: 79%%).
rem Afterwards: the new final at temp 0.5 against its start (800 games) and the panel (400 each), how much it waits at the
rem user's pushes, and its spells on the sleeping king from the user's openings, in raw\h1_cycle7_after.txt (ends with
rem "=== all done"). Keep the start if the final is not better.
rem   scripts\train_h1_cycle7.bat [hours] [start] [the other one]   (default 8, runs/h1/final_cycle6.pt, final_cycle5.pt)
rem Progress: runs\h1\train.log (appended to cycles 1-6).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
set START=%~2
if "%START%"=="" set START=runs/h1/final_cycle6.pt
set PREV=%~3
if "%PREV%"=="" set PREV=runs/h1/final_cycle5.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set C4=runs/h1/final_cycle4.pt
set C3=runs/h1/final_cycle3.pt
set C2=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set OUT=raw\h1_cycle7_after.txt
set PY=".venv\Scripts\python.exe"
echo h1 cycle 7 from %START% (the other: %PREV%)
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 10
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume %START% --progress 0.5 ^
  --giant-starts 0.1 --giant-lead 2 4 --push-starts 0.2 --back-starts 0.15 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --anchor %CLONE% --anchor-coef 0.03 0.01 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --anneal 0.2 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --strong-weights 1 1 1 1 2 --strong-share 0.4 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %C2% %C3% %C4% %PREV% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --eval-minutes 90 --eval-games 150 --final-games 400
echo ### %date% %time% (cycle 7 from %START%) > %OUT%
echo === cycle 7 final vs its start %START%, 800 games >> %OUT%
%PY% python\evaluate.py --model runs/h1/final.pt --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %PREV% %C4% %CLONE%) do (
  echo === cycle 7 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model runs/h1/final.pt --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 7 final, then its start >> %OUT%
%PY% tools\push_defence_check.py --model runs/h1/final.pt --rollouts 4 >> %OUT% 2>&1
%PY% tools\push_defence_check.py --model %START% --rollouts 4 >> %OUT% 2>&1
echo === spells on the sleeping king from the user's openings: cycle 7 final, then its start >> %OUT%
%PY% tools\king_wake_check.py --model runs/h1/final.pt --back-starts --games 400 >> %OUT% 2>&1
%PY% tools\king_wake_check.py --model %START% --back-starts --games 400 >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
