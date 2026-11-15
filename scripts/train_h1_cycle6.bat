@echo off
rem h1 cycle 6 (2026-11-12 night): from the better of cycle 5's final and cycle 4's (tools\pick_start.py picks it after
rem cycle 5's checks: scripts\night_20261112.bat), half-way through the schedule, cycles 3-5's recipe with two changes:
rem  - the twin no longer reads a unit's level badge ("11", "1") walking past an untouched king as the king's HP
rem    (python\live\session.py): 10%% of the recorded positions had the user's king at 1 or 11 HP, so the push starts
rem    of cycles 3-5 had some where one spell on the king won; live the net spelled such a king (2026-11-09..11);
rem  - 15%% of the episodes start where the user has troops behind his king or a princess tower while his king sleeps
rem    (--back-starts: the user's openings, which games against nets never reach; live cycle 4 put every giant at the
rem    bridge against them); back giant starts 15%% -> 10%%, push starts 20%% (rebuilt by the fixed twin).
rem Afterwards: the new final at temp 0.5 against its start (800 games) and the panel (400 each), how much it waits at the
rem user's pushes, and its spells on the sleeping king from the user's openings, in raw\h1_cycle6_after.txt (ends with
rem "=== all done"). Keep the start if the final is not better.
rem   scripts\train_h1_cycle6.bat [hours] [start] [the other one]   (default 8, runs/h1/final_cycle5.pt, final_cycle4.pt)
rem Progress: runs\h1\train.log (appended to cycles 1-5).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
set START=%~2
if "%START%"=="" set START=runs/h1/final_cycle5.pt
set PREV=%~3
if "%PREV%"=="" set PREV=runs/h1/final_cycle4.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set C3=runs/h1/final_cycle3.pt
set C2=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set OUT=raw\h1_cycle6_after.txt
set PY=".venv\Scripts\python.exe"
echo h1 cycle 6 from %START% (the other: %PREV%)
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
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %C2% %C3% %PREV% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --eval-minutes 90 --eval-games 150 --final-games 400
echo ### %date% %time% (cycle 6 from %START%) > %OUT%
echo === cycle 6 final vs its start %START%, 800 games >> %OUT%
%PY% python\evaluate.py --model runs/h1/final.pt --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %PREV% %CLONE%) do (
  echo === cycle 6 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model runs/h1/final.pt --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 6 final, then its start >> %OUT%
%PY% tools\push_defence_check.py --model runs/h1/final.pt --rollouts 4 >> %OUT% 2>&1
%PY% tools\push_defence_check.py --model %START% --rollouts 4 >> %OUT% 2>&1
echo === spells on the sleeping king from the user's openings: cycle 6 final, then its start >> %OUT%
%PY% tools\king_wake_check.py --model runs/h1/final.pt --back-starts --games 400 >> %OUT% 2>&1
%PY% tools\king_wake_check.py --model %START% --back-starts --games 400 >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
