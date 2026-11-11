@echo off
rem h1 cycle 4 (2026-11-11 day): cycle 3's recipe again, from its final (runs\h1\final_cycle3.pt: at temp 0.5 it beats
rem cycle 2 62.3%%, v7e top 60.5%%, si2 58.2%%, sx1 85.0%%, v11 76.8%%, cycle 1 70.5%%, the user's clone v3 94.8%%),
rem half-way through the schedule (~4.8 h at the full rate, then ~3.2 h of annealing).
rem Against cycle 3: the clone v3 picked 2x (not 4x) as often as each other strong net - cycle 3 beats it 95%%, so
rem those games teach little; cycle 2 and cycle 3 are strong opponents and in the pool.
rem Kept: the clone as a weak anchor (0.03 -> 0.01), 20%% of the episodes at the user's real pushes on the bot's side,
rem 15%% right after a back giant with blue 2-4 elixir ahead.
rem Cycle 3 still wants to wait at the user's pushes (median 99%%; >= 90%% at 60%% of them, cycle 2 71%%), and in the
rem simulator that wait is still not clearly worse (B - A +0.023 +- 0.058).
rem Afterwards: the new final at temp 0.5 against cycle 3 (800 games) and the panel (400 each), and how much it wants to
rem wait at the user's pushes, in raw\h1_cycle4_after.txt (ends with "=== all done"). Keep final_cycle3.pt if not better.
rem   scripts\train_h1_cycle4.bat [hours]   (default 8)
rem Detached:  scripts\run_detached.bat train_h1_cycle4.bat 8
rem Progress: runs\h1\train.log (appended to cycles 1-3).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
set CLONE=runs/clone/user_v3_kl0.05.pt
set START=runs/h1/final_cycle3.pt
set C2=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set OUT=raw\h1_cycle4_after.txt
set PY=".venv\Scripts\python.exe"
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 10
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume %START% --progress 0.5 ^
  --giant-starts 0.15 --giant-lead 2 4 --push-starts 0.2 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --anchor %CLONE% --anchor-coef 0.03 0.01 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --anneal 0.2 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt %C2% %START% %CLONE% --strong-weights 1 1 1 1 2 --strong-share 0.4 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %C2% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %C2% %START% %CLONE% --eval-minutes 90 --eval-games 150 --final-games 400
echo ### %date% %time% > %OUT%
echo === cycle 4 final vs cycle 3, 800 games >> %OUT%
%PY% python\evaluate.py --model runs/h1/final.pt --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %C2% %CLONE%) do (
  echo === cycle 4 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model runs/h1/final.pt --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 4 final >> %OUT%
%PY% tools\push_defence_check.py --model runs/h1/final.pt --rollouts 4 >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
