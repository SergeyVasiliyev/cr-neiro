@echo off
rem h1 cycle 2 (2026-11-10): h1 again from its annealed final (runs\h1\final_cycle1.pt: 76.5%% vs sx1, 55.5%% vs v7e
rem top, 52.7%% vs si2 at temp 0.5), from half-way through its schedule: ~4.8 h at the full rate, then ~3.2 h of
rem annealing (as v10's cycle 2, which gained). Its greedy evals were still rising when cycle 1 ended.
rem One change: 20%% of the episodes start right after blue put a giant behind its king at a moment the user would
rem (9+ elixir, the giant in hand, no enemy troop on its side; positions from cycle 1's games against itself).
rem tools\back_giant_check.py found such a giant neutral for cycle 1 (-0.6%% +- 2.6 vs itself, -2.4%% +- 2.3 vs sx1):
rem punished in its first 20 s, and a push that did not pay - but h1 never practised following one up.
rem v7e top, si2, sx1 and cycle 1's final are the strong opponents (sx1 alone is beaten 76%%: little to learn there).
rem Afterwards: the new final at temp 0.5 against cycle 1's final (800 games) and the panel (400 each), and the back
rem giant check for both finals (the same seed), in raw\h1_cycle2_after.txt (ends with "=== all done").
rem   scripts\train_h1_cycle2.bat [hours]   (default 8)
rem Detached:  scripts\run_detached.bat train_h1_cycle2.bat 8
rem Progress: runs\h1\train.log (appended to cycle 1's).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
set CLONE=runs/clone/user_v2_kl0.05.pt
set START=runs/h1/final_cycle1.pt
if not exist runs\h1\final_cycle1.pt copy runs\h1\final.pt runs\h1\final_cycle1.pt
set OUT=raw\h1_cycle2_after.txt
set PY=".venv\Scripts\python.exe"
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 11
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume %START% --progress 0.5 --giant-starts 0.2 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --anchor %CLONE% --anchor-coef 0.1 0.03 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --anneal 0.2 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt runs/sx1/best.pt %START% --strong-share 0.25 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/sx1/best.pt %START% --eval-minutes 90 --eval-games 150 --final-games 400
echo ### %date% %time% > %OUT%
echo === cycle 2 final vs cycle 1 final, 800 games >> %OUT%
%PY% python\evaluate.py --model runs/h1/final.pt --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt) do (
  echo === cycle 2 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model runs/h1/final.pt --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === back giant check: cycle 2 final vs itself >> %OUT%
%PY% tools\back_giant_check.py --model runs/h1/final.pt >> %OUT% 2>&1
echo === back giant check: cycle 1 final vs itself, the same seed >> %OUT%
%PY% tools\back_giant_check.py --model %START% >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
