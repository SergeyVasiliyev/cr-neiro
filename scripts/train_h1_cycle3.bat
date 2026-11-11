@echo off
rem h1 cycle 3 (2026-11-11 night): from cycle 2's last save (runs\h1\cycle2_stopped.pt: at temp 0.5 it beats cycle 1's
rem final 66.4%%, v7e top 58.5%%, si2 54.0%%, sx1 79.8%%), half-way through the schedule (~4.8 h at the full rate, then
rem ~3.2 h of annealing). What changed against cycle 2, after the live games of 2026-11-10 (h1 lost nearly all; it waited
rem 15 s at 10 elixir while the user's giant + mini pekka + musketeer took a tower):
rem  - the user's clone v3 (31 two-sided battles, raw\bc\user_v3.npz) as anchor and as a strong opponent, picked 4x as
rem    often as each other strong net (strong share 0.4: the clone ~20%% of the pool games, before ~1-3%%);
rem  - the anchor weaker, 0.03 -> 0.01: the clones (v2 and its start sx1 all-tiles) want to wait at the user's real pushes
rem    (100%% / 96%%; tools\push_defence_check.py), v7e top and si2 defend (1%%), and h1 copies the wait;
rem  - 20%% of the episodes start at the user's real pushes on the bot's side (--push-starts: the recorded battles every
rem    0.5 s, red troops worth 5+ elixir on blue's side), 15%% right after a back giant with blue 2-4 elixir ahead (the
rem    user's idea; --giant-lead 2 3 tested: the back giant -2.4%% +- 2.6 against the net's own play, -5.8%% with no lead).
rem Caveat: in the simulator h1's waiting at those pushes is not clearly worse than defending (B - A -2.3%% +- 3.8 with
rem its own best other move, -4.6%% +- 4.2 with v7e top's): the practice may still not make it defend.
rem v10's cycle 3 regressed: judge this one by its annealed final at temp 0.5; keep cycle2_stopped.pt if it is not better.
rem Afterwards: the new final at temp 0.5 against cycle 2 (800 games) and the panel (400 each), and how much it wants to
rem wait at the user's pushes, in raw\h1_cycle3_after.txt (ends with "=== all done").
rem   scripts\train_h1_cycle3.bat [hours]   (default 8)
rem Detached:  scripts\run_detached.bat train_h1_cycle3.bat 8
rem Progress: runs\h1\train.log (appended to cycles 1 and 2).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
set CLONE=runs/clone/user_v3_kl0.05.pt
set START=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set OUT=raw\h1_cycle3_after.txt
set PY=".venv\Scripts\python.exe"
if not exist runs\clone\user_v3_kl0.05.pt set CLONE=runs/clone/user_v2_kl0.05.pt
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 10
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume %START% --progress 0.5 ^
  --giant-starts 0.15 --giant-lead 2 4 --push-starts 0.2 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --anchor %CLONE% --anchor-coef 0.03 0.01 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --anneal 0.2 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt %C1% %START% %CLONE% --strong-weights 1 1 1 1 4 --strong-share 0.4 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %C1% %START% %CLONE% --eval-minutes 90 --eval-games 150 --final-games 400
echo ### %date% %time% > %OUT%
echo === cycle 3 final vs cycle 2, 800 games >> %OUT%
%PY% python\evaluate.py --model runs/h1/final.pt --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %C1% %CLONE%) do (
  echo === cycle 3 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model runs/h1/final.pt --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 3 final, then cycle 2 >> %OUT%
%PY% tools\push_defence_check.py --model runs/h1/final.pt --rollouts 4 >> %OUT% 2>&1
%PY% tools\push_defence_check.py --model %START% --rollouts 4 >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
