@echo off
rem pool1: the first net of the Classic Decks stage, from scratch, on all 8 decks of the mode (data\classic_decks.json):
rem every episode draws a deck for each side (64 matchups, mirrors too). The simulator has the pool's mechanics since
rem 2026-11-21 (tools\import_pool.py, game files 16.635.013): buildings, spawners, death spawns and damage, charge,
rem shields, river jumps, dashes, the inferno ramp, stuns, slows, curses, area and rolling spells. Not calibrated
rem against the game yet: a first run to see that it learns at all.
rem The recipe is v10's (the one from-scratch run that worked): self-play on both sides plus pool games against the
rem bots and its own snapshots; bots only in the first 8%%, then 20%% of the pool games whatever else; every card in hand
rem keeps 3%% of its plays (1%% at the end); the rate at most 2e-4, annealed over the last 15%%. No old nets: they never
rem saw these cards (their vocabulary is smaller).
rem   scripts\train_pool1.bat [hours] [resume | checkpoint]   (default 9)
rem Detached:  scripts\run_detached.bat train_pool1.bat 9
rem Progress: runs\pool1\train.log (">>> eval" lines: greedy games against the heuristic and pusher bots, all decks).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=9
set RESUME=
if /i "%~2"=="resume" set RESUME=--resume runs/pool1/latest.pt
if not "%~2"=="" if /i not "%~2"=="resume" set RESUME=--resume %~2
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\pool1\train.log -Until "=== all done" -Hours 12
".venv\Scripts\python.exe" python\train_v7.py --run pool1 --hours %HOURS% %RESUME% --decks data/classic_decks.json ^
  --size 160,3,4,48,320 --spots 2 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --lr 2e-4 --lr-range 1e-5 2e-4 --target-kl 0.01 --anneal 0.15 --ent-place 1.2 0.7 ^
  --ent-time 0 0 --self-envs 32 --pool-envs 64 --bot-share 0.2 --active-nets 3 --active-every 10 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 --reward-trade 0.03 --reward-leak 0.005 --leak-relative ^
  --elixir-bank 0.0005 --pool heuristic pusher --fixed-share 0.2 --bots-early 0.08 --max-snapshots 12 ^
  --snapshot-every 4e6 --card-floor 0.03 0.01 --bench heuristic pusher --eval-minutes 60 --eval-games 200 ^
  --final-games 400
