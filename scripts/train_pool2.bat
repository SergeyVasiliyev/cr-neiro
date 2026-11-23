@echo off
rem pool2: the Classic Decks net from scratch on the 8 known decks (data\classic_decks.json, 64 matchups, mirrors too),
rem with the opponent's deck as an input (2026-11-22, the user's wish): the cards he has played, the pool deck they fit
rem (all 8 cards once it is pinned down) and his cycle - what is in his hand, what comes back next (rl.h EnemyDeckFeature).
rem The sim's pool mechanics were checked against the recordings (tools\classic_check.py). Not in the pool yet: the
rem 9th deck seen in c016/c017/c019 (mega knight, miner, skeleton barrel, bats, spear goblins, inferno dragon, goblin
rem gang, giant snowball) - its first three cards are not simulated yet.
rem The recipe is pool1's = v10's (self-play on both sides plus pool games against the bots and its own snapshots; bots
rem only in the first 8%%, then 20%% of the pool games; every card in hand keeps 3%% of its plays (1%% at the end); the
rem rate at most 2e-4, annealed over the last 15%%).
rem   scripts\train_pool2.bat [hours] [resume | checkpoint]   (default 9)
rem Detached:  scripts\run_detached.bat train_pool2.bat 9
rem Progress: runs\pool2\train.log (">>> eval" lines: greedy games against the heuristic and pusher bots, all decks).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=9
set RESUME=
if /i "%~2"=="resume" set RESUME=--resume runs/pool2/latest.pt
if not "%~2"=="" if /i not "%~2"=="resume" set RESUME=--resume %~2
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\pool2\train.log -Until "=== all done" -Hours 12
".venv\Scripts\python.exe" python\train_v7.py --run pool2 --hours %HOURS% %RESUME% --decks data/classic_decks.json ^
  --enemy-deck on ^
  --size 160,3,4,48,320 --spots 2 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --lr 2e-4 --lr-range 1e-5 2e-4 --target-kl 0.01 --anneal 0.15 --ent-place 1.2 0.7 ^
  --ent-time 0 0 --self-envs 32 --pool-envs 64 --bot-share 0.2 --active-nets 3 --active-every 10 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 --reward-trade 0.03 --reward-leak 0.005 --leak-relative ^
  --elixir-bank 0.0005 --pool heuristic pusher --fixed-share 0.2 --bots-early 0.08 --max-snapshots 12 ^
  --snapshot-every 4e6 --card-floor 0.03 0.01 --bench heuristic pusher --eval-minutes 60 --eval-games 200 ^
  --final-games 400
