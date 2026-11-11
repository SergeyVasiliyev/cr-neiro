@echo off
rem h1: training from the user's clone (tools/clone_user.py on the 22 two-sided battles of 2026-11-09, every tile
rem allowed) instead of from the self-play line that keeps landing on the same level (v7e top, si2, v10, sx1, v11).
rem The clone starts weak (it copies the user's habits from ~380 plays: the elixir it waits for, giants behind the
rem king) and PPO makes it strong again; a KL pull to the clone, weaker towards the end but never gone, keeps what
rem it took from the user while it learns. Every tile is allowed (--spots 0): the clone already knows where people
rem put cards, the reason the key spots were for. Opponents: itself, the pool and the strong nets as v10/v11.
rem   scripts\train_h1.bat [hours] [clone]   (default 8 h from runs\clone\user_v2_kl0.2.pt)
rem Detached:  scripts\run_detached.bat train_h1.bat 8
rem Progress: runs\h1\train.log (">>> eval" lines: greedy games against the bots, the strong nets, the clone, sx1).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
set CLONE=%~2
if "%CLONE%"=="" set CLONE=runs/clone/user_v2_kl0.2.pt
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\h1\train.log -Until "=== all done" -Hours 12
".venv\Scripts\python.exe" python\train_v7.py --run h1 --hours %HOURS% --init %CLONE% --size 160,3,4,48,320 --spots 0 ^
  --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 --noise 0 0 --endgame 0 ^
  --anchor %CLONE% --anchor-coef 0.1 0.03 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --anneal 0.2 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt --strong-share 0.25 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt %CLONE% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/sx1/best.pt %CLONE% --eval-minutes 90 --eval-games 150 --final-games 400
