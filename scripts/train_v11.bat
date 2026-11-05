@echo off
rem v11: v10's training (python\train_v7.py) from the search-distilled runs\sx1\best.pt, with 35% of the episodes
rem starting at moments of the recorded human battles (python\human_starts.py: ~5700 positions from 47 battles, one
rem every 1.5 s): its own games never reach a push built behind the king (the user 48% of giants, v10 0%), elixir
rem leads or finishing a low tower, so its critics could not judge them and its policy never tried them. Nothing
rem tells it what to play there; it plays them out against itself and the pool like any other game.
rem The clone gets twice the share of v7e top and si2 among the strong nets (25% of the pool games between them).
rem Schedule as v10's cycles: from 0.72 at the full rate, then the anneal (the last 8%).
rem   scripts\train_v11.bat [hours] [resume | checkpoint] [progress]   (default 18 h from runs\sx1\best.pt)
rem Detached:  scripts\run_detached.bat train_v11.bat 18
rem Progress: runs\v11\train.log (">>> eval" lines: greedy games against the bots, v7e top, si2, the clone and the
rem start, the overtime finishing test, and its style - giants behind the king among them).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=18
set RESUME=--resume runs/sx1/best.pt --progress 0.72
if /i "%~2"=="resume" set RESUME=--resume runs/v11/latest.pt
if not "%~2"=="" if /i not "%~2"=="resume" set RESUME=--resume %~2 --progress 0.72
if not "%~3"=="" set RESUME=%RESUME% --progress %~3
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\v11\train.log -Until "=== all done" -Hours 26
".venv\Scripts\python.exe" python\train_v7.py --run v11 --hours %HOURS% %RESUME% --size 160,3,4,48,320 --spots 2 ^
  --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 --noise 0 0 --endgame 0 ^
  --human-starts 0.35 --human-every 1.5 ^
  --lr 2e-4 --lr-range 1e-5 2e-4 --target-kl 0.01 --anneal 0.08 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt --strong-weights 1 1 2 --strong-share 0.25 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/ppo_v6/best.pt runs/v7b/final.pt runs/v7d/final.pt runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/sx1/best.pt --eval-minutes 90 --eval-games 150 --final-games 400
