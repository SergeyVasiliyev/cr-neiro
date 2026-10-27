@echo off
rem v10: a new net from scratch (no old weights) on the corrected simulator of 2026-10-23 (troops get onto the
rem bridge's line before the river, go round buildings, a mini pekka slides past its giant unless right behind it,
rem groups face forward, fireball knockback takes 0.3 s, no getting stuck at a bridge entrance), seeing each
rem unit's target and next hit, and placing also on the tactical spots (behind our tanks, beside / behind / in front
rem of enemy troops, pulling them to the middle).
rem Opponents: itself on both sides, and in the pool games its own snapshots (up to 12), the bots, the old nets and
rem the user's clone - picked by how close the games are, 20% spread evenly whatever the win rates; in the first 8%
rem of the run the pool games are against the bots only (the basics first: v7 from scratch went round in circles).
rem The opponent's elixir it sees is off a little, as live (tools\elixir_check.py: the twin ~0.8 off on average):
rem an offset per match of 0.3 +- 0.3 and 0.6 of jitter per step.
rem Every card in hand keeps at least 3% of its plays (1% by the end): without it v10 stopped casting spells
rem within 5M steps (random spells wasted elixir) and could never learn when they pay.
rem After its bots-only start, playing mostly itself, v10 drifted off beating the bots (heuristic 92% -> 18% in 3 h,
rem as v7 did): the bots now get 30% of the pool games whatever else, and the pool games are two thirds of the
rem learning; at most 4 opponent nets at a time (each is a GPU pass of its own per step: ~18 of them made it 3-4x
rem slower); the rate at most 2e-4 (v7 fell apart with its rate stuck at the cap).
rem From 2026-10-25 13:30 (278M, bots beaten 97-99%): the bots down to 5% of the pool games, and v7e top, si2 and the
rem clone (beaten 1-10%, so hardly ever picked by win rate) 15% between them; 3 other opponent nets at a time.
rem   scripts\train_v10.bat [hours] [resume | checkpoint]   (default 72: the schedule runs on the clock and ends
rem        annealed; resume: go on from runs\v10\latest.pt, or from the checkpoint given; hours = the time left)
rem   scripts\train_v10.bat [hours] resume [progress]   (progress: where in the schedule to go on from, e.g. 0.72 -
rem        made longer on 2026-10-27 from 89% (inside the anneal) by 15 h: ~11 h more at the full rate, then a shorter anneal (8% of the schedule, ~4 h; was 15%))
rem Waits for a detector round still running (raw\detector_round_log.txt without "=== done") before it starts.
rem Detached:  set CR_TASK=cr-neiro-train ^&^& scripts\run_detached.bat train_v10.bat 72
rem Progress: runs\v10\train.log (">>> eval" lines: games against the bots, v7e top, si2 best, the user's clone).
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=72
set RESUME=
if /i "%~2"=="resume" set RESUME=--resume runs/v10/latest.pt
if not "%~2"=="" if /i not "%~2"=="resume" set RESUME=--resume %~2
if not "%~3"=="" set RESUME=%RESUME% --progress %~3
:wait_detector
findstr /c:"=== start" raw\detector_round_log.txt >nul 2>&1 || goto train
findstr /c:"=== done" raw\detector_round_log.txt >nul 2>&1 && goto train
findstr /c:"!!! failed" raw\detector_round_log.txt >nul 2>&1 && goto train
timeout /t 60 /nobreak >nul
goto wait_detector
:train
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log runs\v10\train.log -Until "=== all done" -Hours 80
".venv\Scripts\python.exe" python\train_v7.py --run v10 --hours %HOURS% %RESUME% --size 160,3,4,48,320 --spots 2 ^
  --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 --noise 0 0 --endgame 0 ^
  --lr 2e-4 --lr-range 1e-5 2e-4 --target-kl 0.01 --anneal 0.08 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt --strong-share 0.15 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/ppo_v6/best.pt runs/v7b/final.pt runs/v7d/final.pt runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt ^
  --fixed-share 0.2 --bots-early 0.08 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt --eval-minutes 90 --eval-games 150 --final-games 400
