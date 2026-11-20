@echo off
rem Expert iteration from h1 cycle 9 aimed at the user's pushes (2026-11-16 night). tools\search_iter.py rounds:
rem cycle 9 with lookahead plays itself, the net learns what the lookahead found (tools\distill_search.py), the result
rem is kept only if it holds up. Changes from sx1 (2026-10-30, only its round 1 promoted):
rem   --fair            each side looks ahead only from what it can know (the other's blueprints unseen, its elixir
rem                     off like the live twin's): sx1's search peeked, and the net cannot learn what it cannot see;
rem   --horizon 6       6 s ahead instead of 4 (at the giant + mini pekka pushes 4 s and 8 s chose differently);
rem   --push-share 0.6  60%% of the games start at the user's recorded pushes (2705 positions of 32 two-sided battles),
rem   --follow-user 15  red putting down his real next cards for 15 s first;
rem   --push-eval-games a round's net is judged also from those positions against the best: 3+ points better there
rem                     (and 48%%+ from fresh starts), or 52%%+ from fresh starts and not 2 points worse there.
rem Not used: searching at every push (search.py think_at_push): in the simulator it lost 1/11 games where the plain
rem fair lookahead won ~70%% - at moments the net is sure to wait the short lookahead over-rates playing at once.
rem   scripts\train_sx2.bat [hours]   (default 14)
rem Detached:  scripts\run_detached.bat train_sx2.bat 14
rem Progress: raw\train_sx2_log.txt (">>>" lines); the best: runs\sx2\best.pt.
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=14
set START=runs/h1/final_cycle9.pt
set NAME=sx2
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\train_%NAME%_log.txt -Until "=== search iteration done" -Hours 17
".venv\Scripts\python.exe" tools\search_iter.py --name %NAME% --start %START% --hours %HOURS% --promote 0.52 --workers 3 --games 60 --top %START% --panel runs/h1/final_cycle8.pt runs/h1/final_cycle6.pt runs/clone/user_v3_kl0.05.pt --push-eval-games 1000 "--games-extra=--fair --explore 2 --think-anyway 0.05 --horizon 6 --push-share 0.6 --follow-user 15"
