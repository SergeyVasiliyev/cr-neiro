@echo off
rem h1 cycle 11, the defence school (the user's idea, 2026-11-21): cycle 10 trained on with many more hard pushes to
rem defend than ordinary games (the user: more drills than plain games). Live his big pushes break the bot: it waits,
rem then puts its troops in front of the tower instead of pulling them to the middle.
rem 40%% of the episodes start at a defence drill (python\defence_starts.py, 4000 positions): red begins a hard push -
rem a giant with a mini pekka and a musketeer behind, a back giant with support, two lanes at once, 11 kinds - its
rem cards put down by script in single, double or triple elixir, blue behind or ahead on elixir. A net plays red too:
rem in 60%% of them from the push's last card on, in the rest throughout (net against net, the user's wish). Both
rem sides have really played their 8 cards before, so each knows the other's cycle from the history it sees.
rem 20%% start at his real pushes with his real next cards for 15 s, half of them with blue 1-3 elixir poorer.
rem Back and giant starts 5%% each; 30%% ordinary games. Otherwise cycle 10's recipe, half-way through the schedule;
rem 9 h so that it ends with the checks by ~12:00 when started at ~01:45.
rem Nothing tells it how to defend: it learns what pays there.
rem Afterwards scripts\check_h1_cycle11.bat (it may still be changed while this one runs) writes
rem raw\h1_cycle11_after.txt, ending with "=== all done".
rem   scripts\train_h1_cycle11.bat [hours]   (default 9)
rem Detached:  scripts\run_detached.bat train_h1_cycle11.bat 9
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=9
set START=runs/h1/final_cycle10.pt
set PREV=runs/h1/final_cycle9.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set C8=runs/h1/final_cycle8.pt
set C7=runs/h1/final_cycle7.pt
set C6=runs/h1/final_cycle6.pt
set C5=runs/h1/final_cycle5.pt
set C4=runs/h1/final_cycle4.pt
set C3=runs/h1/final_cycle3.pt
set C2=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set LIVE=0.035 0.03 0.9 0.05 0.3
set OUT=raw\h1_cycle11_after.txt
set PY=".venv\Scripts\python.exe"
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 11
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume %START% --progress 0.5 ^
  --giant-starts 0.05 --giant-lead 2 4 --push-starts 0.2 --push-deficit 0.5 1 3 --back-starts 0.05 ^
  --drill-starts 0.4 --drill-count 4000 --follow-user 15 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0.15 --live-view %LIVE% --endgame 0 --anchor %CLONE% --anchor-coef 0.03 0.01 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --anneal 0.2 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --strong-weights 1 1 1 1 2 --strong-share 0.4 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0 0 0 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %C2% %C3% %C4% %C5% %C6% %C7% %C8% %PREV% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --eval-minutes 90 --eval-games 150 --final-games 400
copy /y runs\h1\final.pt runs\h1\final_cycle11.pt >nul
call scripts\check_h1_cycle11.bat
