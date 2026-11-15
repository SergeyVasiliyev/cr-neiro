@echo off
rem h1 cycle 8, resumed: the 20-h run was stopped on 2026-11-13 13:50 at the user's request (to play), at 95.2%% of
rem its schedule, in the anneal (runs\h1\latest.pt kept as runs\h1\cycle8_paused.pt: 245.9M steps, lr 3.84e-5).
rem The rest of the schedule: --progress 0.952 --anneal 0.048 makes lr fall from 3.84e-5 to 0 on the same line as
rem before (lr_at_anneal = the checkpoint's lr), over 1.92 h (the run's 2.5%% of the schedule an hour). Then the same
rem checks as scripts\train_h1_cycle8.bat, in raw\h1_cycle8_after.txt (ends with "=== all done").
rem Detached:  scripts\run_detached.bat train_h1_cycle8_resume.bat
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=1.92
set START=runs/h1/final_cycle6.pt
set PREV=runs/h1/final_cycle7.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set C5=runs/h1/final_cycle5.pt
set C4=runs/h1/final_cycle4.pt
set C3=runs/h1/final_cycle3.pt
set C2=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set OUT=raw\h1_cycle8_after.txt
set PY=".venv\Scripts\python.exe"
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 4
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume runs/h1/cycle8_paused.pt --progress 0.952 --anneal 0.048 ^
  --giant-starts 0.1 --giant-lead 2 4 --push-starts 0.2 --back-starts 0.15 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0 --endgame 0 --anchor %CLONE% --anchor-coef 0.03 0.01 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --strong-weights 1 1 1 1 2 --strong-share 0.4 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0.3 0.3 0.6 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %C2% %C3% %C4% %C5% %PREV% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --eval-minutes 120 --eval-games 150 --final-games 400
copy /y runs\h1\final.pt runs\h1\final_cycle8.pt >nul
set NEW=runs/h1/final_cycle8.pt
echo ### %date% %time% (cycle 8, the long run from %START%, resumed after a pause) > %OUT%
echo === cycle 8 final vs its start %START%, 800 games >> %OUT%
%PY% python\evaluate.py --model %NEW% --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %PREV% %C5% %C4% %CLONE%) do (
  echo === cycle 8 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model %NEW% --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 8 final >> %OUT%
%PY% tools\push_defence_check.py --model %NEW% --rollouts 4 >> %OUT% 2>&1
echo === spells on the sleeping king from the user's openings: cycle 8 final >> %OUT%
%PY% tools\king_wake_check.py --model %NEW% --back-starts --games 400 >> %OUT% 2>&1
echo === pick >> %OUT%
%PY% tools\pick_start.py --new %NEW% --save-as %NEW% --old %START% --result raw/h1_cycle8_after.txt --out raw/h1_cycle9_start.txt --cycle 8 --old-cycle 6 >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
