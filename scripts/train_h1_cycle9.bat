@echo off
rem h1 cycle 9 (2026-11-15 night): cycle 8's final trained against the user's real follow-ups, seeing him like live.
rem The h1 recipe had plateaued (cycle 7 lost to 6, the 20-h cycle 8 tied it): the nets learn only against nets, and a
rem net's follow-up never punished the bot's waiting at the user's pushes (against his real one, defending at once was
rem better by +0.10: tools\push_follow_check.py). Two changes from cycle 8's recipe:
rem   --follow-user 15  at the push and back starts (positions of his recorded battles) red puts down his real next
rem                     cards for 15 s, when and where he did (BatchEnv scripts), then plays on as a net;
rem   --live-view       the opponent seen like the live twin sees him (~3.5%% of plays missed, ~3%% phantoms, new troops
rem                     up to 0.9 s late, a few HP off: tools\twin_plays_check.py after the twin fixes of 2026-11-15).
rem Otherwise cycle 7's recipe: 8 h half-way through the schedule, push starts 20%%, back starts 15%%, giant starts 10%%.
rem Afterwards (raw\h1_cycle9_after.txt, ends with "=== all done"): the new final against its start seen like live (800
rem games), from the user's positions with his real cards (the new and the old one as blue against the old one, 800
rem each), exactly (800), the panel (400 each), the wait at his pushes, the king spells; then pick_start: the bot
rem scripts play the new one if it won the head-to-head (not worse from his positions by 0.03), or if it scored 45%%+
rem there and is 0.03+ better from his positions.
rem   scripts\train_h1_cycle9.bat [hours]   (default 8)
rem Detached:  scripts\run_detached.bat train_h1_cycle9.bat 8
cd /d "%~dp0.."
set HOURS=%~1
if "%HOURS%"=="" set HOURS=8
set START=runs/h1/final_cycle8.pt
set PREV=runs/h1/final_cycle6.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set C7=runs/h1/final_cycle7.pt
set C5=runs/h1/final_cycle5.pt
set C4=runs/h1/final_cycle4.pt
set C3=runs/h1/final_cycle3.pt
set C2=runs/h1/cycle2_stopped.pt
set C1=runs/h1/final_cycle1.pt
set LIVE=0.035 0.03 0.9 0.05 0.3
set OUT=raw\h1_cycle9_after.txt
set PY=".venv\Scripts\python.exe"
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %OUT% -Until "=== all done" -Hours 11
%PY% python\train_v7.py --run h1 --hours %HOURS% --resume %START% --progress 0.5 ^
  --giant-starts 0.1 --giant-lead 2 4 --push-starts 0.2 --back-starts 0.15 --follow-user 15 ^
  --size 160,3,4,48,320 --spots 0 --decision-ticks 4 --gamma 0.99933 0.99332 0.99332 --lam 0.966 --delay 3 9 ^
  --noise 0 0.15 --live-view %LIVE% --endgame 0 --anchor %CLONE% --anchor-coef 0.03 0.01 --anchor-until 1.0 ^
  --lr 1e-4 --lr-range 1e-5 1e-4 --target-kl 0.01 --anneal 0.2 --ent-place 1.2 0.7 --ent-time 0 0 ^
  --self-envs 32 --pool-envs 64 --bot-share 0.05 --active-nets 3 --active-every 10 ^
  --strong runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --strong-weights 1 1 1 1 2 --strong-share 0.4 ^
  --w-shaping 1.0 0.1 --w-style 1.0 0.3 --w-until 0.6 ^
  --reward-trade 0.03 --reward-leak 0.005 --leak-relative --elixir-bank 0.0005 --elixir-noise 0 0 0 ^
  --pool heuristic pusher runs/v7e/top.pt runs/si2/best.pt runs/clone/user_v1.pt runs/v10/final_cycle1.pt runs/sx1/best.pt runs/v11/final.pt %CLONE% %C1% %C2% %C3% %C4% %C5% %C7% %PREV% %START% ^
  --fixed-share 0.2 --bots-early 0 --max-snapshots 12 --snapshot-every 4e6 --card-floor 0.03 0.01 ^
  --bench heuristic pusher runs/v7e/top.pt runs/si2/best.pt %PREV% %START% %CLONE% --eval-minutes 90 --eval-games 150 --final-games 400
copy /y runs\h1\final.pt runs\h1\final_cycle9.pt >nul
set NEW=runs/h1/final_cycle9.pt
echo ### %date% %time% (cycle 9 from %START%: the user's real follow-ups, seeing like live) > %OUT%
echo === cycle 9 final vs its start %START%, 800 games, both seeing like live >> %OUT%
%PY% python\evaluate.py --model %NEW% --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm --live-view %LIVE% >> %OUT% 2>&1
echo === user starts (both, his real cards for 15 s, seen like live): new as blue vs %START%, 800 games >> %OUT%
%PY% python\evaluate.py --model %NEW% --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm --live-view %LIVE% --user-starts both --follow-user 15 >> %OUT% 2>&1
echo === user starts (both, his real cards for 15 s, seen like live): old as blue vs %START%, 800 games >> %OUT%
%PY% python\evaluate.py --model %START% --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm --live-view %LIVE% --user-starts both --follow-user 15 >> %OUT% 2>&1
echo === cycle 9 final vs its start %START%, 800 games, exact views >> %OUT%
%PY% python\evaluate.py --model %NEW% --opponent %START% --games 800 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %PREV% %C7% %CLONE%) do (
  echo === cycle 9 final vs %%O, 400 games >> %OUT%
  %PY% python\evaluate.py --model %NEW% --opponent %%O --games 400 --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 9 final >> %OUT%
%PY% tools\push_defence_check.py --model %NEW% --rollouts 4 >> %OUT% 2>&1
echo === spells on the sleeping king from the user's openings: cycle 9 final >> %OUT%
%PY% tools\king_wake_check.py --model %NEW% --back-starts --games 400 >> %OUT% 2>&1
echo === pick >> %OUT%
%PY% tools\pick_start.py --new %NEW% --save-as %NEW% --old %START% --result raw/h1_cycle9_after.txt --out raw/h1_cycle10_start.txt --cycle 9 --old-cycle 8 --user-gain 0.03 --min-score 0.45 --bot-scripts scripts/play_vs_bot.bat scripts/play_recorded.bat scripts/play_search_ab.bat >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
