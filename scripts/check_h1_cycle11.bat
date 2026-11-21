@echo off
rem Checks after h1 cycle 11 (the defence school), called by scripts\train_h1_cycle11.bat once the training is done
rem (this file may still be changed until then). Writes raw\h1_cycle11_after.txt, ending with "=== all done":
rem the new final against its start like live and exactly, from the user's positions with his real cards, the drills
rem (hard pushes it must defend, new and old as blue: the score and where its first troop goes), the panel, the wait
rem at his pushes, the king spells; then pick_start, as after cycle 10 (the bot scripts and the sim window play the new
rem one if it won the head-to-head and is not 0.03 worse from his positions, or scored 45%% and is 0.03+ better there).
cd /d "%~dp0.."
set NEW=runs/h1/final_cycle11.pt
set START=runs/h1/final_cycle10.pt
set PREV=runs/h1/final_cycle9.pt
set C6=runs/h1/final_cycle6.pt
set CLONE=runs/clone/user_v3_kl0.05.pt
set LIVE=0.035 0.03 0.9 0.05 0.3
set OUT=raw\h1_cycle11_after.txt
set PY=".venv\Scripts\python.exe"
set EV=%PY% python\evaluate.py --envs 100 --model-temp 0.5 --opp-temp 0.5 --native-rhythm
echo ### %date% %time% (cycle 11 from %START%: the defence school) > %OUT%
echo === cycle 11 final vs its start %START%, 800 games, both seeing like live >> %OUT%
%EV% --model %NEW% --opponent %START% --games 800 --live-view %LIVE% >> %OUT% 2>&1
echo === user starts (both, his real cards for 15 s, seen like live): new as blue vs %START%, 800 games >> %OUT%
%EV% --model %NEW% --opponent %START% --games 800 --live-view %LIVE% --user-starts both --follow-user 15 >> %OUT% 2>&1
echo === user starts (both, his real cards for 15 s, seen like live): old as blue vs %START%, 800 games >> %OUT%
%EV% --model %START% --opponent %START% --games 800 --live-view %LIVE% --user-starts both --follow-user 15 >> %OUT% 2>&1
echo === drills (800 hard pushes not trained on, seen like live): new as blue vs %START%, 800 games >> %OUT%
%EV% --model %NEW% --opponent %START% --games 800 --live-view %LIVE% --drill 800 >> %OUT% 2>&1
echo === drills (800 hard pushes not trained on, seen like live): old as blue vs %START%, 800 games >> %OUT%
%EV% --model %START% --opponent %START% --games 800 --live-view %LIVE% --drill 800 >> %OUT% 2>&1
echo === cycle 11 final vs its start %START%, 800 games, exact views >> %OUT%
%EV% --model %NEW% --opponent %START% --games 800 >> %OUT% 2>&1
for %%O in (runs/sx1/best.pt runs/v7e/top.pt runs/si2/best.pt runs/v11/final.pt %PREV% %C6% %CLONE%) do (
  echo === cycle 11 final vs %%O, 400 games >> %OUT%
  %EV% --model %NEW% --opponent %%O --games 400 >> %OUT% 2>&1
)
echo === waiting at the user's pushes: cycle 11 final >> %OUT%
%PY% tools\push_defence_check.py --model %NEW% --rollouts 4 >> %OUT% 2>&1
echo === spells on the sleeping king from the user's openings: cycle 11 final >> %OUT%
%PY% tools\king_wake_check.py --model %NEW% --back-starts --games 400 >> %OUT% 2>&1
echo === pick >> %OUT%
%PY% tools\pick_start.py --new %NEW% --save-as %NEW% --old %START% --result raw/h1_cycle11_after.txt --out raw/h1_cycle12_start.txt --cycle 11 --old-cycle 10 --user-gain 0.03 --min-score 0.45 --bot-scripts scripts/play_vs_bot.bat scripts/play_recorded.bat scripts/play_search_ab.bat python/play_sim.py >> %OUT% 2>&1
echo === all done %date% %time% >> %OUT%
