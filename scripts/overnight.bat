@echo off
rem Overnight: re-read recordings 1-7 with the latest detector (+ unit HP bars), write match reports and the
rem opponent-play checks, then train the policy on the corrected simulator (starter deck mirror, decisions
rem every 0.3 s) until morning. Progress: raw\overnight_log.txt and runs\starter_v2\train.log.
cd /d "%~dp0.."
set LOG=raw\overnight_log.txt
echo === start %date% %time% > %LOG%
for %%v in (video1 video2 video3 video4 video5 video6 video7) do (
  echo --- read %%v %time% >> %LOG%
  ".venv\Scripts\python.exe" tools\perceive_video.py --video raw\videos\%%v.mp4 --conf 0.15 >> %LOG% 2>&1
)
echo --- reports %time% >> %LOG%
".venv\Scripts\python.exe" tools\match_report.py > raw\perception\report_all.txt 2>&1
".venv\Scripts\python.exe" tools\compare_red_truth.py --blue raw\perception\cache\video8.json --red raw\perception\cache\video8_red.json > raw\perception\truth_video8.txt 2>&1
".venv\Scripts\python.exe" tools\compare_red_truth.py --blue raw\perception\cache\video9.json --red raw\perception\cache\video9_red.json > raw\perception\truth_video9.txt 2>&1
echo --- policy training %time% >> %LOG%
".venv\Scripts\python.exe" python\train_ppo.py --run starter_v2 --deck knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer --opponent heuristic:0.4,selfplay:0.6 --decision-ticks 6 --total-steps 60e6 --envs 64 >> %LOG% 2>&1
echo === done %date% %time% >> %LOG%
