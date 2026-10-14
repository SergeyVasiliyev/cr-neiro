@echo off
rem Trains the policy on the starter deck mirror with the elixir-trade reward (decisions every 0.3 s), on its
rem own (survives closing Claude). Progress: runs\starter_v3\train.log; raw\train_policy_log.txt ends with "=== done".
cd /d "%~dp0.."
set LOG=raw\train_policy_log.txt
echo === start %date% %time% > %LOG%
".venv\Scripts\python.exe" python\train_ppo.py --run starter_v3 --deck knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer --opponent heuristic:0.5,selfplay:0.5 --decision-ticks 6 --reward-trade 0.1 --trade-floor 0.25 --reward-leak 0.02 --total-steps 60e6 --envs 64 >> %LOG% 2>&1
echo === done %date% %time% >> %LOG%
