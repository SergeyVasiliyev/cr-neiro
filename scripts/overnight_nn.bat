@echo off
rem Overnight: finish detector v5 (and compare it with v4), then the policy - behavior cloning of the heuristic
rem bot (1 h), then PPO from it (win, tower damage, crowns, a little for elixir trades) with a pull towards the
rem cloned play that fades out. Progress: raw\overnight_nn_log.txt; runs\ppo_v4\train.log. PC stays awake.
cd /d "%~dp0.."
set LOG=raw\overnight_nn_log.txt
echo === start %date% %time% > %LOG%
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %LOG% -Until "=== overnight done" -Hours 16
echo --- detector v5 %time% >> %LOG%
".venv\Scripts\python.exe" tools\detector_round.py --holdout video14 --resume >> %LOG% 2>&1
echo --- behavior cloning %time% >> %LOG%
".venv\Scripts\python.exe" python\train_bc.py --run bc_heuristic --minutes 60 >> %LOG% 2>&1
echo --- PPO %time% >> %LOG%
".venv\Scripts\python.exe" python\train_ppo.py --run ppo_v4 --resume runs\bc_heuristic\latest.pt --anchor runs\bc_heuristic\latest.pt --anchor-coef 0.1 --anchor-coef-end 0.01 --deck knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer --opponent heuristic:0.5,selfplay:0.5 --decision-ticks 6 --envs 64 --lr 1e-4 --reward-win 1 --reward-tower 0.5 --reward-crown 0.1 --reward-trade 0.03 --trade-floor 0.25 --reward-leak 0.005 --total-steps 40e6 >> %LOG% 2>&1
echo === overnight done %date% %time% >> %LOG%
