@echo off
rem Continues ppo_v5 (stopped at 25M decisions) after the 2026-10-15 fixes: spells aim where troops will be,
rem never at a sleeping king, a tower together with its troops; more defensive spots; a troop put on top of
rem one of ours lands in front of it; the pusher also counter-pushes with the giant at the bridge.
rem Progress: raw\train_nn_v5c_log.txt (ends with "=== all done"); runs\ppo_v5c. PC stays awake.
cd /d "%~dp0.."
set LOG=raw\train_nn_v5c_log.txt
set DECK=knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer
echo === start %date% %time% > %LOG%
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %LOG% -Until "=== all done" -Hours 14
echo --- PPO %time% >> %LOG%
".venv\Scripts\python.exe" python\train_ppo.py --run ppo_v5c --resume runs\ppo_v5b\latest.pt --anchor runs\bc_v5\latest.pt --anchor-coef 0.02 --anchor-coef-end 0.0 --key-spots --deck %DECK% --opponent heuristic:0.25,pusher:0.35,selfplay:0.4 --decision-ticks 6 --envs 64 --lr 1e-4 --ent 0.04 --reward-win 1 --reward-tower 0.5 --reward-crown 0.1 --reward-trade 0.03 --trade-floor 0.25 --reward-leak 0.005 --total-steps 55e6 >> %LOG% 2>&1
echo --- comparison %time% >> %LOG%
for %%o in (heuristic pusher runs\ppo_v4\latest.pt runs\ppo_v5\stopped_25M.pt) do (
  echo vs %%o >> %LOG%
  ".venv\Scripts\python.exe" python\evaluate.py --model runs\ppo_v5c\latest.pt --opponent %%o --deck %DECK% --games 300 --envs 100 --decision-ticks 6 --greedy >> %LOG% 2>&1
)
echo === all done %date% %time% >> %LOG%
