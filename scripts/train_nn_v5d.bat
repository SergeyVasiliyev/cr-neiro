@echo off
rem Night of 2026-10-16: waits for ppo_v5c (to 55M decisions) to finish, then keeps training it until ~10:00
rem (82M decisions), unhooked from the imitated play, mostly against its own past versions (the bots are ~95%+ beaten by then), and compares the result.
rem Progress: raw\train_nn_v5d_log.txt (ends with "=== all done"); runs\ppo_v5d. PC stays awake.
cd /d "%~dp0.."
set LOG=raw\train_nn_v5d_log.txt
set DECK=knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer
echo === start %date% %time% (waiting for ppo_v5c) > %LOG%
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %LOG% -Until "=== all done" -Hours 12
powershell -NoProfile -Command "while (-not (Select-String -Path raw\train_nn_v5c_continue_log.txt -SimpleMatch '=== all done' -Quiet)) { Start-Sleep 60 }"
echo --- PPO %time% >> %LOG%
".venv\Scripts\python.exe" python\train_ppo.py --run ppo_v5d --resume runs\ppo_v5c\latest.pt --anchor runs\bc_v5\latest.pt --anchor-coef 0.01 --anchor-coef-end 0.0 --key-spots --deck %DECK% --opponent heuristic:0.1,pusher:0.2,selfplay:0.7 --decision-ticks 6 --envs 64 --lr 7e-5 --ent 0.04 --reward-win 1 --reward-tower 0.5 --reward-crown 0.1 --reward-trade 0.03 --trade-floor 0.25 --reward-leak 0.005 --total-steps 82e6 >> %LOG% 2>&1
echo --- comparison %time% >> %LOG%
for %%o in (heuristic pusher runs\ppo_v4\latest.pt runs\ppo_v5c\latest.pt) do (
  echo vs %%o >> %LOG%
  ".venv\Scripts\python.exe" python\evaluate.py --model runs\ppo_v5d\latest.pt --opponent %%o --deck %DECK% --games 300 --envs 100 --decision-ticks 6 --greedy >> %LOG% 2>&1
)
echo === all done %date% %time% >> %LOG%
