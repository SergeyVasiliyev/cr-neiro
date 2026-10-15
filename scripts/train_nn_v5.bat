@echo off
rem The policy on key spots: examples of the user's own games (recorded 2026-10-15: 4 against the bot - only the
rem user's side - and 3 against himself - both sides), then imitation of the pusher bot and the user (75 min),
rem then PPO from it against the heuristic bot, the pusher and its own past versions, pulled towards the imitated
rem play at first, and a comparison with the previous policy at the end.
rem Progress: raw\train_nn_v5_log.txt (ends with "=== all done"); runs\bc_v5, runs\ppo_v5. PC stays awake.
cd /d "%~dp0.."
set LOG=raw\train_nn_v5_log.txt
set DECK=knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer
echo === start %date% %time% > %LOG%
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log %LOG% -Until "=== all done" -Hours 14
echo --- recordings read %time% >> %LOG%
".venv\Scripts\python.exe" tools\ensure_caches.py video22 video22_red video23 video23_red video24 video24_red video25 video25_red video26 video26_red video27 video27_red video28 video28_red >> %LOG% 2>&1
echo --- examples of the user's play %time% >> %LOG%
".venv\Scripts\python.exe" tools\human_examples.py --players video22_red video23_red video24_red video25_red video26 video26_red video27 video27_red video28 video28_red --out raw\bc\human_v1.npz >> %LOG% 2>&1
echo --- imitation: pusher bot + the user %time% >> %LOG%
".venv\Scripts\python.exe" python\train_bc.py --run bc_v5 --key-spots --teacher pusher --human raw\bc\human_v1.npz --human-share 0.2 --minutes 75 >> %LOG% 2>&1
echo --- PPO %time% >> %LOG%
".venv\Scripts\python.exe" python\train_ppo.py --run ppo_v5 --resume runs\bc_v5\latest.pt --anchor runs\bc_v5\latest.pt --anchor-coef 0.1 --anchor-coef-end 0.01 --key-spots --deck %DECK% --opponent heuristic:0.25,pusher:0.35,selfplay:0.4 --decision-ticks 6 --envs 64 --lr 1e-4 --ent 0.02 --reward-win 1 --reward-tower 0.5 --reward-crown 0.1 --reward-trade 0.03 --trade-floor 0.25 --reward-leak 0.005 --total-steps 40e6 >> %LOG% 2>&1
echo --- comparison %time% >> %LOG%
for %%o in (heuristic pusher runs\ppo_v4\latest.pt) do (
  echo vs %%o >> %LOG%
  ".venv\Scripts\python.exe" python\evaluate.py --model runs\ppo_v5\latest.pt --opponent %%o --deck %DECK% --games 300 --envs 100 --decision-ticks 6 --greedy >> %LOG% 2>&1
)
echo === all done %date% %time% >> %LOG%
