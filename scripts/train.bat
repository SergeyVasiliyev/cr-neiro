@echo off
rem Starts PPO training. All arguments go to python\train_ppo.py, e.g.:
rem   scripts\train.bat --run first --opponent heuristic --total-steps 20e6
cd /d "%~dp0.."
".venv\Scripts\python.exe" python\train_ppo.py %*
