@echo off
rem The policy plays itself: one bot on each LDPlayer instance (0 and 1, both guest accounts).
rem   scripts\bot_vs_bot.bat                          runs\v7b\final.pt on both sides
rem   scripts\bot_vs_bot.bat runs\ppo_v4\latest.pt    a particular one on both sides
rem Then start a friendly battle between the two guests (one invites in the clan chat, the other accepts).
rem In each bot's preview window: P = pause, Q = quit (or close it). Logs: raw\live_logs.
cd /d "%~dp0.."
set BRAIN=%~1
if "%BRAIN%"=="" set BRAIN=runs\v7b\final.pt
echo brain: %BRAIN%
start "bot 0" cmd /k .venv\Scripts\python.exe tools\play_live.py --brain "%BRAIN%" --act --instance 0
rem a second apart, so the two logs get different names
ping -n 3 127.0.0.1 >nul
start "bot 1" cmd /k .venv\Scripts\python.exe tools\play_live.py --brain "%BRAIN%" --act --instance 1
