@echo off
rem Starts the bot on LDPlayer instance 0 (the guest account) with the newest policy, or the one given:
rem   scripts\play_vs_bot.bat                          the newest of runs\ppo_v5, runs\ppo_v4
rem   scripts\play_vs_bot.bat runs\ppo_v4\latest.pt    a particular one
rem Then call the guest account to a friendly battle (starter deck on both sides); the bot plays the battle.
rem In its window: P = pause, Q = quit. Log: raw\live_logs. Keep LDPlayer's window open (covered is fine).
cd /d "%~dp0.."
set BRAIN=%~1
if "%BRAIN%"=="" if exist runs\ppo_v5\latest.pt set BRAIN=runs\ppo_v5\latest.pt
if "%BRAIN%"=="" set BRAIN=runs\ppo_v4\latest.pt
echo brain: %BRAIN%
".venv\Scripts\python.exe" tools\play_live.py --brain %BRAIN% --act --instance 0
pause
