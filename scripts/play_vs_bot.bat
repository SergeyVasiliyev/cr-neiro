@echo off
rem Starts the bot on LDPlayer instance 0 (the guest account) with the best policy so far, or the one given:
rem   scripts\play_vs_bot.bat                          runs\v7b\final.pt (the best one so far: beats ppo_v6 72%)
rem   scripts\play_vs_bot.bat runs\ppo_v4\latest.pt    a particular one
rem Then call the guest account to a friendly battle (starter deck on both sides); the bot plays the battle.
rem In its window: P = pause, Q = quit (or close the preview window). Log: raw\live_logs.
rem Keep LDPlayer's window open (covered is fine).
cd /d "%~dp0.."
set BRAIN=%~1
if "%BRAIN%"=="" set BRAIN=runs\v7b\final.pt
echo brain: %BRAIN%
".venv\Scripts\python.exe" tools\play_live.py --brain "%BRAIN%" --act --instance 0
pause
