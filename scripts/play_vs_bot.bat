@echo off
rem Starts the bot on LDPlayer instance 0 (the guest account) with the best policy so far, or the one given:
rem   scripts\play_vs_bot.bat                          runs\h1\final_cycle11.pt (the best one so far, 2026-11-21: PPO from the
rem                                                    user's clone, 11 cycles; at temp 0.5 beats cycle 10 58%, v7e top 72%, si2 72%)
rem   scripts\play_vs_bot.bat runs\ppo_v4\latest.pt    a particular one
rem   scripts\play_vs_bot.bat runs\si2\best.pt --search  with lookahead before its moves (python\search.py)
rem Then call the guest account to a friendly battle (starter deck on both sides); the bot plays the battle.
rem In its window: P = pause, Q = quit (or close the preview window). Log: raw\live_logs.
rem Keep LDPlayer's window open (covered is fine).
cd /d "%~dp0.."
set BRAIN=%~1
if "%BRAIN%"=="" set BRAIN=runs\h1\final_cycle11.pt
echo brain: %BRAIN%
".venv\Scripts\python.exe" tools\play_live.py --brain "%BRAIN%" --act --instance 0 %2 %3 %4
pause
