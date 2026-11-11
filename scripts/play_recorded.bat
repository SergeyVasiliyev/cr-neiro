@echo off
rem The bot on LDPlayer instance 0 as in play_vs_bot.bat, with its window recorded at the same time (raw\videos\videoN.mp4),
rem to check the simulator copy against the game afterwards (tools\match_report.py, tools\twin_replay.py).
rem   scripts\play_recorded.bat [checkpoint]   (default runs\h1\final_cycle4.pt)
rem Several battles in a row are fine. After the last one press Ctrl+C in the "recording" window (it stops by itself
rem after 60 minutes).
cd /d "%~dp0.."
set BRAIN=%~1
if "%BRAIN%"=="" set BRAIN=runs\h1\final_cycle4.pt
start "recording" ".venv\Scripts\python.exe" tools\record_windows.py --instances 0 --minutes 60
call scripts\play_vs_bot.bat "%BRAIN%" %2 %3 %4
