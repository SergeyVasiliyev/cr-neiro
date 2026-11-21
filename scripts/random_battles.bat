@echo off
rem Random bots on both LDPlayer instances (0 = blue, 1 = red), every battle recorded from both windows:
rem raw\classic\cNNN.mp4 + cNNN_red.mp4, the bots' touches in cNNN_plays.jsonl, card pictures in cNNN_cards\.
rem You start each Classic Decks friendly battle yourself; the bots play it at random.
rem   scripts\random_battles.bat                 stops after 10 battles
rem   scripts\random_battles.bat --battles 3
rem   scripts\random_battles.bat --check         what the bots see on both screens now
rem Stop early with Ctrl+C (closing the window also finishes the files).
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\random_battles.py %*
pause
