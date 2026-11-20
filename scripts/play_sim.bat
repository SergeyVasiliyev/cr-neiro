@echo off
rem Play the bot yourself in the simulator (python\play_sim.py): a window like the game, you at the bottom.
rem   scripts\play_sim.bat                          against h1 cycle 9
rem   scripts\play_sim.bat --bot runs\h1\final_cycle8.pt   against another net
rem Every game is saved to raw\sim_games; the window shows your score against that bot.
cd /d "%~dp0.."
if not exist raw\perception\card_art\knight.png ".venv\Scripts\python.exe" tools\card_art.py
".venv\Scripts\python.exe" python\play_sim.py %*
