@echo off
rem Lookahead against the user (2026-11-16): the bot with python\search.py's lookahead in every other battle (the 1st,
rem 3rd, ... with it, the 2nd, 4th, ... without; the log says which), and both LDPlayer windows recorded for the twin
rem (raw\videos\videoN.mp4 + videoN_red.mp4), so each battle also adds to the user's recorded games.
rem In the simulator, seeing only what the live twin can (search_eval.py --fair, 2026-11-16), cycle 9 with lookahead
rem beat cycle 9 68%%, cycle 8 85%% (47%% without), cycle 6 70%% (60%% without): against a human, this test tells.
rem   scripts\play_search_ab.bat [checkpoint] [one more play_live.py option, e.g. --sync-capture]
rem   (default runs\h1\final_cycle11.pt; --sync-capture: if the screen is not read right with the background capture)
rem Play as many battles as you like (an even number is best). Afterwards press Ctrl+C in the "recording" window.
cd /d "%~dp0.."
set BRAIN=%~1
if "%BRAIN%"=="" set BRAIN=runs\h1\final_cycle11.pt
start "recording" cmd /c scripts\record_both.bat --minutes 120
call scripts\play_vs_bot.bat "%BRAIN%" --search-ab %2
