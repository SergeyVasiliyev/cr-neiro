@echo off
rem Records both LDPlayer windows (instance 0 = blue, 1 = red) into raw\videos\videoN.mp4 + videoN_red.mp4.
rem Start before the battle; after it press Ctrl+C (do not just close the window: the files would be unreadable). Stops by itself after 10 minutes.
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\record_windows.py %*
pause
