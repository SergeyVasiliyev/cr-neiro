@echo off
rem Opens the unit labelling tool (frames from tools\extract_label_frames.py).
cd /d "%~dp0.."
".venv\Scripts\python.exe" python\label_units.py %*
