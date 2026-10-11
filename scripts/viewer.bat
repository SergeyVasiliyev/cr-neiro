@echo off
rem Opens the debug viewer. Extra arguments are passed through, e.g.:
rem   scripts\viewer.bat --blue-bot --seed 5
cd /d "%~dp0.."
".venv\Scripts\python.exe" python\viewer.py %*
