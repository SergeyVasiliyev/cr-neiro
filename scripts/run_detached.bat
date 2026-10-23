@echo off
rem Runs a long job (a training script with its arguments) through the Windows Task Scheduler, so it does not belong
rem to whatever started it - a Claude session that closes, a console that is shut - and keeps running on its own.
rem   scripts\run_detached.bat train_hunter.bat 3
rem   scripts\run_detached.bat train_v8.bat 7
rem The task is "cr-neiro-job" (replaced by the next one); remove it with:  schtasks /Delete /TN cr-neiro-job /F
cd /d "%~dp0.."
set SCRIPT=%~1
set ARGS=%2 %3 %4 %5 %6 %7 %8
schtasks /Create /TN cr-neiro-job /TR "cmd /c \"\"%CD%\scripts\%SCRIPT%\" %ARGS%\"" /SC ONCE /ST 00:00 /SD 01/01/2030 /F >nul || exit /b 1
schtasks /Run /TN cr-neiro-job
