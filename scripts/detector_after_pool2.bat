@echo off
rem Detector round 2 (classic1) after the pool2 policy run: they cannot share the GPU (pool2 holds ~3.6 GB of the 6).
rem Waits for runs\pool2\train.log to say "=== all done", then trains scripts\train_classic1.bat on raw\detector\classic1
rem (real frames labelled by the round-0 detector and the twin, tools\classic_round.py, plus synthetic pictures).
rem Detached, as a task of its own (not run_detached's cr-neiro-job, which pool2 runs as):
rem   schtasks /Create /TN cr-neiro-detector /TR "cmd /c \"<repo>\scripts\detector_after_pool2.bat\"" /SC ONCE /ST 00:00 /SD 01/01/2030 /F
rem   schtasks /Run /TN cr-neiro-detector
rem Log: raw\detector\classic1_train_log.txt
cd /d "%~dp0.."
powershell -NoProfile -Command "while (-not ((Test-Path 'runs\pool2\train.log') -and (Select-String -Path 'runs\pool2\train.log' -SimpleMatch -Pattern '=== all done' -Quiet))) { Start-Sleep -Seconds 60 }"
rem (pool2's own stay-awake ends with it: this one keeps the PC up through the detector's training)
start "stay awake" /min powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stay_awake.ps1 -Log raw\detector\classic1_train_log.txt -Until "Results saved to" -Hours 4
call scripts\train_classic1.bat > raw\detector\classic1_train_log.txt 2>&1
