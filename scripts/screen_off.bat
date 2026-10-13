@echo off
rem Turns the monitor(s) off now; the computer keeps working. Move the mouse or press a key to wake them.
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0screen_off.ps1"
