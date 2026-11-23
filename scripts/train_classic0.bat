@echo off
rem Round 0 of the Classic Decks unit detector: synthetic crowds of real units cut out of the random-bot battles
rem (tools\classic_labels.py seeds + clean, tools\classic_synth.py towers + build --exclude c013), then YOLO11s.
rem About 2 hours on the RTX 2060. Check after: tools\classic_eval.py --weights raw\detector\runs\classic0\weights\best.pt c013
cd /d "%~dp0.."
".venv\Scripts\python.exe" tools\train_detector.py --data raw\detector\classic0\data.yaml --model yolo11s.pt --epochs 35 --imgsz 960 --batch 8 --name classic0 --patience 10 %*
