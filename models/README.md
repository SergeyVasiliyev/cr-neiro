# Saved weights

Copies of the checkpoints that matter, kept in Git LFS (`runs/` and `raw/` stay out of git). After a fresh clone:
`git lfs pull`, then copy a file back where the scripts expect it.

| File | From | What |
|---|---|---|
| policy/v10_final.pt | runs/v10/final_cycle2.pt | v10 (from scratch, 2026-10-23..29, 574M steps: 460M to cycle 1's anneal, then 24 h more from 72% of the schedule with the anneal again), with the optimizer state. At temperature 0.5 over 400 games: v7e top 52.5%, si2 49%, the clone 59.3%, cycle 1's final 53.7% (cycle 1's final: 27.5%, 31%, 38.5%). Train on from it: `scripts\train_v10.bat <hours> models/policy/v10_final.pt <progress>` |
| policy/si2_best.pt | runs/si2/best.pt | v7e top after search distillation (expert iteration): the strongest net so far, the default of play_vs_bot / bot_vs_bot |
| policy/v7e_top.pt | runs/v7e/top.pt | the best of the v7 runs |
| policy/clone_user_v1.pt | runs/clone/user_v1.pt | a clone of the user's own play |
| policy/ppo_v6_best.pt, v7b_final.pt, v7d_final.pt | runs/... | older nets in v10's opponent pool |
| detector/units_v5_best.pt | raw/detector/runs/units_v5/weights/best.pt | the live unit detector (python/perception/units.py) |
