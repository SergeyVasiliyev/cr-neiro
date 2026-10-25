# Saved weights

Copies of the checkpoints that matter, kept in Git LFS (`runs/` and `raw/` stay out of git). After a fresh clone:
`git lfs pull`, then copy a file back where the scripts expect it.

| File | From | What |
|---|---|---|
| policy/v10_latest.pt | runs/v10/latest.pt | v10 (from scratch, 2026-10-23..26) at 293.9M steps, 62% of its schedule, with the optimizer state: `scripts\train_v10.bat <hours left> models/policy/v10_latest.pt` goes on from it |
| policy/si2_best.pt | runs/si2/best.pt | v7e top after search distillation (expert iteration): the strongest net so far, the default of play_vs_bot / bot_vs_bot |
| policy/v7e_top.pt | runs/v7e/top.pt | the best of the v7 runs |
| policy/clone_user_v1.pt | runs/clone/user_v1.pt | a clone of the user's own play |
| policy/ppo_v6_best.pt, v7b_final.pt, v7d_final.pt | runs/... | older nets in v10's opponent pool |
| detector/units_v5_best.pt | raw/detector/runs/units_v5/weights/best.pt | the live unit detector (python/perception/units.py) |
