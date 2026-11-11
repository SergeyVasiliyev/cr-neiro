# Saved weights

Copies of the checkpoints that matter, kept in Git LFS (`runs/` and `raw/` stay out of git). After a fresh clone:
`git lfs pull`, then copy a file back where the scripts expect it.

| File | From | What |
|---|---|---|
| policy/h1_cycle4_final.pt | runs/h1/final_cycle4.pt | h1 cycle 4 (2026-11-10..11): PPO from the user's clone on every tile, 4 cycles of 8 h, with starts at the user's real pushes and after a back giant with an elixir lead. At temperature 0.5: cycle 3 59%, v7e top 79%, si2 71%, sx1 81%. The best net, the default of play_vs_bot / play_recorded. Train on from it: `scripts\train_h1_cycle5.bat` (its START) |
| policy/clone_user_v3.pt | runs/clone/user_v3_kl0.05.pt | the user's clone v3: sx1 on every tile cloned on 32 two-sided recorded battles (3077 examples); h1's anchor and sparring partner |
| policy/v10_final.pt | runs/v10/final_cycle2.pt | v10 (from scratch, 2026-10-23..29, 574M steps: 460M to cycle 1's anneal, then 24 h more from 72% of the schedule with the anneal again), with the optimizer state. At temperature 0.5 over 400 games: v7e top 52.5%, si2 49%, the clone 59.3%, cycle 1's final 53.7% (cycle 1's final: 27.5%, 31%, 38.5%). Train on from it: `scripts\train_v10.bat <hours> models/policy/v10_final.pt <progress>` |
| policy/sx1_best.pt | runs/sx1/best.pt | v10 cycle 2 after one round of search distillation (tools/search_iter.py, 2026-10-30): 52.6% vs cycle 2. Level with v7e top, si2 and cycle 2 within noise; the start of v11 (no gain, not saved) and of the next clone |
| policy/v10_cycle1_final.pt | runs/v10/final_cycle1.pt | v10 cycle 1's annealed final, with the optimizer state; in the opponent pool and the search panel |
| policy/si2_best.pt | runs/si2/best.pt | v7e top after search distillation (expert iteration): the strongest net before h1 |
| policy/v7e_top.pt | runs/v7e/top.pt | the best of the v7 runs |
| policy/clone_user_v1.pt | runs/clone/user_v1.pt | a clone of the user's own play |
| policy/ppo_v6_best.pt, v7b_final.pt, v7d_final.pt | runs/... | older nets in v10's opponent pool |
| detector/units_v5_best.pt | raw/detector/runs/units_v5/weights/best.pt | the live unit detector (python/perception/units.py) |
