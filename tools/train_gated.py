"""v7 training in gated chunks: each chunk starts from the best policy so far, trains --chunk-hours with
python/train_v7.py (ending annealed), and becomes the new best only if it beats the best fairly (each policy at
its own decision rhythm) and does no worse than the best against a panel of other policies - so long runs can't
drift into beating some opponents while losing to others (v7d: 81% vs ppo_v6 but 39.5% vs v7b).

    .venv\\Scripts\\python.exe tools\\train_gated.py --start runs\\v7c\\step_9M.pt --hours 20

Everything in runs/<name> (best.pt = the best so far, chunk_N.pt = each chunk's result); chunk runs in
runs/<name>_cN. Progress: raw/train_<name>_log.txt (">>>" lines), ends with "=== all done". Stopping at any time
is safe; started again it goes on from best.pt.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DECK = "knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer"
# v7c's settings without endgame starts: decisions every 0.2 s, moves landing 0.15-0.45 s later, the reward mostly
# the result, a light pull to the chunk's start
CHUNK_ARGS = ["--decision-ticks", "4", "--gamma", "0.99933", "0.99332", "0.99332", "--lam", "0.966",
              "--delay", "3", "9", "--endgame", "0", "--anchor-coef", "0.02", "0", "--anchor-until", "0.3",
              "--w-shaping", "0.1", "0.1", "--w-style", "0.3", "0.3", "--ent-place", "0.8", "0.6", "--anneal", "0.25",
              "--eval-minutes", "999", "--bench", "heuristic", "--final-games", "50"]
POOL = ["heuristic", "pusher", "runs/ppo_v6/best.pt", "runs/v7b/final.pt", "runs/v7d/final.pt", "runs/v7c/step_9M.pt"]
PANEL = ["runs/v7b/final.pt", "runs/ppo_v6/best.pt", "runs/v7d/final.pt", "pusher"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="v7e")
    p.add_argument("--start", default="runs/v7c/step_9M.pt")
    p.add_argument("--hours", type=float, default=20)
    p.add_argument("--chunk-hours", type=float, default=1.5)
    p.add_argument("--games", type=int, default=400, help="games against the best")
    p.add_argument("--panel-games", type=int, default=200)
    p.add_argument("--promote", type=float, default=0.55, help="win rate over the best to replace it")
    p.add_argument("--panel-tol", type=float, default=0.03, help="how much worse than the best on the panel is allowed")
    args = p.parse_args()

    out_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(ROOT, "raw", f"train_{args.name}_log.txt")
    if os.path.exists(log_path):
        os.replace(log_path, log_path.replace(".txt", time.strftime("_%Y%m%d_%H%M%S.txt")))

    def log(text):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text, flush=True)

    def score(model, opponent, games):
        cmd = [PY, "python/evaluate.py", "--model", model, "--opponent", opponent, "--deck", DECK, "--games",
               str(games), "--envs", "100", "--greedy"]
        cmd += ["--decision-ticks", "4"] if opponent in ("heuristic", "pusher") else ["--native-rhythm"]
        out = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        m = re.search(r"score ([\d.]+)", out)
        if not m:
            log(f"!!! evaluation failed: {out[-1500:]}")
            raise SystemExit(1)
        return float(m.group(1))

    def panel_scores(model):
        return {opp: score(model, opp, args.panel_games) for opp in PANEL}

    best = os.path.join(out_dir, "best.pt")
    if not os.path.exists(best):
        shutil.copy(os.path.join(ROOT, args.start), best)
    started = time.time()
    log(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}, from {args.start}")
    best_panel = panel_scores(best)
    log(">>> best on the panel: " + ", ".join(f"{os.path.basename(os.path.dirname(o)) or o} {s:.0%}"
                                              for o, s in best_panel.items()))
    done = [int(m.group(1)) for m in (re.fullmatch(r"chunk_(\d+)\.pt", n) for n in os.listdir(out_dir)) if m]
    chunk = max(done, default=0)
    bests = []  # earlier bests, as opponents in the pool
    while time.time() - started < args.hours * 3600:
        chunk += 1
        run = f"{args.name}_c{chunk}"
        pool = POOL + bests[-3:] + [os.path.relpath(best, ROOT).replace("\\", "/")]
        log(f"--- chunk {chunk} ({time.strftime('%H:%M')}): {args.chunk_hours} h from best.pt")
        with open(log_path, "a", encoding="utf-8") as f:
            code = subprocess.run([PY, "python/train_v7.py", "--run", run, "--init", best, "--anchor", best,
                                   "--hours", str(args.chunk_hours), "--seed", str(100 + chunk), *CHUNK_ARGS,
                                   "--pool", *pool], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if code:
            log(f"!!! chunk {chunk} training failed (exit {code})")
            raise SystemExit(code)
        result = os.path.join(out_dir, f"chunk_{chunk}.pt")
        shutil.copy(os.path.join(ROOT, "runs", run, "final.pt"), result)
        vs_best = score(result, best, args.games)
        cand_panel = panel_scores(result)
        avg_c = sum(cand_panel.values()) / len(cand_panel)
        avg_b = sum(best_panel.values()) / len(best_panel)
        promoted = vs_best >= args.promote and avg_c >= avg_b - args.panel_tol
        detail = ", ".join(f"{os.path.basename(os.path.dirname(o)) or o} {s:.0%} (best {best_panel[o]:.0%})"
                           for o, s in cand_panel.items())
        if promoted:
            keep = os.path.join(out_dir, f"best_before_{chunk}.pt")
            shutil.copy(best, keep)
            bests.append(os.path.relpath(keep, ROOT).replace("\\", "/"))
            shutil.copy(result, best)
            best_panel = cand_panel
        log(f">>> chunk {chunk}: vs best {vs_best:.1%}, panel avg {avg_c:.1%} (best {avg_b:.1%}) -> "
            f"{'NEW BEST' if promoted else 'kept the best'} | {detail}  ({time.strftime('%H:%M')})")
    log(f"=== all done {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
