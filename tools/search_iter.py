"""Expert iteration: the network learns what its own lookahead finds, round after round. Each round
  1. the best policy so far plays itself with lookahead (tools/search_games.py, several workers side by side),
  2. a copy of it is trained on those games (tools/distill_search.py: the search's view of the moves it compared),
  3. the copy becomes the best only if it beats the best fairly (each at its own rhythm, --games games) and holds
     up on a panel of other policies, as in tools/train_gated.py.
A better net makes a better search, which makes better games to learn from.

    .venv\\Scripts\\python.exe tools\\search_iter.py --name si --start runs/v7e/top.pt --hours 10

Everything in runs/<name> (best.pt, round_N.pt); games in raw/search_games/<name>_rN. Progress:
raw/train_<name>_log.txt (">>>" lines), ends with "=== search iteration done".
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
PANEL = ["runs/v7b/final.pt", "runs/ppo_v6/best.pt", "pusher"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="si")
    p.add_argument("--start", default="runs/v7e/top.pt")
    p.add_argument("--hours", type=float, default=10)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--games", type=int, default=150, help="games per worker per round")
    p.add_argument("--keep-rounds", type=int, default=3, help="train on this many latest rounds' games")
    p.add_argument("--eval-games", type=int, default=800)
    p.add_argument("--panel-games", type=int, default=200)
    p.add_argument("--promote", type=float, default=0.53,
                   help="win rate over the best to replace it (a distilled step gains a few %%, not 10)")
    p.add_argument("--panel-tol", type=float, default=0.03)
    p.add_argument("--distill-extra", default="", help="more tools/distill_search.py arguments")
    args = p.parse_args()

    out_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(ROOT, "raw", f"train_{args.name}_log.txt")

    def log(text):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text, flush=True)

    def run(cmd, tag):
        out = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        with open(os.path.join(ROOT, "raw", f"train_{args.name}_{tag}.txt"), "w", encoding="utf-8") as f:
            f.write(out)
        return out

    def score(model, opponent, games):
        # as in live play: moves sampled at temperature 0.5 (greedy play hid the distilled nets' gain: ~50% greedy,
        # +5% over the start at 0.5)
        cmd = [PY, "python/evaluate.py", "--model", model, "--opponent", opponent, "--deck", DECK, "--games",
               str(games), "--envs", "100", "--model-temp", "0.5"]
        cmd += ["--decision-ticks", "4"] if opponent in ("heuristic", "pusher") else ["--native-rhythm", "--opp-temp", "0.5"]
        out = run(cmd, "eval")
        m = re.search(r"score ([\d.]+)", out)
        if not m:
            log(f"!!! evaluation failed: {out[-1500:]}")
            raise SystemExit(1)
        return float(m.group(1))

    def panel(model):
        return {o: score(model, o, args.panel_games) for o in PANEL}

    best = os.path.join(out_dir, "best.pt")
    if not os.path.exists(best):
        shutil.copy(os.path.join(ROOT, args.start), best)
    rel_best = os.path.relpath(best, ROOT)
    started = time.time()
    done = [int(m.group(1)) for m in (re.fullmatch(r"round_(\d+)\.pt", n) for n in os.listdir(out_dir)) if m]
    rnd = max(done, default=0) + 1
    best_is_top = args.start.replace("\\", "/") == "runs/v7e/top.pt" and not done  # no need to play it twice
    best_panel = panel(rel_best)
    log(f">>> start: best = {args.start}; panel " + ", ".join(f"{o} {s:.0%}" for o, s in best_panel.items()))
    while time.time() - started < args.hours * 3600:
        t0 = time.time()
        games_dir = os.path.join("raw", "search_games", f"{args.name}_r{rnd}")
        shards = -(-args.games // 20) * args.workers  # search_games.py saves a shard per 20 games
        if os.path.isdir(os.path.join(ROOT, games_dir)) and len(os.listdir(os.path.join(ROOT, games_dir))) >= shards:
            log(f">>> round {rnd}: its games are there already (a restart), on to training")
            logs = []
        else:
            logs = [open(os.path.join(ROOT, "raw", f"train_{args.name}_gen{w}.txt"), "w") for w in range(args.workers)]
        workers = [subprocess.Popen([PY, "tools/search_games.py", "--model", rel_best, "--out", games_dir,
                                     "--games", str(args.games), "--seed", str(rnd * 100 + w)], cwd=ROOT,
                                    stdout=f, stderr=subprocess.STDOUT)
                   for w, f in enumerate(logs)]
        codes = [w.wait() for w in workers]
        for f in logs:
            f.close()
        if any(codes):
            log(f"!!! round {rnd}: a game worker failed ({codes}), see raw/train_{args.name}_gen*.txt")
            raise SystemExit(1)
        if logs:
            log(f">>> round {rnd}: {args.workers * args.games} games with lookahead in {(time.time() - t0) / 60:.0f} min")
        data = [os.path.join("raw", "search_games", f"{args.name}_r{r}") for r in range(max(1, rnd - args.keep_rounds + 1), rnd + 1)]
        data = [d for d in data if os.path.isdir(os.path.join(ROOT, d))]
        out = run([PY, "tools/distill_search.py", "--data", *data, "--start", rel_best, "--run",
                   f"{args.name}_r{rnd}"] + args.distill_extra.split(), "distill")
        if "=== distill done" not in out:
            log(f"!!! round {rnd}: distillation failed: {out[-1500:]}")
            raise SystemExit(1)
        log(">>> round " + str(rnd) + " trained: " + " | ".join(l for l in out.splitlines() if l.startswith("epoch")))
        cand = os.path.join(out_dir, f"round_{rnd}.pt")
        shutil.copy(os.path.join(ROOT, "runs", f"{args.name}_r{rnd}", "final.pt"), cand)
        rel_cand = os.path.relpath(cand, ROOT)
        vs_best = score(rel_cand, rel_best, args.eval_games)
        vs_top = vs_best if best_is_top else score(rel_cand, "runs/v7e/top.pt", args.eval_games)
        promoted = False
        if vs_best >= args.promote:
            cand_panel = panel(rel_cand)
            avg_c = sum(cand_panel.values()) / len(cand_panel)
            avg_b = sum(best_panel.values()) / len(best_panel)
            promoted = avg_c >= avg_b - args.panel_tol
            log(f">>> round {rnd}: vs best {vs_best:.1%}, vs v7e top {vs_top:.1%}, panel avg {avg_c:.1%} "
                f"(best {avg_b:.1%}) -> {'PROMOTED' if promoted else 'kept the best'}")
            if promoted:
                shutil.copy(cand, best)
                best_panel, best_is_top = cand_panel, False
        else:
            log(f">>> round {rnd}: vs best {vs_best:.1%}, vs v7e top {vs_top:.1%} -> kept the best "
                f"({(time.time() - t0) / 60:.0f} min a round)")
        rnd += 1
    log("=== search iteration done")


if __name__ == "__main__":
    main()
