"""Keeps training the best policy, mostly against itself (its recent versions) and a little against the
scripted bots, in chunks; a chunk's result replaces the best policy only if it beats it.

    .venv\\Scripts\\python.exe tools\\train_more.py --start runs\\ppo_v5d\\latest.pt --hours 12

Each chunk trains --chunk decisions with the learning rate going down from --lr to 5%, so it ends settled (a
policy stopped half-way at a high rate plays clearly worse), pulled towards the best policy at first (--anchor).
Then it plays the best policy so far (greedy, --games games): at --promote or more it becomes the new best,
runs/ppo_v6/best.pt. The next chunk goes on from the chunk's result, or from the best one if the result fell
clearly behind (below --fallback). Progress: raw/train_v6_log.txt, ends with "=== all done". Stopping at any
time is safe: best.pt is always the best policy so far.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG = os.path.join(ROOT, "raw", "train_v6_log.txt")
RUN = os.path.join(ROOT, "runs", "ppo_v6")
DECK = "knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer"
COMMON = ["--deck", DECK, "--decision-ticks", "6", "--envs", "64", "--key-spots",
          "--reward-win", "1", "--reward-tower", "0.5", "--reward-crown", "0.1", "--reward-trade", "0.03",
          "--trade-floor", "0.25", "--reward-leak", "0.005"]


def log(text):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text, flush=True)


def train(title, *args):
    log(f"--- {title}  ({time.strftime('%H:%M:%S')})")
    with open(LOG, "a", encoding="utf-8") as f:
        code = subprocess.run([PY, "python/train_ppo.py", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        log(f"!!! training failed (exit {code})")
        raise SystemExit(code)


def win_rate(model, opponent, games):
    out = subprocess.run([PY, "python/evaluate.py", "--model", model, "--opponent", opponent, "--deck", DECK,
                          "--games", str(games), "--envs", "100", "--decision-ticks", "6", "--greedy"],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    m = re.search(r"wins \d+ \(([\d.]+)%\)", out)
    if not m:
        log(f"!!! evaluation failed:\n{out[-2000:]}")
        raise SystemExit(1)
    return float(m.group(1)) / 100


def steps_of(path):
    return int(torch.load(path, map_location="cpu", weights_only=False)["global_step"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="runs/ppo_v5d/latest.pt", help="the policy to start from (ignored if "
                   "runs/ppo_v6/best.pt exists: then it goes on from there)")
    p.add_argument("--hours", type=float, default=12)
    p.add_argument("--chunk", type=float, default=6e6, help="decisions per chunk (~1 h)")
    p.add_argument("--opponent", default="heuristic:0.05,pusher:0.05,selfplay:0.9")
    p.add_argument("--lr", type=float, default=3e-5, help="learning rate at the start of each chunk")
    p.add_argument("--ent", type=float, default=0.02)
    p.add_argument("--anchor", type=float, default=0.02, help="KL pull towards the best policy, fading out in the chunk")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--promote", type=float, default=0.55)
    p.add_argument("--fallback", type=float, default=0.45)
    args = p.parse_args()

    started = time.time()
    os.makedirs(RUN, exist_ok=True)
    best = os.path.join(RUN, "best.pt")
    if not os.path.exists(best):
        shutil.copy(os.path.join(ROOT, args.start), best)
    if os.path.exists(LOG):  # a fresh log each start (the keep-awake helper waits for its "=== all done")
        os.replace(LOG, LOG.replace(".txt", time.strftime("_%Y%m%d_%H%M%S.txt", time.localtime(os.path.getmtime(LOG)))))
    with open(LOG, "w", encoding="utf-8") as f:
        f.write(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}, best so far: {steps_of(best) / 1e6:.1f}M decisions\n")
    current = best
    done = [int(m.group(1)) for m in (re.fullmatch(r"chunk_(\d+)\.pt", n) for n in os.listdir(RUN)) if m]
    chunk = max(done, default=0)  # numbering goes on after an earlier start
    while time.time() - started < args.hours * 3600:
        chunk += 1
        total = steps_of(current) + int(args.chunk)
        train(f"chunk {chunk}: {args.chunk / 1e6:.0f}M decisions from {os.path.basename(current)}",
              "--run", "ppo_v6", "--resume", current, "--total-steps", str(total), "--lr", str(args.lr),
              "--ent", str(args.ent), "--anchor", best, "--anchor-coef", str(args.anchor), "--anchor-coef-end", "0",
              "--opponent", args.opponent, "--seed", str(100 + chunk), *COMMON)
        result = os.path.join(RUN, f"chunk_{chunk}.pt")
        shutil.copy(os.path.join(RUN, "latest.pt"), result)
        wr = win_rate(result, best, args.games)
        bots = ", ".join(f"vs {b} {win_rate(result, b, 200):.0%}" for b in ("heuristic", "pusher"))
        if wr >= args.promote:
            shutil.copy(result, best)
            verdict, current = "NEW BEST", result
        elif wr >= args.fallback:
            verdict, current = "kept going (not better yet)", result
        else:
            verdict, current = "fell behind: back to the best", best
        log(f">>> chunk {chunk}: beats the best {wr:.1%} ({bots}) -> {verdict}  ({time.strftime('%H:%M:%S')})")
    log(f"=== all done {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
