"""One round of detector improvement from new two-sided recordings, start to finish (hours: run it overnight).

    .venv\\Scripts\\python.exe tools\\detector_round.py [--name units_v5] [--base <weights>] [--holdout video16]

1. Reads every two-sided recording (raw/videos/videoN.mp4 + videoN_red.mp4) that has no perception cache yet,
   with the current detector.
2. Labels all two-sided recordings automatically (tools/auto_label.py), except the held-out one: the newest
   new battle when there are at least two new ones (or --holdout), kept for a fair comparison.
3. Builds the dataset (hand labels + automatic ones) and trains the new detector, starting from the current one.
4. Compares old and new: on the hand-labelled validation battle (tools/detector_errors.py) and on the
   held-out battle (read again with the new detector; tools/auto_label.py --score).
Everything is logged to raw/detector_round_log.txt, which ends with "=== done". If the training was cut off
(PC restarted), --resume continues it from its last epoch and then does the comparisons.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.units import DEFAULT_WEIGHTS  # noqa: E402

PY = sys.executable
LOG = os.path.join(ROOT, "raw", "detector_round_log.txt")
CACHE = os.path.join(ROOT, "raw", "perception", "cache")


def log(text):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text, flush=True)


def step(title, *args):
    log(f"--- {title}  ({time.strftime('%H:%M:%S')})")
    with open(LOG, "a", encoding="utf-8") as f:
        code = subprocess.call([PY, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if code:
        log(f"!!! failed (exit {code})")
        raise SystemExit(code)


def two_sided():
    names = []
    for path in glob.glob(os.path.join(ROOT, "raw", "videos", "video*_red.mp4")):
        name = os.path.basename(path)[:-len("_red.mp4")]
        if os.path.exists(os.path.join(ROOT, "raw", "videos", f"{name}.mp4")):
            names.append(name)
    return sorted(names, key=lambda n: int(re.sub(r"\D", "", n) or 0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="units_v5")
    p.add_argument("--base", default=DEFAULT_WEIGHTS, help="detector to start from (and to compare with)")
    p.add_argument("--holdout", default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--resume", action="store_true",
                   help="the training of --name was cut off: continue it, then compare (data steps skipped)")
    args = p.parse_args()

    with open(LOG, "a" if args.resume else "w", encoding="utf-8") as f:
        f.write(f"=== start {time.strftime('%Y-%m-%d %H:%M:%S')}{' (resumed)' if args.resume else ''}\n")
    matches = two_sided()
    new = [m for m in matches if not all(os.path.exists(os.path.join(CACHE, f"{v}.json")) for v in (m, f"{m}_red"))]
    holdout = args.holdout or (new[-1] if len(new) >= 2 else None)
    log(f"two-sided battles: {', '.join(matches)}; new: {', '.join(new) or 'none'}; held out: {holdout or 'none'}")

    if args.resume:
        step(f"train {args.name} (continued)", "tools/train_detector.py", "--name", args.name, "--resume")
    else:
        for m in new:
            for v in (m, f"{m}_red"):
                step(f"read {v}", "tools/perceive_video.py", "--video", f"raw/videos/{v}.mp4", "--conf", "0.15",
                     "--weights", args.base)
        train_on = [m for m in matches if m != holdout]
        step("automatic labels", "tools/auto_label.py", "--write", "--matches", *train_on)
        step("dataset", "tools/build_detector_dataset.py", "--auto")
        step(f"train {args.name}", "tools/train_detector.py", "--model", args.base, "--name", args.name,
             "--epochs", str(args.epochs), "--patience", str(args.patience))
    new_weights = os.path.join(ROOT, "raw", "detector", "runs", args.name, "weights", "best.pt")

    step("old detector on the hand-labelled validation battle", "tools/detector_errors.py", "--weights", args.base)
    step("new detector on the hand-labelled validation battle", "tools/detector_errors.py", "--weights", new_weights)
    if holdout:
        new_cache = os.path.join(ROOT, "raw", "perception", f"cache_{args.name}")
        os.makedirs(new_cache, exist_ok=True)
        for v in (holdout, f"{holdout}_red"):
            step(f"read {v} with the new detector", "tools/perceive_video.py", "--video", f"raw/videos/{v}.mp4",
                 "--conf", "0.15", "--weights", new_weights, "--out", os.path.join(new_cache, f"{v}.json"))
        step(f"old detector on {holdout} (not trained on)", "tools/auto_label.py", "--score", "--matches", holdout)
        step(f"new detector on {holdout} (not trained on)", "tools/auto_label.py", "--score", "--matches", holdout,
             "--cache", new_cache)
    log(f"=== done {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
