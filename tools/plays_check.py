"""How well the live twin finds the opponent's plays, over every two-sided recording, from both sides.

    .venv\\Scripts\\python.exe tools\\plays_check.py [--matches video8 video9 ...] [--cache raw/perception/cache]

A battle recorded by both players is checked twice: the blue player's twin looking for red's plays (red's
own hand is the truth), and the red player's twin looking for blue's. Per battle and side: plays found
right, missed, extra (found but not played), and how far off in time and position the right ones were.
"""
import argparse
import glob
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402
from compare_red_truth import found_plays, match_plays, true_plays  # noqa: E402
from crsim_env import load_data  # noqa: E402


def two_sided(cache):
    names = [os.path.basename(p)[:-len("_red.json")] for p in glob.glob(os.path.join(cache, "video*_red.json"))]
    return sorted((n for n in names if os.path.exists(os.path.join(cache, f"{n}.json"))),
                  key=lambda n: int(re.sub(r"\D", "", n) or 0))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=os.path.join(ROOT, "raw", "perception", "cache"))
    p.add_argument("--matches", nargs="*", default=None)
    p.add_argument("--details", action="store_true", help="list every missed and extra play")
    args = p.parse_args()
    data = load_data()
    fmt = lambda q: "   ?      " if q is None else f"({q[0]:4.1f},{q[1]:4.1f})"  # noqa: E731
    total = {"truth": 0, "right": 0, "missed": 0, "extra": 0}
    dts, dps = [], []
    print("battle      twin of  finds   played  right  missed  extra")
    for m in args.matches or two_sided(args.cache):
        frames = {v: twin.load_cache(os.path.join(args.cache, f"{v}.json"))["frames"] for v in (m, f"{m}_red")}
        for me, other, label in ((m, f"{m}_red", "blue     red's"), (f"{m}_red", m, "red      blue's")):
            if not frames[me] or not frames[other]:
                continue
            truth = true_plays(frames[other], data)
            found, rejected = found_plays(frames[me], data)
            hits, missed, extra = match_plays(truth, found)
            total["truth"] += len(truth)
            total["right"] += len(hits)
            total["missed"] += len(missed)
            total["extra"] += len(extra)
            dts += [abs(h[3][0] - h[0]) for h in hits]
            dps += [math.dist(h[2], h[3][2]) for h in hits if h[2] and h[3][2]]
            print(f"{m:10s}  {label}  {len(truth):6d}  {len(hits):5d}  {len(missed):6d}  {len(extra):5d}")
            if args.details:
                for t, card, pos in missed:
                    near = [(abs(g - t), f"rejected {g:6.1f}s {fmt(q)} ({why})") for g, c, q, why in rejected
                            if c == card and abs(g - t) < 8]
                    print(f"      missed {t:6.1f}s {card:10s} {fmt(pos)}  " + (min(near)[1] if near else ""))
                for g, card, pos in extra:
                    print(f"      extra  {g:6.1f}s {card:10s} {fmt(pos)}")
    n = max(1, total["truth"])
    print(f"\nall: {total['truth']} plays, {total['right']} found right ({total['right'] / n:.1%}), "
          f"{total['missed']} missed, {total['extra']} extra")
    if dts:
        print(f"right ones: time off by median {statistics.median(dts):.2f}s, position off by median "
              f"{statistics.median(dps):.2f} tiles" if dps else "")


if __name__ == "__main__":
    main()
