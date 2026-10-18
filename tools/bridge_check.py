"""Where on a bridge troops cross, in the recordings: the x at which a ground troop's track passes the middle of
the river (y = 16), relative to the bridge centre, by where the troop came from (its x 3 tiles before the river).

    .venv\\Scripts\\python.exe tools\\bridge_check.py [caches...]
"""
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_check import tracks_of  # noqa: E402

GROUND = ("knight", "archer", "giant", "mini_pekka", "musketeer")
BRIDGES = (3.5, 14.5)


def crossings(track):
    """(x before the river, x on the river's middle) for a track that walks across y = 16."""
    for k in range(len(track) - 1):
        (t0, x0, y0), (t1, x1, y1) = track[k], track[k + 1]
        if (y0 - 16) * (y1 - 16) <= 0 and y0 != y1 and t1 - t0 < 0.5:
            x_mid = x0 + (x1 - x0) * (16 - y0) / (y1 - y0)
            back = 3 if y1 > y0 else -3  # where it came from: 3 tiles before the river on its way
            before = [p for p in track[:k + 1] if abs(p[2] - (16 - back)) < 0.5]
            if before:
                return before[-1][1], x_mid
    return None


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json")))
    rows = []
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        for cls in GROUND:
            for team in ("blue", "red"):
                for tr in tracks_of(frames, cls, team):
                    c = crossings(tr)
                    if c is not None:
                        rows.append((cls,) + c)
    print(f"{len(rows)} crossings in {len(files)} recordings")
    print("x on the bridge - bridge centre (tiles), by where the troop came from (x 3 tiles before, - bridge centre):")
    bins = [(-9, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 9)]
    for lo, hi in bins:
        offs = []
        for cls, xb, xm in rows:
            bridge = min(BRIDGES, key=lambda b: abs(b - xm))
            if lo <= xb - bridge < hi:
                offs.append(xm - bridge)
        if len(offs) >= 5:
            q = sorted(offs)
            print(f"  came from {lo:+5.1f}..{hi:+5.1f}: n={len(offs):4d}  median {statistics.median(offs):+.2f}  "
                  f"10%..90% {q[len(q) // 10]:+.2f}..{q[len(q) * 9 // 10]:+.2f}")
    allx = sorted(xm - min(BRIDGES, key=lambda b: abs(b - xm)) for _, _, xm in rows)
    print(f"all: 5%..95% of crossings at {allx[len(allx) // 20]:+.2f}..{allx[len(allx) * 19 // 20]:+.2f} tiles "
          f"from the centre (the simulator: bridge half width 1.0, troops aim at the centre)")


if __name__ == "__main__":
    main()
