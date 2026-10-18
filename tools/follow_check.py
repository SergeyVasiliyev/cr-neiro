"""A troop walking right behind its own giant: does it get held up (walk at the giant's pace) or go round it?
Game: walking windows of knights / mini pekkas / musketeers / archers with a friendly giant just ahead of them,
vs the same kind walking freely. Simulator: a giant with each kind put right behind it.

    .venv\\Scripts\\python.exe tools\\follow_check.py
"""
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from speed_check import tracks_of  # noqa: E402

KINDS = {"knight": "knight", "mini_pekka": "mini_pekka", "musketeer": "musketeer", "archer": "archers"}


def game():
    out = {k: {"behind": [], "free": []} for k in KINDS}
    for path in sorted(glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json"))):
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        by_t = {f["t"]: f for f in frames}
        for cls in KINDS:
            for team in ("blue", "red"):
                for tr in tracks_of(frames, cls, team):
                    for k, (t0, x0, y0) in enumerate(tr):
                        later = [p for p in tr[k + 1:] if p[0] - t0 >= 1.0]
                        if not later or later[0][0] - t0 > 1.25:
                            continue
                        t1, x1, y1 = later[0]
                        step = math.dist((x0, y0), (x1, y1))
                        if abs(y1 - y0) < 0.3 or abs(x1 - x0) > abs(y1 - y0):
                            continue  # walking up or down the lane only
                        up = 1 if y1 > y0 else -1
                        giants = [(d[7], d[8]) for d in by_t[t0].get("det", [])
                                  if d[0] == "giant" and d[1] == team and d[2] >= 0.6]
                        ahead = any(abs(gx - x0) < 0.9 and 0.3 < (gy - y0) * up < 1.8 for gx, gy in giants)
                        out[cls]["behind" if ahead else "free"].append(step / (t1 - t0))
    return out


def simulated():
    data = ce.load_data()
    out = {}
    for cls, card in KINDS.items():
        b = crsim.Battle(data, ce.DEFAULT_DECK, ce.DEFAULT_DECK, 0)
        gid = b.twin_spawn(crsim.Team.BLUE, "giant", crsim.Vec2(3.5, 9.0))
        uid = b.twin_spawn(crsim.Team.BLUE, card, crsim.Vec2(3.5, 7.8))
        b.steps(30)  # both deployed and walking
        u = b.find(uid)
        y0 = u.pos.y
        b.steps(40)  # 2 s
        u, g = b.find(uid), b.find(gid)
        out[cls] = ((u.pos.y - y0) / 2.0, u.pos.y > g.pos.y, round(u.pos.x - 3.5, 2))
    return out


def main():
    g = game()
    s = simulated()
    print("tiles/s walking up or down a lane: game free / game right behind own giant  |  simulator behind a giant")
    for cls in KINDS:
        free, behind = g[cls]["free"], g[cls]["behind"]
        fm = f"{statistics.median(free):.2f} ({len(free)})" if len(free) >= 10 else "-"
        bm = f"{statistics.median(behind):.2f} ({len(behind)})" if len(behind) >= 10 else f"- ({len(behind)})"
        sp, passed, dx = s[cls]
        print(f"  {cls:11s} game free {fm:>13s}   behind giant {bm:>12s}   |  sim {sp:.2f}"
              f"{', got past it' if passed else ''} (side step {dx:+.2f})")
    print("  (giant: game ~0.92, simulator 0.92)")


if __name__ == "__main__":
    main()
