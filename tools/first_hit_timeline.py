"""Timelines of troops put near a tower one by one (a first-hit test recording): when each troop first shows up
near a tower, and when the tower's HP number (read the same 3 times in a row) goes down after that.

    .venv\\Scripts\\python.exe tools\\first_hit_timeline.py raw/perception/cache/video31.json [more...]
Prints, per episode: kind, first seen (its deploy starts about then), first steady drop and its size, the next
drops. The simulator's figure: deploy + first-attack delay + first hit (+ the projectile's flight for ranged).
"""
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim_env as ce  # noqa: E402
from tower_hits_check import CARD, TOWER_POS  # noqa: E402


def steady_drops(frames, tower, need=3):
    """(time, hp before, hp after) of steady changes of a tower's HP number."""
    out, cur, run, last_t, cand = [], None, 0, None, None
    for f in frames:
        hp, exact = f.get("towers", {}).get(tower, [None, False])
        if not (exact and hp):
            continue
        if hp == cand:
            run += 1
        else:
            cand, run, last_t = hp, 1, f["t"]
        if run == need and hp != cur:
            if cur is not None and hp < cur:
                out.append((last_t, cur, hp))
            cur = hp
    return out


def main():
    data = ce.load_data()
    for path in sys.argv[1:]:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        print(f"== {os.path.basename(path)}")
        for tower, (tx, ty) in TOWER_POS.items():
            side = "blue" if tower.startswith("red") else "red"
            drops = steady_drops(frames, tower)
            if not drops:
                continue
            # first sightings near this tower (within 4 tiles) of each kind, one per 6 s at most
            seen = []
            for f in frames:
                for d in f.get("det", []):
                    if d[1] == side and d[0] in CARD and d[2] >= 0.6 and math.dist((d[7], d[8]), (tx, ty)) < 4.5:
                        if not any(s[1] == d[0] and f["t"] - s[0] < 6.0 for s in seen):
                            seen.append((f["t"], d[0], round(math.dist((d[7], d[8]), (tx, ty)), 1)))
            for t0, cls, dist in seen:
                after = [dr for dr in drops if t0 < dr[0] < t0 + 6.0]
                if not after:
                    continue
                u = data.card(CARD[cls]).unit
                sim = (u.deploy_time_ms + getattr(u, "first_attack_delay_ms", 0) + u.attack.first_hit_ms) / 1000
                firsts = ", ".join(f"{dr[0] - t0:.2f}s (-{dr[1] - dr[2]})" for dr in after[:4])
                print(f"  {tower:10s} {cls:10s} seen at {t0:6.1f}s, {dist:3.1f} tiles away; drops after: {firsts}"
                      f"   | sim from landing: {sim:.2f}s (+flight if ranged)")


if __name__ == "__main__":
    main()
