"""Troops hitting a tower in the recordings vs the simulator: per kind, the HP one hit takes off, the time
between hits and the damage per second, from a tower's HP number while units of one kind only are at it.

    .venv\\Scripts\\python.exe tools\\tower_hits_check.py [caches...]   (default: every raw/perception/cache/video*.json)
"""
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim_env as ce  # noqa: E402

CARD = {"knight": "knight", "archer": "archers", "minion": "minions", "giant": "giant", "mini_pekka": "mini_pekka",
        "musketeer": "musketeer"}
TOWER_POS = {"red_left": (3.5, 25.5), "red_right": (14.5, 25.5), "blue_left": (3.5, 6.5), "blue_right": (14.5, 6.5)}


def attackers(frame, tower, reach):
    """Kinds (and counts) of the other side's troops within reach of a tower in this frame."""
    side = "blue" if tower.startswith("red") else "red"
    tx, ty = TOWER_POS[tower]
    out = {}
    for d in frame.get("det", []):
        if d[1] == side and d[0] in CARD and d[2] >= 0.5 and math.dist((d[7], d[8]), (tx, ty)) <= reach[d[0]]:
            out[d[0]] = out.get(d[0], 0) + 1
    return out


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json")))
    data = ce.load_data()
    # how close a troop of each kind stands to a tower it hits: its range + its radius + the tower's ~1.5
    reach = {cls: data.card(card).unit.attack.range + data.card(card).unit.radius + 2.0 for cls, card in CARD.items()}
    drops = {cls: [] for cls in CARD}     # HP taken off by one reading-to-reading drop, one kind alone at the tower
    gaps = {cls: [] for cls in CARD}      # s between drops (single troops only)
    dps = {cls: [] for cls in CARD}       # (damage, seconds, units) over stretches with one kind alone at the tower
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        for tower in TOWER_POS:
            reads = []
            for f in frames:
                hp, exact = f.get("towers", {}).get(tower, [None, False])
                if exact and hp:
                    reads.append((f["t"], hp, attackers(f, tower, reach)))
            # readings agreeing 2 in a row (a single misread number is not a hit)
            stable = [r for k, r in enumerate(reads[1:], 1) if r[1] == reads[k - 1][1]]
            last_drop = {}
            stretch = None
            for (t0, h0, a0), (t1, h1, a1) in zip(stable, stable[1:]):
                kinds = set(a0) | set(a1)
                alone = len(kinds) == 1 and t1 - t0 < 1.0
                cls = next(iter(kinds)) if alone else None
                if alone and h1 < h0:
                    drops[cls].append(h0 - h1)
                    n = max(a0.get(cls, 0), a1.get(cls, 0))
                    if n == 1 and cls in last_drop and t0 - last_drop[cls] < 4.0:
                        gaps[cls].append(t1 - last_drop[cls])
                    last_drop[cls] = t1
                if alone:
                    if stretch is None or stretch[0] != cls:
                        if stretch is not None and stretch[3] > 1.5:
                            dps[stretch[0]].append(stretch[2:])
                        stretch = [cls, t0, 0, 0.0, 0]
                    stretch[2] += max(0, h0 - h1)
                    stretch[3] = t1 - stretch[1]
                    stretch[4] = max(stretch[4], a0.get(cls, 0), a1.get(cls, 0))
                else:
                    if stretch is not None and stretch[3] > 1.5:
                        dps[stretch[0]].append(stretch[2:])
                    stretch = None
    print(f"{len(files)} recordings")
    print(f"{'unit':11s} {'one hit: game':>14s} {'sim':>6s}   {'time between hits: game':>24s} {'sim':>5s}   "
          f"{'tower damage/s per unit: game':>30s} {'sim':>6s}")
    for cls, card in CARD.items():
        u = data.card(card).unit
        a = u.attack
        sim_hit = a.damage - a.damage * (100 - a.tower_damage_pct) // 100 if a.tower_damage_pct != 100 else a.damage
        sim_dps = sim_hit / (a.hit_speed_ms / 1000.0)
        d = drops[cls]
        hit = f"{statistics.median(d):6.0f} ({len(d):3d})" if len(d) >= 3 else f"{'-':>14s}"
        g = gaps[cls]
        gap = f"{statistics.median(g):5.2f}s ({len(g):3d})" if len(g) >= 3 else f"{'-':>24s}"
        s = dps[cls]
        if s:
            tot_d = sum(x[0] for x in s)
            tot_t = sum(x[1] * max(1, x[2]) for x in s)
            per = f"{tot_d / tot_t:6.0f} ({len(s):2d} stretches, {sum(x[1] for x in s):4.0f}s)"
        else:
            per = f"{'-':>30s}"
        print(f"{cls:11s} {hit:>14s} {sim_hit:6d}   {gap:>24s} {a.hit_speed_ms / 1000:5.2f}   {per:>30s} {sim_dps:6.0f}")
    print("one hit: a drop between two readings (two hits can merge into one, so the median can come out higher);"
          " damage/s divides by the number of that kind seen at the tower (some may be hitting something else)")


if __name__ == "__main__":
    main()
