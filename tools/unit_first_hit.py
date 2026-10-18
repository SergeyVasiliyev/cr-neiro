"""First hits between troops, in the recordings vs the simulator: two enemies come into reach of each other,
and each one's first damage shows as an HP bar appearing over it (a troop at full HP shows none).

    .venv\\Scripts\\python.exe tools\\unit_first_hit.py [caches...]   (default: every cache)
Per attacker kind: the time from contact to its victim's first bar (its first hit, plus ~0.1 s of reading lag)
vs the simulator's first-hit time. Only meetings where the victim had no bar just before, one of each kind.
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
import twin  # noqa: E402
from perception.unit_hp import assign  # noqa: E402
from perception.units import Detection  # noqa: E402

CARD = {"knight": "knight", "mini_pekka": "mini_pekka", "giant": "giant", "musketeer": "musketeer",
        "archer": "archers"}
VICTIMS = {"knight", "mini_pekka", "giant", "musketeer", "archer"}


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json")))
    data = ce.load_data()
    unit = {cls: data.card(card).unit for cls, card in CARD.items()}
    delays = {cls: [] for cls in CARD}
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        dets, barred = [], []
        for f in frames:
            ds = [Detection(d[0], d[1], d[2], tuple(d[3:7]), (d[7], d[8])) for d in f.get("det", [])]
            bars = twin.bars_from_cache(f)
            dets.append(ds)
            barred.append(set(assign(bars, ds)) if bars else set())
        done = set()
        for k, f in enumerate(frames):
            ds = dets[k]
            for a in ds:
                if a.cls not in CARD:
                    continue
                for v_i, v in enumerate(ds):
                    if v.team == a.team or v.cls not in VICTIMS or v.team == "any":
                        continue
                    if sum(1 for o in ds if o.cls == a.cls and o.team == a.team) > 1 or \
                            sum(1 for o in ds if o.cls == v.cls and o.team == v.team) > 1:
                        continue  # one of each kind only: the bar is surely this one's
                    gap = math.dist(a.tile, v.tile) - unit[a.cls].radius - unit.get(v.cls, unit["knight"]).radius
                    if gap > unit[a.cls].attack.range + 0.2:
                        continue
                    key = (a.cls, a.team, v.cls)
                    if any(key == d[0] and f["t"] - d[1] < 8.0 for d in done):
                        continue
                    done.add((key, f["t"]))
                    # the victim must have had no bar for the 1 s before contact
                    def victim_index(j):
                        same = [i for i, o in enumerate(dets[j]) if o.cls == v.cls and o.team == v.team]
                        return same[0] if len(same) == 1 else None
                    before = [j for j in range(max(0, k - 8), k) if frames[k]["t"] - frames[j]["t"] <= 1.0]
                    if any(victim_index(j) is not None and victim_index(j) in barred[j] for j in before):
                        continue
                    for j in range(k, len(frames)):
                        if frames[j]["t"] - f["t"] > 3.0:
                            break
                        vi = victim_index(j)
                        if vi is not None and vi in barred[j]:
                            delays[a.cls].append(frames[j]["t"] - f["t"])
                            break
    print(f"{len(files)} recordings: first hit on an enemy troop, s from coming into reach")
    print(f"{'attacker':11s} {'game median':>12s} {'25%..75%':>14s} {'n':>4s}   {'sim first hit':>13s}")
    for cls in CARD:
        v = sorted(delays[cls])
        sim = unit[cls].attack.first_hit_ms / 1000
        if len(v) >= 3:
            print(f"{cls:11s} {statistics.median(v):12.2f} {v[len(v) // 4]:6.2f}..{v[len(v) * 3 // 4]:5.2f} {len(v):4d}   {sim:13.2f}")
        else:
            print(f"{cls:11s} {'-':>12s} {'':>14s} {len(v):4d}   {sim:13.2f}")
    print("(ranged: + the projectile's flight; reading lag ~0.1 s; a just-deployed troop waits longer)")


if __name__ == "__main__":
    main()
