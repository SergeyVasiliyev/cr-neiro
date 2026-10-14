"""Finds two boxes on one unit in the hand labels and keeps the right one.

    .venv\\Scripts\\python.exe tools\\fix_duplicate_labels.py [--apply]

Pre-labelling puts the detector's boxes on a frame; where the detector had boxed one unit twice (once per
side, or as two kinds), correcting one box and missing the other leaves a frame that says the unit is both
a blue and a red knight - which teaches the detector exactly that mistake. Boxes that cover each other
almost entirely are taken as one unit; which kind and side it is, is voted by the detector over the
recording around that moment (the same unit seen in some 20 frames either side, by its position), and the
box that loses the vote is dropped. Pairs the vote can't tell apart are listed and left alone.
Without --apply it only lists what it would do (labels are backed up before any change).
"""
import argparse
import math
import os
import shutil
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from label_units import LABELS, load_labels, save_labels  # noqa: E402
from perception.layout import Layout  # noqa: E402
from perception.units import UnitDetector  # noqa: E402

LAYOUT = Layout(scale=1.0)
SIMILAR = {"knight", "mini_pekka", "musketeer", "archer"}  # about the same size on screen
WINDOW = 1.5   # s either side of the frame
CLEAR = 1.5    # the winner needs this many times the loser's support


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_duplicate(o, p):
    if "hit" in o["cls"] or "hit" in p["cls"] or o["cls"] in ("fireball", "arrows") or p["cls"] in ("fireball", "arrows"):
        return False
    v = iou(o["box"], p["box"])
    if v >= 0.8:
        return True
    return v >= 0.6 and (o["cls"] == p["cls"] or (o["cls"] in SIMILAR and p["cls"] in SIMILAR))


def support(frames, t, tile):
    """Detector confidence summed per (kind, side) near this spot around time t (the frame itself left out)."""
    votes = defaultdict(float)
    for fr in frames:
        dt = abs(fr["t"] - t)
        if dt > WINDOW or dt < 0.05:
            continue
        gate = 0.8 + 1.2 * dt  # it may have walked
        for d in fr["det"]:
            if d[0] in ("fireball", "arrows", "fireball_hit", "arrows_hit") or d[2] < 0.25:
                continue
            if math.dist(UnitDetector.box_to_tile(d[0], d[3:7], LAYOUT), tile) <= gate:
                votes[(d[0], d[1])] += d[2]
    return votes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    doc = load_labels()
    caches = {}
    changed = unsure = 0
    for name, entry in sorted(doc["frames"].items()):
        if entry.get("status") != "labeled":
            continue
        objs = entry["objects"]
        drop = set()
        for a in range(len(objs)):
            for b in range(a + 1, len(objs)):
                if a in drop or b in drop or not is_duplicate(objs[a], objs[b]):
                    continue
                o, q = objs[a], objs[b]
                if (o["cls"], o["team"]) == (q["cls"], q["team"]):
                    drop.add(b)
                    print(f"{name}: {o['team']} {o['cls']} boxed twice - one dropped")
                    continue
                rec, t = name[:-4].rsplit("_", 1)
                if rec not in caches:
                    path = os.path.join(ROOT, "raw", "perception", "cache", f"{rec}.json")
                    caches[rec] = twin.load_cache(path)["frames"] if os.path.exists(path) else None
                if caches[rec] is None:
                    print(f"{name}: no perception cache for {rec}")
                    unsure += 1
                    continue
                tile = UnitDetector.box_to_tile(o["cls"], o["box"], LAYOUT)
                votes = support(caches[rec], float(t), tile)
                so, sq = votes.get((o["cls"], o["team"]), 0.0), votes.get((q["cls"], q["team"]), 0.0)
                if max(so, sq) < 2.0 or max(so, sq) < CLEAR * min(so, sq):
                    print(f"{name}: {o['team']} {o['cls']} ({so:.1f}) or {q['team']} {q['cls']} ({sq:.1f})? left as is")
                    unsure += 1
                    continue
                keep, lose = (a, b) if so > sq else (b, a)
                drop.add(lose)
                print(f"{name}: {objs[keep]['team']} {objs[keep]['cls']} ({max(so, sq):.1f}) over "
                      f"{objs[lose]['team']} {objs[lose]['cls']} ({min(so, sq):.1f})")
        if drop:
            changed += 1
            entry["objects"] = [o for k, o in enumerate(objs) if k not in drop]
    print(f"\n{changed} frames {'fixed' if args.apply else 'to fix'}; {unsure} pairs left as they are")
    if args.apply and changed:
        shutil.copy(LABELS, LABELS.replace(".json", "_before_duplicates.json"))
        save_labels(doc)


if __name__ == "__main__":
    main()
