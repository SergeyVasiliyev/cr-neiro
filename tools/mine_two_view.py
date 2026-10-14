"""Finds the detector's mistakes with two recordings of one battle (one per side) and makes label frames of them.

    .venv\\Scripts\\python.exe tools\\mine_two_view.py [--matches video8 video9 ...] [--count 150]

The same unit is seen from both sides at once (positions turn 180 degrees, sides swap). Where one view
sees a unit confidently and the other misses it, gives it the other side, or another kind, that view's
detector was wrong there. The frames with most such mistakes (from both recordings) are written to
raw/labels/frames and pre-labelled in raw/labels/labels.json: the detector's own boxes, plus the units it
missed, carried over from the other view - but only where it had at least a weak detection of its own
(a unit fully hidden in that view is not labelled: the detector must not learn to imagine). Correct them
in scripts\\label.bat as usual.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from label_units import FRAMES, load_labels, save_labels  # noqa: E402
from perception.layout import Layout  # noqa: E402
from perception.units import FEET_OFFSET_TILES, FLYING, FLYING_OFFSET_TILES, Detection, UnitDetector  # noqa: E402

W, H = 18.0, 32.0
SURE = 0.6      # a detection this confident in the other view is taken as true
WEAK = 0.15     # the lowest confidence the caches keep
NEAR = 0.8      # tiles: the same unit in the two views
LAYOUT = Layout(scale=1.0)


def all_dets(frame):
    return [Detection(d[0], d[1], d[2], tuple(d[3:7]), UnitDetector.box_to_tile(d[0], d[3:7], LAYOUT))
            for d in frame["det"]]


def flip(tile):
    return W - tile[0], H - tile[1]


def box_at(d_other, tile):
    """The other view's box for this unit, moved to where it is in this view (sizes are the same)."""
    x0, y0, x1, y1 = d_other.box
    w, h = x1 - x0, y1 - y0
    if d_other.cls in FLYING:
        cx, cy = LAYOUT.tile_to_px(tile[0], tile[1] + FLYING_OFFSET_TILES)
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    cx, by = LAYOUT.tile_to_px(tile[0], tile[1] - FEET_OFFSET_TILES.get(d_other.cls, 0.0))
    return cx - w / 2, by - h, cx + w / 2, by


def compare(this, other):
    """Mistakes of `this` view against `other` (both lists of Detection; other's tiles already flipped).
    Returns (score, [(kind, fix)]): fix = (cls, team, box) to add / correct in this view."""
    fixes = []
    for o in other:
        if o.is_spell or o.conf < SURE:
            continue
        team_here = "red" if o.team == "blue" else "blue"
        same = [d for d in this if not d.is_spell and math.dist(d.tile, o.tile) < NEAR]
        sure_here = [d for d in same if d.conf >= 0.35]
        if not sure_here:
            weak = [d for d in same if d.cls == o.cls]
            if weak:  # seen a little here: a real miss, worth a label
                fixes.append(("missed", (o.cls, team_here, box_at(o, o.tile))))
            continue
        best = max(sure_here, key=lambda d: d.conf)
        if best.cls != o.cls and o.conf > best.conf + 0.15:
            fixes.append(("wrong kind", (o.cls, team_here, best.box)))
        elif best.cls == o.cls and best.team != team_here and o.conf > best.conf:
            fixes.append(("wrong side", (o.cls, team_here, best.box)))
    return len(fixes), fixes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matches", nargs="+", default=["video8", "video9", "video10", "video11", "video12"])
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--min-gap", type=float, default=1.5)
    args = p.parse_args()

    cache = os.path.join(ROOT, "raw", "perception", "cache")
    candidates = []  # (score, recording, frame time, fixes, frame)
    for m in args.matches:
        views = {}
        for name in (m, f"{m}_red"):
            frames = twin.load_cache(os.path.join(cache, f"{name}.json"))["frames"]
            start, _ = twin.battle_start(frames)
            views[name] = (frames, start)
        (fb, sb), (fr, sr) = views[m], views[f"{m}_red"]
        red_by_g = {round(f["t"] - sr, 1): f for f in fr}
        for f in fb:
            o = red_by_g.get(round(f["t"] - sb, 1))
            if o is None:
                continue
            A, B = all_dets(f), all_dets(o)
            Bf = [Detection(d.cls, d.team, d.conf, d.box, flip(d.tile)) for d in B]
            Af = [Detection(d.cls, d.team, d.conf, d.box, flip(d.tile)) for d in A]
            sa, fa = compare(A, Bf)  # mistakes of the blue recording
            sb_, fb_ = compare(B, Af)  # mistakes of the red recording
            if sa:
                candidates.append((sa, m, f["t"], fa, A))
            if sb_:
                candidates.append((sb_, f"{m}_red", o["t"], fb_, B))
        print(f"{m}: {sum(1 for c in candidates if c[1].split('_')[0] == m)} frames with mistakes", flush=True)

    doc = load_labels()
    labelled = defaultdict(list)
    for f in os.listdir(FRAMES):
        if f.endswith(".jpg") and "_" in f:
            name, t = f[:-4].rsplit("_", 1)
            labelled[name].append(float(t))
    picked = defaultdict(list)
    for score, rec, t, fixes, dets in sorted(candidates, key=lambda c: -c[0]):
        if sum(len(v) for v in picked.values()) >= args.count:
            break
        if any(abs(t - u) < args.min_gap for u in labelled[rec]) or any(abs(t - u) < args.min_gap for u, *_ in picked[rec]):
            continue
        picked[rec].append((t, fixes, dets))

    kinds = defaultdict(int)
    for rec, items in picked.items():
        cap = cv2.VideoCapture(os.path.join(ROOT, "raw", "videos", f"{rec}.mp4"))
        for t, fixes, dets in items:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                continue
            if img.shape[1] != 1080:
                img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_LINEAR)
            name = f"{rec}_{t:06.1f}.jpg"
            cv2.imwrite(os.path.join(FRAMES, name), img)
            objects = []
            fixed_boxes = [fix[2] for _, fix in fixes]
            for d in dets:  # the detector's own boxes (dashed in the tool), except those a fix replaces
                if d.conf < 0.25 or any(abs(d.box[0] - b[0]) < 1 and abs(d.box[1] - b[1]) < 1 for b in fixed_boxes):
                    continue
                objects.append({"cls": d.cls, "team": d.team if d.team != "any" else "blue",
                                "box": [round(v, 1) for v in d.box], "auto": True, "conf": round(d.conf, 2)})
            for kind, (cls, team, box) in fixes:
                kinds[kind] += 1
                objects.append({"cls": cls, "team": team, "box": [round(v, 1) for v in box], "auto": True,
                                "conf": 0.99, "note": f"other view: {kind}"})
            doc["frames"][name] = {"status": "prelabeled", "objects": objects}
    save_labels(doc)
    total = sum(len(v) for v in picked.values())
    print(f"{total} frames -> {FRAMES}: " + ", ".join(f"{len(v)} from {k}" for k, v in sorted(picked.items())))
    print("fixes from the other view: " + ", ".join(f"{k} {n}" for k, n in kinds.items()))


if __name__ == "__main__":
    main()
