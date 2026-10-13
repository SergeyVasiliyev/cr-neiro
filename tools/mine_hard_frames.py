"""Picks the frames most worth labelling next: where the unit detector probably goes wrong.

    .venv\\Scripts\\python.exe tools\\mine_hard_frames.py [--count 150] [--caches "raw/perception/cache/video?.json"]

Uses the perception caches (tools/perceive_video.py) and the tracker. A frame scores for:
  - a unit the tracker follows that the detector missed for a moment in the open (seen before and after);
  - a unit that changes kind or side (a track ends and another one starts in the same spot);
  - spells: any spell detection, and the second after one of our spells is cast (hand reader);
  - the second after one of our plays (the card name / level text covers the unit);
  - crowded frames (many units, overlaps).
The best frames (at least --min-gap seconds apart, away from frames already labelled) are written to
raw/labels/frames; then pre-label them with tools/prelabel_frames.py and correct them in scripts\\label.bat.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from label_units import FRAMES  # noqa: E402

SPELL_CLASSES = {"fireball", "arrows", "fireball_hit", "arrows_hit"}


def near_cover(x, y, frame_dets):
    """Next to a giant (can hide units) - a miss there is expected, not a detector error."""
    return any(d[0] == "giant" and abs(d[7] - x) < 1.5 and -0.5 < y - d[8] < 2.5 for d in frame_dets)


def score_frames(frames):
    times = [fr["t"] for fr in frames]
    index = {round(t, 3): k for k, t in enumerate(times)}
    score = defaultdict(float)
    why = defaultdict(set)

    def add(k, s, reason):
        score[k] += s
        why[k].add(reason)

    tracks = twin.run_tracker(frames)
    for tr in tracks:
        h = tr.history
        for (t1, x1, y1), (t2, x2, y2) in zip(h, h[1:]):
            if 0.15 <= t2 - t1 <= 0.6:  # missed for 1-5 frames, then found again
                for k in range(index.get(round(t1, 3), -1) + 1, index.get(round(t2, 3), -1)):
                    if 0 <= k < len(frames) and not near_cover(x1, y1, frames[k]["det"]) and y1 < 30.5:
                        add(k, 2.0, "missed")
    ends = [(tr.last_seen, tr.history[-1][1:], tr) for tr in tracks]
    for tr in tracks:  # a track starting where another just ended, as something else
        for te, (x, y), other in ends:
            if other is tr or not (0 <= tr.first_seen - te <= 0.4):
                continue
            if math.dist((x, y), tr.history[0][1:]) < 1.2 and (other.cls != tr.cls or other.team != tr.team):
                k = index.get(round(tr.first_seen, 3))
                if k is not None:
                    add(k, 3.0, "mixed up")
                    if k > 0:
                        add(k - 1, 1.5, "mixed up")
    for k, fr in enumerate(frames):
        spells = [d for d in fr["det"] if d[0] in SPELL_CLASSES]  # any confidence the cache kept
        if spells:
            add(k, 3.0, "spell")
            if any(d[1] == "red" for d in spells):
                add(k, 1.5, "red spell")  # the rarest in the data so far
        if sum(1 for d in fr["det"] if d[2] >= 0.35) >= 8:
            add(k, 1.0, "crowded")
    for play in twin.own_plays_from_hand(frames):
        lo, hi = (1.0, 2.8) if play.card in twin.SPELL_CARDS else (0.3, 1.2)
        for k, t in enumerate(times):
            if play.t + lo <= t <= play.t + hi:
                add(k, 2.5 if play.card in twin.SPELL_CARDS else 1.0,
                    "our spell" if play.card in twin.SPELL_CARDS else "just played")
    return score, why


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caches", default=os.path.join(ROOT, "raw", "perception", "cache", "video?.json"))
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--min-gap", type=float, default=1.5, help="seconds between picked frames of one video")
    args = p.parse_args()

    labelled = defaultdict(list)  # video -> times already in raw/labels/frames
    for f in os.listdir(FRAMES):
        if f.endswith(".jpg") and "_" in f:
            name, t = f[:-4].rsplit("_", 1)
            labelled[name].append(float(t))

    candidates = []
    for path in sorted(glob.glob(args.caches)):
        cache = twin.load_cache(path)
        video = os.path.splitext(cache["video"])[0]
        frames = cache["frames"]
        score, why = score_frames(frames)
        for k, s in score.items():
            t = frames[k]["t"]
            if any(abs(t - u) < 0.7 for u in labelled[video]):
                continue
            candidates.append((s, video, t, sorted(why[k])))
        print(f"{video}: {sum(1 for c in candidates if c[1] == video)} candidate frames", flush=True)

    picked = defaultdict(list)
    for s, video, t, reasons in sorted(candidates, reverse=True):
        if sum(len(v) for v in picked.values()) >= args.count:
            break
        if any(abs(t - u) < args.min_gap for u, _ in picked[video]):
            continue
        picked[video].append((t, reasons))

    tally = defaultdict(int)
    os.makedirs(FRAMES, exist_ok=True)
    for video, items in picked.items():
        cap = cv2.VideoCapture(os.path.join(ROOT, "raw", "videos", video + ".mp4"))
        for t, reasons in sorted(items):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if ok and img.shape[1] != 1080:  # window recordings: labels are in 1080x1920 pixels
                img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_LINEAR)
            if ok:
                cv2.imwrite(os.path.join(FRAMES, f"{video}_{t:06.1f}.jpg"), img)
            for r in reasons:
                tally[r] += 1
    total = sum(len(v) for v in picked.values())
    print(f"{total} frames -> {FRAMES}: " + ", ".join(f"{len(v)} from {k}" for k, v in sorted(picked.items())))
    print("why: " + ", ".join(f"{r} {n}" for r, n in sorted(tally.items(), key=lambda kv: -kv[1])))
    with open(os.path.join(ROOT, "raw", "labels", "mined_frames.json"), "w", encoding="utf-8") as f:
        json.dump({v: items for v, items in picked.items()}, f, indent=1)


if __name__ == "__main__":
    main()
