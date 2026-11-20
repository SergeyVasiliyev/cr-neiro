"""Deploy clocks (python/perception/clock.py) for cached recordings, so the tools that replay the live twin over the
caches (twin_plays_check.py) see them as the live bot does: raw/perception/cache/<name>_clocks.json, one list per
cached frame, [[x, y, score, side], ...], looked for around the red troops the detector saw in that frame (the live
bot looks around the new ones only: the same clocks, there).

    .venv\\Scripts\\python.exe tools\\clock_cache.py [names ...] [--workers 6] [--again]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402
from perception.clock import Clock, find_clocks  # noqa: E402

CACHE = os.path.join(ROOT, "raw", "perception", "cache")
VIDEOS = os.path.join(ROOT, "raw", "videos")


def clocks_path(name):
    return os.path.join(CACHE, f"{name}_clocks.json")


def load_clocks(name):
    """[[Clock] per cached frame] of a recording, or None if not made yet."""
    path = clocks_path(name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)["frames"]
    return [[Clock((c[0], c[1]), c[2], c[3]) for c in row] for row in rows]


def make(name):
    cache = twin.load_cache(os.path.join(CACHE, f"{name}.json"))
    frames = cache["frames"]
    cap = cv2.VideoCapture(os.path.join(VIDEOS, cache["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pos, img, out, found = -1, None, [], 0
    first = max(0, int(round(frames[0]["t"] * fps)) - 1) if frames else 0
    if first > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        pos = first - 1
    for fr in frames:
        k = int(round(fr["t"] * fps))
        near = [d.tile for d in twin.detections(fr) if d.team == "red" and d.cls in twin.CARD_OF_UNIT]
        while pos < k:
            if not cap.grab():
                break
            pos += 1
            img = None
        if not near or pos != k:
            out.append([])
            continue
        if img is None:
            ok, img = cap.retrieve()
            if not ok:
                out.append([])
                continue
            if img.shape[1] != 1080:
                img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        clocks = find_clocks(img, near=near)
        found += len(clocks)
        out.append([[round(c.tile[0], 2), round(c.tile[1], 2), round(c.score, 3), c.side] for c in clocks])
    with open(clocks_path(name), "w", encoding="utf-8") as f:
        json.dump({"video": cache["video"], "frames": out}, f)
    return name, len(frames), found


def main():
    from push_defence_check import two_sided_battles

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="cached recordings (default: the bot's side of every two-sided battle)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--again", action="store_true", help="make them again even if there")
    args = p.parse_args()
    names = [n for n in (args.names or two_sided_battles()) if args.again or not os.path.exists(clocks_path(n))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, n, found in pool.map(make, names):
            print(f"{name}: {n} frames, {found} clock sightings ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
