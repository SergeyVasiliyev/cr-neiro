"""Re-reads only the towers' HP and the elixir in perception caches (after a fix to their reading), from their
videos - no detector, so it is quick.

    .venv\\Scripts\\python.exe tools\\reread_towers.py [video29 video29_red ...]   (default: every cache)
"""
import glob
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.layout import Layout  # noqa: E402
from perception.reader import ScreenReader  # noqa: E402

CACHE = os.path.join(ROOT, "raw", "perception", "cache")


def main():
    names = sys.argv[1:] or [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(CACHE, "video*.json")))
                             if "_v1_" not in p]
    reader = ScreenReader()
    for name in names:
        path = os.path.join(CACHE, name + ".json")
        video = os.path.join(ROOT, "raw", "videos", name + ".mp4")
        if not os.path.exists(path) or not os.path.exists(video):
            print(f"{name}: no cache or video, skipped")
            continue
        cache = json.load(open(path, encoding="utf-8"))
        frames = cache["frames"]
        reader.numbers.bar_dy.clear()  # (where each tower's bar was last seen: another match, other skins)
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        k, n_frame, before, after = 0, 0, 0, 0
        while k < len(frames):
            ok, img = cap.read()
            if not ok:
                break
            t = n_frame / fps
            n_frame += 1
            if t + 0.5 / fps < frames[k]["t"]:
                continue
            img = cv2.resize(img, (1080, 1920))
            towers = {tw: list(v) for tw, v in reader.read_towers(img).items()}
            before += sum(1 for v in frames[k].get("towers", {}).values() if v[1])
            after += sum(1 for v in towers.values() if v[1])
            frames[k]["towers"] = towers
            frames[k]["elixir"] = reader._elixir(img, Layout.for_frame(img))
            k += 1
        json.dump(cache, open(path, "w", encoding="utf-8"))
        print(f"{name}: {k}/{len(frames)} frames, exact tower readings {before} -> {after}", flush=True)


if __name__ == "__main__":
    main()
