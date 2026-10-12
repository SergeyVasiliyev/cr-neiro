"""Picks varied battle frames from the recordings for labelling units.

    .venv\\Scripts\\python.exe tools\\extract_label_frames.py [--per-video 45]

Only frames inside a battle (the timer is readable) are used. Each video's empty arena is estimated as
the median frame; frames are scored by how much of the arena differs from it (units, spells), and
the most active frames are taken while keeping them spread out in time.
Output: raw/labels/frames/<video>_<seconds>.jpg (full resolution).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.reader import ScreenReader  # noqa: E402

ARENA = (120, 1540)  # rows of the arena at 1080x1920


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", default=os.path.join(ROOT, "raw", "videos", "*.mp4"))
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "labels", "frames"))
    p.add_argument("--per-video", type=int, default=45)
    p.add_argument("--step", type=float, default=0.5, help="seconds between candidate frames")
    p.add_argument("--min-gap", type=float, default=2.5, help="minimum seconds between picked frames")
    args = p.parse_args()

    reader = ScreenReader()
    os.makedirs(args.out, exist_ok=True)
    for path in sorted(glob.glob(args.videos)):
        name = os.path.splitext(os.path.basename(path))[0]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        times, smalls = [], []
        for i in range(0, n, max(1, int(round(args.step * fps)))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                break
            if reader.numbers.read_timer(frame)[0] is None:
                continue  # menus, loading, result screen
            times.append(i / fps)
            smalls.append(cv2.resize(frame[ARENA[0]:ARENA[1]], (135, 178), interpolation=cv2.INTER_AREA))
        if not smalls:
            print(f"{name}: no battle frames found")
            continue
        stack = np.stack(smalls).astype(np.int16)
        background = np.median(stack, axis=0)
        activity = (np.abs(stack - background).max(axis=3) > 40).mean(axis=(1, 2))
        picked = []
        for k in np.argsort(-activity):
            if len(picked) >= args.per_video:
                break
            if all(abs(times[k] - times[j]) >= args.min_gap for j in picked):
                picked.append(k)
        for k in sorted(picked):
            cap.set(cv2.CAP_PROP_POS_MSEC, times[k] * 1000)
            ok, frame = cap.read()
            cv2.imwrite(os.path.join(args.out, f"{name}_{times[k]:06.1f}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"{name}: {len(picked)} frames (activity {activity[picked].min():.3f}..{activity[picked].max():.3f})")


if __name__ == "__main__":
    main()
