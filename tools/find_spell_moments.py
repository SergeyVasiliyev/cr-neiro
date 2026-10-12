"""Finds the moments our own spells were cast in the recordings, from the hand reader.

    .venv\\Scripts\\python.exe tools\\find_spell_moments.py [--videos "raw/videos/*.mp4"]

A spell card leaving its hand slot = played. The spell appears ~1 s later (blueprint delay) and flies
from our king tower, so frames shortly after show it in flight and landing. Prints the events and
saves them to raw/labels/own_spells.json.
"""
import argparse
import glob
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.reader import ScreenReader  # noqa: E402

SPELLS = ("arrows", "fireball")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", default=os.path.join(ROOT, "raw", "videos", "*.mp4"))
    p.add_argument("--step", type=float, default=0.1)
    args = p.parse_args()

    reader = ScreenReader()
    events = []
    for path in sorted(glob.glob(args.videos)):
        name = os.path.basename(path)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        stride = max(1, int(round(args.step * fps)))
        last_card = [None] * 4  # last stable card seen in each slot
        index, found = 0, 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            index += 1
            if index % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = index / fps
            try:
                hand = reader.read_hand(frame)
            except ValueError:
                continue
            for slot, h in enumerate(hand):
                if h["unknown"]:
                    continue  # card moving (selected and dragged, or sliding in)
                if h["card"] is None:
                    if last_card[slot] in SPELLS:
                        events.append({"video": name, "time": round(t, 2), "spell": last_card[slot]})
                        found += 1
                    last_card[slot] = None
                else:
                    last_card[slot] = h["card"]
        print(f"{name}: {found} own spells")
    out = os.path.join(ROOT, "raw", "labels", "own_spells.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=1)
    print(f"{len(events)} events -> {out}")


if __name__ == "__main__":
    main()
