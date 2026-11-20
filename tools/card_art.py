"""Cuts a picture of each card out of a recorded battle for the sim game window (python/play_sim.py).

It uses the hand crops at the moments listed in data/perception/card_anchors.json and keeps the most colourful
one per card: an affordable card, not the greyed one. Output: raw/perception/card_art/<card>.png. The pictures
stay outside git, as they are cut from game art.

    .venv\\Scripts\\python.exe tools\\card_art.py
"""
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.layout import Layout  # noqa: E402
from perception.reader import saturation  # noqa: E402

OUT = os.path.join(ROOT, "raw", "perception", "card_art")


def main():
    with open(os.path.join(ROOT, "data", "perception", "card_anchors.json"), encoding="utf-8") as f:
        anchors = json.load(f)
    cap = cv2.VideoCapture(os.path.join(ROOT, anchors["video"]))
    if not cap.isOpened():
        sys.exit(f"cannot open {anchors['video']}")
    os.makedirs(OUT, exist_ok=True)
    for card, spots in anchors["cards"].items():
        best = None
        for t, slot in spots:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            crop = Layout.for_frame(frame).hand_slot(frame, slot)
            sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            score = (round(saturation(crop) / 5.0), sharp)  # colour first (affordable), then the sharpest
            if best is None or score > best[0]:
                best = (score, crop, t)
        if best is None:
            print(f"{card}: no frame read")
            continue
        cv2.imwrite(os.path.join(OUT, f"{card}.png"), best[1])
        print(f"{card}: {best[2]:.1f} s, saturation ~{best[0][0] * 5}, {best[1].shape[1]}x{best[1].shape[0]}")


if __name__ == "__main__":
    main()
