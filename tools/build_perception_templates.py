"""Builds the card templates for python/perception from a recorded battle.

    .venv\\Scripts\\python.exe tools\\build_perception_templates.py [--anchors data/perception/card_anchors.json]

Hand templates: the average edge features of the crops listed in the anchors file (video time +
hand slot for each card, colour and greyed-out). The colour/grey threshold is set between the two.
Next-card templates are labelled automatically: when an empty hand slot fills with card X, the
"next" icon just before showed X.
Digit templates (tower HP, timer, x2/x3 badge): glyphs of the numbers read by eye in
data/perception/number_anchors.json, averaged per digit.
Output: raw/perception/templates.npz (outside git: it is cut from game art).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.layout import Layout  # noqa: E402
from perception.numbers import REGIONS, glyphs  # noqa: E402
from perception.reader import DEFAULT_TEMPLATES, EMPTY, ScreenReader, edge_features, hand_art, saturation  # noqa: E402


def build_digits(anchors_path):
    """Average glyph per digit from the hand-read numbers."""
    doc = json.load(open(anchors_path, encoding="utf-8"))
    per_digit = {}
    for video_name, t, values in doc["anchors"]:
        frame = Video(os.path.join(ROOT, "raw", "videos", video_name)).frame(t)
        for region, text in values.items():
            x0, y0, x1, y1 = REGIONS[region]
            feats, _ = glyphs(frame[y0:y1, x0:x1])
            digits = text.replace(":", "")
            if len(feats) != len(digits):
                print(f"  skipped {video_name} {t}s {region}: {len(feats)} glyphs for {text!r}")
                continue
            for d, f in zip(digits, feats):
                per_digit.setdefault(d, []).append(f)
    labels = sorted(per_digit)
    templates = []
    for d in labels:
        m = np.mean(per_digit[d], axis=0)
        templates.append((m - m.mean()) / (m.std() + 1e-6))
    print("digit templates:", {d: len(per_digit[d]) for d in labels})
    return np.array(labels), np.array(templates, dtype=np.float32)


class Video:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open {path}")

    def frame(self, t):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = self.cap.read()
        if not ok:
            raise ValueError(f"no frame at {t}s")
        return f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors", default=os.path.join(ROOT, "data", "perception", "card_anchors.json"))
    p.add_argument("--number-anchors", default=os.path.join(ROOT, "data", "perception", "number_anchors.json"))
    p.add_argument("--out", default=DEFAULT_TEMPLATES)
    p.add_argument("--step", type=float, default=0.25, help="seconds between frames for next-card labelling")
    args = p.parse_args()

    anchors = json.load(open(args.anchors, encoding="utf-8"))
    video = Video(os.path.join(ROOT, anchors["video"]))

    # --- hand templates ---
    names, templates, sats = [], [], {True: [], False: []}
    for name, points in anchors["cards"].items():
        feats = []
        for t, slot in points:
            frame = video.frame(t)
            crop = Layout.for_frame(frame).hand_slot(frame, slot)
            feats.append(edge_features(hand_art(crop)))
            if name != EMPTY:
                s = saturation(hand_art(crop))
                sats[s > 60].append(s)
        m = np.mean(feats, axis=0)
        names.append(name)
        templates.append((m - m.mean()) / (m.std() + 1e-6))
    color_threshold = (np.min(sats[True]) + np.max(sats[False])) / 2 if sats[True] and sats[False] else 60.0
    digit_labels, digit_templates = build_digits(args.number_anchors)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, hand_names=np.array(names), hand_templates=np.array(templates, dtype=np.float32),
             next_names=np.array(["?"]), next_templates=np.zeros((1, templates[0].size), np.float32),
             color_threshold=np.float32(color_threshold), digit_labels=digit_labels, digit_templates=digit_templates)
    print(f"hand templates: {names}; colour threshold {color_threshold:.1f} "
          f"(colour min {min(sats[True]):.1f}, grey max {max(sats[False]):.1f})")

    # --- next-card templates, labelled from hand refills ---
    reader = ScreenReader(args.out)
    cap = video.cap
    cap.set(cv2.CAP_PROP_POS_MSEC, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, int(round(args.step * fps)))
    # When a card is played, the "next" icon at that moment is the card that will refill the slot.
    prev_hand, prev_next_feat, labelled, waiting = None, None, {}, {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % stride:
            continue
        try:
            read = reader.read_hand(frame)
        except ValueError:
            continue
        if any(h["unknown"] for h in read):
            continue  # a card is moving: skip the frame rather than mislabel
        hand = [h["card"] for h in read]
        next_feat = edge_features(Layout.for_frame(frame).next_card(frame))
        if prev_hand is not None:
            for slot in range(4):
                if prev_hand[slot] is not None and hand[slot] is None and prev_next_feat is not None:
                    waiting[slot] = prev_next_feat  # just played
                elif prev_hand[slot] is None and hand[slot] is not None and slot in waiting:
                    labelled.setdefault(hand[slot], []).append(waiting.pop(slot))
        prev_hand, prev_next_feat = hand, next_feat
    next_names = sorted(labelled)
    next_templates = []
    for n in next_names:
        m = np.mean(labelled[n], axis=0)
        next_templates.append((m - m.mean()) / (m.std() + 1e-6))
    np.savez(args.out, hand_names=np.array(names), hand_templates=np.array(templates, dtype=np.float32),
             next_names=np.array(next_names), next_templates=np.array(next_templates, dtype=np.float32),
             color_threshold=np.float32(color_threshold), digit_labels=digit_labels, digit_templates=digit_templates)
    print("next-card templates:", {n: len(v) for n, v in labelled.items()})
    print("saved", args.out)


if __name__ == "__main__":
    main()
