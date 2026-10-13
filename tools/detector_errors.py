"""Where the unit detector goes wrong, at the confidence the live bot uses.

    .venv\\Scripts\\python.exe tools\\detector_errors.py [--weights ...] [--conf 0.35] [--split val] [--save]

For the labelled images of a dataset split: per class how many objects were found, missed, found with
the wrong side (blue <-> red) or the wrong unit, and how many detections match nothing. With --save,
the images with mistakes are written to raw/detector/errors/ with the mistakes drawn.
"""
import argparse
import glob
import os
import sys
from collections import Counter

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.units import DEFAULT_WEIGHTS  # noqa: E402

DATASET = os.path.join(ROOT, "raw", "detector", "dataset")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_labels(path, w, h):
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            out.append((int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--split", default="val")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    stats = {n: Counter() for n in names.values()}
    swaps = Counter()
    out_dir = os.path.join(ROOT, "raw", "detector", "errors")
    if args.save:
        os.makedirs(out_dir, exist_ok=True)
    for img_path in sorted(glob.glob(os.path.join(DATASET, "images", args.split, "*.jpg"))):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gts = load_labels(os.path.join(DATASET, "labels", args.split,
                                       os.path.splitext(os.path.basename(img_path))[0] + ".txt"), w, h)
        res = model.predict(img, imgsz=args.imgsz, conf=args.conf, device=0, verbose=False)[0]
        preds = [(int(c), tuple(float(v) for v in b), float(s)) for b, c, s in
                 zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy())]
        # Greedy matching by IoU, any class: then judge the class of each pair.
        pairs = sorted(((iou(g[1], q[1]), gi, pi) for gi, g in enumerate(gts) for pi, q in enumerate(preds)),
                       reverse=True)
        used_g, used_p, mistakes = set(), set(), []
        for v, gi, pi in pairs:
            if v < 0.4 or gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            g, q = names[gts[gi][0]], names[preds[pi][0]]
            if g == q:
                stats[g]["found"] += 1
                continue
            gu, gt = g.rsplit("_", 1)
            qu, qt = q.rsplit("_", 1)
            kind = "wrong side" if gu == qu else "wrong unit"
            stats[g][kind] += 1
            swaps[(g, q)] += 1
            mistakes.append((gts[gi][1], f"{g} -> {q}", (0, 165, 255)))
        for gi, g in enumerate(gts):
            if gi not in used_g:
                stats[names[g[0]]]["missed"] += 1
                mistakes.append((g[1], f"missed {names[g[0]]}", (0, 0, 255)))
        for pi, q in enumerate(preds):
            if pi not in used_p:
                stats[names[q[0]]]["false"] += 1
                mistakes.append((q[1], f"false {names[q[0]]} {q[2]:.2f}", (255, 0, 255)))
        if args.save and mistakes:
            for box, text, color in mistakes:
                x0, y0, x1, y1 = map(int, box)
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
                cv2.putText(img, text, (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                cv2.putText(img, text, (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imwrite(os.path.join(out_dir, os.path.basename(img_path)), img)

    print(f"conf {args.conf}, split {args.split}")
    print("class               objects  found  missed  wrong side  wrong unit   false   recall  precision")
    tot = Counter()
    for n in names.values():
        s = stats[n]
        objects = s["found"] + s["missed"] + s["wrong side"] + s["wrong unit"]
        if objects == 0 and s["false"] == 0:
            continue
        tot.update(s)
        predicted_as_n = s["found"] + s["false"] + sum(v for (g, q), v in swaps.items() if q == n)
        rec = s["found"] / objects if objects else float("nan")
        prec = s["found"] / predicted_as_n if predicted_as_n else float("nan")
        print(f"{n:18s}  {objects:7d}  {s['found']:5d}  {s['missed']:6d}  {s['wrong side']:10d}  {s['wrong unit']:10d}"
              f"  {s['false']:6d}   {rec:6.0%}   {prec:6.0%}")
    objects = tot["found"] + tot["missed"] + tot["wrong side"] + tot["wrong unit"]
    print(f"{'all':18s}  {objects:7d}  {tot['found']:5d}  {tot['missed']:6d}  {tot['wrong side']:10d}  "
          f"{tot['wrong unit']:10d}  {tot['false']:6d}")
    print("\nmost common mix-ups:", ", ".join(f"{g}->{q} x{v}" for (g, q), v in swaps.most_common(8)))


if __name__ == "__main__":
    main()
