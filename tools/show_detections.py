"""Draws the unit detector's output on frames, next to the hand labels if there are any.

    .venv\\Scripts\\python.exe tools\\show_detections.py [--weights raw/detector/runs/units/weights/best.pt]
        [--frames raw/detector/dataset/images/val] [--count 6]

Writes raw/detector/preview.jpg: thin boxes = labels, thick boxes = detections (blue = ours, red = enemy).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=os.path.join(ROOT, "raw", "detector", "runs", "units", "weights", "best.pt"))
    p.add_argument("--frames", default=os.path.join(ROOT, "raw", "detector", "dataset", "images", "val"))
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "detector", "preview.jpg"))
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    files = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(args.count, len(files))).astype(int)]
    tiles = []
    for path in files:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        label_path = path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
        if os.path.exists(label_path):
            for line in open(label_path).read().split("\n"):
                if not line.strip():
                    continue
                c, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                color = (255, 200, 120) if int(c) % 2 == 0 else (120, 120, 255)
                cv2.rectangle(img, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), color, 1)
        res = model.predict(img, conf=args.conf, verbose=False)[0]
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            name = model.names[int(cls)]
            color = (255, 120, 0) if name.endswith("_blue") else (0, 0, 255)
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
            cv2.putText(img, f"{name.rsplit('_', 1)[0][:5]} {conf:.2f}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        tiles.append(cv2.resize(img, (540, 710)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
