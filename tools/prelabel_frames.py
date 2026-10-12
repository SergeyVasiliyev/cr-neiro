"""Pre-labels new frames with the current unit detector, for correcting in scripts\\label.bat.

    .venv\\Scripts\\python.exe tools\\prelabel_frames.py [--weights raw/detector/runs/units/weights/best.pt] [--conf 0.3]

Only frames that have no entry in raw/labels/labels.json yet are touched. Their detections are stored
with "auto": true and status "prelabeled"; the labelling tool shows them dashed until the frame is
reviewed (moving on to the next frame accepts the corrected set).
"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from label_units import ARENA, FRAMES, LABELS, load_labels, save_labels  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=os.path.join(ROOT, "raw", "detector", "runs", "units", "weights", "best.pt"))
    p.add_argument("--conf", type=float, default=0.3)
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    doc = load_labels()
    new = [f for f in sorted(os.listdir(FRAMES)) if f.lower().endswith(".jpg") and f not in doc["frames"]]
    total = 0
    for name in new:
        img = cv2.imread(os.path.join(FRAMES, name))
        res = model.predict(img[ARENA[0]:ARENA[1]], imgsz=960, conf=args.conf, verbose=False)[0]
        objects = []
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
            unit, team = model.names[int(cls)].rsplit("_", 1)
            x0, y0, x1, y1 = (float(v) for v in box)
            objects.append({"cls": unit, "team": team, "box": [round(x0, 1), round(y0 + ARENA[0], 1),
                                                                round(x1, 1), round(y1 + ARENA[0], 1)],
                            "auto": True, "conf": round(float(conf), 2)})
        doc["frames"][name] = {"status": "prelabeled", "objects": objects}
        total += len(objects)
    save_labels(doc)
    print(f"pre-labelled {len(new)} frames with {total} detections -> {LABELS}")


if __name__ == "__main__":
    main()
