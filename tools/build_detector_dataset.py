"""Turns the unit labels (raw/labels/labels.json) into a YOLO dataset.

    .venv\\Scripts\\python.exe tools\\build_detector_dataset.py [--val-video video3] [--all] [--auto]

Images are the arena part of each frame (rows 120..1540 at 1080x1920). Classes are unit x side,
e.g. knight_blue / knight_red, so the detector also tells whose unit it is. Spell hits look the same
whoever cast them, so they have no side (fireball_hit_any): the bot knows its own spells from its hand. Frames from --val-video
go to validation, which measures how well it works on a battle it never saw; --all puts every frame
in training (for the final model) and keeps the same validation frames for reporting.
Output: raw/detector/dataset/{images,labels}/{train,val} + data.yaml
"""
import argparse
import json
import os
import shutil
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from label_units import ARENA, CLASSES  # noqa: E402

TEAMS = ("blue", "red")
TEAMLESS = ("fireball_hit", "arrows_hit")  # an explosion does not show who threw it
NAMES = [f"{c[0]}_{t}" for c in CLASSES for t in (("any",) if c[0] in TEAMLESS else TEAMS)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default=os.path.join(ROOT, "raw", "labels", "labels.json"))
    p.add_argument("--frames", default=os.path.join(ROOT, "raw", "labels", "frames"))
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "detector", "dataset"))
    p.add_argument("--val-video", default="video3")
    p.add_argument("--all", action="store_true", help="train on every frame (validation frames included)")
    p.add_argument("--auto", action="store_true",
                   help="also train on the automatic labels of two-sided recordings (tools/auto_label.py)")
    args = p.parse_args()

    doc = json.load(open(args.labels, encoding="utf-8"))
    sources = {name: args.frames for name in doc["frames"]}
    auto_path = os.path.join(os.path.dirname(args.labels), "auto_labels.json")
    if args.auto and os.path.exists(auto_path):
        auto = json.load(open(auto_path, encoding="utf-8"))
        for name, entry in auto["frames"].items():
            if name not in doc["frames"]:  # a person's labels win
                doc["frames"][name] = {"status": "labeled", "objects": entry["objects"]}
                sources[name] = os.path.join(os.path.dirname(args.frames), "auto_frames")
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split))
        os.makedirs(os.path.join(args.out, "labels", split))

    counts = {"train": 0, "val": 0}
    for name, entry in sorted(doc["frames"].items()):
        if entry.get("status") != "labeled":
            continue
        img = cv2.imread(os.path.join(sources[name], name))
        if img is None:
            continue
        crop = img[ARENA[0]:ARENA[1]]
        h, w = crop.shape[:2]
        lines = []
        for o in entry["objects"]:
            x0, y0, x1, y1 = o["box"]
            y0, y1 = y0 - ARENA[0], y1 - ARENA[0]
            x0, x1 = max(0.0, x0), min(float(w), x1)
            y0, y1 = max(0.0, y0), min(float(h), y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            cls = NAMES.index(f"{o['cls']}_{'any' if o['cls'] in TEAMLESS else o['team']}")
            lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        is_val = name.startswith(args.val_video + "_")
        splits = ["val", "train"] if (is_val and args.all) else (["val"] if is_val else ["train"])
        stem = os.path.splitext(name)[0]
        for split in splits:
            cv2.imwrite(os.path.join(args.out, "images", split, stem + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            counts[split] += 1

    yaml = os.path.join(args.out, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {args.out.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, n in enumerate(NAMES):
            f.write(f"  {i}: {n}\n")
    print(f"train {counts['train']} images, val {counts['val']} images -> {yaml}")


if __name__ == "__main__":
    main()
