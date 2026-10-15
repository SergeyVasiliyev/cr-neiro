"""Reads a whole recording once and saves what was seen, so later steps (twin replay, measurements)
run in seconds without the detector.

    .venv\\Scripts\\python.exe tools\\perceive_video.py --video raw/videos/video1.mp4 [--fps 10] [--start 0 --end 60]

Output raw/perception/cache/<video>.json:
    {"video", "fps", "weights", "frames": [{"t", "hand": [card | null | "?"] x4, "next", "elixir",
     "timer": [seconds, overtime], "mult", "towers": {name: [hp, exact]},
     "det": [[cls, team, conf, x0, y0, x1, y1, tile_x, tile_y], ...],
     "bars": [[team, x0, y0, x1, y1, length, filled], ...]}]}   (HP bars over damaged troops)
"""
import argparse
import json
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.reader import ScreenReader  # noqa: E402
from perception.unit_hp import UnitHpReader  # noqa: E402
from perception.units import DEFAULT_WEIGHTS, UnitDetector  # noqa: E402


def cache_path(video):
    name = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(ROOT, "raw", "perception", "cache", f"{name}.json")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=0.35, help="lowest detection confidence kept (the twin "
                                                            "uses >= 0.35 anyway; lower helps finding frames to label)")
    p.add_argument("--device", default="0", help="0 = GPU, cpu = processor")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    reader = ScreenReader()
    hp_reader = UnitHpReader()
    detector = UnitDetector(args.weights, conf=args.conf, device=0 if args.device == "0" else args.device)
    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(vfps / args.fps)))
    first = int(round(args.start * vfps))
    last = total if args.end is None else min(total, int(round(args.end * vfps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    started = time.time()
    for index in range(first, last):
        if not cap.grab():
            break
        if (index - first) % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        if frame.shape[1] != 1080:  # a window recording (tools/record_windows.py): bring it to 1080x1920
            frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        seen = reader.read(frame)
        dets = detector.detect(frame)
        frames.append({
            "t": round(index / vfps, 3),
            "hand": [h["card"] if not h["unknown"] else "?" for h in seen["hand"]],
            "next": seen["next"]["card"],
            "elixir": round(seen["elixir"], 2),
            "timer": [seen["timer"]["seconds"], seen["timer"]["overtime"]],
            "mult": seen["elixir_multiplier"],
            "towers": {k: [v[0], v[1]] for k, v in seen["towers"].items()},
            "det": [[d.cls, d.team, round(d.conf, 3), *(round(v, 1) for v in d.box),
                     round(d.tile[0], 2), round(d.tile[1], 2)] for d in dets],
            "bars": [[b.team, *b.badge, b.length, b.filled] for b in hp_reader.bars(frame, dets)],
        })
        if len(frames) % 200 == 0:
            rate = len(frames) / (time.time() - started)
            print(f"  {index / vfps:6.1f}s  {rate:.1f} frames/s", flush=True)
    out = args.out or cache_path(args.video)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"video": os.path.basename(args.video), "fps": args.fps, "weights": args.weights,
                   "frames": frames}, f, separators=(",", ":"),
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))  # numpy scalars
    print(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
