"""Runs the unit detector + tracker on a recording and draws the result.

    .venv\\Scripts\\python.exe tools\\track_overlay.py --video raw/videos/video5.mp4 --start 60 --end 90 [--fps 10]

Output raw/perception/tracks.mp4 (half size): each tracked unit with its id and a short trail;
units not seen in this frame but still remembered are drawn dashed with "?" at their predicted spot.
"""
import argparse
import math
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.layout import Layout  # noqa: E402
from perception.tracker import Tracker  # noqa: E402
from perception.units import DEFAULT_WEIGHTS, UnitDetector  # noqa: E402

COLORS = {"blue": (255, 140, 40), "red": (40, 40, 255)}
SHORT = {"knight": "kn", "archer": "ar", "minion": "mn", "giant": "GI", "mini_pekka": "pk", "musketeer": "mu",
         "fireball": "fb", "arrows": "aw", "fireball_hit": "FB", "arrows_hit": "AW"}


def draw(frame, detections, tracker, t):
    L = Layout.for_frame(frame)
    out = frame.copy()
    for d in detections:
        x0, y0, x1, y1 = map(int, d.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), COLORS[d.team], 1)
    for tr in tracker.tracks:
        if not tr.confirmed:
            continue
        px, py = tr.predicted(t)
        sx, sy = (int(v) for v in L.tile_to_px(px, py))
        color = COLORS[tr.team]
        seen = tr.last_seen == t
        pts = [tuple(int(v) for v in L.tile_to_px(x, y)) for _, x, y in tr.history[-15:]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, color, 2)
        if seen:
            cv2.circle(out, (sx, sy), 9, color, -1)
        else:  # remembered, not visible: dashed ring
            for k in range(0, 360, 40):
                a0, a1 = math.radians(k), math.radians(k + 20)
                cv2.line(out, (int(sx + 16 * math.cos(a0)), int(sy + 16 * math.sin(a0))),
                         (int(sx + 16 * math.cos(a1)), int(sy + 16 * math.sin(a1))), color, 3)
        label = f"{SHORT.get(tr.cls, tr.cls)}{tr.id}{'' if seen else '?'}"
        cv2.putText(out, label, (sx + 12, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
        cv2.putText(out, label, (sx + 12, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=os.path.join(ROOT, "raw", "videos", "video5.mp4"))
    p.add_argument("--start", type=float, default=60.0)
    p.add_argument("--end", type=float, default=90.0)
    p.add_argument("--fps", type=float, default=10.0, help="frames per second fed to the detector")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="0", help="0 = GPU, cpu = processor")
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "perception", "tracks.mp4"))
    args = p.parse_args()

    detector = UnitDetector(args.weights, device=0 if args.device == "0" else args.device)
    tracker = Tracker()
    cap = cv2.VideoCapture(args.video)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, int(round(vfps / args.fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    writer = None
    i = 0
    while True:
        ok, frame = cap.read()
        t = args.start + i / vfps
        if not ok or t > args.end:
            break
        i += 1
        if (i - 1) % stride:
            continue
        detections = detector.detect(frame)
        tracker.update(t, detections)
        img = draw(frame, detections, tracker, t)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        if writer is None:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer:
        writer.release()
    print(f"wrote {args.out}; {tracker.next_id - 1} tracks created")
    # How the memory was used: per confirmed track, its life and the longest stretch it went unseen.
    print("track  unit         side  from    to      seen  longest gap")
    for tr in sorted(tracker.finished + tracker.tracks, key=lambda tr: tr.id):
        if not tr.confirmed:
            continue
        times = [h[0] for h in tr.history]
        gap = max((b - a for a, b in zip(times, times[1:])), default=0.0)
        print(f"{tr.id:5d}  {tr.cls:11s}  {tr.team:4s}  {times[0]:6.1f}  {times[-1]:6.1f}  {len(times):4d}  {gap:5.1f}s")


if __name__ == "__main__":
    main()
