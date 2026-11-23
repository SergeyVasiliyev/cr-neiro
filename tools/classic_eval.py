"""The Classic Decks unit detector on real frames of a battle left out of its training (tools/classic_synth.py
build --exclude), against that battle's seeds (tools/classic_labels.py): units the simulator and a track named.

    .venv\\Scripts\\python.exe tools\\classic_eval.py --weights raw/detector/runs/classic0/weights/best.pt c013
    .venv\\Scripts\\python.exe tools\\classic_eval.py --weights ... c013 --frames 30 60 90   pictures of whole frames
    .venv\\Scripts\\python.exe tools\\classic_eval.py --weights ... c013 --round   also the round labels (crowds)

Seeds are lone units only, so this measures how often a lone unit is found (a box over it, IoU >= 0.5) and named
right (its look and its colour); boxes on other things are not counted here - the pictures show them.
"""
import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classic_labels import ARENA_ROWS, SEEDS, video_path  # noqa: E402
from classic_replay import OUT  # noqa: E402

REF_W = 1080


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Detector:
    def __init__(self, weights, conf=0.3, imgsz=960):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.weights = weights
        self.names = self.model.names
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, frame):
        """[(class name, conf, (x0, y0, x1, y1) in the frame's own pixels)]"""
        s = REF_W / frame.shape[1]
        big = cv2.resize(frame, (REF_W, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
        crop = big[ARENA_ROWS[0]:ARENA_ROWS[1]]
        res = self.model.predict(crop, imgsz=self.imgsz, conf=self.conf, verbose=False, device=0)[0]
        out = []
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            x0, y0, x1, y1 = b
            out.append((self.names[int(k)], float(c), (x0 / s, (y0 + ARENA_ROWS[0]) / s, x1 / s, (y1 + ARENA_ROWS[0]) / s)))
        return out


def frames_of(name, view, wanted):
    cap = cv2.VideoCapture(video_path(name, view))
    k, out = -1, {}
    wanted = set(wanted)
    last = max(wanted) if wanted else -1
    while k < last:
        ok, f = cap.read()
        if not ok:
            break
        k += 1
        if k in wanted:
            out[k] = f
    return out


def evaluate(det, name):
    recs = [json.loads(line) for line in open(os.path.join(SEEDS, f"{name}.jsonl"), encoding="utf-8")]
    stats = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    confusion = collections.Counter()
    for view in ("blue", "red"):
        rs = [r for r in recs if r["view"] == view]
        frames = frames_of(name, view, [r["frame"] for r in rs])
        cache = {}
        for r in rs:
            if r["frame"] not in frames:
                continue
            if r["frame"] not in cache:
                cache[r["frame"]] = det(frames[r["frame"]])
            dets = cache[r["frame"]]
            truth = f"{r['unit']}_{r['seen']}"
            best = max(dets, key=lambda d: iou(d[2], r["box"]), default=None)
            stats["seeds"] += 1
            per[truth]["seeds"] += 1
            if best is None or iou(best[2], r["box"]) < 0.5:
                stats["missed"] += 1
                per[truth]["missed"] += 1
                continue
            stats["found"] += 1
            per[truth]["found"] += 1
            if best[0] == truth:
                stats["right"] += 1
                per[truth]["right"] += 1
            else:
                confusion[(truth, best[0])] += 1
                if best[0].rsplit("_", 1)[0] == r["unit"]:
                    stats["colour_wrong"] += 1
    n = max(1, stats["seeds"])
    print(f"{name}: {stats['seeds']} lone units; found {stats['found'] / n:.1%}, named right {stats['right'] / n:.1%} "
          f"(of found {stats['right'] / max(1, stats['found']):.1%}; right kind, wrong colour {stats['colour_wrong']})")
    for cls, c in sorted(per.items(), key=lambda kv: -kv[1]["seeds"]):
        print(f"  {cls:26s} {c['seeds']:4d}  found {c['found'] / c['seeds']:.0%}  right {c['right'] / c['seeds']:.0%}")
    if confusion:
        print("  most common wrong names: " + ", ".join(f"{a} as {b} x{n}" for (a, b), n in confusion.most_common(12)))


def evaluate_round(det, name):
    """Against the battle's round labels (tools/classic_round.py: real frames the twin checked, crowds too): the
    labels found with the right name (IoU >= 0.5), and the sure boxes (conf >= 0.5) that are a label."""
    from classic_round import ROUND
    recs = [json.loads(line) for line in open(os.path.join(ROUND, f"{name}.jsonl"), encoding="utf-8")]
    recs = [r for r in recs if not r["problems"] and not r.get("paint")]
    stats = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    for view in ("blue", "red"):
        rs = [r for r in recs if r["view"] == view]
        frames = frames_of(name, view, [r["frame"] for r in rs])
        for r in rs:
            if r["frame"] not in frames:
                continue
            dets = det(frames[r["frame"]])
            used = set()
            for o in r["objects"]:
                stats["labels"] += 1
                per[o["cls"]]["labels"] += 1
                best = max(((iou(d[2], o["box"]), j) for j, d in enumerate(dets) if j not in used), default=(0, -1))
                if best[0] >= 0.5:
                    used.add(best[1])
                    stats["found"] += 1
                    if dets[best[1]][0] == o["cls"]:
                        stats["right"] += 1
                        per[o["cls"]]["right"] += 1
            sure = [j for j, d in enumerate(dets) if d[1] >= 0.5]
            stats["sure"] += len(sure)
            stats["sure_on_label"] += sum(1 for j in sure if j in used)
    n = max(1, stats["labels"])
    print(f"{name} round labels: {len(recs)} frames, {stats['labels']} labels; found {stats['found'] / n:.1%}, "
          f"named right {stats['right'] / n:.1%}; sure boxes on a label {stats['sure_on_label'] / max(1, stats['sure']):.1%}")
    for cls, c in sorted(per.items(), key=lambda kv: -kv[1]["labels"])[:16]:
        print(f"  {cls:26s} {c['labels']:4d}  right {c['right'] / c['labels']:.0%}")


def pictures(det, name, times, fps=15.0):
    rows = []
    for t in times:
        k = int(round(t * fps))
        pair = []
        for view in ("blue", "red"):
            f = frames_of(name, view, [k]).get(k)
            if f is None:
                continue
            vis = f.copy()
            for cls, conf, (x0, y0, x1, y1) in det(f):
                col = (255, 120, 0) if cls.endswith("_blue") else (0, 0, 255)
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
                cv2.putText(vis, f"{cls.rsplit('_', 1)[0][:12]} {conf:.2f}", (int(x0), int(y0) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            pair.append(vis)
        if pair:
            rows.append(np.hstack(pair))
    out = os.path.join(OUT, f"{name}_detector.jpg")
    cv2.imwrite(out, np.vstack(rows))
    print("pictures:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("battles", nargs="+")
    p.add_argument("--weights", required=True)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--frames", type=float, nargs="*", help="video seconds to draw whole frames at")
    p.add_argument("--round", action="store_true", help="also score against the battle's round labels (crowds)")
    args = p.parse_args()
    det = Detector(args.weights, conf=args.conf)
    for b in args.battles:
        if args.frames:
            pictures(det, b, args.frames)
        else:
            evaluate(det, b)
            if args.round:
                evaluate_round(det, b)


if __name__ == "__main__":
    main()
