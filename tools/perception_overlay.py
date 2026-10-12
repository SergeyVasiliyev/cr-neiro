"""Draws what the screen reader sees on top of a recording, to check it by eye.

    .venv\\Scripts\\python.exe tools\\perception_overlay.py [--video raw/videos/video1.mp4] [--start 20 --end 80]

Writes raw/perception/overlay.mp4 (half size): tile grid on the arena, each hand slot's card with its
match score (green = affordable, grey = not enough elixir, red = unsure), the next card and elixir.
"""
import argparse
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.layout import Layout  # noqa: E402
from perception.reader import ScreenReader  # noqa: E402


def draw(frame, reader):
    L = Layout.for_frame(frame)
    out = frame.copy()
    for tx in range(19):
        cv2.line(out, tuple(map(int, L.tile_to_px(tx, 0))), tuple(map(int, L.tile_to_px(tx, 32))), (255, 255, 255), 1)
    for ty in range(33):
        cv2.line(out, tuple(map(int, L.tile_to_px(0, ty))), tuple(map(int, L.tile_to_px(18, ty))), (255, 255, 255), 1)
    out = cv2.addWeighted(out, 0.35, frame, 0.65, 0)
    res = reader.read(frame)
    for slot, h in enumerate(res["hand"]):
        xa, xb = Layout.HAND_SLOTS_X[slot]
        if h["unknown"]:
            text, color = "?", (0, 0, 255)
        elif h["card"] is None:
            text, color = "empty", (200, 200, 200)
        else:
            text, color = h["card"], (0, 220, 0) if h["affordable"] else (160, 160, 160)
        y = L.s(Layout.HAND_Y[0] - 12)
        cv2.putText(out, text, (L.s(xa), y), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * L.scale, color, 3)
        cv2.putText(out, f"{h['score']:.2f}", (L.s(xa), y - L.s(32)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * L.scale, color, 2)
    x0, y0, _, _ = Layout.NEXT_CARD
    cv2.putText(out, f"next: {res['next']['card']}", (L.s(x0), L.s(y0 - 12)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * L.scale, (0, 255, 255), 2)
    cv2.putText(out, f"elixir {res['elixir']:.2f}", (L.s(90), L.s(1530)), cv2.FONT_HERSHEY_SIMPLEX,
                1.3 * L.scale, (255, 80, 255), 3)
    if "timer" in res:
        secs, ot = res["timer"]["seconds"], res["timer"]["overtime"]
        clock = "--" if secs is None else f"{secs // 60}:{secs % 60:02d}"
        cv2.putText(out, f"{'OT ' if ot else ''}{clock} x{res['elixir_multiplier']}", (L.s(640), L.s(140)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * L.scale, (0, 255, 255), 3)
        # HP next to each tower: green = number read and confirmed by the bar, orange = bar estimate.
        spots = {"red_king": (700, 60), "red_left": (60, 320), "red_right": (640, 320),
                 "blue_king": (700, 1440), "blue_left": (60, 1280), "blue_right": (640, 1280)}
        for name, (hp, exact) in res["towers"].items():
            if hp is None:
                continue
            x, y = spots[name]
            cv2.putText(out, f"{hp}{'' if exact else '~'}", (L.s(x), L.s(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1 * L.scale, (0, 230, 0) if exact else (0, 160, 255), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=os.path.join(ROOT, "raw", "videos", "video1.mp4"))
    p.add_argument("--start", type=float, default=20.0)
    p.add_argument("--end", type=float, default=80.0)
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "perception", "overlay.mp4"))
    args = p.parse_args()

    reader = ScreenReader()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or args.start + n / fps > args.end:
            break
        n += 1
        img = draw(frame, reader)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        if writer is None:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer:
        writer.release()
    print(f"wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()
