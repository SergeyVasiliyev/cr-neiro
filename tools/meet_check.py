"""Head-on meetings in test recordings: two ground troops of opposite sides that walk into each other and fight, in
the game vs the same test replayed in the simulator (tools/test_replay.py): where they meet, how far apart their
centres are while they fight, and whether the pair drifts (one pushing the other back).

    .venv\\Scripts\\python.exe tools\\meet_check.py --match video36 video37

A meeting starts when the two first come within 2.5 tiles (centre to centre); the fight is the time after it while
both live. Pictures: raw/test_replay/meet_<match>_<n>.png, zoomed on the meeting - game thick, simulator thin, a dot
every 0.5 s from 1.5 s before the meeting to 4 s after it (white numbers: seconds from the meeting).
"""
import argparse
import math
import os
import statistics
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402
from crsim_env import load_data  # noqa: E402
from test_replay import BLUE, COLORS, RED, UNIT, at, own_plays, pair_up, replay, segments, tower_states  # noqa: E402

GROUND = {"knight", "giant", "mini_pekka", "musketeer", "archer"}
MEET = 2.5


def first_meeting(a, b):
    """The first moment two ways come within MEET tiles."""
    for t, x, y in a:
        p = at(b, t)
        if p and math.dist((x, y), p) < MEET:
            return t
    return None


def fight_stats(a, b, t0, span=4.0):
    """Centre distance (median) and the pair's midpoint drift, from t0 to t0 + span while both are there."""
    gaps, mids = [], []
    for t, x, y in a:
        if t < t0 or t > t0 + span:
            continue
        p = at(b, t)
        if p:
            gaps.append(math.dist((x, y), p))
            mids.append((t, (x + p[0]) / 2, (y + p[1]) / 2))
    if len(mids) < 3:
        return None
    return statistics.median(gaps), mids


def draw(path, t0, items, center):
    S = 60  # pixels per tile
    half = 4.0
    img = np.full((int(2 * half * S), int(2 * half * S), 3), (60, 110, 60), np.uint8)
    cx, cy = center

    def px(x, y):
        return int((x - cx + half) * S), int((cy - y + half) * S)
    for gx in range(int(cx - half), int(cx + half) + 2):
        cv2.line(img, px(gx, cy - half), px(gx, cy + half), (70, 125, 70), 1)
    for gy in range(int(cy - half), int(cy + half) + 2):
        cv2.line(img, px(cx - half, gy), px(cx + half, gy), (70, 125, 70), 1)
    if 14.5 < cy + half and cy - half < 17.5:  # the river and bridges
        cv2.rectangle(img, px(cx - half, 17), px(cx + half, 15), (160, 110, 40), -1)
        for bx in (3.5, 14.5):
            cv2.rectangle(img, px(bx - 1, 17), px(bx + 1, 15), (90, 140, 170), -1)
    for label, unit, gw, sw in items:
        color = (255, 140, 40) if label.startswith("B") else (60, 60, 255)  # blue / red side (BGR)
        for way, thick in ((gw, 3), (sw, 1)):
            pts = [(t, x, y) for t, x, y in way if t0 - 1.5 <= t <= t0 + 4.0]
            for (_, x0, y0), (_, x1, y1) in zip(pts, pts[1:]):
                cv2.line(img, px(x0, y0), px(x1, y1), color if thick == 3 else (255, 255, 255), thick, cv2.LINE_AA)
            for k in range(-3, 9):
                p = at(way, t0 + k * 0.5)
                if p:
                    cv2.circle(img, px(*p), 6 if thick == 3 else 7, color, -1 if thick == 3 else 2, cv2.LINE_AA)
                    if thick == 3 and k % 2 == 0:
                        cv2.putText(img, f"{k * 0.5:+.0f}", (px(*p)[0] + 7, px(*p)[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if gw:
            p = at(gw, t0) or gw[0][1:]
            cv2.putText(img, label, (px(*p)[0] + 8, px(*p)[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--match", nargs="+", default=["video36", "video37"])
    args = p.parse_args()
    data = load_data()
    cache = os.path.join(ROOT, "raw", "perception", "cache")
    out_dir = os.path.join(ROOT, "raw", "test_replay")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for match in args.match:
        blue_frames = twin.load_cache(os.path.join(cache, f"{match}.json"))["frames"]
        red_frames = twin.load_cache(os.path.join(cache, f"{match}_red.json"))["frames"]
        blue, start = own_plays(blue_frames, flip=False)
        red, _ = own_plays(red_frames, flip=True)
        plays = [dict(q, team=BLUE) for q in blue] + [dict(q, team=RED) for q in red]
        n_meet = 0
        for t0, t1, group in segments(plays):
            frame = min(blue_frames, key=lambda f: abs(f["t"] - start - t0))
            sims = replay(data, t0, t1, group, tower_states(frame))
            units = []  # (side, unit, label, game way, sim way)
            for i, q in enumerate(group):
                unit = UNIT.get(q["card"], q["card"])
                if unit not in GROUND:
                    continue
                for gw, sw in pair_up(q["ways"], sims.get(i, [])):
                    side = "B" if q["team"] == BLUE else "R"
                    units.append((side, unit, f"{side} {unit}", gw, sw))
            seen_pairs = set()
            for a in units:
                for b in units:
                    if a[0] != "B" or b[0] != "R" or (id(a[3]), id(b[3])) in seen_pairs:
                        continue
                    seen_pairs.add((id(a[3]), id(b[3])))
                    tg = first_meeting(a[3], b[3])
                    if tg is None or not (t0 <= tg <= t1):
                        continue
                    ts = first_meeting(a[4], b[4])
                    g = fight_stats(a[3], b[3], tg)
                    s = fight_stats(a[4], b[4], ts) if ts is not None else None
                    if g is None:
                        continue
                    ga, gb = at(a[3], tg), at(b[3], tg)
                    line = (f"{match} {tg:6.1f}s  {a[2]:13s} vs {b[2]:13s} game: meet at ({(ga[0] + gb[0]) / 2:4.1f},"
                            f"{(ga[1] + gb[1]) / 2:4.1f}), centres {g[0]:.2f} apart, pair drifts "
                            f"{(g[1][-1][2] - g[1][0][2]):+.2f} (y) in {g[1][-1][0] - g[1][0][0]:.1f}s")
                    if s is not None:
                        sa, sb = at(a[4], ts), at(b[4], ts)
                        line += (f" | sim: meet {ts - tg:+.1f}s at ({(sa[0] + sb[0]) / 2:4.1f},{(sa[1] + sb[1]) / 2:4.1f}), "
                                 f"centres {s[0]:.2f} apart, drifts {(s[1][-1][2] - s[1][0][2]):+.2f}")
                    else:
                        line += " | sim: they never meet"
                    rows.append((a[1], b[1], g, s))
                    print(line)
                    for k in (-2, 0, 2, 4, 6):
                        tk = tg + k * 0.5
                        cells = []
                        for way in (a[3], a[4], b[3], b[4]):
                            q = at(way, tk if way in (a[3], b[3]) else tk + ((ts or tg) - tg) * 0)
                            cells.append(f"({q[0]:4.1f},{q[1]:4.1f})" if q else "     -     ")
                        print(f"      {k * 0.5:+.0f}s  {a[2]} game {cells[0]} sim {cells[1]}   "
                              f"{b[2]} game {cells[2]} sim {cells[3]}")
                    center = ((ga[0] + gb[0]) / 2, (ga[1] + gb[1]) / 2)
                    draw(os.path.join(out_dir, f"meet_{match}_{n_meet:02d}.png"), tg,
                         [(a[2], a[1], a[3], a[4]), (b[2], b[1], b[3], b[4])], center)
                    n_meet += 1


if __name__ == "__main__":
    main()
