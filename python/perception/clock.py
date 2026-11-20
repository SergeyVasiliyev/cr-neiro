"""Deploy clocks: the stopwatch the game draws at the spot of every troop just put down, for its deploy second.

It shows up together with the troop (about the blueprint second after the card was dropped) and is easy to see
for sure: a gold ring with a little loop on top, a white face that a wedge fills (red for the opponent's troops,
blue for ours). The live twin took a new red troop for a play only once it had been seen 0.5 s (a flicker of the
detector must not count): with its clock under it, the troop is surely just put down, there, so it can count at
once (python/live/session.py).

Found by the shape of the gold ring in the gold-colour mask (the ring is drawn the same way every time; the face
changes as the wedge fills). The ring is made here, not cut from the game's art. Frames are 1080 wide (live:
play_live.full_size; recordings are brought to it).
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .layout import Layout

RING_OUT, RING_IN = 31.0, 22.0       # px (1080-wide frame): the gold ring around the face
LOOP_DY, LOOP_OUT, LOOP_IN = -35.0, 8.0, 3.5  # the loop on top of it
GOLD_LO, GOLD_HI = (15, 120, 150), (32, 255, 255)  # HSV
MIN_SCORE = 0.62    # ring correlation: true clocks 0.84-0.93 (video47_b02), the arena rarely above 0.5


def _ring_template():
    r = int(RING_OUT) + 2
    top = int(-LOOP_DY + LOOP_OUT) + 1
    h, w = top + r + 1, 2 * r + 1
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = float(r), float(top)
    d = np.hypot(xx - cx, yy - cy)
    dl = np.hypot(xx - cx, yy - (cy + LOOP_DY))
    t = ((d >= RING_IN) & (d <= RING_OUT)) | ((dl >= LOOP_IN) & (dl <= LOOP_OUT) & (yy < cy - RING_OUT + 2))
    return t.astype(np.float32), (cx, cy)


TEMPLATE, CENTRE = _ring_template()


@dataclass
class Clock:
    tile: tuple       # (x, y): the spot the troop was put down
    score: float      # ring match, 0..1
    side: str         # "red" / "blue" by the wedge's colour, "?" while the face is still (nearly) white
    px: tuple = (0.0, 0.0)


def gold_mask(img):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), GOLD_LO, GOLD_HI)


def _wedge_side(frame, cx, cy):
    """The wedge's colour inside the face: red (theirs), blue (ours), or ? (white yet / unclear)."""
    r = int(RING_IN - 4)
    x0, y0 = int(cx) - r, int(cy) - r
    face = frame[max(0, y0):y0 + 2 * r + 1, max(0, x0):x0 + 2 * r + 1]
    if face.size == 0:
        return "?"
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    strong = (s > 110) & (v > 120)
    red = int((strong & ((h <= 8) | (h >= 170))).sum())
    blue = int((strong & (h >= 95) & (h <= 125)).sum())
    n = max(1, face.shape[0] * face.shape[1])
    if max(red, blue) < 0.04 * n:
        return "?"
    return "red" if red > blue else "blue"


def find_clocks(frame, near=None, reach=(1.4, 3.2, 0.9), min_score=MIN_SCORE, layout=None):
    """Deploy clocks in a 1080-wide frame: [Clock]. near: tiles to look around only (the spots of new troops:
    reach = (x either way, y below, y above) in tiles - a troop drops in from above, so its clock may lie up to
    ~2.5 tiles below where it is first seen); None: the whole arena (slow, ~25 ms)."""
    layout = layout or Layout.for_frame(frame)
    H, W = frame.shape[:2]
    th, tw = TEMPLATE.shape
    boxes = []
    if near is None:
        boxes.append((0, layout.s(120), W, layout.s(1540)))
    else:
        for tx, ty in near:
            x0, y1 = layout.tile_to_px(tx - reach[0], ty - reach[1])
            x1, y0 = layout.tile_to_px(tx + reach[0], ty + reach[2])
            boxes.append((int(x0) - tw // 2, int(y0) - th, int(x1) + tw // 2, int(y1) + th // 2))
    boxes = _merge([(max(0, a), max(0, b), min(W, c), min(H, d)) for a, b, c, d in boxes])
    found = []
    for x0, y0, x1, y1 in boxes:
        if x1 - x0 < tw or y1 - y0 < th:
            continue
        gm = gold_mask(frame[y0:y1, x0:x1]).astype(np.float32) * (1.0 / 255.0)
        res = cv2.matchTemplate(gm, TEMPLATE, cv2.TM_CCORR_NORMED)
        while True:
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < min_score:
                break
            cx, cy = x0 + loc[0] + CENTRE[0], y0 + loc[1] + CENTRE[1]
            if all(math.hypot(cx - c.px[0], cy - c.px[1]) > RING_OUT for c in found):
                found.append(Clock(layout.px_to_tile(cx, cy), float(mx), _wedge_side(frame, cx, cy), (cx, cy)))
            k = int(RING_OUT)
            res[max(0, loc[1] - k):loc[1] + k + 1, max(0, loc[0] - k):loc[0] + k + 1] = 0.0
    return found


def _merge(boxes):
    """Overlapping search boxes as one (each spot searched once)."""
    boxes = sorted(boxes)
    out = []
    for b in boxes:
        for i, o in enumerate(out):
            if b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3]:
                out[i] = (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
                break
        else:
            out.append(b)
    return out
