"""Unit HP from the bars over damaged units.

A troop that has taken damage shows its level badge (a small square with "11") and an HP bar to the
right of it, in its side's colour: blue for ours, red for theirs. Undamaged troops show nothing. The
bar's length differs between units (a giant's is longer than a mini pekka's), so the HP fraction is the
filled part over the whole bar, found by scanning from the badge until the bar ends.

The badge is found by template matching on grey levels (the white "11" on the coloured square looks the
same for both sides); its hue tells the side. The template is cut from a recording once
(build_badge_template) and kept in raw/perception/unit_badge.png (game imagery: not in git).
"""
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .layout import Layout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE = os.path.join(ROOT, "raw", "perception", "unit_badge.png")
ANCHOR = os.path.join(ROOT, "data", "perception", "unit_badge_anchor.json")
ARENA_ROWS = (120, 1540)
MATCH = 0.8          # badge template correlation (other things on the arena stay below ~0.72)
MAX_BAR = 170        # px at 1080 wide: longest bar scanned
# The whole bar is one of three sizes by unit (px at 1080 wide, measured over 3 recordings): the HP
# fraction is the filled part over this, more robust than finding where the bar ends on a busy background.
BAR_LENGTH = {"archer": 32, "minion": 32, "knight": 56, "mini_pekka": 56, "musketeer": 56, "giant": 84}


@dataclass
class Bar:
    team: str        # "blue" or "red"
    badge: tuple     # x0, y0, x1, y1 (full-frame px)
    length: int      # whole bar, px
    filled: int      # px
    score: float

    @property
    def fraction(self):
        return self.filled / self.length if self.length else 0.0

    @property
    def center_x(self):
        return (self.badge[0] + self.badge[2] + self.length) / 2


def build_badge_template(video=None, t=None, box=None):
    """Cuts the badge template from the recording named in data/perception/unit_badge_anchor.json."""
    with open(ANCHOR, encoding="utf-8") as f:
        a = json.load(f)
    video, t, box = video or a["video"], t if t is not None else a["time"], box or a["box"]
    cap = cv2.VideoCapture(os.path.join(ROOT, "raw", "videos", video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read {video} at {t}s")
    x, y, w, h = box
    os.makedirs(os.path.dirname(TEMPLATE), exist_ok=True)
    cv2.imwrite(TEMPLATE, cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY))


def _side(hsv_patch):
    """'red' / 'blue' from the coloured (not white, not dark) pixels of a badge."""
    h, s, v = hsv_patch[..., 0], hsv_patch[..., 1], hsv_patch[..., 2]
    coloured = (s > 90) & (v > 90)
    if coloured.sum() < 20:
        return None
    hue = h[coloured]
    red = np.count_nonzero((hue >= 160) | (hue <= 10))
    blue = np.count_nonzero((hue >= 95) & (hue <= 125))
    if max(red, blue) < 0.5 * coloured.sum():
        return None
    return "red" if red > blue else "blue"


class UnitHpReader:
    def __init__(self, template=TEMPLATE):
        if not os.path.exists(template):
            build_badge_template()
        self.tpl = cv2.imread(template, cv2.IMREAD_GRAYSCALE)

    def bars(self, frame, detections=None):
        """Every HP bar over a troop in a full frame (any size with 9:16; read at 1080 wide). With the
        frame's detections (1080-wide boxes), only where their badges can be (assign() takes no other):
        a few ms instead of ~45 for the whole arena."""
        if frame.shape[1] != 1080:
            frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)
        y0, y1 = ARENA_ROWS
        th, tw = self.tpl.shape
        if detections is None:
            windows = [(0, y0, frame.shape[1], y1)]
        else:  # around each unit's badge (left edge, bottom) as BADGE_OFFSET says, with its tolerance and more
            windows = []
            mx, my = BADGE_TOLERANCE[0] + 8, BADGE_TOLERANCE[1] + 8
            for d in detections:
                if d.cls not in BADGE_OFFSET:
                    continue
                ox, oy = BADGE_OFFSET[d.cls]
                bx = (d.box[0] + d.box[2]) / 2 + ox      # badge left edge
                by = d.box[1] - oy - th                  # badge top
                windows.append((int(max(0, bx - mx)), int(max(y0, by - my)),
                                int(min(frame.shape[1], bx + tw + mx)), int(min(y1, by + th + my))))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        found = []  # (score, frame y, frame x)
        for wx0, wy0, wx1, wy1 in windows:
            if wx1 - wx0 < tw or wy1 - wy0 < th:
                continue
            gray = cv2.cvtColor(frame[wy0:wy1, wx0:wx1], cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, self.tpl, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= MATCH)
            found += [(float(res[y, x]), int(y) + wy0, int(x) + wx0) for y, x in zip(ys, xs)]
        found.sort(reverse=True)
        out = []
        kept = []
        for score, fy, fx in found:
            if any(abs(fy - ky) < th // 2 and abs(fx - kx) < tw // 2 for ky, kx in kept):
                continue  # the same badge (a neighbouring position, or found from two windows)
            kept.append((fy, fx))
            team = _side(hsv[fy:fy + th, fx:fx + tw])
            if team is None:
                continue
            bar = self._scan(hsv, fx + tw, fy + th // 2, team)
            if bar is None:
                continue
            out.append(Bar(team, (fx, fy, fx + tw, fy + th), bar[0], bar[1], score))
        return out

    @staticmethod
    def _scan(hsv, x0, yc, team):
        """(length, filled) of the bar starting right of the badge, from 3 rows around its middle."""
        rows = hsv[yc - 2:yc + 3, x0:x0 + MAX_BAR].astype(int)
        h, s, v = rows[..., 0], rows[..., 1], rows[..., 2]
        # Filled part: bright and strongly coloured. Empty part: dark, still tinted (semi-transparent, so
        # over a bright explosion it gets lighter, never as bright as the fill). Sampled on green and pink
        # arenas: the pink floor has the red bar's hue but is paler (S 80-150) than the red fill (S > 200).
        # (the purple goo around the pink arena is H 155-156, S/V ~200: neither)
        if team == "red":
            fill = ((h >= 160) | (h <= 10)) & (s >= 175) & (v >= 175)
            hue = (h >= 150) | (h <= 10)
        else:
            hue = (h >= 95) & (h <= 125)
            fill = hue & (s >= 90) & (v >= 200)
        empty = hue & (s >= 60) & (v < 145)
        fill_col = fill.sum(axis=0) >= 3
        bar_col = fill_col | (empty.sum(axis=0) >= 3)
        length = filled = 0
        misses = 0
        for i in range(bar_col.shape[0]):
            if bar_col[i]:
                length = i + 1
                misses = 0
            else:
                misses += 1
                if misses > 2:
                    break
        if length < 25:
            return None
        filled = int(fill_col[:length].sum())
        return length, filled


def fraction(bar, cls):
    """HP fraction of a unit of kind cls from its bar."""
    full = BAR_LENGTH.get(cls)
    if full is None or abs(bar.length - full) <= 0.15 * full:
        full = full or bar.length
    return max(0.0, min(1.0, bar.filled / full))


# Where a unit's badge sits relative to its detection box (px at 1080 wide): badge left edge minus the box
# centre, box top minus the badge bottom. Measured on isolated units over 11 recordings (the spread is a few px).
BADGE_OFFSET = {"archer": (-29, -2), "minion": (-30, -1), "knight": (-44, 13), "mini_pekka": (-44, 0),
                "musketeer": (-43, -3), "giant": (-56, 17)}
BADGE_TOLERANCE = (15, 16)  # px: further than this from where its own badge would be, a bar is a neighbour's


def assign(bars, detections):
    """{detection index: Bar} - each bar to the unit of its side whose badge would be exactly there, with a
    bar of the right size for that kind of unit (in a crowd, a neighbour's bar is otherwise easily taken)."""
    pairs = []
    for bi, bar in enumerate(bars):
        for di, d in enumerate(detections):
            if d.team != bar.team or d.is_spell or d.cls not in BADGE_OFFSET:
                continue
            full = BAR_LENGTH[d.cls]
            if not 0.8 * full <= bar.length <= 1.2 * full:
                continue
            ox, oy = BADGE_OFFSET[d.cls]
            x0, y0, x1, y1 = d.box
            ddx = abs(bar.badge[0] - ((x0 + x1) / 2 + ox))
            ddy = abs((y0 - bar.badge[3]) - oy)
            if ddx <= BADGE_TOLERANCE[0] and ddy <= BADGE_TOLERANCE[1]:
                pairs.append((ddx + ddy, bi, di))
    pairs.sort()
    used_b, used_d, out = set(), set(), {}
    for _, bi, di in pairs:
        if bi in used_b or di in used_d:
            continue
        used_b.add(bi)
        used_d.add(di)
        out[di] = bars[bi]
    return out
