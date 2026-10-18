"""Reads the numbers on screen: tower HP, the battle timer and the x2/x3 elixir badge.

All of them use the same font: white (or light blue on blue bars) digits with a dark outline. A region
is thresholded to "white", split into glyphs (connected components on one text line), and each glyph
is matched against digit templates built from hand-read anchors (data/perception/number_anchors.json).
"""
import cv2
import numpy as np

# Reference regions at 1080x1920: x0, y0, x1, y1.
REGIONS = {
    "red_king": (440, 5, 720, 55),      # shown only once the king tower has taken damage
    "blue_king": (440, 1445, 720, 1500),
    "red_left": (150, 225, 380, 275),
    "red_right": (700, 225, 930, 275),
    "blue_left": (150, 1180, 380, 1235),
    "blue_right": (700, 1180, 930, 1235),
    "timer": (900, 45, 1070, 100),
}
TIMER_BOX = (905, 10, 1072, 45)      # the "Осталось" / "Доп. время" caption: red in overtime
ELIXIR_BADGE = (950, 170, 1035, 230)  # pink drop with x2 / x3
GLYPH_SHAPE = (12, 18)                # w, h of a normalised glyph


def white_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > 200) & (hsv[..., 1] < 75)).astype(np.uint8)


def glyph_feature(mask_crop):
    h, w = mask_crop.shape
    side = max(h, int(np.ceil(w * 1.5)))  # pad to the 2:3 glyph box, keeping thin "1" thin
    canvas = np.zeros((side, int(np.ceil(side / 1.5))), np.uint8)
    oy, ox = (side - h) // 2, (canvas.shape[1] - w) // 2
    canvas[oy:oy + h, ox:ox + w] = mask_crop
    g = cv2.resize(canvas.astype(np.float32), GLYPH_SHAPE, interpolation=cv2.INTER_AREA).ravel()
    return (g - g.mean()) / (g.std() + 1e-6)


def glyphs(img, min_h=12, max_h=48):
    """Glyph features of the main text line in img, left to right."""
    m = white_mask(img)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = [tuple(s[:4]) for s in stats[1:] if min_h <= s[3] <= max_h and 3 <= s[2] <= 45 and s[4] > 20]
    if not boxes:
        return [], []

    def same_line(a, b):
        return abs((a[1] + a[3] / 2) - (b[1] + b[3] / 2)) < a[3] * 0.35 and 0.6 < b[3] / a[3] < 1.6

    # The text line is the row with the most glyphs (a stray mark can be taller than the digits).
    anchor = max(boxes, key=lambda a: (sum(same_line(a, b) for b in boxes), a[3]))
    line = sorted((b for b in boxes if same_line(anchor, b)), key=lambda b: b[0])
    # Drop stray marks far from the rest of the line.
    if len(line) > 1:
        kept = [line[0]]
        for b in line[1:]:
            if b[0] - (kept[-1][0] + kept[-1][2]) < anchor[3] * 1.2:
                kept.append(b)
            elif len(kept) == 1:
                kept = [b]
        line = kept
    # Video blur can glue neighbouring digits together: split components wider than one digit.
    split = []
    for x, y, w, h in line:
        n = max(1, int(round(w / (0.8 * h))))
        if w > 1.25 * h and n > 1:
            step = w / n
            split += [(int(x + i * step), y, int(round(step)), h) for i in range(n)]
        else:
            split.append((x, y, w, h))
    line = split
    feats = [glyph_feature(m[y:y + h, x:x + w]) for x, y, w, h in line]
    return feats, line


# Tower HP bars at 1080x1920: x0, x1, centre rows. Towers never move, so neither do their bars.
BARS = {
    "red_king": (482, 638, (52, 54, 56)),  # lower half: the HP digits are drawn over the top half
    "blue_king": (486, 638, (1448, 1450, 1452)),
    "red_left": (216, 336, (265, 267, 270)),
    "red_right": (782, 902, (265, 267, 270)),
    "blue_left": (218, 335, (1190, 1192, 1194)),
    "blue_right": (784, 902, (1190, 1192, 1194)),
}


#  The opponent's bars sit lower than BARS says in recent recordings (princess towers ~20 px, the king ~12:
#  video31), maybe a game update; ours did not move. A bar is looked for this far below (and a little above).
BAR_SEARCH = {"red": range(0, 36, 3), "blue": range(0, 1)}


def bar_fraction(frame, name, scale=1.0):
    """HP fraction from a tower's bar, or None if the bar is not shown (destroyed tower, or a king
    tower that has not been hit yet). The bar fills from the left (pink for the opponent, light blue
    for us); the rest of the bar is dark."""
    frac, _ = bar_and_shift(frame, name, scale)
    return frac


def bar_and_shift(frame, name, scale=1.0):
    """(HP fraction, rows below BARS where the bar was found) or (None, 0)."""
    for dy in BAR_SEARCH[name.split("_")[0]]:
        frac = _bar_at(frame, name, scale, dy)
        if frac is not None:
            return frac, dy
    return None, 0


def _bar_at(frame, name, scale, dy):
    x0, x1, rows = BARS[name]
    xa, xb = int(x0 * scale), int(x1 * scale)
    fracs = []
    for row in rows:
        row += dy
        seg = cv2.cvtColor(frame[int(row * scale):int(row * scale) + 1, xa:xb], cv2.COLOR_BGR2HSV)[0].astype(int)
        h, s, v = seg[:, 0], seg[:, 1], seg[:, 2]
        if name.startswith("red"):
            fill = ((h > 160) | (h < 4)) & (s > 110) & (v > 150)
        else:
            fill = (h > 92) & (h < 112) & (s > 90) & (v > 160)
        dark = (v < 175) & ~fill  # the empty part is dark (dark red in overtime, ~140)
        n = len(fill)
        end = 0
        while end < n and (fill[end] or (end + 1 < n and fill[end + 1])):  # contiguous fill from the left
            end += 1
        rest = n - end
        # A real bar: the unfilled remainder is dark (or there is none).
        if rest > 6 and dark[end:].mean() < 0.7:
            continue
        if end == 0 and dark.mean() < 0.9:
            continue
        fracs.append(end / n)
    if len(fracs) < 2:
        return None
    return float(np.median(fracs))


class NumberReader:
    MIN_SCORE = 0.55
    BAR_TOLERANCE = 0.12  # how far the read number may be from the HP bar fill

    def __init__(self, digit_templates, digit_labels):
        self.templates = np.asarray(digit_templates, np.float32)
        self.labels = [str(d) for d in digit_labels]

    def classify(self, feat):
        s = self.templates @ feat / len(feat)
        k = int(np.argmax(s))
        return self.labels[k], float(s[k])

    def read_digits(self, img):
        """Digit string and the worst glyph score ('' if nothing readable)."""
        feats, _ = glyphs(img)
        out, worst = "", 1.0
        for f in feats:
            d, score = self.classify(f)
            out += d
            worst = min(worst, score)
        return out, worst

    def region(self, frame, name, scale):
        x0, y0, x1, y1 = (REGIONS[name] if isinstance(name, str) else name)
        return frame[int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)]

    def read_tower_hp(self, frame, name, max_hp, scale=1.0):
        """(hp, exact) for a tower: the number if it agrees with the HP bar, else an estimate from the
        bar (exact=False); (None, False) when no bar is shown (destroyed tower, untouched king)."""
        frac, dy = bar_and_shift(frame, name, scale)
        if frac is None:
            return None, False
        x0, y0, x1, y1 = REGIONS[name]  # the number moves with its bar
        digits, score = self.read_digits(self.region(frame, (x0, y0 + dy, x1, y1 + dy), scale))
        if digits and score >= self.MIN_SCORE and len(digits) <= 4:
            hp = int(digits)
            if hp <= max_hp and abs(hp / max_hp - frac) <= self.BAR_TOLERANCE:
                return hp, True
        return int(round(frac * max_hp)), False

    def read_timer(self, frame, scale=1.0):
        """(seconds left in the current phase, overtime?) or (None, overtime?)."""
        x0, y0, x1, y1 = TIMER_BOX
        box = frame[int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)]
        b, g, r = (box[..., c].mean() for c in range(3))
        overtime = r > 120 and r > 1.6 * b
        digits, score = self.read_digits(self.region(frame, "timer", scale))
        if len(digits) != 3 or score < self.MIN_SCORE:
            return None, overtime
        return int(digits[0]) * 60 + int(digits[1:]), overtime

    def read_elixir_multiplier(self, frame, scale=1.0):
        """1, 2 or 3 from the badge under the timer (no badge = x1)."""
        x0, y0, x1, y1 = ELIXIR_BADGE
        badge = frame[int(y0 * scale):int(y1 * scale), int(x0 * scale):int(x1 * scale)]
        hsv = cv2.cvtColor(badge, cv2.COLOR_BGR2HSV)
        if ((hsv[..., 0] > 135) & (hsv[..., 0] < 172) & (hsv[..., 1] > 90)).mean() < 0.2:
            return 1
        feats, _ = glyphs(badge, min_h=18)
        best = (1, 0.0)
        for f in feats:
            d, score = self.classify(f)
            if d in ("2", "3") and score > best[1]:
                best = (int(d), score)
        return best[0] if best[1] >= self.MIN_SCORE else 1
