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
    "elixir": (280, 1822, 380, 1872),   # the elixir count over the drop left of the bar
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

    def cluster(a):
        """The longest run of glyphs on a's line with no gap wider than a digit or so (stray marks further off,
        or a mark a little out of line with the digits, are not part of the number)."""
        line = sorted((b for b in boxes if same_line(a, b)), key=lambda b: b[0])
        runs = [[line[0]]]
        for b in line[1:]:
            if b[0] - (runs[-1][-1][0] + runs[-1][-1][2]) < a[3] * 1.2:
                runs[-1].append(b)
            else:
                runs.append([b])
        return max(runs, key=len)

    # The text line is the one with the most glyphs side by side (a stray mark can be taller than the digits, or
    # line up with some of them: the "7" of "762" was left out when a mark at the bar's end was the anchor).
    line = max((cluster(a) for a in boxes), key=lambda run: (len(run), max(b[3] for b in run)))
    # Video blur can glue neighbouring digits together: split components wider than one digit, each cut at the
    # thinnest column near where an even split would put it (digits differ in width: "30" cut in halves read "1"
    # for the "3"). A piece that is not a digit's height is a mark glued to the line (a bright blob behind the
    # last digit), not a digit.
    split = []
    for x, y, w, h in line:
        n = max(1, int(round(w / (0.8 * h))))
        if w > 1.25 * h and n > 1:
            cols = m[y:y + h, x:x + w].sum(axis=0)
            cuts, prev, reach = [], 0, max(1, int(0.3 * h))
            for i in range(1, n):
                mid = int(round(i * w / n))
                lo, hi = max(prev + 3, mid - reach), min(w - 3, mid + reach)
                cut = lo + int(np.argmin(cols[lo:hi + 1])) if hi > lo else mid
                cuts.append(cut)
                prev = cut
            edges = [0] + cuts + [w]
            for a, b in zip(edges, edges[1:]):
                on = np.flatnonzero(m[y:y + h, x + a:x + b].any(axis=1))
                if b - a >= 3 and len(on) and on[-1] - on[0] + 1 >= 0.6 * h:
                    split.append((x + a, y, b - a, h))
        else:
            split.append((x, y, w, h))
    line = split
    feats = [glyph_feature(m[y:y + h, x:x + w]) for x, y, w, h in line]
    return feats, line


# Tower HP bars at 1080x1920: x0, x1, centre rows. Towers never move, but their skins put the bars higher or lower
# (BAR_SEARCH).
BARS = {
    "red_king": (482, 638, (52, 54, 56)),  # lower half: the HP digits are drawn over the top half
    "blue_king": (486, 638, (1448, 1450, 1452)),
    "red_left": (216, 336, (265, 267, 270)),
    "red_right": (782, 902, (265, 267, 270)),
    "blue_left": (218, 335, (1190, 1192, 1194)),
    "blue_right": (784, 902, (1190, 1192, 1194)),
}


#  The opponent's bars sit where their towers' skin puts them: from ~13 px above BARS (a garden skin, video40)
#  to ~20 px below (the default towers in recent recordings, video31; a wagon skin ~0, video39). Ours (the
#  guest account's default towers) do not move. A bar is looked for over these rows around BARS.
BAR_SEARCH = {"red": (-24, 40), "blue": (-6, 8)}
BAR_MIN_ROWS = 4  # rows in a row that look like a bar (its body is ~10 px tall; one stray pink line is not one)


def bar_fraction(frame, name, scale=1.0):
    """HP fraction from a tower's bar, or None if the bar is not shown (destroyed tower, or a king
    tower that has not been hit yet). The bar fills from the left (pink for the opponent, light blue
    for us); the rest of the bar is dark."""
    frac, _ = bar_and_shift(frame, name, scale)
    return frac


def bar_and_shift(frame, name, scale=1.0, hint=None):
    """(HP fraction, rows below BARS where the bar's top was found - negative: above) or (None, 0). The bar is
    the longest run of bar-like rows within BAR_SEARCH of BARS. A bar with next to no HP left has no fill to find
    it by: with hint (rows below BARS where this tower's bar was last seen) a dark bar there is read as empty."""
    x0, x1, rows = BARS[name]
    lo, hi = BAR_SEARCH[name.split("_")[0]]
    ya, yb = max(0, int((rows[0] + lo) * scale)), int((rows[-1] + hi) * scale)
    hsv = cv2.cvtColor(frame[ya:yb, int(x0 * scale):int(x1 * scale)], cv2.COLOR_BGR2HSV).astype(int)
    fracs = [_bar_row(row, name.startswith("red")) for row in hsv]
    best, start = None, None
    for i, f in enumerate(fracs + [None]):
        if f is not None:
            start = i if start is None else start
        elif start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is not None and best[1] - best[0] >= BAR_MIN_ROWS * scale:
        top = (ya + best[0]) / scale
        return float(np.median(fracs[best[0]:best[1]])), int(round(top - rows[0]))
    if hint is not None:
        dark = [(hsv[int((r + hint) * scale) - ya, :, 2] < 175).mean() for r in rows
                if 0 <= int((r + hint) * scale) - ya < len(hsv)]
        if len(dark) == len(rows) and min(dark) >= 0.9:
            return 0.0, hint
    return None, 0


def _bar_row(seg, red):
    """Fill fraction of one pixel row across a tower's bar, or None if the row does not look like one."""
    h, s, v = seg[:, 0], seg[:, 1], seg[:, 2]
    if red:
        fill = ((h > 160) | (h < 4)) & (s > 110) & (v > 150)
    else:
        fill = (h > 92) & (h < 112) & (s > 90) & (v > 160)
    dark = (v < 175) & ~fill  # the empty part is dark (dark red in overtime, ~140)
    n = len(fill)
    first = 0
    while first < min(4, n) and not fill[first]:  # the bar may start a pixel or two right of BARS' x0
        first += 1
    end = 0
    if first < min(4, n):
        end = first
        while end < n and (fill[end] or (end + 1 < n and fill[end + 1])):  # contiguous fill from the left
            end += 1
    # A real bar: some fill (a standing tower has HP; a dark row elsewhere in the searched rows is no empty bar),
    # and the unfilled remainder is dark (or there is none).
    if end == 0 or (n - end > 6 and dark[end:].mean() < 0.7):
        return None
    return end / n


class NumberReader:
    MIN_SCORE = 0.55
    BAR_TOLERANCE = 0.12  # how far the read number may be from the HP bar fill

    def __init__(self, digit_templates, digit_labels):
        self.templates = np.asarray(digit_templates, np.float32)
        self.labels = [str(d) for d in digit_labels]
        self.bar_dy = {}  # tower -> rows below BARS where its bar was last found

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
        frac, dy = bar_and_shift(frame, name, scale, hint=self.bar_dy.get(name, 0 if name.startswith("blue") else None))
        if frac is None:
            return None, False
        self.bar_dy[name] = dy  # (the opponent's towers keep their skin all match)
        x0, y0, x1, y1 = REGIONS[name]  # the number moves with its bar (a few rows either way on some skins)
        for oy in (dy, dy - 5, dy + 5):
            digits, score = self.read_digits(self.region(frame, (x0, max(0, y0 + oy), x1, y1 + oy), scale))
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
