"""Reads the parts of the screen that look the same every time: hand, next card, elixir.

Cards are recognised by the edges of their artwork, so the colour version and the greyed-out
"not enough elixir" version (with its lighter clock wedge) match the same template. Templates are
built from a recording by tools/build_perception_templates.py and live outside git
(raw/perception/templates.npz) because they are cut from game art.
"""
import os

import cv2
import numpy as np

from .layout import Layout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TEMPLATES = os.path.join(ROOT, "raw", "perception", "templates.npz")
EMPTY = "_empty"
FEAT_SHAPE = (32, 40)  # w, h


def edge_features(img):
    """Normalised gradient magnitude of a crop, resized to FEAT_SHAPE."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, FEAT_SHAPE, interpolation=cv2.INTER_AREA).astype(np.float32)
    sx, sy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.sqrt(sx * sx + sy * sy).ravel()
    return (m - m.mean()) / (m.std() + 1e-6)


def hand_art(slot_crop):
    """The artwork part of a hand card (no frame, no elixir drop)."""
    h, w = slot_crop.shape[:2]
    return slot_crop[int(h * 0.07):int(h * 0.70), int(w * 0.09):int(w * 0.91)]


def saturation(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())


class ScreenReader:
    MIN_SCORE = 0.5  # template correlation below this = card in motion / unreadable

    def __init__(self, templates_path=DEFAULT_TEMPLATES):
        if not os.path.exists(templates_path):
            raise FileNotFoundError(
                f"{templates_path} not found: run tools/build_perception_templates.py first")
        t = np.load(templates_path, allow_pickle=False)
        self.hand_names = [str(n) for n in t["hand_names"]]
        self.hand_templates = t["hand_templates"]
        self.next_names = [str(n) for n in t["next_names"]]
        self.next_templates = t["next_templates"]
        self.color_threshold = float(t["color_threshold"])

    # --- hand ---------------------------------------------------------------------------
    def classify_hand_crop(self, crop):
        f = edge_features(hand_art(crop))
        scores = self.hand_templates @ f / len(f)  # correlation with each template
        best = int(np.argmax(scores))
        return self.hand_names[best], float(scores[best])

    def read_hand(self, frame, layout=None):
        """[{card, score, raise, affordable}] for the 4 slots; card is None for an empty slot.
        The selected card is drawn raised, so a few vertical offsets are tried."""
        layout = layout or Layout.for_frame(frame)
        out = []
        for slot in range(4):
            best = (None, -1.0, 0)
            for raise_px in range(0, Layout.HAND_MAX_RAISE + 1, 5):
                crop = layout.hand_slot(frame, slot, raise_px)
                name, score = self.classify_hand_crop(crop)
                if score > best[1]:
                    best = (name, score, raise_px)
            name, score, raise_px = best
            crop = layout.hand_slot(frame, slot, raise_px)
            out.append({
                # A card sliding into its slot matches nothing well: report it as unknown.
                "card": None if name == EMPTY or score < self.MIN_SCORE else name,
                "unknown": score < self.MIN_SCORE,
                "score": score,
                "raise": raise_px,
                "affordable": name != EMPTY and saturation(hand_art(crop)) > self.color_threshold,
            })
        return out

    # --- next card ----------------------------------------------------------------------
    def read_next(self, frame, layout=None):
        """The small "next" icon is the same artwork as in the hand: match it with the hand templates."""
        layout = layout or Layout.for_frame(frame)
        f = edge_features(hand_art(layout.next_card(frame)))
        scores = self.hand_templates @ f / len(f)
        scores[self.hand_names.index(EMPTY)] = -1.0
        best = int(np.argmax(scores))
        return {"card": self.hand_names[best], "score": float(scores[best])}

    # --- elixir -------------------------------------------------------------------------
    @staticmethod
    def read_elixir(frame, layout=None):
        """Whole elixir = number of fully lit segments; the segment being filled is paler and adds
        its filled fraction."""
        layout = layout or Layout.for_frame(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        full = 0
        partial = 0.0
        for row in Layout.ELIXIR_ROWS:
            y = layout.s(row)
            h, s, v = (hsv[y, :, c].astype(int) for c in range(3))
            pink = (h > 135) & (h < 172) & (v > 110)
            bright = pink & (s > 90)
            pale = pink & (s > 35)
            seg_full = 0
            for n in range(10):
                # Judge a segment by its right part: the elixir digit is drawn over the first one.
                xa = layout.s(Layout.ELIXIR_X0 + (n + 0.6) * Layout.ELIXIR_SEG)
                xb = layout.s(Layout.ELIXIR_X0 + (n + 1) * Layout.ELIXIR_SEG - 6)
                if bright[xa:xb].mean() > 0.6:
                    seg_full = n + 1
                else:
                    xa_all = layout.s(Layout.ELIXIR_X0 + n * Layout.ELIXIR_SEG + (45 if n == 0 else 6))
                    partial = max(partial, pale[xa_all:xb].mean())
                    break
            full = max(full, seg_full)
        return full + (min(partial, 0.99) if full < 10 else 0.0)

    def read(self, frame):
        layout = Layout.for_frame(frame)
        return {
            "hand": self.read_hand(frame, layout),
            "next": self.read_next(frame, layout),
            "elixir": self.read_elixir(frame, layout),
        }
