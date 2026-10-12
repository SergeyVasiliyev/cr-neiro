"""Where things are on the Clash Royale screen.

Measured on a 1080x1920 portrait recording (raw/videos/video1.mp4, LDPlayer). Other resolutions with
the same 9:16 aspect are handled by scaling; anything else needs a new calibration.

Arena mapping (checked against the grass checkerboard, river and towers): the view is a slightly
tilted top-down camera, so a tile is 51.06 px wide but only 41.6 px tall. Tile coordinates are the
simulator's: x 0..18 left to right, y 0..32 from the bottom (our side) up.
"""
from dataclasses import dataclass

REF_W, REF_H = 1080, 1920


@dataclass(frozen=True)
class Layout:
    scale: float = 1.0  # screen width / 1080

    # arena: px = X0 + tx * TILE_W,  py = Y0 - ty * TILE_H  (reference pixels)
    X0 = 80.5
    TILE_W = 51.06
    Y0 = 1476.0
    TILE_H = 41.625

    # hand: 4 slots (x ranges) on one row; the selected card is raised, see HAND_MAX_RAISE
    HAND_SLOTS_X = ((250, 425), (447, 632), (652, 836), (855, 1040))
    HAND_Y = (1600, 1815)
    HAND_MAX_RAISE = 40
    NEXT_CARD = (60, 1792, 142, 1895)  # x0, y0, x1, y1

    # elixir bar: full segments end at ELIXIR_X0 + n * ELIXIR_SEG
    ELIXIR_X0 = 297.0
    ELIXIR_SEG = 74.2
    ELIXIR_ROWS = (1866, 1874)

    @classmethod
    def for_frame(cls, frame):
        h, w = frame.shape[:2]
        if abs(w / h - REF_W / REF_H) > 0.01:
            raise ValueError(f"expected a 9:16 portrait frame, got {w}x{h}")
        return cls(scale=w / REF_W)

    def s(self, v):
        return int(round(v * self.scale))

    # --- arena -----------------------------------------------------------------------------
    def tile_to_px(self, tx, ty):
        return (self.X0 + tx * self.TILE_W) * self.scale, (self.Y0 - ty * self.TILE_H) * self.scale

    def px_to_tile(self, px, py):
        return (px / self.scale - self.X0) / self.TILE_W, (self.Y0 - py / self.scale) / self.TILE_H

    # --- crops ----------------------------------------------------------------------------
    def hand_slot(self, frame, slot, raise_px=0):
        xa, xb = self.HAND_SLOTS_X[slot]
        ya, yb = self.HAND_Y
        return frame[self.s(ya - raise_px):self.s(yb - raise_px), self.s(xa):self.s(xb)]

    def next_card(self, frame):
        x0, y0, x1, y1 = self.NEXT_CARD
        return frame[self.s(y0):self.s(y1), self.s(x0):self.s(x1)]
