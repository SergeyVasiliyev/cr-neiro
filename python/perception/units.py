"""Units and spells on the arena: detector boxes -> positions in simulator tiles.

The detector (YOLO, trained by tools/train_detector.py) works on the arena crop and returns one class
per unit x side, e.g. "knight_red". Ground units stand where the bottom of their box is; flying units
(minions) are drawn above their shadow, so their tile is taken a bit below the box centre; spells in
flight have no ground position, spell hits are centred on where the spell landed.
"""
import os
from dataclasses import dataclass

from .layout import Layout

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS = os.path.join(ROOT, "raw", "detector", "runs", "units_v4", "weights", "best.pt")
ARENA_ROWS = (120, 1540)  # the crop the detector was trained on (1080x1920 reference)

FLYING = {"minion"}
SPELLS_IN_FLIGHT = {"fireball", "arrows"}
SPELL_HITS = {"fireball_hit", "arrows_hit"}  # side "any": an explosion does not show who threw it
FLYING_OFFSET_TILES = 1.0  # minions are drawn about a tile above the ground they fly over
HIT_MERGE_TILES = 4.5      # spell hit boxes closer than this are one explosion (arrows: 3 waves, one spot)


def merge_spells(dets):
    """One volley of arrows is many small arrows spread over a large area, and the detector may box it in
    pieces. Each deck has one Arrows and one Fireball, so all in-flight boxes of one spell and side in a
    frame are one spell: merge them (union box, mean position). Hits of one spell close together merge too."""
    out, groups = [], {}
    for d in dets:
        if d.cls in SPELLS_IN_FLIGHT:
            groups.setdefault((d.cls, d.team), []).append(d)
        elif d.cls in SPELL_HITS:
            for key, g in groups.items():
                if key[0] == d.cls and any(abs(o.tile[0] - d.tile[0]) + abs(o.tile[1] - d.tile[1]) < HIT_MERGE_TILES
                                           for o in g):
                    g.append(d)
                    break
            else:
                groups[(d.cls, d.team, len(groups))] = [d]
        else:
            out.append(d)
    for g in groups.values():
        if len(g) == 1:
            out.append(g[0])
            continue
        box = (min(o.box[0] for o in g), min(o.box[1] for o in g), max(o.box[2] for o in g), max(o.box[3] for o in g))
        tile = (sum(o.tile[0] for o in g) / len(g), sum(o.tile[1] for o in g) / len(g))
        out.append(Detection(g[0].cls, g[0].team, max(o.conf for o in g), box, tile))
    return out


@dataclass
class Detection:
    cls: str          # knight, archer, ..., fireball_hit
    team: str         # "blue" (ours), "red", or "any" for spell hits
    conf: float
    box: tuple        # x0, y0, x1, y1 in full-frame pixels
    tile: tuple       # (x, y) in simulator tiles

    @property
    def is_spell(self):
        return self.cls in SPELLS_IN_FLIGHT or self.cls in SPELL_HITS


class UnitDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, conf=0.35, device=0, imgsz=960):
        from ultralytics import YOLO  # heavy import, only when the detector is used

        if not os.path.exists(weights):
            raise FileNotFoundError(f"{weights} not found: train it with tools/train_detector.py")
        self.model = YOLO(weights)
        self.conf, self.device, self.imgsz = conf, device, imgsz

    def detect(self, frame):
        layout = Layout.for_frame(frame)
        y0, y1 = layout.s(ARENA_ROWS[0]), layout.s(ARENA_ROWS[1])
        res = self.model.predict(frame[y0:y1], imgsz=self.imgsz, conf=self.conf, device=self.device,
                                 verbose=False)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
            unit, team = self.model.names[int(cls)].rsplit("_", 1)
            bx0, by0, bx1, by1 = (float(v) for v in box)
            by0, by1 = by0 + y0, by1 + y0
            if unit in FLYING:
                tx, ty = layout.px_to_tile((bx0 + bx1) / 2, (by0 + by1) / 2)
                ty -= FLYING_OFFSET_TILES
            elif unit in SPELLS_IN_FLIGHT or unit in SPELL_HITS:
                tx, ty = layout.px_to_tile((bx0 + bx1) / 2, (by0 + by1) / 2)
            else:  # ground troop: feet at the bottom centre of the box
                tx, ty = layout.px_to_tile((bx0 + bx1) / 2, by1)
            out.append(Detection(unit, team, float(conf), (bx0, by0, bx1, by1), (tx, ty)))
        return merge_spells(out)
