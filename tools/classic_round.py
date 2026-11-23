"""Real frames of the random-bot battles labelled for the Classic Decks unit detector: the last round's detector
finds the units and the simulator twin names them. The twin knows both bots' plays exactly (tools/classic_replay.py)
and follows the boxes, so it knows each unit even in a crowd, the bats and the brawler too. It is the starter
deck's tools/auto_label.py idea, with exact plays instead of plays read off the hands.

    .venv\\Scripts\\python.exe tools\\classic_round.py label --weights raw/detector/runs/classic0/weights/best.pt c003 ...
    .venv\\Scripts\\python.exe tools\\classic_round.py build --name classic1 [--synthetic 2500] [--exclude c013]
    .venv\\Scripts\\python.exe tools\\classic_round.py show c008 [--n 6]
    .venv\\Scripts\\python.exe tools\\classic_round.py debug --weights ... c003 --frames 330 1140   (blue frame numbers)

label: per battle, raw/classic/round/<battle>.jsonl, one line per frame (every second, both views) with its boxes
and, when the detector and the twin disagree, the problems; only frames with none go into a dataset.
- A box on a twin unit gets the twin's name; a sure box (conf >= SURE) with another name that the other view's box
  backs, or a sure box on nothing the twin has, is a problem.
- A twin unit that the other view sees surely and this one misses in the open is a problem.
- A unit just put down (its deploy and the 0.75 s its clock stays) is exactly at its logged tile: missed by the
  detector, it gets a box of its kind's usual size there.
- A box with a look neither deck has is the detector's mistake: the twin names it when no other twin unit is near.
- A sure box no twin unit took is a unit the twin lost (a play it missed, a unit it let die): when the other view
  has a sure box of the same look and side on the same ground, it is labelled as the detector says, and the
  twin's free unit of that look (if any, within 8 tiles) is moved there.
- Boxes on the ghost of a card the view's own side is putting down (its blueprint second) are no units.
- A twin unit of a kind the detector has not learnt is painted over with the empty arena in the dataset when its
  background-difference blob stands apart; otherwise its frame is left out.
build: the accepted frames (1080-wide arena crops) plus synthetic pictures (tools/classic_synth.py) into
raw/detector/<name>.
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import shutil
import statistics
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classic_eval import Detector  # noqa: E402
from classic_labels import (ARENA_ROWS, CLOCK_LINGER, COUNTDOWN, FOOT_DEFAULT, LOOK, SEEDS, TOWERS,  # noqa: E402
                            anchor, blobs, clean_arena, exclusion, touching_labels, video_path, view_tile)
from classic_replay import FLYING_OFFSET, OUT, ROOT, Replay, team_name  # noqa: E402
from classic_synth import GROUP_LOOKS, class_names, looks_of  # noqa: E402

import crsim  # noqa: E402
from perception.layout import Layout  # noqa: E402

ROUND = os.path.join(OUT, "round")
DETECT_EVERY = 3     # frames: the detector runs 5 times a second, so the twin keeps up
LABEL_EVERY = 15     # frames: one labelled frame a second per view
MIN_CONF = 0.15      # boxes below this are dropped
SURE = 0.6           # a box this sure is trusted against the twin
GATE, FAR_GATE = 2.0, 4.0           # tiles: a box and a twin unit may be this far apart (any kind / same kind)
KIND_COST, SIDE_COST = 1.0, 3.0     # matching penalties (tiles) for another kind / colour
MOVE_MIN, MOVE_MAX = 0.3, 4.0
HUD_FROM = 11.0                     # battle seconds: the "VS" intro is gone
BANNERS = ((119.5, 123.5),)          # battle seconds with "60 s left" / "Elixir x2" across the arena
CONFIRMED_S = 2.0                    # a unit a box had in the last 2 s must have a box now (unless hidden)
FRESH_FOOT_S = 1.5                   # s after a deploy: the twin has the unit exactly (feet offsets are measured)
AGREE = 1.5                          # tiles: two views' boxes on the same ground
RESYNC = 8.0                         # tiles: how far the twin may have drifted from a unit both views agree on
BLUEPRINT = {"troop": (3.3, 3.3, 4.0), "spell": (5.8, 5.8, 6.0)}   # tiles: half width, below, above the spot
REACH = 1.2                          # tiles: blobs this near an unlabelled unit's spot (or its shadow's) are its picture
PAINT_PAD = 4                        # px around a painted blob that no labelled box may enter
NO_PAINT = {"bats"}                  # swarms the twin cannot place well enough to paint them over
GROUP_BOX_TILES = (1.5, 2.6)          # a unit of a group look is at most this wide and tall (tiles): a bigger box
                                     # holds two of them (the round-0 detector learnt pair-boxes from group seeds)
BIG = {"golem", "electro_giant", "royal_giant", "goblin_giant", "pekka", "lava_hound", "balloon", "sparky",
       "goblin_cage", "inferno_tower", "tombstone", "goblin_drill", "battle_ram"}


def air_looks():
    cards = json.load(open(os.path.join(ROOT, "data", "cards.json"), encoding="utf-8"))
    out = set()
    for cid, c in cards.items():
        if isinstance(c, dict) and c.get("type") != "spell" and (c.get("unit") or {}).get("air"):
            out.add(LOOK.get(cid, cid))
    return out


def look_sizes():
    """Median box size (w, h in recording pixels) per class, from the clean seeds."""
    sizes = collections.defaultdict(list)
    path = os.path.join(SEEDS, "clean.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["unit"] in GROUP_LOOKS:  # (their seeds are the group, not a unit: classic_synth.GROUP_LOOKS)
                continue
            x0, y0, x1, y1 = r["box"]
            sizes[f"{r['unit']}_{r['seen']}"].append((x1 - x0, y1 - y0))
    return {k: (statistics.median(w for w, _ in v), statistics.median(h for _, h in v)) for k, v in sizes.items()}


def known_classes(det):
    """Classes the detector was trained on (its dataset's summary next to the weights' run), else all of them."""
    try:
        args = os.path.join(os.path.dirname(os.path.dirname(det.weights)), "args.yaml")
        data = [line.split(":", 1)[1].strip() for line in open(args, encoding="utf-8") if line.startswith("data:")][0]
        summary = json.load(open(os.path.join(os.path.dirname(data), "summary.json"), encoding="utf-8"))
        return set(summary.get("sprites") or {}) | set(summary.get("boxes") or {}) | set(summary.get("synthetic_boxes") or {})
    except (OSError, IndexError, ValueError):
        return None


def greedy(pairs):
    out, used = {}, set()
    for cost, i, j in sorted(pairs):
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


class Box:
    def __init__(self, cls, conf, box, view, layout, air, foot):
        self.look, colour = cls.rsplit("_", 1)
        self.colour = colour
        self.conf = conf
        self.box = box
        self.view = view
        self.team = view if colour == "blue" else ("red" if view == "blue" else "blue")
        self.air = self.look in air
        x0, y0, x1, y1 = box
        if self.air:
            tx, ty = layout.px_to_tile((x0 + x1) / 2, (y0 + y1) / 2)
            ty -= FLYING_OFFSET
        else:
            tx, self.bottom = layout.px_to_tile((x0 + x1) / 2, y1)
            ty = self.bottom + foot(self.look)
        self.ground = view_tile(tx, ty, view)  # blue's tiles


def match(units, boxes):
    """{unit index: box index}"""
    pairs = []
    for i, u in enumerate(units):
        for j, b in enumerate(boxes):
            if b.air != u["air"]:
                continue
            d = math.dist((u["x"], u["y"]), b.ground)
            if d > GATE:
                continue
            look = LOOK.get(u["unit"], u["unit"])
            pairs.append((d + KIND_COST * (b.look != look) + SIDE_COST * (b.team != u["team"]) - 0.3 * b.conf, i, j))
    out = greedy(pairs)
    used = set(out.values())
    pairs = [(math.dist((u["x"], u["y"]), b.ground) + SIDE_COST * (b.team != u["team"]), i, j)
             for i, u in enumerate(units) if i not in out
             for j, b in enumerate(boxes) if j not in used and b.look == LOOK.get(u["unit"], u["unit"])
             and math.dist((u["x"], u["y"]), b.ground) <= FAR_GATE]
    out.update(greedy(pairs))
    return out


def hidden(u, view, boxes, layout):
    """Could unit u be out of sight in this view: behind a tower (above it on screen), behind a big unit, or
    under the top of the arena."""
    x, y = view_tile(u["x"], u["y"], view)
    for cx, cy, kind in TOWERS:
        half = 2.0 if kind == "king" else 1.5
        if abs(x - cx) < half + 0.2 and -0.5 < y - cy < 3.0:
            return True
    px, py = layout.tile_to_px(x, y)
    for b in boxes:
        if b.look in BIG and b.conf >= SURE and b.box[0] - 8 <= px <= b.box[2] + 8 and b.box[1] - 8 <= py <= b.box[3] + 20:
            return True
    return y > 31.0


class Labeller:
    def __init__(self, name, det):
        self.name, self.det = name, det
        self.replay = Replay(name)
        self.fps = self.replay.tl["fps"]
        self.red_shift = int(round((self.replay.tl["start_red"] - self.replay.tl["start"]) * self.fps))
        self.air = air_looks()
        self.sizes = look_sizes()
        self.known = known_classes(det)
        self.first_active = {}
        self.layout = None
        self.feet = collections.defaultdict(list)   # look -> measured feet offsets (tiles below the box bottom)
        self.last_seen = {}                          # entity id -> battle second a box last had it
        self.last_read = self.replay.tl.get("last_read") or 1e9
        self.looks = set(looks_of(sorted({c for _, cards in self.replay.decks.values() for c in cards})))
        self.bg = clean_arena()
        self.excl = self.excl_near = None
        self.tile_px = 1.0
        self.pending = []
        self.resyncs = 0

    def foot(self, look):
        f = self.feet[look]
        return statistics.median(f[-60:]) if len(f) >= 3 else FOOT_DEFAULT

    def view_boxes(self, frame, view):
        return [Box(c, conf, box, view, self.layout, self.air, self.foot) for c, conf, box in self.det(frame)
                if conf >= MIN_CONF]

    def run(self, debug=None):
        """debug: blue frame numbers to draw (raw/classic/round/<battle>_debug_<frame>.jpg) instead of writing labels"""
        os.makedirs(ROUND, exist_ok=True)
        caps = {"blue": cv2.VideoCapture(video_path(self.name, "blue")), "red": cv2.VideoCapture(video_path(self.name, "red"))}
        for _ in range(max(0, self.red_shift)):
            caps["red"].read()
        out = None if debug else open(os.path.join(ROUND, f"{self.name}.jsonl"), "w", encoding="utf-8")
        k, n_frames, n_ok, moves = -1, 0, 0, 0
        problems_seen = collections.Counter()
        while True:
            frames = {v: c.read()[1] for v, c in caps.items()}
            if any(f is None for f in frames.values()):
                break
            k += 1
            if debug and k > max(debug):
                break
            t = k / self.fps
            if not self.replay.advance(t) or k % DETECT_EVERY:
                continue
            if self.layout is None:
                self.layout = Layout.for_frame(frames["blue"])
                h, w = frames["blue"].shape[:2]
                self.excl = exclusion(self.layout, h, w)
                self.excl_near = cv2.dilate(self.excl, np.ones((7, 7), np.uint8))
                self.tile_px = self.layout.TILE_W * self.layout.scale
            st = self.replay.state()
            self.pending = list(self.replay.battle.pending)
            g = st["g"]
            for u in st["units"]:
                if not u["deploying"]:
                    self.first_active.setdefault(u["id"], g)
            if g < HUD_FROM or t > self.last_read or any(a <= g < b for a, b in COUNTDOWN + BANNERS):
                continue
            units = [u for u in st["units"] if not u["rider"] and u["hp"] > 0]
            boxes = {v: self.view_boxes(frames[v], v) for v in ("blue", "red")}
            matched = {v: match(units, boxes[v]) for v in ("blue", "red")}
            # the twin follows the boxes (both views averaged)
            obs = collections.defaultdict(list)
            for v in ("blue", "red"):
                for i, j in matched[v].items():
                    u, b = units[i], boxes[v][j]
                    same_look = b.look == LOOK.get(u["unit"], u["unit"])
                    if same_look or b.conf >= SURE:
                        obs[i].append(b.ground)
                        self.last_seen[u["id"]] = g
                    age = g - self.first_active.get(u["id"], g)
                    if same_look and not b.air and not u["deploying"] and age <= FRESH_FOOT_S and not u["building"]:
                        _, sy = view_tile(u["x"], u["y"], v)
                        self.feet[b.look].append(sy - b.bottom)
            for i, gs in obs.items():
                u = units[i]
                gx, gy = sum(p[0] for p in gs) / len(gs), sum(p[1] for p in gs) / len(gs)
                d = math.dist((gx, gy), (u["x"], u["y"]))
                if MOVE_MIN < d < MOVE_MAX and not u["building"] and not u["deploying"]:
                    self.replay.battle.twin_move(u["id"], crsim.Vec2(gx, gy))
                    moves += 1
            self.resync(units, boxes, matched)
            if k % LABEL_EVERY and not (debug and k in debug):
                continue
            labels = {}
            for v in ("blue", "red"):
                other = "red" if v == "blue" else "blue"
                objects, problems, paint = self.label(units, boxes[v], matched[v], boxes[other], matched[other], v, g,
                                                      frames[v])
                labels[v] = (objects, problems)
                if debug:
                    continue
                n_frames += 1
                n_ok += not problems
                for p in problems:
                    problems_seen[p.split(":")[0]] += 1
                out.write(json.dumps({"battle": self.name, "view": v, "frame": k + (self.red_shift if v == "red" else 0),
                                      "g": round(g, 2), "objects": objects, "problems": problems,
                                      "paint": paint}) + "\n")
            if debug and k in debug:
                self.draw_debug(k, g, frames, units, boxes, matched, labels)
        if debug:
            return
        out.close()
        print(f"{self.name}: {n_ok} of {n_frames} frames without problems, {moves} twin moves, {self.resyncs} "
              f"resyncs; problems "
              f"{dict(problems_seen.most_common(6))}", flush=True)

    def draw_debug(self, k, g, frames, units, boxes, matched, labels):
        """Both views side by side: the detector's boxes (thin, look and conf, blue/red by colour), the twin's units
        (circles at their feet or body: cyan for the view's own side, yellow for the other; thin while deploying),
        a white line from each matched unit to its box, and the frame's problems."""
        pics = []
        for v in ("blue", "red"):
            f = frames[v].copy()
            for b in boxes[v]:
                col = (255, 140, 0) if b.colour == "blue" else (60, 60, 255)
                x0, y0, x1, y1 = (int(c) for c in b.box)
                cv2.rectangle(f, (x0, y0), (x1, y1), col, 1)
                cv2.putText(f, f"{b.look[:10]} {b.conf:.2f}", (x0, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            for i, u in enumerate(units):
                px, py = anchor(u, v, self.layout)
                col = (255, 255, 0) if u["team"] == v else (0, 255, 255)
                cv2.circle(f, (int(px), int(py)), 6, col, 1 if u["deploying"] else 2)
                cv2.putText(f, LOOK.get(u["unit"], u["unit"])[:10], (int(px) + 7, int(py) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
                j = matched[v].get(i)
                if j is not None:
                    b = boxes[v][j]
                    cv2.line(f, (int(px), int(py)), (int((b.box[0] + b.box[2]) / 2), int(b.box[3])), (255, 255, 255), 1)
            y = 130
            cv2.putText(f, f"{self.name} {v} frame {k} g {g:.1f}", (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for p in labels[v][1][:12]:
                y += 15
                cv2.putText(f, p[:75], (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
            pics.append(f[60:1180])
        path = os.path.join(ROUND, f"{self.name}_debug_{k:05d}.jpg")
        cv2.imwrite(path, np.hstack(pics))
        print("debug:", path, flush=True)

    def resync(self, units, boxes, matched):
        """Sure boxes no twin unit took that both views agree on (same look and side, same ground): the twin's free
        unit of that look and side nearest to them (within RESYNC tiles) has drifted, so it is moved there."""
        free = {v: [b for j, b in enumerate(boxes[v]) if j not in set(matched[v].values()) and b.conf >= SURE
                    and b.look in self.looks] for v in ("blue", "red")}
        taken = set(matched["blue"]) | set(matched["red"])
        for a in free["blue"]:
            b = min(free["red"], key=lambda o: math.dist(o.ground, a.ground) if o.look == a.look and o.team == a.team
                    else 1e9, default=None)
            if b is None or b.look != a.look or b.team != a.team or math.dist(a.ground, b.ground) > AGREE:
                continue
            gx, gy = (a.ground[0] + b.ground[0]) / 2, (a.ground[1] + b.ground[1]) / 2
            cands = [(math.dist((u["x"], u["y"]), (gx, gy)), i) for i, u in enumerate(units) if i not in taken
                     and LOOK.get(u["unit"], u["unit"]) == a.look and u["team"] == a.team and not u["building"]
                     and not u["deploying"]]
            if cands and min(cands)[0] <= RESYNC:
                i = min(cands)[1]
                self.replay.battle.twin_move(units[i]["id"], crsim.Vec2(gx, gy))
                taken.add(i)
                self.resyncs += 1

    def blueprints(self, view):
        """Pixel boxes around the cards this view's own side is putting down: the card's ghost, its name and level
        and its range circle show there for the blueprint second (the other side sees nothing of it)."""
        out = []
        for p in self.pending:
            if team_name(p.team) != view:
                continue
            kind = "spell" if self.replay.data.card(self.replay.data.card_ids[p.card_index]).is_spell else "troop"
            half, below, above = BLUEPRINT[kind]
            x, y = view_tile(p.pos.x, p.pos.y, view)
            xa, ya = self.layout.tile_to_px(x - half, y + above)
            xb, yb = self.layout.tile_to_px(x + half, y - below)
            out.append((xa, ya, xb, yb))
        return out

    def paint_region(self, u, view, frame, objects, cache):
        """The blob (its box, frame pixels) of a twin unit that gets no label, to paint over with the empty arena:
        () when nothing shows near its spot (gone, or behind something), None when its blob is cut by a tower or
        the arena's edge or touches a labelled box (it cannot be painted without harm)."""
        if "blobs" not in cache:
            lab, found = blobs(frame, self.bg, self.excl)
            cache["blobs"] = (found, touching_labels(lab, self.excl_near))
        found, touching = cache["blobs"]
        points = [anchor(u, view, self.layout)]
        if u["air"]:  # its shadow on the ground too
            x, y = view_tile(u["x"], u["y"], view)
            points.append(self.layout.tile_to_px(x, y))
        near = []
        for bl in found:
            x0, y0, x1, y1 = bl["box"]
            if any(math.hypot(max(x0 - px, 0.0, px - x1), max(y0 - py, 0.0, py - y1)) <= REACH * self.tile_px
                   for px, py in points):
                near.append(bl)
        out = []
        for bl in near:
            if bl["i"] in touching:
                return None
            x0, y0, x1, y1 = bl["box"]
            region = (x0 - PAINT_PAD, y0 - PAINT_PAD, x1 + PAINT_PAD, y1 + PAINT_PAD)
            for o in objects:
                ox0, oy0, ox1, oy1 = o["box"]
                inter = (max(0.0, min(region[2], ox1) - max(region[0], ox0))
                         * max(0.0, min(region[3], oy1) - max(region[1], oy0)))
                if inter > 0.05 * max(1.0, (ox1 - ox0) * (oy1 - oy0)):
                    return None
            out.append(list(bl["box"]))
        return tuple(out)

    def label(self, units, boxes, matched, other_boxes, other_matched, view, g, frame):
        objects, problems, to_paint = [], [], []
        taken = set()
        for i, j in matched.items():
            u, b = units[i], boxes[j]
            look = LOOK.get(u["unit"], u["unit"])
            colour = "blue" if u["team"] == view else "red"
            same = b.look == look and b.colour == colour
            if b.look != look and b.conf >= SURE:
                if b.look in self.looks:
                    o = other_matched.get(i)
                    disputed = u["spawned"] or (o is not None and other_boxes[o].look == b.look
                                                and other_boxes[o].conf >= SURE)
                else:  # a look neither deck has: the detector is wrong, and the twin names it unless another is near
                    disputed = sum(math.dist((w["x"], w["y"]), b.ground) <= GATE for w in units
                                   if w["air"] == u["air"]) > 1
                if disputed:
                    problems.append(f"name: twin {look} {colour}, detector {b.look} {b.colour} {b.conf:.2f}")
                    taken.add(j)
                    continue
            objects.append({"cls": f"{look}_{colour}", "box": [round(float(c), 1) for c in b.box], "conf": round(b.conf, 2),
                            "src": "det" if same else f"twin (det {b.look}_{b.colour})"})
            taken.add(j)
        labelled = [o["box"] for o in objects]
        zones = self.blueprints(view)
        other_free = [o for jj, o in enumerate(other_boxes) if jj not in set(other_matched.values()) and o.conf >= SURE]
        for j, b in enumerate(boxes):
            if j in taken or b.conf < SURE:
                continue
            if any(iou(b.box, o["box"]) > 0.45 or (o["cls"].rsplit("_", 1)[0] == b.look and ios(b.box, o["box"]) >= 0.5)
                   for o in objects):
                continue  # (a second box on a unit already labelled)
            cx, cy = (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2
            if any(xa <= cx <= xb and ya <= cy <= yb for xa, ya, xb, yb in zones):
                continue  # the ghost of a card this side is putting down
            if b.look in self.looks and any(o.look == b.look and o.team == b.team and math.dist(o.ground, b.ground) <= AGREE
                                            for o in other_free):
                objects.append({"cls": f"{b.look}_{b.colour}", "box": [round(float(c), 1) for c in b.box],
                                "conf": round(b.conf, 2), "src": "two views"})
                labelled.append(objects[-1]["box"])
                continue
            problems.append(f"extra: {b.look} {b.colour} {b.conf:.2f} where the twin has nothing")
        for i, u in enumerate(units):
            if i in matched:
                continue
            look = LOOK.get(u["unit"], u["unit"])
            colour = "blue" if u["team"] == view else "red"
            age = g - self.first_active.get(u["id"], g)
            fresh = u["deploying"] or age < CLOCK_LINGER
            if fresh and not u["spawned"]:
                size = self.sizes.get(f"{look}_{colour}")
                if hidden(u, view, boxes, self.layout):
                    continue
                if size:
                    x, y = view_tile(u["x"], u["y"], view)
                    if u["air"]:
                        px, py = self.layout.tile_to_px(x, y + FLYING_OFFSET)
                        box = (px - size[0] / 2, py - size[1] / 2, px + size[0] / 2, py + size[1] / 2)
                    else:
                        px, py = self.layout.tile_to_px(x, y - FOOT_DEFAULT)
                        box = (px - size[0] / 2, py - size[1], px + size[0] / 2, py)
                    objects.append({"cls": f"{look}_{colour}", "box": [round(float(c), 1) for c in box], "conf": 0.0,
                                    "src": "deploy"})
                else:
                    to_paint.append((u, f"deploy: {look} {colour} put down, no size known, not apart"))
                continue
            if hidden(u, view, boxes, self.layout):
                continue
            if self.known is not None and f"{look}_{colour}" not in self.known:
                if look in NO_PAINT:
                    problems.append(f"unknown: {look} {colour} in view, not learnt")
                else:
                    to_paint.append((u, f"unknown: {look} {colour} in view, not learnt, not apart"))
                continue
            o = other_matched.get(i)
            if o is not None and other_boxes[o].conf >= SURE:
                problems.append(f"missed: {look} {colour} seen in the other view")
            elif g - self.last_seen.get(u["id"], -1e9) < CONFIRMED_S:
                problems.append(f"missed: {look} {colour} had a box {g - self.last_seen[u['id']]:.1f} s ago")
            elif any(j in matched for j in group_mates(i, units)):
                # one box over two of a group (skeletons walk bunched up): its mate got the box, it got none
                problems.append(f"missed: {look} {colour} beside its group")
        paint, cache = [], {}
        for u, why in to_paint:
            region = self.paint_region(u, view, frame, objects, cache)
            if region is None:
                problems.append(why)
            for r in region or ():
                if r not in paint:
                    paint.append(r)
        return objects, problems, paint


def group_mates(i, units):
    """Indices of the units that came with units[i]: the same play, or (spawned) the same kind close by."""
    u = units[i]
    if u["play"] >= 0:
        return [j for j, w in enumerate(units) if j != i and w["team"] == u["team"] and w["play"] == u["play"]]
    return [j for j, w in enumerate(units) if j != i and w["team"] == u["team"] and w["spawned"]
            and w["unit"] == u["unit"] and math.dist((w["x"], w["y"]), (u["x"], u["y"])) <= 2.0]


def ios(a, b):
    """Intersection over the smaller box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# --- dataset ----------------------------------------------------------------------------------------------------
def build(name, synthetic, exclude, val_share=0.1, seed=0, synth_from="classic0"):
    rng = random.Random(seed)
    names = class_names()
    index = {c: i for i, c in enumerate(names)}
    out = os.path.join(ROOT, "raw", "detector", name)
    for sub in ("images", "labels"):  # (a rebuilt set starts empty: its frames change with the labels)
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)
    frames = []
    for f in sorted(glob.glob(os.path.join(ROUND, "c*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r["problems"] and r["battle"] not in exclude:
                frames.append(r)
    by_video = collections.defaultdict(list)
    for r in frames:
        by_video[(r["battle"], r["view"])].append(r)
    counts = collections.Counter()
    n = n_painted = n_group_off = 0
    bg = clean_arena()
    layout = Layout.for_frame(bg)
    excl = exclusion(layout, bg.shape[0], bg.shape[1])
    for (battle, view), rs in sorted(by_video.items()):
        cap = cv2.VideoCapture(video_path(battle, view))
        want = {r["frame"]: r for r in rs}
        k = -1
        while want:
            ok, f = cap.read()
            if not ok:
                break
            k += 1
            r = want.pop(k, None)
            if r is None:
                continue
            if not group_boxes_ok(r["objects"], f.shape[1]):
                n_group_off += 1
                continue
            if r.get("paint"):
                f = paint_over(f, bg, excl, r["paint"])
                n_painted += 1
            s = 1080 / f.shape[1]
            crop = cv2.resize(f, (1080, int(round(f.shape[0] * s))), interpolation=cv2.INTER_CUBIC)[ARENA_ROWS[0]:ARENA_ROWS[1]]
            h, w = crop.shape[:2]
            lines = []
            for o in r["objects"]:
                if o["cls"] not in index:
                    continue
                x0, y0, x1, y1 = o["box"]
                bx0, by0, bx1, by1 = max(0, x0 * s), max(0, y0 * s - ARENA_ROWS[0]), min(w, x1 * s), min(h, y1 * s - ARENA_ROWS[0])
                if bx1 - bx0 < 4 or by1 - by0 < 4:
                    continue
                lines.append(f"{index[o['cls']]} {(bx0 + bx1) / 2 / w:.6f} {(by0 + by1) / 2 / h:.6f} {(bx1 - bx0) / w:.6f} {(by1 - by0) / h:.6f}")
                counts[o["cls"]] += 1
            split = "val" if rng.random() < val_share else "train"
            stem = f"{battle}_{view}_{k:05d}"
            cv2.imwrite(os.path.join(out, "images", split, stem + ".jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(os.path.join(out, "labels", split, stem + ".txt"), "w").write("\n".join(lines))
            n += 1
    # synthetic pictures from round 0's set, for the kinds the real frames still lack
    synth = os.path.join(ROOT, "raw", "detector", synth_from)
    pics = sorted(glob.glob(os.path.join(synth, "images", "train", "*.jpg")))
    rng.shuffle(pics)
    synth_counts = collections.Counter()
    for p in pics[:synthetic]:
        stem = os.path.basename(p)[:-4]
        shutil.copy(p, os.path.join(out, "images", "train", stem + ".jpg"))
        lab = os.path.join(synth, "labels", "train", stem + ".txt")
        shutil.copy(lab, os.path.join(out, "labels", "train", stem + ".txt"))
        for line in open(lab).read().splitlines():
            if line.strip():
                synth_counts[names[int(line.split()[0])]] += 1
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(names):
            f.write(f"  {i}: {c}\n")
    json.dump({"real_frames": n, "synthetic": min(synthetic, len(pics)), "boxes": counts,
               "synthetic_boxes": synth_counts, "synthetic_from": synth_from, "excluded": sorted(exclude)},
              open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"{name}: {n} real frames ({sum(counts.values())} boxes, {n_painted} with units painted over; {n_group_off} "
          f"left out for a group look's box too big or only a deploy guess) + "
          f"{min(synthetic, len(pics))} synthetic")


def group_boxes_ok(objects, frame_w):
    """No box of a group look that may hold two of them, and none that is only a deploy guess (its size would be the
    group seeds')."""
    layout = Layout(scale=frame_w / 1080.0)  # (perception.layout.REF_W)
    tw, th = layout.TILE_W * layout.scale, layout.TILE_H * layout.scale
    for o in objects:
        if o["cls"].rsplit("_", 1)[0] not in GROUP_LOOKS:
            continue
        x0, y0, x1, y1 = o["box"]
        if o["src"] == "deploy" or (x1 - x0) > GROUP_BOX_TILES[0] * tw or (y1 - y0) > GROUP_BOX_TILES[1] * th:
            return False
    return True


def paint_over(frame, bg, excl, regions):
    """The frame with the given blobs (their boxes, as paint_region found them) painted over with the empty arena."""
    lab, found = blobs(frame, bg, excl)
    want = {tuple(int(v) for v in r) for r in regions}
    mask = np.zeros(frame.shape[:2], np.uint8)
    for bl in found:
        if tuple(int(v) for v in bl["box"]) in want:
            mask[lab == bl["i"]] = 1
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    mask[excl > 0] = 0
    out = frame.copy()
    out[mask > 0] = bg[mask > 0]
    return out


def show(battle, n):
    recs = [json.loads(line) for line in open(os.path.join(ROUND, f"{battle}.jsonl"), encoding="utf-8")]
    rng = random.Random(1)
    good = [r for r in recs if not r["problems"]]
    bad = [r for r in recs if r["problems"]]
    pick = rng.sample(good, min(n, len(good))) + rng.sample(bad, min(n // 2, len(bad)))
    bg = clean_arena()
    excl = exclusion(Layout.for_frame(bg), bg.shape[0], bg.shape[1])
    tiles = []
    for r in pick:
        cap = cv2.VideoCapture(video_path(battle, r["view"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, f = cap.read()
        if not ok:
            continue
        if r.get("paint"):
            f = paint_over(f, bg, excl, r["paint"])
        for o in r["objects"]:
            x0, y0, x1, y1 = (int(v) for v in o["box"])
            col = ((0, 200, 0) if o["src"] == "det" else (0, 165, 255) if o["src"] == "deploy"
                   else (255, 255, 0) if o["src"] == "two views" else (255, 0, 255))
            cv2.rectangle(f, (x0, y0), (x1, y1), col, 2)
            cv2.putText(f, o["cls"][:16], (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        y = 140
        for p in r["problems"][:6]:
            cv2.putText(f, p[:60], (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            y += 18
        tiles.append(cv2.resize(f[80:1100], None, fx=0.5, fy=0.5))
    rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4]))) for i in range(0, len(tiles), 4)]
    path = os.path.join(ROUND, f"{battle}_show.jpg")
    cv2.imwrite(path, np.vstack(rows))
    print("show:", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("label")
    s1.add_argument("battles", nargs="+")
    s1.add_argument("--weights", required=True)
    s2 = sub.add_parser("build")
    s2.add_argument("--name", required=True)
    s2.add_argument("--synthetic", type=int, default=2500)
    s2.add_argument("--exclude", nargs="*", default=[])
    s2.add_argument("--synth-from", default="classic0", help="the synthetic set (tools/classic_synth.py build --name)")
    s4 = sub.add_parser("debug")
    s4.add_argument("battle")
    s4.add_argument("--weights", required=True)
    s4.add_argument("--frames", type=int, nargs="+", required=True)
    s3 = sub.add_parser("show")
    s3.add_argument("battle")
    s3.add_argument("--n", type=int, default=8)
    args = p.parse_args()
    if args.cmd == "label":
        det = Detector(args.weights, conf=MIN_CONF)
        for b in args.battles:
            Labeller(b, det).run()
    elif args.cmd == "debug":
        Labeller(args.battle, Detector(args.weights, conf=MIN_CONF)).run(debug=set(args.frames))
    elif args.cmd == "build":
        build(args.name, args.synthetic, set(args.exclude), synth_from=args.synth_from)
    else:
        show(args.battle, args.n)


if __name__ == "__main__":
    main()
