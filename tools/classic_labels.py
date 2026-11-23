"""Detector labels for the Classic Decks pool from the random-bot battles, with no hand labelling (round 0).

Each battle is replayed in the simulator (tools/classic_replay.py), which knows every unit on the field, the ones
nobody played too (bats, the brawler from a cage, golemites). The arena of these friendly battles is plain, so the
median of a battle's frames is a clean arena and a unit standing alone is a clean blob of difference from it.

A unit gets its name where the simulator is sure of it: just after its deploy (the spot and the moment come from the
bots' logs), or anywhere for a building. From there its blob is followed frame by frame in each view (a track),
until it meets another unit, vanishes or changes size too much. Every 5th frame of a track, with nothing else near,
is a seed: its box, its picture with a mask (a sprite for synthetic crowds), its kind and the colour it shows
(blue = the side at the bottom of that view). Tracks also pull the simulated units to where the game has them, so
the units they make later (bats, golemites) come out in the right place.

    .venv\\Scripts\\python.exe tools\\classic_labels.py seeds c008 c009 ...      seeds into raw/classic/seeds
    .venv\\Scripts\\python.exe tools\\classic_labels.py debug c008 40           one frame: blobs, tracks, simulated units
    .venv\\Scripts\\python.exe tools\\classic_labels.py clean                    drops seeds unlike the rest of their kind
    .venv\\Scripts\\python.exe tools\\classic_labels.py sheet [--clean]          a contact sheet of every kind's seeds
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classic_replay import FLYING_OFFSET, OUT, Replay, mirror  # noqa: E402

import crsim  # noqa: E402  (classic_replay put python/ on the path)
from perception.layout import Layout  # noqa: E402

SEEDS = os.path.join(OUT, "seeds")
STRIDE = 5            # frames: seeds from every 5th frame (3 a second at 15 fps)
DIFF = 28             # a pixel differs from the arena when a channel is off by more than this
MIN_AREA = 80         # px (764-wide recording): smaller blobs are arrows, sparks, numbers
PAD = 0.4             # tiles around a blob's box where a unit's anchor still counts as in it
FOOT_DEFAULT = 0.35   # tiles a ground unit's blob reaches below where it stands, until measured per kind
MOVE_MIN, MOVE_MAX = 0.3, 4.0  # tiles: corrections smaller are noise, larger are a wrong match
CLOCK_LINGER = 0.75   # s after a troop's deploy ends: its clock is still on screen (shown 0.67 s past the deploy)
CHILD_LINGER = 1.0    # s after a unit nobody played (bats, a brawler) appears: the puff it comes out of
COUNTDOWN = ((169.5, 183.5), (289.5, 1e9))  # battle seconds with big gold numbers (and "overtime") mid-arena
CHILD_TRUST = (1.5, 1.5)  # a unit nobody played is named only if a unit a track saw in the last 1.5 s (or a
                          # building) stood within 1.5 tiles of where it came out: else its parent may be elsewhere
START_WINDOW = 4.0    # s after a troop's deploy ends while the simulator still has it where it is: tracks start then
GATE = 0.9            # tiles a tracked blob's centre may be from where it was going
AREA_JUMP = (0.45, 2.2)  # a tracked blob's area against its recent median: outside, no seed; for 5 frames, the end
OFF_SIZE_END = 5
SEED_ALONE = 0.5      # tiles: no unexplained unit this close to a seed's box
ARENA_ROWS = (120, 1540)       # reference (1080-wide) rows of the arena
TOWERS = ((3.5, 6.5, "princess"), (14.5, 6.5, "princess"), (3.5, 25.5, "princess"), (14.5, 25.5, "princess"),
          (9.0, 3.0, "king"), (9.0, 29.0, "king"))
# tiles: half width, reach below and above the centre; the near towers' HP bars sit low, the far ones' above them
TOWER_BOX = {("princess", "near"): (1.9, 1.8, 2.9), ("princess", "far"): (1.9, 1.7, 3.9),
             ("king", "near"): (2.5, 3.2, 3.4), ("king", "far"): (2.5, 2.3, 3.6)}
K3, K7 = np.ones((3, 3), np.uint8), np.ones((7, 7), np.uint8)
# what a simulated unit looks like when its id is no look of its own (the Goblin Gang's goblins are plain goblins)
LOOK = {"goblin_gang": "goblins", "spear_goblin_rider": "spear_goblins"}
BODY_SHARE = 0.35     # a flying unit's blob must be at least this much body (not shadow), else it is its shadow


# --- the arena --------------------------------------------------------------------------------------------
def video_path(name, view):
    return os.path.join(OUT, f"{name}{'' if view == 'blue' else '_red'}.mp4")


def clean_arena():
    """The empty arena (raw/classic/arena_clean.png). The friendly battles' arena is the same in both views and in
    every battle, pixel for pixel; it is empty from just after the timer shows until the first troop lands, so the
    median of those moments over all battles is the arena with nothing on it (a per-battle median keeps ghosts of
    units that stood long at the bridges)."""
    path = os.path.join(OUT, "arena_clean.png")
    if os.path.exists(path):
        return cv2.imread(path)
    from perception.reader import ScreenReader
    reader = ScreenReader()
    stack = []
    for f in sorted(glob.glob(os.path.join(OUT, "c*_plays.jsonl"))):
        name = os.path.basename(f)[:-12]
        plays = [json.loads(line) for line in open(f, encoding="utf-8")]
        if not plays:
            continue
        first_play = min(pl["t"] for pl in plays)
        for view in ("blue", "red"):
            cap = cv2.VideoCapture(video_path(name, view))
            fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
            first, k = None, 0
            while k < fps * 15:
                ok, fr = cap.read()
                if not ok:
                    break
                t = k / fps
                k += 1
                if first is None:
                    if reader.numbers.read_timer(cv2.resize(fr, (1080, 1920)))[0] is not None:
                        first = t
                elif first + 0.6 <= t <= min(first + 2.0, first_play + 0.15):
                    stack.append(fr)
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    cv2.imwrite(path, bg)
    return bg


def exclusion(layout, h, w):
    """Pixels where no seed may be: outside the arena, the side walls, the towers with their HP bars and
    archers (the same in both views: the towers stand mirror-symmetric)."""
    m = np.zeros((h, w), np.uint8)
    m[:layout.s(ARENA_ROWS[0])] = 1
    m[layout.s(ARENA_ROWS[1]):] = 1
    xl, _ = layout.tile_to_px(0.0, 0.0)
    xr, _ = layout.tile_to_px(18.0, 0.0)
    m[:, :max(0, int(xl))] = 1
    m[:, int(xr):] = 1
    for cx, cy, kind in TOWERS:
        dx, below, above = TOWER_BOX[(kind, "far" if cy > 16 else "near")]
        xa, ya = layout.tile_to_px(cx - dx, cy + above)
        xb, yb = layout.tile_to_px(cx + dx, cy - below)
        m[max(0, int(ya)):int(yb) + 1, max(0, int(xa)):int(xb) + 1] = 1
    return m


def blobs(frame, bg, excl):
    """Connected blobs of difference from the clean arena: (labels image, [blob dict])."""
    d = cv2.absdiff(frame, bg).max(axis=2)
    m = ((d > DIFF) & (excl == 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K7)
    m[excl > 0] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in stats[i])
        if a >= MIN_AREA:
            out.append({"i": i, "box": (x, y, x + w, y + h), "area": a, "c": (x + w / 2.0, y + h / 2.0),
                        "claims": []})
    return lab, out


def shadow_mask(frame, bg):
    """Pixels darkened alike in every channel: shadows on the arena (no part of a flying unit's body)."""
    r = (frame.astype(np.float32) + 1.0) / (bg.astype(np.float32) + 1.0)
    lo, hi = r.min(axis=2), r.max(axis=2)
    return (hi < 0.88) & (lo > 0.3) & (hi - lo < 0.15)


def touching_labels(lab, excl_near):
    """Blob labels with a pixel next to the excluded area (cut by a tower or the arena's edge)."""
    return set(np.unique(lab[excl_near > 0]).tolist()) - {0}


# --- the simulated units on screen ----------------------------------------------------------------------------
def view_tile(x, y, view):
    return mirror(x, y) if view == "red" else (x, y)


def anchor(u, view, layout):
    """Where a simulated unit shows in a view: its feet (ground) or its body (flying), in pixels."""
    x, y = view_tile(u["x"], u["y"], view)
    return layout.tile_to_px(x, y + (FLYING_OFFSET if u["air"] else 0.0))


def mark_busy(state, pending, view, layout, found, touching, tile_px):
    """Marks blobs cut by the excluded area, and blobs near a spell or a blueprint (their pictures are no unit)."""
    busy = []
    for s in state["spells"]:
        x, y = view_tile(s["x"], s["y"], view)
        px, py = layout.tile_to_px(x, y)
        busy.append((px, py, (s.get("radius") or s.get("half_width") or 1.0) * tile_px + 1.0 * tile_px))
    boxes = []  # blueprints: the card's name and level over the spot in the player's own view
    for p in pending:
        x, y = view_tile(p.pos.x, p.pos.y, view)
        xa, ya = layout.tile_to_px(x - 3.2, y + 4.0)
        xb, yb = layout.tile_to_px(x + 3.2, y - 1.5)
        boxes.append((xa, ya, xb, yb))
    for b in found:
        x0, y0, x1, y1 = b["box"]
        b["touching"] = b["i"] in touching
        near_spell = any(x0 - r <= px <= x1 + r and y0 - r <= py <= y1 + r for px, py, r in busy)
        b["busy"] = near_spell or any(x0 <= xb and xa <= x1 and y0 <= yb and ya <= y1 for xa, ya, xb, yb in boxes)


# --- tracks and seeds -----------------------------------------------------------------------------------------
class Track:
    """One unit followed through the frames of one view, from a moment the simulator had it exactly."""

    def __init__(self, u, view, k, blob):
        self.uid, self.unit, self.team, self.air = u["id"], u["unit"], u["team"], u["air"]
        self.view = view
        self.obs = [(k, blob["c"], blob["area"])]
        self.miss = 0
        self.off = 0          # frames in a row with the blob's size off
        self.done = False
        self.start_k = k

    def predict(self, k):
        k1, c1, _ = self.obs[-1]
        v = (0.0, 0.0)
        if len(self.obs) >= 3:
            k0, c0, _ = self.obs[-3]
            v = ((c1[0] - c0[0]) / max(1, k1 - k0), (c1[1] - c0[1]) / max(1, k1 - k0))
        return c1[0] + v[0] * (k - k1), c1[1] + v[1] * (k - k1)

    def area(self):
        return statistics.median(a for _, _, a in self.obs[-8:])


class Seeder:
    def __init__(self, name):
        self.name = name
        self.replay = Replay(name)
        tl = self.replay.tl
        self.fps = tl["fps"]
        self.red_shift = int(round((tl["start_red"] - tl["start"]) * self.fps))
        arena = clean_arena()
        self.bg = {"blue": arena, "red": arena}
        h, w = self.bg["blue"].shape[:2]
        self.layout = Layout.for_frame(self.bg["blue"])
        self.tile_px = self.layout.TILE_W * self.layout.scale
        self.excl = exclusion(self.layout, h, w)
        self.excl_near = cv2.dilate(self.excl, np.ones((7, 7), np.uint8))
        self.first_active = {}   # entity id -> battle second it was first seen deployed
        self.foot = collections.defaultdict(list)  # unit kind -> feet offsets (tiles) seen when a track started
        self.tracks = {"blue": {}, "red": {}}       # entity id -> Track (the live one, or the last)
        self.moves = []
        self.ended = collections.Counter()
        self.confirmed = {}       # entity id -> (battle second, x, y): where a track last saw it (blue's tiles)
        self.child_ok = {}        # spawned entity id -> whether its parent's spot was sure when it came out
        self.born = {}            # entity id -> (battle second, x, y) when first seen

    def foot_of(self, kind):
        f = self.foot[kind]
        return statistics.median(f) if len(f) >= 3 else FOOT_DEFAULT

    def ground_seen(self, blob, u, view):
        """Where a blob says unit u stands, in the view's tiles: feet below a ground unit's body, the ground under
        a flying one's (its box without the shadow)."""
        if u["air"]:
            x0, y0, x1, y1 = blob.get("body", blob["box"])
            tx, ty = self.layout.px_to_tile((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            return tx, ty - FLYING_OFFSET
        x0, y0, x1, y1 = blob["box"]
        tx, ty_bottom = self.layout.px_to_tile((x0 + x1) / 2.0, y1)
        return tx, ty_bottom + self.foot_of(u["unit"])

    def siblings(self, u, units):
        """Units that came with u: the same play, or (nobody played them) the same kind out at the same moment
        and spot (a pair of bats, golemites, lava pups)."""
        if u["play"] >= 0:
            return [v for v in units if v["id"] != u["id"] and v["team"] == u["team"] and v["play"] == u["play"]]
        bu = self.born.get(u["id"])
        out = []
        for v in units:
            bv = self.born.get(v["id"])
            if v["id"] != u["id"] and v["spawned"] and v["unit"] == u["unit"] and v["team"] == u["team"] and bu and bv \
                    and abs(bu[0] - bv[0]) < 0.3 and ((bu[1] - bv[1]) ** 2 + (bu[2] - bv[2]) ** 2) ** 0.5 < 2.5:
                out.append(v)
        return out

    def siblings_apart(self, u, blob, units, found, view):
        """A group's unit is named alone only when each of the others stands in a blob of its own: a crowd of
        skeletons is one blob, and the simulator's spread of a group is not the game's to a few pixels."""
        pad = PAD * self.tile_px
        for v in self.siblings(u, units):
            px, py = anchor(v, view, self.layout)
            inside = [b for b in found if b["box"][0] - pad <= px <= b["box"][2] + pad
                      and b["box"][1] - pad <= py <= b["box"][3] + pad]
            if not inside or any(b is blob for b in inside):
                return False
        return True

    def note_births(self, state):
        """Where each unit first showed, and whether a spawned one came from a parent whose spot was sure."""
        g = state["g"]
        for u in state["units"]:
            if u["id"] in self.born:
                continue
            self.born[u["id"]] = (g, u["x"], u["y"])
            if u["spawned"]:
                near = [c for c in self.confirmed.values() if g - c[0] <= CHILD_TRUST[0]
                        and ((c[1] - u["x"]) ** 2 + (c[2] - u["y"]) ** 2) ** 0.5 <= CHILD_TRUST[1]]
                self.child_ok[u["id"]] = bool(near)
        for u in state["units"]:
            if u["building"] and not u["deploying"]:
                self.confirmed[u["id"]] = (g, u["x"], u["y"])

    def trusted(self, u, g):
        """Whether the simulator's spot for unit u is good: a building, a troop just deployed, or one a track
        follows in some view."""
        if u["building"] or g - self.first_active.get(u["id"], g) <= START_WINDOW:
            return True
        return any(u["id"] in tr and not tr[u["id"]].done for tr in self.tracks.values())

    def strangers(self, blob, uid, units, found, view, margin):
        """Simulated units (not uid) with their anchor within margin tiles of the blob's box that no other blob
        explains: they may be hidden in it."""
        x0, y0, x1, y1 = blob["box"]
        m, pad = margin * self.tile_px, PAD * self.tile_px
        out = []
        for u in units:
            if u["id"] == uid:
                continue
            px, py = anchor(u, view, self.layout)
            if not (x0 - m <= px <= x1 + m and y0 - m <= py <= y1 + m):
                continue
            if any(b is not blob and b["box"][0] - pad <= px <= b["box"][2] + pad
                   and b["box"][1] - pad <= py <= b["box"][3] + pad for b in found):
                continue
            out.append(u)
        return out

    def step_view(self, view, k, frame, state, pending):
        layout = self.layout
        lab, found = blobs(frame, self.bg[view], self.excl)
        mark_busy(state, pending, view, layout, found, touching_labels(lab, self.excl_near), self.tile_px)
        units = [u for u in state["units"] if not u["rider"] and u["hp"] > 0]  # (riders are drawn on their host)
        by_id = {u["id"]: u for u in units}
        g = state["g"]
        tracks = self.tracks[view]
        gate = GATE * self.tile_px
        # 1. carry on the live tracks
        for tr in [t for t in tracks.values() if not t.done]:
            if tr.uid not in by_id:
                tr.done = True
                self.ended["died"] += 1
                continue
            px, py = tr.predict(k)
            best, bd = None, gate
            for b in found:
                x0, y0, x1, y1 = b["box"]
                d = 0.0 if (x0 <= px <= x1 and y0 <= py <= y1) else ((b["c"][0] - px) ** 2 + (b["c"][1] - py) ** 2) ** 0.5
                if d < bd:
                    best, bd = b, d
            if best is None:
                tr.miss += 1
                if tr.miss > 2:
                    tr.done = True
                    self.ended["lost"] += 1
                continue
            if not AREA_JUMP[0] <= best["area"] / max(1.0, tr.area()) <= AREA_JUMP[1]:
                tr.off += 1
                if tr.off >= OFF_SIZE_END:
                    tr.done = True
                    self.ended["area"] += 1
                    continue
                best["off_size"] = True
            else:
                tr.off = 0
            best["claims"].append(tr)
        for b in found:
            if len(b["claims"]) > 1:  # two tracked units in one blob: they met
                for tr in b["claims"]:
                    tr.done = True
                    self.ended["met"] += 1
                b["claims"] = []
            elif b["claims"]:
                tr = b["claims"][0]
                strange = self.strangers(b, tr.uid, units, found, view, 0.0)
                if strange:
                    sure = any(self.trusted(u, g) for u in strange) or b["area"] > 1.3 * tr.area()
                    if sure:
                        tr.done = True
                        self.ended["stranger"] += 1
                        b["claims"] = []
                        continue
                    b["doubt"] = True
                if not b.get("off_size"):
                    tr.obs.append((k, b["c"], b["area"]))
                else:
                    tr.obs.append((k, b["c"], tr.area()))  # (keep the size it had)
                tr.miss = 0
        # 2. start tracks on units the simulator has exactly: troops just deployed (their clock gone), buildings
        for u in units:
            if u["deploying"] or u["id"] in tracks and (not tracks[u["id"]].done or not u["building"]):
                continue  # (a troop whose track ended: what the simulator says now is less sure than it was)
            age = g - self.first_active.get(u["id"], g)
            if age < (CHILD_LINGER if u["spawned"] else CLOCK_LINGER) or (age > START_WINDOW and not u["building"]):
                continue
            if u["spawned"] and not self.child_ok.get(u["id"], False):
                continue
            px, py = anchor(u, view, layout)
            for b in found:
                x0, y0, x1, y1 = b["box"]
                if not (x0 <= px <= x1 and y0 <= py <= y1) or b["claims"] or b["touching"] or b["busy"]:
                    continue
                if self.strangers(b, u["id"], units, found, view, SEED_ALONE):
                    continue
                if not self.siblings_apart(u, b, units, found, view):
                    continue
                tr = Track(u, view, k, b)
                tracks[u["id"]] = tr
                b["claims"] = [tr]
                if not u["air"]:
                    _, sy = view_tile(u["x"], u["y"], view)
                    _, ty_bottom = layout.px_to_tile(b["c"][0], y1)
                    self.foot[u["unit"]].append(sy - ty_bottom)
                break
        # 3. flying units' bodies: their blob without its shadow
        fliers = [b for b in found if b["claims"] and b["claims"][0].air]
        if fliers:
            sh = shadow_mask(frame, self.bg[view])
            for b in fliers:
                x0, y0, x1, y1 = b["box"]
                m = (lab[y0:y1, x0:x1] == b["i"]) & ~sh[y0:y1, x0:x1]
                m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, K3).astype(bool)
                ys, xs = np.nonzero(m)
                if len(xs) >= BODY_SHARE * b["area"]:
                    b["body"] = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
                    b["body_mask"] = m
                else:
                    b["shadow_only"] = True
        return lab, found

    def run(self, write=True, only_frame=None):
        os.makedirs(os.path.join(SEEDS, "sprites"), exist_ok=True)
        caps = {"blue": cv2.VideoCapture(video_path(self.name, "blue")),
                "red": cv2.VideoCapture(video_path(self.name, "red"))}
        for _ in range(max(0, self.red_shift)):
            caps["red"].read()
        records, k, frames_done = [], -1, 0
        while True:
            frames = {}
            for v, cap in caps.items():
                ok, f = cap.read()
                frames[v] = f if ok else None
            if frames["blue"] is None or frames["red"] is None:
                break
            k += 1
            t = k / self.fps
            if not self.replay.advance(t):
                continue
            frames_done += 1
            state = self.replay.state()
            g = state["g"]
            for u in state["units"]:
                if not u["deploying"]:
                    self.first_active.setdefault(u["id"], g)
            pending = list(self.replay.battle.pending)
            self.note_births(state)
            seen = {v: self.step_view(v, k, frames[v], state, pending) for v in ("blue", "red")}
            by_id = {u["id"]: u for u in state["units"]}
            # pull the simulated units to where their tracks have them
            obs = collections.defaultdict(list)
            for view, (lab, found) in seen.items():
                for b in found:
                    if b["claims"] and not b["touching"]:
                        tr = b["claims"][0]
                        if tr.obs[-1][0] == k and len(tr.obs) > 1 and tr.uid in by_id:
                            obs[tr.uid].append(view_tile(*self.ground_seen(b, by_id[tr.uid], view), view))
            for uid, gs in obs.items():
                u = by_id[uid]
                gx = sum(p[0] for p in gs) / len(gs)
                gy = sum(p[1] for p in gs) / len(gs)
                self.confirmed[uid] = (g, gx, gy)
                d = ((gx - u["x"]) ** 2 + (gy - u["y"]) ** 2) ** 0.5
                if MOVE_MIN < d < MOVE_MAX and not u["building"]:
                    self.replay.battle.twin_move(uid, crsim.Vec2(gx, gy))
                    self.moves.append(d)
            if only_frame is not None:
                if t >= only_frame - 0.5 / self.fps:
                    return seen, state
                continue
            if k % STRIDE or any(a <= g < b for a, b in COUNTDOWN):
                continue
            # seeds: tracked blobs alone, off the towers, with no spell or blueprint on them
            units = [u for u in state["units"] if not u["rider"] and u["hp"] > 0]
            for view, (lab, found) in seen.items():
                for b in found:
                    if not b["claims"] or b["touching"] or b["busy"] or b.get("off_size") or b.get("shadow_only")                             or b.get("doubt"):
                        continue
                    tr = b["claims"][0]
                    if tr.obs[-1][0] != k or tr.uid not in by_id:
                        continue
                    if self.strangers(b, tr.uid, units, found, view, SEED_ALONE):
                        continue
                    records.append(self.seed(frames[view], lab, b, by_id[tr.uid], view, k, t, g, write, tr))
        # both views: a seed whose unit the other view tracked at the same moment, at the same spot
        where = {}
        for view, tracks in self.tracks.items():
            for tr in tracks.values():
                for kk, c, _ in tr.obs:
                    where[(view, tr.uid, kk)] = view_tile(*self.layout.px_to_tile(*c), view)
        for rec in records:
            other = "red" if rec["view"] == "blue" else "blue"
            a, o = where.get((rec["view"], rec["id"], rec["frame"])), where.get((other, rec["id"], rec["frame"]))
            rec["both"] = bool(a and o and ((a[0] - o[0]) ** 2 + (a[1] - o[1]) ** 2) ** 0.5 < 1.2)
        if write:
            with open(os.path.join(SEEDS, f"{self.name}.jsonl"), "w", encoding="utf-8") as out:
                for rec in records:
                    out.write(json.dumps(rec) + "\n")
        return records, frames_done

    def seed(self, frame, lab, b, u, view, k, t, g, write, tr):
        x0, y0, x1, y1 = b.get("body", b["box"]) if u["air"] else b["box"]
        seen_colour = "blue" if u["team"] == view else "red"
        look = LOOK.get(u["unit"], u["unit"])
        rec = {"battle": self.name, "view": view, "frame": k, "t": round(t, 3), "g": round(g, 3), "unit": look,
               "sim_unit": u["unit"],
               "seen": seen_colour, "id": u["id"], "box": [x0, y0, x1, y1], "area": b["area"], "air": u["air"],
               "building": u["building"], "hp": round(u["hp"] / max(1, u["max_hp"]), 3),
               "age": round(g - self.first_active.get(u["id"], g), 2), "track": k - tr.start_k}
        if write:
            m = 2
            h, w = frame.shape[:2]
            xa, ya, xb, yb = max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)
            mask = lab == b["i"]
            if u["air"] and "body_mask" in b:
                bx0, by0 = b["box"][0], b["box"][1]
                mask = np.zeros(lab.shape, bool)
                mask[by0:by0 + b["body_mask"].shape[0], bx0:bx0 + b["body_mask"].shape[1]] = b["body_mask"]
            alpha = cv2.dilate(mask[ya:yb, xa:xb].astype(np.uint8), K3) * 255
            d = os.path.join(SEEDS, "sprites", f"{look}_{seen_colour}")
            os.makedirs(d, exist_ok=True)
            fname = f"{self.name}_{view}_{k:05d}_{u['id']}.png"
            cv2.imwrite(os.path.join(d, fname), np.dstack([frame[ya:yb, xa:xb], alpha]))
            rec["sprite"] = f"{look}_{seen_colour}/{fname}"
        return rec


# --- cleaning ---------------------------------------------------------------------------------------------------
def colour_hist(im):
    """HSV histogram of a sprite's masked pixels (12 x 4 x 4 bins, sums to 1)."""
    hsv = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2HSV)
    mask = (im[:, :, 3] > 0).astype(np.uint8)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return h / max(1.0, h.sum())


def clean(max_dist=0.55, mad_k=3.0):
    """Drops seeds unlike the rest of their kind and colour: size off the group's median (area x0.5..x2, height
    x0.65..x1.55), or colours far from the group's mean (Bhattacharyya distance above both max_dist and median +
    mad_k * MAD). Writes raw/classic/seeds/clean.jsonl."""
    recs = []
    for f in sorted(glob.glob(os.path.join(SEEDS, "c[0-9]*.jsonl"))):  # (not clean.jsonl itself)
        recs += [json.loads(line) for line in open(f, encoding="utf-8")]
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["unit"], r["seen"])].append(r)
    kept, dropped = [], collections.Counter()
    for key, rs in sorted(groups.items()):
        hs = []
        for r in rs:
            im = cv2.imread(os.path.join(SEEDS, "sprites", r["sprite"]), cv2.IMREAD_UNCHANGED)
            hs.append(colour_hist(im) if im is not None else None)
        areas = [(r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) for r in rs]
        heights = [r["box"][3] - r["box"][1] for r in rs]
        ma, mh = statistics.median(areas), statistics.median(heights)
        valid = [h for h in hs if h is not None]
        mean = np.mean(valid, axis=0).astype(np.float32) if valid else None
        dists = [cv2.compareHist(mean, h.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA) if h is not None else 9.0
                 for h in hs]
        md = statistics.median(dists)
        mad = statistics.median(abs(d - md) for d in dists) or 0.05
        for r, a, ht, d in zip(rs, areas, heights, dists):
            if len(rs) >= 5 and not (0.5 <= a / ma <= 2.0 and 0.65 <= ht / mh <= 1.55):
                dropped["size"] += 1
            elif len(rs) >= 5 and d > max(max_dist, md + mad_k * mad):
                dropped["colour"] += 1
            else:
                r["colour_dist"] = round(float(d), 3)
                kept.append(r)
    with open(os.path.join(SEEDS, "clean.jsonl"), "w", encoding="utf-8") as out:
        for r in kept:
            out.write(json.dumps(r) + "\n")
    print(f"clean: kept {len(kept)} of {len(recs)}; dropped {dict(dropped)}")


# --- looking at it ------------------------------------------------------------------------------------------
def frame_of(name, view, t):
    cap = cv2.VideoCapture(video_path(name, view))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    return cap.read()[1]


def debug(name, t):
    s = Seeder(name)
    seen, state = s.run(write=False, only_frame=t)
    rows = []
    for view, (lab, found) in seen.items():
        vis = frame_of(name, view, t).copy()
        vis[s.excl > 0] = (vis[s.excl > 0] * 0.6).astype(np.uint8)
        for b in found:
            x0, y0, x1, y1 = b["box"]
            col = (0, 200, 0) if b["claims"] else (0, 0, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2 if b["claims"] else 1)
            if b["claims"]:
                cv2.putText(vis, b["claims"][0].unit, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        for u in state["units"]:
            if not u["rider"]:
                px, py = anchor(u, view, s.layout)
                cv2.circle(vis, (int(px), int(py)), 3, (255, 0, 255), -1)
        rows.append(vis)
    path = os.path.join(OUT, f"{name}_seeds_debug.jpg")
    cv2.imwrite(path, np.hstack(rows))
    print("debug:", path)


def sheet(per_kind=10, size=72, only=None, cleaned=False):
    """One picture: a row per kind and colour with its seed count and some random seeds."""
    recs = []
    for f in ([os.path.join(SEEDS, "clean.jsonl")] if cleaned else glob.glob(os.path.join(SEEDS, "c[0-9]*.jsonl"))):
        recs += [json.loads(line) for line in open(f, encoding="utf-8")]
    groups = collections.defaultdict(list)
    for r in recs:
        if only is None or r["unit"] == only:
            groups[(r["unit"], r["seen"])].append(r)
    rng = random.Random(0)
    rows = []
    for key in sorted(groups):
        rs = groups[key]
        pick = rng.sample(rs, min(per_kind, len(rs)))
        row = np.full((size + 16, size * (per_kind + 2), 3), 255, np.uint8)
        cv2.putText(row, f"{key[0][:12]} {key[1]}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(row, f"{len(rs)}", (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        for j, r in enumerate(pick):
            im = cv2.imread(os.path.join(SEEDS, "sprites", r["sprite"]), cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            h, w = im.shape[:2]
            sc = min(size / w, size / h, 1.5)
            im = cv2.resize(im, (max(1, int(w * sc)), max(1, int(h * sc))))
            a = im[:, :, 3:4] / 255.0
            tile = (im[:, :, :3] * a + 200 * (1 - a)).astype(np.uint8)
            x = size * (j + 2)
            row[16:16 + tile.shape[0], x:x + tile.shape[1]] = tile
        rows.append(row)
    path = os.path.join(SEEDS, ("sheet" if only is None else f"sheet_{only}") + ("_clean" if cleaned else "") + ".jpg")
    cv2.imwrite(path, np.vstack(rows))
    print("sheet:", path, len(rows), "rows")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("seeds")
    s1.add_argument("battles", nargs="+")
    s2 = sub.add_parser("debug")
    s2.add_argument("battle")
    s2.add_argument("t", type=float)
    s3 = sub.add_parser("sheet")
    s3.add_argument("--unit")
    s3.add_argument("--clean", action="store_true", help="the cleaned seeds (raw/classic/seeds/clean.jsonl)")
    sub.add_parser("clean")
    args = p.parse_args()
    if args.cmd == "seeds":
        for name in args.battles:
            s = Seeder(name)
            recs, n = s.run()
            kinds = collections.Counter(r["unit"] for r in recs)
            both = sum(1 for r in recs if r["both"])
            moves = f"{len(s.moves)} moves, median {statistics.median(s.moves):.2f}" if s.moves else "no moves"
            print(f"{name}: {n} frames, {len(recs)} seeds ({both} in both views), {moves}; tracks ended {dict(s.ended)}")
            print(f"   {dict(kinds.most_common())}", flush=True)
    elif args.cmd == "debug":
        debug(args.battle, args.t)
    elif args.cmd == "clean":
        clean()
    else:
        sheet(only=args.unit, cleaned=args.clean)


if __name__ == "__main__":
    main()
