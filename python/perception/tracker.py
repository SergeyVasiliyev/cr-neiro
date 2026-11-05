"""Keeps track of every unit on the arena over time, including ones that are briefly hidden.

The detector looks at single frames: units disappear behind a Giant, blink out for a frame or two, and
two archers swap places. The tracker gives each unit a stable id, smooths its position and velocity,
and keeps a unit that stopped being seen for a while (longer when a big unit could be covering it),
moving it along its last velocity, with a confidence that fades until it is seen again or dropped.

Later the simulator twin will take over the "move it while unseen" part (it knows where a unit walks).
"""
from dataclasses import dataclass, field

from .units import SPELL_HITS, SPELLS_IN_FLIGHT

BIG_UNITS = {"giant"}  # can hide smaller units drawn behind them
# A unit put on the field drops in from above: for its first ~0.3 s it is drawn higher than where it
# stands (measured over 5 recordings: giant 2.3 tiles at first sight, knight 1.1, minions 0.9, archers
# 0.2; on the spot after 0.3 s). Its position and speed only count after that.
DROP_TIME = 0.3


@dataclass
class Track:
    id: int
    cls: str
    team: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    hits: int = 1
    conf: float = 1.0  # detection confidence when seen, fading while unseen
    history: list = field(default_factory=list)
    team_votes: dict = field(default_factory=dict)  # side -> summed detection confidence
    still: bool = False  # something drawn on the arena that the detector takes for a unit (Tracker.STILL_*)

    @property
    def confirmed(self):
        return self.hits >= 2 or self.cls in SPELL_HITS

    def predicted(self, t):
        dt = max(0.0, t - self.last_seen)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def landed(self, t=None):
        """Still dropping in (spells and their hits don't drop)."""
        return self.cls in SPELLS_IN_FLIGHT or self.cls in SPELL_HITS or \
            (self.last_seen if t is None else t) - self.first_seen >= DROP_TIME

    def landing_pos(self):
        """Where the unit stands once it has dropped in (the first position after DROP_TIME)."""
        if self.cls in SPELLS_IN_FLIGHT or self.cls in SPELL_HITS:
            return self.history[0][1], self.history[0][2]
        for t, x, y in self.history:
            if t - self.first_seen >= DROP_TIME:
                return x, y
        return self.history[-1][1], self.history[-1][2]


class Tracker:
    FLIP_CONF = 0.8      # a detection of the other side's colour below this confidence may be the same unit
    FLIP_GATE = 0.8      # tiles from the track's prediction
    FLIP_PENALTY = 0.5   # a same-side match is preferred
    # Something drawn on the arena that the detector takes for a unit stays put to the pixel: a tower skin's
    # cannon was a "mini pekka" in half the frames of a match (video39), its 3 s spread 0.005-0.08 tiles; a unit
    # standing to fight still moves with its animation (no 3 s window of a real unit under 0.1 tile in 4
    # matches). From then on that spot is the thing's: its detections there (same kind and side) go to a still
    # track, which is no unit (not in tracks).
    STILL_WINDOW = 3.0   # s
    STILL_SEEN = 15      # sightings in the window, at least
    STILL_SPREAD = 0.05  # tiles: spread over the window below this = something drawn there
    STILL_SPOT = 0.3     # tiles: a detection this close to a still spot is the thing again

    def __init__(self, gate=1.6, max_speed=3.0, keep_unseen=1.5, keep_hidden=4.0, keep_spell=0.4):
        self.gate = gate              # tiles: how far a detection may be from a track's prediction
        self.max_speed = max_speed    # tiles/s, widens the gate after a gap
        self.keep_unseen = keep_unseen  # seconds a unit may go unseen in the open
        self.keep_hidden = keep_hidden  # ... next to a big unit that may cover it
        self.keep_spell = keep_spell
        self.tracks = []
        self.still = []     # tracks of things drawn on the arena (see STILL_*)
        self.still_spots = []  # (cls, team, x, y) where such things were
        self.finished = []  # dropped tracks, kept for inspection
        self.next_id = 1

    def _at_still_spot(self, cls, team, x, y):
        return any(c == cls and side == team and abs(sx - x) < self.STILL_SPOT and abs(sy - y) < self.STILL_SPOT
                   for c, side, sx, sy in self.still_spots)

    def _is_still(self, tr, t):
        recent = []  # newest first
        for h in reversed(tr.history):
            if t - h[0] > self.STILL_WINDOW:
                break
            recent.append(h)
        if len(recent) < self.STILL_SEEN or recent[0][0] - recent[-1][0] < 0.8 * self.STILL_WINDOW:
            return False
        spread = max(max(h[1] for h in recent) - min(h[1] for h in recent),
                     max(h[2] for h in recent) - min(h[2] for h in recent))
        return spread < self.STILL_SPREAD

    def _keep_for(self, track, t):
        if track.cls in SPELLS_IN_FLIGHT or track.cls in SPELL_HITS:
            return self.keep_spell
        px, py = track.predicted(t)
        for other in self.tracks:
            if other is not track and other.cls in BIG_UNITS and track.cls not in BIG_UNITS:
                ox, oy = other.predicted(t)
                if abs(ox - px) < 1.5 and -0.5 < py - oy < 2.5:  # behind (above on screen) a giant
                    return self.keep_hidden
        return self.keep_unseen

    def update(self, t, detections):
        """Feeds the detections of one frame taken at time t (seconds); returns the live tracks."""
        # Greedy matching by distance to each track's predicted position, same class. The detector now and
        # then gives a unit the other side's colour for a frame, at low confidence (both-side recordings:
        # median 0.61 vs 0.90): such a detection right on an established track is that unit, and the
        # track's side is decided by a vote over its life.
        # A detection at a known still spot is that thing again: only still tracks take it, and units' tracks
        # never do (a unit passing by would carry the thing off with it, or be carried off by it).
        at_spot = [self._at_still_spot(d.cls, d.team, *d.tile) for d in detections]
        pool = self.tracks + self.still
        pairs = []
        for ti, tr in enumerate(pool):
            px, py = tr.predicted(t)
            gate = self.gate + self.max_speed * min(1.0, max(0.0, t - tr.last_seen))
            for di, d in enumerate(detections):
                if d.cls != tr.cls or at_spot[di] != tr.still:
                    continue
                dist = ((d.tile[0] - px) ** 2 + (d.tile[1] - py) ** 2) ** 0.5
                if d.team == tr.team:
                    if dist <= gate:
                        pairs.append((dist, ti, di))
                elif (tr.hits >= 3 and d.conf < self.FLIP_CONF and dist <= self.FLIP_GATE
                      and "any" not in (d.team, tr.team)):
                    pairs.append((dist + self.FLIP_PENALTY, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr, d = pool[ti], detections[di]
            dt = t - tr.last_seen
            if dt > 1e-6 and t - tr.first_seen > DROP_TIME:  # the drop-in is not walking
                a = 0.5  # smoothing of the velocity estimate
                tr.vx = (1 - a) * tr.vx + a * (d.tile[0] - tr.x) / dt
                tr.vy = (1 - a) * tr.vy + a * (d.tile[1] - tr.y) / dt
            tr.x, tr.y = d.tile
            tr.last_seen = t
            tr.hits += 1
            tr.conf = d.conf
            tr.history.append((t, tr.x, tr.y))
            tr.team_votes[d.team] = tr.team_votes.get(d.team, 0.0) + d.conf
            tr.team = max(tr.team_votes, key=tr.team_votes.get)
        for di, d in enumerate(detections):
            if di not in used_d:
                pool.append(Track(self.next_id, d.cls, d.team, d.tile[0], d.tile[1], first_seen=t,
                                  last_seen=t, conf=d.conf, history=[(t, *d.tile)],
                                  team_votes={d.team: d.conf}, still=at_spot[di]))
                self.next_id += 1
        # Unseen tracks fade, then go.
        alive = []
        for ti, tr in enumerate(pool):
            if ti in used_t or tr.last_seen == t:
                alive.append(tr)
                continue
            unseen = t - tr.last_seen
            keep = self._keep_for(tr, t)
            if unseen <= keep and (tr.confirmed or unseen < 0.3):
                tr.conf *= 0.85
                alive.append(tr)
            else:
                self.finished.append(tr)
        for tr in alive:
            if not tr.still and self._is_still(tr, t):
                tr.still = True
                if not self._at_still_spot(tr.cls, tr.team, tr.x, tr.y):
                    self.still_spots.append((tr.cls, tr.team, tr.x, tr.y))
        self.tracks = [tr for tr in alive if not tr.still]
        self.still = [tr for tr in alive if tr.still]
        return self.tracks

    def state(self, t):
        """Confirmed units with their (predicted, if unseen) positions at time t."""
        out = []
        for tr in self.tracks:
            if not tr.confirmed:
                continue
            x, y = tr.predicted(t)
            out.append({"id": tr.id, "cls": tr.cls, "team": tr.team, "x": x, "y": y,
                        "seen": tr.last_seen == t, "conf": tr.conf})
        return out
