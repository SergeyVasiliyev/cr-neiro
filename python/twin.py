"""Digital twin: replays a real match (seen through perception) in the simulator and compares the two.

Input is a perception cache (tools/perceive_video.py): per frame the hand, elixir, timer, tower HP and
detected units. Steps:
  1. clock: when the battle started in video time, from the countdown timer;
  2. plays: our plays from the hand (a card leaving its slot) placed where its units / spell hit
     appeared; the opponent's from groups of new red units (and red spell hits) on their side;
  3. replay: the same plays at the same moments in a fresh simulated battle;
  4. compare: tower HP, our elixir and unit positions, real vs simulated.
"""
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crsim  # noqa: E402
from crsim_env import DEFAULT_DECK, load_data  # noqa: E402
from perception.tracker import Tracker  # noqa: E402
from perception.layout import Layout  # noqa: E402
from perception.units import SPELL_HITS, Detection, UnitDetector, merge_spells  # noqa: E402

_LAYOUT = Layout(scale=1.0)

SPELL_SAME_PLAY = (1.5, 4.5)  # hits this close in seconds and tiles belong to one cast (arrows: 3 waves)

# detector class <-> card id
UNIT_OF_CARD = {"knight": "knight", "archers": "archer", "minions": "minion", "giant": "giant",
                "mini_pekka": "mini_pekka", "musketeer": "musketeer"}
CARD_OF_UNIT = {u: c for c, u in UNIT_OF_CARD.items()}
SPELL_CARDS = ("arrows", "fireball")
REGULATION_S = 180
KING = {"blue": (9.0, 3.0), "red": (9.0, 29.0)}
TOWER_NAMES = {("blue", "king"): "blue_king", ("blue", 0): "blue_left", ("blue", 1): "blue_right",
               ("red", "king"): "red_king", ("red", 0): "red_left", ("red", 1): "red_right"}


def load_cache(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detections(frame, min_conf=0.35):
    """The frame's detections as perception.units.Detection, at the live bot's confidence threshold
    (spell pieces merged, as the live detector does)."""
    # Tiles are worked out again from the boxes (1080-wide pixels), so caches made before a calibration
    # change follow it.
    return merge_spells([Detection(d[0], d[1], d[2], tuple(d[3:7]), UnitDetector.box_to_tile(d[0], d[3:7], _LAYOUT))
                         for d in frame["det"] if d[2] >= min_conf])


# --- 1. clock ----------------------------------------------------------------------------------------
def battle_start(frames, display="ceil"):
    """Video time of the battle start (simulator time 0).

    The timer shows whole seconds left. With display="ceil" it shows S while the time left is in
    (S-1, S], i.e. from game second 180-S up to 181-S. Every reading bounds the start; the start that
    agrees with the most readings wins (a few misread frames don't matter)."""
    shift = 0.0 if display == "ceil" else -1.0
    bounds = []
    for fr in frames:
        sec, overtime = fr["timer"]
        if sec is None or overtime:
            continue
        g_lo = REGULATION_S - sec + shift  # game time window for this reading
        bounds.append((fr["t"] - (g_lo + 1.0), fr["t"] - g_lo))  # start in (lo, hi]
    if not bounds:
        raise ValueError("no timer readings")
    candidates = sorted({round(hi - k * 0.02, 2) for lo, hi in bounds for k in range(50)})
    best = max(candidates, key=lambda s: sum(lo < s <= hi for lo, hi in bounds))
    agree = [s for s in candidates if sum(lo < s <= hi for lo, hi in bounds) ==
             sum(lo < best <= hi for lo, hi in bounds)]
    return (min(agree) + max(agree)) / 2, sum(lo < best <= hi for lo, hi in bounds) / len(bounds)


# --- 2. plays ----------------------------------------------------------------------------------------
@dataclass
class Play:
    team: str           # "blue" = us, "red" = opponent
    card: str
    t: float            # video time the card was played (dropped)
    pos: tuple = None   # tiles; None if it could not be found
    source: str = ""
    units: list = field(default_factory=list)  # track ids of the units it made
    rejected: str = None  # opponent candidates the twin decided were not a new play: why


EMPTY_FOR = 0.25  # s a slot must stay empty to count as a play (a played card's slot stays empty ~1.3 s;
                  # a greyed-out card sometimes reads as empty for a frame)


def own_plays_from_hand(frames):
    """A card that leaves its slot (the slot stays empty) was played when it first read empty."""
    plays = []
    last = [None] * 4        # last card read in each slot
    empty_since = [None] * 4  # (time first read empty, the card that was there)
    for fr in frames:
        t = fr["t"]
        for slot, card in enumerate(fr["hand"]):
            if card == "?":
                continue  # moving: being dragged or sliding in
            if card is None:
                if last[slot] is not None and empty_since[slot] is None:
                    empty_since[slot] = (t, last[slot])
                if empty_since[slot] is not None and t - empty_since[slot][0] >= EMPTY_FOR:
                    plays.append(Play("blue", empty_since[slot][1], empty_since[slot][0] - 0.05, source="hand"))
                    empty_since[slot] = None
                    last[slot] = None
            else:
                if empty_since[slot] is not None and card == empty_since[slot][1]:
                    empty_since[slot] = None  # a flicker: the same card is still there
                last[slot] = card
    return plays


def run_tracker(frames):
    tracker = Tracker()
    for fr in frames:
        tracker.update(fr["t"], detections(fr))
    tracks = [tr for tr in tracker.finished + tracker.tracks if tr.confirmed and not tr.still]
    return sorted(tracks, key=lambda tr: tr.first_seen)


def place_own_plays(plays, tracks, delay=1.0):
    """Puts each of our plays where its units (or its spell hit) first appeared."""
    used = set()
    for p in plays:
        if p.card in SPELL_CARDS:
            hits = [tr for tr in tracks if tr.cls == p.card + "_hit"
                    and p.t + delay < tr.first_seen < p.t + delay + 4.0 and tr.id not in used]
            # A hit shows no side (older detectors guessed one, often wrongly): take one marked ours or
            # side-less first, a "red" one only if there is nothing else.
            hits.sort(key=lambda tr: (tr.team == "red", tr.first_seen))
            if hits:
                first = hits[0]
                p.pos = first.landing_pos()
                p.units = [tr.id for tr in hits if tr.first_seen - first.first_seen < SPELL_SAME_PLAY[0]
                           and math.dist(tr.landing_pos(), p.pos) < SPELL_SAME_PLAY[1]]  # every wave is ours
                used.update(p.units)
            continue
        unit = UNIT_OF_CARD.get(p.card)
        count = {"archers": 2, "minions": 3}.get(p.card, 1)
        new = [tr for tr in tracks if tr.team == "blue" and tr.cls == unit and tr.id not in used
               and p.t + delay - 0.5 < tr.first_seen < p.t + delay + 1.5]
        new = new[:count]
        if new:
            used.update(tr.id for tr in new)
            p.pos = (statistics.fmean(tr.landing_pos()[0] for tr in new),
                     statistics.fmean(tr.landing_pos()[1] for tr in new))
            p.units = [tr.id for tr in new]


def enemy_candidates(tracks, own_unit_ids, delay=1.0, spell_speed=None):
    """Possible opponent plays: new red units appearing on their half (we don't see their hand or
    blueprints), grouped by kind and moment; red spell hits for their spells. Whether a candidate is a
    real new play is decided during the replay, against the simulated battle (see judge_enemy)."""
    out = []
    taken = set()
    tracks_by_id = {tr.id: tr for tr in tracks}
    # Spell hits without a side ("any") that none of our spells claimed are theirs.
    red = [tr for tr in tracks if (tr.team == "red" or (tr.team == "any" and tr.cls in SPELL_HITS))
           and tr.id not in own_unit_ids]
    for tr in red:
        if tr.id in taken:
            continue
        x, y = tr.landing_pos()
        if tr.cls.endswith("_hit"):
            spell = tr.cls[:-4]
            taken.add(tr.id)
            same = [p for p in out if p.card == spell and p.source == "hit"
                    and abs(tracks_by_id[p.units[0]].first_seen - tr.first_seen) < SPELL_SAME_PLAY[0]
                    and math.dist(p.pos, (x, y)) < SPELL_SAME_PLAY[1]]
            if same:
                same[0].units.append(tr.id)  # another wave of the same arrows
                continue
            flight = math.dist((x, y), KING["red"]) / (spell_speed or {}).get(spell, 10.0)
            out.append(Play("red", spell, tr.first_seen - flight - delay, (x, y), "hit", [tr.id]))
            continue
        card = CARD_OF_UNIT.get(tr.cls)
        if card is None or not 15.5 <= y <= 31.6:
            continue  # a spell in flight, or first seen where they can't deploy: not a fresh unit
        if any(o.cls == tr.cls and o.last_seen < tr.first_seen < o.last_seen + 4.0
               and math.dist((o.history[-1][1], o.history[-1][2]), (x, y)) < 4.0 for o in red if o is not tr):
            continue  # most likely a unit the tracker lost and found again
        count = {"archers": 2, "minions": 3}.get(card, 1)
        group = [tr] + [o for o in red if o is not tr and o.id not in taken and o.cls == tr.cls
                        and abs(o.first_seen - tr.first_seen) < 0.6
                        and math.dist(o.landing_pos(), (x, y)) < 3.0][:count - 1]
        taken.update(o.id for o in group)
        pos = (statistics.fmean(o.landing_pos()[0] for o in group), statistics.fmean(o.landing_pos()[1] for o in group))
        out.append(Play("red", card, tr.first_seen - delay, pos, "units", [o.id for o in group]))
    # One unit sometimes gets two labels at once (a red mini pekka partly seen as minions): candidates
    # of different cards at the same spot and moment are one play; keep the better-seen one.
    hits = {tr.id: tr.hits for tr in red}
    out.sort(key=lambda p: -sum(hits.get(i, 0) for i in p.units))
    kept = []
    for p in out:
        if not any(q.card != p.card and q.source == p.source == "units" and abs(q.t - p.t) < 0.6
                   and math.dist(q.pos, p.pos) < 2.0 for q in kept):
            kept.append(p)
    return sorted(kept, key=lambda p: p.t)


def judge_enemy(battle, data, cand, accepted, same_unit_radius=3.5, cycle_gap=3, seen_elsewhere=()):
    """Is this candidate a new opponent play? The simulated battle knows what is already on the field:
    a "new" mini pekka next to one the simulator still has walking there is that same unit seen again.
    And a card comes back to the hand only after the others cycle: the same card again within
    cycle_gap of their plays is not possible (a play of theirs we missed makes this a little loose).
    seen_elsewhere: positions of other units of that kind on screen right now - a simulated unit standing
    on one of those is that unit, so it doesn't explain the candidate (a second giant next to the first)."""
    if cand.source == "units":
        near = sum(1 for e in battle.entities
                   if not e.is_building and e.team == crsim.Team.RED and e.def_id == cand.card
                   and math.dist((e.pos.x, e.pos.y), cand.pos) < same_unit_radius
                   and not any(math.dist((e.pos.x, e.pos.y), q) < 1.2 for q in seen_elsewhere))
        near += sum(1 for d in battle.pending if d.team == crsim.Team.RED and d.unit >= 0
                    and d.card_index == data.card(cand.card).index
                    and math.dist((d.pos.x, d.pos.y), cand.pos) < same_unit_radius)
        if near >= len(cand.units):
            return "already in the simulator"
    if cand.card in [p.card for p in accepted[-cycle_gap:]]:
        return "card not back in hand yet"
    return None


# --- 3. replay -----------------------------------------------------------------------------------
def tower_hp(battle):
    out = {}
    for team, tname in ((crsim.Team.BLUE, "blue"), (crsim.Team.RED, "red")):
        king = battle.find(battle.king_tower(team))
        out[TOWER_NAMES[(tname, "king")]] = king.hp if king else 0
        for side in (0, 1):
            e = battle.find(battle.princess_tower(team, side))
            out[TOWER_NAMES[(tname, side)]] = e.hp if e else 0
    return out


def snapshot(battle):
    units = []
    for e in battle.entities:
        if e.is_building:
            continue
        units.append({"id": e.id, "cls": UNIT_OF_CARD.get(e.def_id, e.def_id),
                      "team": "blue" if e.team == crsim.Team.BLUE else "red",
                      "x": e.pos.x, "y": e.pos.y, "hp": e.hp, "max_hp": e.max_hp,
                      "deploying": e.deploy_ms_left > 0})
    spells = [{"team": "blue" if p.team == crsim.Team.BLUE else "red", "x": p.pos.x, "y": p.pos.y}
              for p in battle.projectiles if p.is_spell and p.delay_ms <= 0]
    return {"g": battle.time_ms / 1000.0, "units": units, "spells": spells, "towers": tower_hp(battle),
            "elixir": battle.player(crsim.Team.BLUE).elixir, "red_elixir": battle.player(crsim.Team.RED).elixir}


def replay(own, candidates, frames, start, data=None, deck=DEFAULT_DECK):
    """Runs our plays and the accepted opponent candidates through a fresh battle, in time order.
    Returns per cache frame the simulated state (None before the start) and, per play (by id()), the
    elixir the simulator thinks was missing; rejected candidates get .rejected = the reason."""
    data = data or load_data()
    battle = crsim.Battle(data, deck, deck, 0)
    tick = data.rules.tick_ms / 1000.0
    todo = sorted(own + candidates, key=lambda p: p.t)
    accepted_red = []
    missing = {}
    states = []
    k = 0
    for fr in frames:
        g = fr["t"] - start
        while battle.time_ms / 1000.0 + tick <= g + 1e-9 and not battle.finished:
            now = battle.time_ms / 1000.0
            while k < len(todo) and todo[k].t - start <= now:
                p = todo[k]
                k += 1
                if p.team == "red":
                    p.rejected = judge_enemy(battle, data, p, accepted_red)
                    if p.rejected:
                        continue
                    accepted_red.append(p)
                team = crsim.Team.BLUE if p.team == "blue" else crsim.Team.RED
                missing[id(p)] = battle.force_play(team, p.card, crsim.Vec2(*p.pos) if p.pos else None)
            battle.step()
        states.append(snapshot(battle) if g >= 0 else None)
    return states, missing


# --- 4. compare ----------------------------------------------------------------------------------
def match_units(sim_units, real_tracks, max_dist=3.0, prev=None, keep_bonus=1.0, hp_weight=2.0):
    """Pairs (sim unit, real track state) of the same kind and side, chosen together per kind and side so the
    total distance is smallest: closest-first pairing swaps look-alikes (three minions crossing). A pair kept
    from the last frame (prev: sim key -> track id) counts keep_bonus tiles closer, so identities stay put;
    a unit left unpaired costs max_dist (so does any pair farther than that: never made). Where a track's HP
    bar is read ("hp_frac"), a difference in HP counts too, hp_weight tiles for all of it: of two minions side by
    side, the one the tower hit in the game goes with the one it hit in the simulator (video32: the same two
    minions at 50% and 5%, paired the wrong way round, showed as +45% HP errors)."""
    groups = {}
    for i, s in enumerate(sim_units):
        groups.setdefault((s["cls"], s["team"]), ([], []))[0].append(i)
    for j, r in enumerate(real_tracks):
        groups.setdefault((r["cls"], r["team"]), ([], []))[1].append(j)
    out = []
    for si, rj in groups.values():
        if not si or not rj:
            continue
        dist = {(i, j): math.dist((sim_units[i]["x"], sim_units[i]["y"]), (real_tracks[j]["x"], real_tracks[j]["y"]))
                for i in si for j in rj}
        cost = {}
        for (i, j), d in dist.items():
            if d <= max_dist:
                key = sim_units[i].get("key")
                kept = prev is not None and key is not None and prev.get(key) == real_tracks[j].get("id")
                cost[(i, j)] = d - (keep_bonus if kept else 0.0)
                frac, max_hp = real_tracks[j].get("hp_frac"), sim_units[i].get("max_hp")
                if frac is not None and max_hp:
                    cost[(i, j)] += hp_weight * abs(sim_units[i]["hp"] / max_hp - frac)
        if len(si) > 8 or len(rj) > 12:  # (never in this deck) too many to try every way: closest first
            used_s, used_r = set(), set()
            for (i, j), c in sorted(cost.items(), key=lambda kv: kv[1]):
                if i not in used_s and j not in used_r:
                    used_s.add(i)
                    used_r.add(j)
                    out.append((i, j, dist[(i, j)]))
            continue
        memo = {}

        def best(a, used):  # the cheapest way to pair sim units si[a:] with tracks not in `used`
            if a == len(si):
                return 0.0, ()
            if (a, used) in memo:
                return memo[(a, used)]
            c0, rest0 = best(a + 1, used)
            result = (c0 + max_dist, rest0)  # this one unpaired
            for b, j in enumerate(rj):
                if not used >> b & 1 and (si[a], j) in cost:
                    c, rest = best(a + 1, used | 1 << b)
                    c += cost[(si[a], j)]
                    if c < result[0]:
                        result = (c, ((si[a], j),) + rest)
            memo[(a, used)] = result
            return result

        out.extend((i, j, dist[(i, j)]) for i, j in best(0, 0)[1])
    return out


def bars_from_cache(frame):
    from perception.unit_hp import Bar
    return [Bar(b[0], tuple(b[1:5]), b[5], b[6], 1.0) for b in frame.get("bars", [])]


def seen_from_cache(frame):
    """A perception cache frame in the shape ScreenReader.read() returns (for the live code)."""
    return {
        "hand": [{"card": None if c in (None, "?") else c, "unknown": c == "?"} for c in frame["hand"]],
        "next": {"card": frame["next"]},
        "elixir": frame["elixir"],
        "timer": {"seconds": frame["timer"][0], "overtime": frame["timer"][1]},
        "elixir_multiplier": frame["mult"],
        "towers": {k: tuple(v) for k, v in frame["towers"].items()},
    }
