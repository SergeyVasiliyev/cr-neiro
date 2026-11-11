"""Does an enemy troop go for the troop the user just put down - in the game, and in the simulator - and from how far?

For every troop the user put down in their own half, the simulator plays on by itself from that moment (both
players' real plays, nothing pulled back to the screen; as tools/defense_check.py), and every enemy troop near it
that can hit it is followed:
- in the simulator: when its target becomes one of the new troop's units, and the gap then (edge to edge, tiles);
- in the game (the twin, which follows the screen): when it turns towards the new troop - its heading over the next
  0.5 s points at it within --cone degrees, having pointed elsewhere before - or stops to fight next to it.

    .venv\\Scripts\\python.exe tools\\aggro_check.py video8 video8_red ... [--horizon 6]

It prints each pair and then, per enemy troop kind, how often the two agree and the gaps at the turn.
"""
import argparse
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import twin  # noqa: E402
from compare_red_truth import own_plays_timed, true_plays  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import BLUE, TEAM, LiveMatch, Play  # noqa: E402

CACHE = os.path.join(ROOT, "raw", "perception", "cache")


def troops(b, data):
    """{(user's, play index, unit): (card, x, y, radius, target id, locked, id)} of the troops on the field."""
    out = {}
    for e in b.entities:
        if e.is_building or e.hp <= 0 or e.play < 0:
            continue
        card = data.card_ids[e.card_index] if e.card_index >= 0 else "?"
        out[(e.team == BLUE, e.play, e.unit)] = (card, e.pos.x, e.pos.y, e.radius, e.target, e.locked, e.id)
    return out


def heading(track, g, ahead=0.5):
    """(dx, dy, speed) of a troop over [g, g + ahead] from its track [(g, x, y)], None if not followed then."""
    a = next((p for p in track if p[0] >= g - 1e-9), None)
    b = next((p for p in track if p[0] >= g + ahead - 1e-9), None)
    if a is None or b is None or b[0] - a[0] < 0.3 or b[0] - a[0] > ahead + 0.3:
        return None
    dx, dy = b[1] - a[1], b[2] - a[2]
    d = math.hypot(dx, dy)
    return (dx / d, dy / d, d / (b[0] - a[0])) if d > 1e-6 else (0.0, 0.0, 0.0)


def angle(ux, uy, vx, vy):
    n = math.hypot(ux, uy) * math.hypot(vx, vy)
    if n < 1e-9:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / n))))


def check(view, data, args, rows):
    other = view[:-4] if view.endswith("_red") else f"{view}_red"
    frames = twin.load_cache(os.path.join(CACHE, f"{view}.json"))["frames"]
    other_frames = twin.load_cache(os.path.join(CACHE, f"{other}.json"))["frames"]
    own = sorted(p for p in own_plays_timed(frames, data) if p[2] is not None)
    m = LiveMatch(data, red_truth=true_plays(other_frames, data), blue_truth=own)
    pending = [p for p in own if p[1] not in twin.SPELL_CARDS and p[2][1] < 16.0]
    starts, closed = [], []
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0:
            continue
        while pending and pending[0][0] <= m.now + 0.2:
            p = pending.pop(0)
            if p[0] >= m.now - 0.3:
                starts.append((p, m.now, fr["t"], m.battle.clone()))
        closed.append((m.now, troops(m.battle, data)))
    plays = sorted((ev for ev in m.journal if isinstance(ev, Play) and ev.on), key=lambda ev: ev.g)
    for (gp, card, pos), g0, t0, b in starts:
        at_start = troops(b, data)
        cand = {}
        for k, (c, x, y, r, *_rest) in at_start.items():
            if k[0] or math.hypot(x - pos[0], y - pos[1]) > args.near:
                continue
            rules = data.card(c).unit.attack.targets if c in data.cards else None
            if rules is None or rules.buildings_only:
                continue
            if data.card(card).unit.air and not rules.air:
                continue  # it cannot hit the new troop
            cand[k] = c
        if not cand:
            continue
        todo = [ev for ev in plays if ev.g > g0 + 1e-9]
        ahead = [(g, tr) for g, tr in closed if g0 - 1.0 <= g <= g0 + args.horizon + 0.6]
        # the user's buildings (an enemy troop standing next to the new troop may be hitting one of these)
        towers = []
        for tid in (b.king_tower(BLUE), b.princess_tower(BLUE, 0), b.princess_tower(BLUE, 1)):
            tw = b.find(tid)
            if tw is not None:
                towers.append((tw.pos.x, tw.pos.y, tw.radius))
        # tracks in the game (the twin), by key
        track = defaultdict(list)
        for g, tr in ahead:
            for k, v in tr.items():
                track[k].append((g, v[1], v[2]))
        new_keys, sim_hit, k_ev = None, {}, 0
        sim_tracks = defaultdict(list)
        steps = 0
        while b.time_ms / 1000.0 < g0 + args.horizon - 1e-9:
            now = b.time_ms / 1000.0
            while k_ev < len(todo) and todo[k_ev].g <= now + 1e-9:
                ev = todo[k_ev]
                k_ev += 1
                b.force_play(TEAM[ev.side], ev.card, None if ev.pos is None else crsim.Vec2(*ev.pos),
                             max(0, int(round((now - ev.g) * 1000))))
            b.step()
            steps += 1
            sim = troops(b, data)
            if new_keys is None:
                fresh = [k for k, v in sim.items() if k[0] and k not in at_start and v[0] == card]
                if fresh:
                    first = min(k[1] for k in fresh)
                    new_keys = {k for k in fresh if k[1] == first}
            if new_keys is None:
                continue
            ids = {sim[k][6]: k for k in new_keys if k in sim}
            by_id = {v[6]: v for v in sim.values()}
            for k in cand:
                if k in sim:
                    me = sim[k]
                    us = [sim[u] for u in new_keys if u in sim]
                    gap_u = min((math.hypot(u[1] - me[1], u[2] - me[2]) - u[3] - me[3] for u in us), default=None)
                    t = me[4]
                    after = by_id[t][0] if t in by_id else ("tower" if b.find(t) is not None else "-")
                    sim_tracks[k].append((b.time_ms / 1000.0 - g0, after, me[5], gap_u))
                if k in sim and k not in sim_hit and sim[k][4] in ids:
                    u = sim[ids[sim[k][4]]]
                    gap = math.hypot(u[1] - sim[k][1], u[2] - sim[k][2]) - u[3] - sim[k][3]
                    sim_hit[k] = (b.time_ms / 1000.0 - g0, gap)
        if new_keys is None:
            continue
        # in the game: before the new troop, where was the enemy troop heading? then: when does it turn to it?
        for k, c in cand.items():
            before = heading(track[k], g0 - 0.6, 0.5)
            game_hit = None
            for g, tr in ahead:
                if g < g0 or k not in tr:
                    continue
                us = [tr[u] for u in new_keys if u in tr]
                if not us:
                    continue
                me = tr[k]
                u = min(us, key=lambda v: math.hypot(v[1] - me[1], v[2] - me[2]))
                gap = math.hypot(u[1] - me[1], u[2] - me[2]) - u[3] - me[3]
                h = heading(track[k], g, 0.5)
                if h is None:
                    continue
                # the new troop must be the user's troop nearest to it, and no tower of theirs within its reach,
                # or what it went for is not clear
                others = [v for kk, v in tr.items() if kk[0] and kk not in new_keys]
                reach = data.card(c).unit.attack.range + 0.6
                if any(math.hypot(v[1] - me[1], v[2] - me[2]) - v[3] - me[3] < gap for v in others):
                    continue
                if any(math.hypot(x - me[1], y - me[2]) - r - me[3] <= reach for x, y, r in towers):
                    continue
                to_u = (u[1] - me[1], u[2] - me[2])
                walking_before = before is not None and before[2] >= 0.5
                stopped_by_it = walking_before and h[2] < 0.25 and gap <= reach
                towards = h[2] >= 0.5 and angle(h[0], h[1], *to_u) <= args.cone
                was_elsewhere = before is None or before[2] < 0.25 or angle(before[0], before[1], *to_u) > args.cone + 15
                if stopped_by_it or (towards and was_elsewhere):
                    game_hit = (g - g0, gap, "stops by it" if stopped_by_it and not towards else "turns to it")
                    break
            # how close the two came in each (did the simulator's troop ever have it in sight?)
            then = None  # what the simulator's troop was doing when the game's turned
            if game_hit:
                then = next((x for x in sim_tracks[k] if x[0] >= game_hit[0] - 1e-9), None)
            rows.append(dict(view=view, t=t0, g=gp, card=card, enemy=c, game=game_hit, sim=sim_hit.get(k),
                             sight=data.card(c).unit.sight_range, then=then))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("views", nargs="+")
    p.add_argument("--horizon", type=float, default=6.0)
    p.add_argument("--near", type=float, default=9.0, help="enemy troops this close to the new troop are followed")
    p.add_argument("--cone", type=float, default=30.0, help="degrees: heading this close to it counts as towards it")
    p.add_argument("--quiet", action="store_true", help="only the summary")
    args = p.parse_args()
    data = load_data()
    rows = []
    for view in args.views:
        try:
            check(view, data, args, rows)
        except Exception as e:
            print(f"{view}: skipped ({type(e).__name__}: {e})", flush=True)
    why = defaultdict(Counter)  # game only: what the simulator's troop did instead
    for r in rows:
        if r["game"] and not r["sim"] and r["then"]:
            _, after, locked, gap_u = r["then"]
            sight = r["sight"]
            if gap_u is None:
                why[r["enemy"]]["yours not on yet in the simulator"] += 1
            elif locked:
                why[r["enemy"]]["attacking something else"] += 1
            elif gap_u > sight:
                why[r["enemy"]]["yours out of its sight"] += 1
            else:
                why[r["enemy"]][f"after something closer ({after})" if after != "tower" else "after a tower, yours in sight"] += 1
    by = defaultdict(Counter)
    gaps = defaultdict(lambda: defaultdict(list))
    lags = defaultdict(list)
    for r in rows:
        kind = r["enemy"]
        both = "both" if r["game"] and r["sim"] else "game only" if r["game"] else "simulator only" if r["sim"] else "neither"
        by[kind][both] += 1
        if r["game"]:
            gaps[kind]["game"].append(r["game"][1])
        if r["sim"]:
            gaps[kind]["sim"].append(r["sim"][1])
        if r["game"] and r["sim"]:
            lags[kind].append(r["game"][0] - r["sim"][0])
        if not args.quiet and both in ("game only", "simulator only"):
            mins, secs = divmod(r["t"], 60)
            gm = f"game +{r['game'][0]:.1f} s gap {r['game'][1]:.1f} ({r['game'][2]})" if r["game"] else "game: no"
            sm = f"simulator +{r['sim'][0]:.1f} s gap {r['sim'][1]:.1f}" if r["sim"] else "simulator: no"
            if r["then"]:
                _, after, locked, gap_u = r["then"]
                gu = f"{gap_u:.1f}" if gap_u is not None else "-"
                sm += f" (then after {after}{', attacking' if locked else ''}, gap to yours {gu})"
            print(f"{both:15s} {r['view']} video {int(mins)}:{secs:04.1f}: your {r['card']} <- their {kind} | {gm} | {sm}")
    print()
    for kind, cnt in sorted(by.items(), key=lambda kv: -sum(kv[1].values())):
        med = lambda xs: f"{statistics.median(xs):.1f}" if xs else "-"
        lag = f", the game turns {statistics.median(lags[kind]):+.1f} s after the simulator (median)" if lags[kind] else ""
        print(f"their {kind} (sight {data.card(kind).unit.sight_range}): " + ", ".join(f"{k} {v}" for k, v in cnt.items())
              + f" | gap at the turn: game {med(gaps[kind]['game'])}, simulator {med(gaps[kind]['sim'])}{lag}")
        if why[kind]:
            print("    game only - the simulator's troop then: " + ", ".join(f"{k} {v}" for k, v in why[kind].most_common()))


if __name__ == "__main__":
    main()
