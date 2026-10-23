"""Test recordings, one test at a time: every stretch of a two-sided recording between empty fields is played
again in a fresh simulated battle (same cards, same moments, same spots, towers as they were), and each unit's
way is drawn and measured against its way in the game.

    .venv\\Scripts\\python.exe tools\\test_replay.py --match video36 [--match video37 ...]

Pictures: raw/test_replay/<match>_<n>.png - the arena, blue at the bottom; each unit's way in the game (thick,
a dot a second) and in the simulator (thin, a ring a second, same colour), a label where it starts.
Numbers: per unit, how far the simulator's unit is from the game's 1, 2, 3, 5 and 8 s after it appears, and
the median over its life (forward: + the game's unit further on its way; side: + the game's further right).
"""
import argparse
import math
import os
import statistics
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import twin  # noqa: E402
from crsim_env import DEFAULT_DECK, load_data  # noqa: E402

W, H = 18.0, 32.0
BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
PX = 26  # pixels per tile
COLORS = {"knight": (60, 200, 255), "archer": (255, 120, 220), "minion": (255, 200, 60), "giant": (60, 120, 255),
          "mini_pekka": (255, 90, 90), "musketeer": (120, 255, 120)}
UNIT = {"archers": "archer", "minions": "minion"}
CHECK = (1.0, 2.0, 3.0, 5.0, 8.0)


def own_plays(frames, flip):
    """A player's troop and spell plays with their units' game tracks, in battle s and blue's tiles."""
    start, _ = twin.battle_start(frames)
    tracks = twin.run_tracker(frames)
    by_id = {tr.id: tr for tr in tracks}
    plays = twin.own_plays_from_hand(frames)
    twin.place_own_plays(plays, tracks)

    def tile(x, y):
        return (W - x, H - y) if flip else (x, y)
    out = []
    for p in plays:
        seen = [by_id[u] for u in p.units if u in by_id]
        if p.pos is None:
            continue
        t = (min(tr.first_seen for tr in seen) - 1.0) if seen and p.card not in twin.SPELL_CARDS else p.t
        if p.card in twin.SPELL_CARDS and seen:
            t = min(tr.first_seen for tr in seen) - 1.0 - 0.5  # roughly: its flight from the king tower
        ways = [[(ht - start, *tile(hx, hy)) for ht, hx, hy in tr.history] for tr in seen
                if p.card not in twin.SPELL_CARDS]
        out.append({"t": t - start, "card": p.card, "pos": tile(*p.pos), "ways": ways})
    return out, start


def tower_states(frame):
    return {name: hp for name, (hp, _) in frame["towers"].items()}


def segments(plays, gap=3.0, tail=10.0):
    """Plays in groups split where none follows for gap s; each group runs until the next one starts (or tail s
    after its last play). Units of earlier plays still on the field come along ("carried"), from where they
    are in the game then."""
    plays = sorted(plays, key=lambda p: p["t"])
    starts = [0] + [k for k in range(1, len(plays)) if plays[k]["t"] - plays[k - 1]["t"] > gap]
    groups = []
    for n, k in enumerate(starts):
        last = starts[n + 1] if n + 1 < len(starts) else len(plays)
        t0 = plays[k]["t"] - 0.5
        t1 = min(plays[last]["t"] - 0.5 if last < len(plays) else 1e9, plays[last - 1]["t"] + tail)
        group = []
        for q in plays[:k]:  # carried: alive in the game at t0
            ways = [[pt for pt in w if t0 <= pt[0] <= t1] for w in q["ways"] if w and w[0][0] < t0 <= w[-1][0]]
            ways = [w for w in ways if len(w) > 3]
            if ways and q["card"] not in twin.SPELL_CARDS:
                group.append(dict(q, ways=ways, carried=True))
        for q in plays[k:last]:
            group.append(dict(q, ways=[[pt for pt in w if pt[0] <= t1] for w in q["ways"]], carried=False))
        groups.append((t0, t1, group))
    return groups


def set_towers(battle, hp):
    for team, tname in ((BLUE, "blue"), (RED, "red")):
        ids = {"king": battle.king_tower(team), 0: battle.princess_tower(team, 0), 1: battle.princess_tower(team, 1)}
        for key, eid in ids.items():
            value = hp.get(twin.TOWER_NAMES[(tname, key)])
            if value is not None and battle.find(eid) is not None:
                battle.twin_set_hp(eid, max(0, int(value)))


def replay(data, t0, t1, group, towers):
    """The group in a fresh battle from t0 to t1: carried units put where they are in the game, the plays made
    at their moments. {group index: [sim way per unit]} (battle s, x, y)."""
    battle = crsim.Battle(data, DEFAULT_DECK, DEFAULT_DECK, 0)
    battle.set_follow_real(True)
    set_towers(battle, towers)
    battle.step()
    owner = {}
    for i, q in enumerate(group):
        if q["carried"]:
            for w in q["ways"]:
                uid = battle.twin_spawn(q["team"], q["card"], crsim.Vec2(w[0][1], w[0][2]))
                owner[uid] = i
    pending = [(i, q) for i, q in enumerate(group) if not q["carried"]]
    ways, k = {}, 0
    base = battle.time_ms / 1000.0
    while t0 + battle.time_ms / 1000.0 - base < t1:
        g = t0 + battle.time_ms / 1000.0 - base
        while k < len(pending) and pending[k][1]["t"] <= g:
            q = pending[k][1]
            battle.force_play(q["team"], q["card"], crsim.Vec2(*q["pos"]))
            k += 1
        battle.step()
        g = t0 + battle.time_ms / 1000.0 - base
        for e in battle.entities:
            if e.is_building:
                continue
            if e.id not in owner:
                for j in range(k - 1, -1, -1):
                    i, q = pending[j]
                    if q["card"] == e.def_id and q["team"] == e.team:
                        owner[e.id] = i
                        break
            if e.id in owner and e.hp > 0 and battle.time_ms % 100 == 0:
                ways.setdefault(e.id, []).append((g, e.pos.x, e.pos.y))
    out = {}
    for eid, way in ways.items():
        out.setdefault(owner[eid], []).append(way)
    return out


def at(way, t):
    near = min(way, key=lambda p: abs(p[0] - t))
    return near[1:] if abs(near[0] - t) <= 0.2 else None


def pair_up(game_ways, sim_ways):
    """Each game unit with the simulator's unit closest to it at the start."""
    pairs, used = [], set()
    for gw in game_ways:
        best = None
        for j, sw in enumerate(sim_ways):
            if j in used:
                continue
            p = at(sw, gw[0][0])
            d = math.dist(p, gw[0][1:]) if p else 1e9
            if best is None or d < best[0]:
                best = (d, j)
        if best is not None and best[0] < 3.0:
            used.add(best[1])
            pairs.append((gw, sim_ways[best[1]]))
    return pairs


def draw(path, group, sims, title):
    img = np.full((int(H * PX) + 40, int(W * PX), 3), (60, 110, 60), np.uint8)

    def px(x, y):
        return int(x * PX), int((H - y) * PX) + 40
    cv2.rectangle(img, px(0, 17), px(W, 15), (160, 110, 40), -1)  # river
    for bx in (3.5, 14.5):
        cv2.rectangle(img, px(bx - 1, 17), px(bx + 1, 15), (90, 140, 170), -1)
    for (x, y, r) in ((3.5, 6.5, 1.5), (14.5, 6.5, 1.5), (9, 3, 2), (3.5, 25.5, 1.5), (14.5, 25.5, 1.5), (9, 29, 2)):
        cv2.rectangle(img, px(x - r, y + r), px(x + r, y - r), (130, 130, 130), -1)
    for gx in range(int(W) + 1):
        cv2.line(img, px(gx, 0), px(gx, H), (70, 125, 70), 1)
    for gy in range(int(H) + 1):
        cv2.line(img, px(0, gy), px(W, gy), (70, 125, 70), 1)
    for i, p in enumerate(group):
        unit = UNIT.get(p["card"], p["card"])
        color = COLORS.get(unit, (255, 255, 255))
        if p["card"] in twin.SPELL_CARDS:
            cv2.circle(img, px(*p["pos"]), int(PX * (2.5 if p["card"] == "fireball" else 3.0)), (0, 140, 255), 2)
            cv2.putText(img, f"{p['card']} {p['t']:.1f}s", px(p["pos"][0] - 1, p["pos"][1] + 0.3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            continue
        for w in p["ways"]:
            pts = [px(x, y) for _, x, y in w]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, color, 3, cv2.LINE_AA)
            for k, (t, x, y) in enumerate(w):
                if k == 0 or int(t) != int(w[k - 1][0]):
                    cv2.circle(img, px(x, y), 4, color, -1)
        for w in sims.get(i, []):
            pts = [px(x, y) for _, x, y in w]
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, (255, 255, 255), 1, cv2.LINE_AA)
            for k, (t, x, y) in enumerate(w):
                if k == 0 or int(t) != int(w[k - 1][0]):
                    cv2.circle(img, px(x, y), 5, color, 1, cv2.LINE_AA)
        side = "B" if p["team"] == BLUE else "R"
        label = (p["ways"][0][0][1:] if p["carried"] and p["ways"] else p["pos"])
        cv2.putText(img, f"{side} {unit} " + ("(was there)" if p["carried"] else f"{p['t']:.1f}s"),
                    px(label[0] + 0.4, label[1] - 0.2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, title, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--match", nargs="+", default=["video36"])
    args = p.parse_args()
    data = load_data()
    out_dir = os.path.join(ROOT, "raw", "test_replay")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(ROOT, "raw", "perception", "cache")
    for match in args.match:
        blue_frames = twin.load_cache(os.path.join(cache, f"{match}.json"))["frames"]
        red_frames = twin.load_cache(os.path.join(cache, f"{match}_red.json"))["frames"]
        blue, start = own_plays(blue_frames, flip=False)
        red, _ = own_plays(red_frames, flip=True)
        plays = [dict(q, team=BLUE) for q in blue] + [dict(q, team=RED) for q in red]
        for n, (t0, t1, group) in enumerate(segments(plays)):
            frame = min(blue_frames, key=lambda f: abs(f["t"] - start - t0))
            sims = replay(data, t0, t1, group, tower_states(frame))
            title = f"{match} test {n}: {t0:.0f}-{t1:.0f}s  game thick / sim thin"
            draw(os.path.join(out_dir, f"{match}_{n:02d}.png"), group, sims, title)
            print(f"\n{match} test {n} ({t0:.1f}-{t1:.1f}s): " + ", ".join(
                f"{'B' if q['team'] == BLUE else 'R'} {q['card']} " +
                ("(on the field)" if q["carried"] else f"{q['t']:.1f}s at ({q['pos'][0]:.1f},{q['pos'][1]:.1f})")
                for q in group))
            for i, q in enumerate(group):
                if q["card"] in twin.SPELL_CARDS:
                    continue
                fwd = 1 if q["team"] == BLUE else -1
                for gw, sw in pair_up(q["ways"], sims.get(i, [])):
                    t0 = gw[0][0]
                    cols, offs = [], []
                    for t, gx, gy in gw:
                        s = at(sw, t)
                        if s:
                            offs.append(math.dist((gx, gy), s))
                    for dt in CHECK:
                        g, s = at(gw, t0 + dt), at(sw, t0 + dt)
                        if g and s:
                            cols.append(f"{dt:.0f}s {math.dist(g, s):.1f} (fwd {(g[1] - s[1]) * fwd:+.1f} "
                                        f"side {(g[0] - s[0]) * fwd:+.1f})")
                    life_g = gw[-1][0] - t0
                    life_s = sw[-1][0] - sw[0][0]
                    if life_g < 1.0:
                        continue
                    print(f"   {'B' if q['team'] == BLUE else 'R'} {UNIT.get(q['card'], q['card']):10s} "
                          f"life game {life_g:4.1f}s sim {life_s:4.1f}s, off median "
                          f"{statistics.median(offs) if offs else float('nan'):.2f}: " + " | ".join(cols))


if __name__ == "__main__":
    main()
