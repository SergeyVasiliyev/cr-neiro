"""The way a troop walks from where it is put down, alone, in the recordings vs the simulator: the same troop put
down on the same spot in an empty simulated battle, compared 1..6 s after it lands - how far on its way (forward)
and how far off to the side (towards the bridge it heads for) each one is.

    .venv\\Scripts\\python.exe tools\\solo_path_check.py [caches...]   (default: every raw/perception/cache/video*.json)

Tracks start where a troop is first seen on its own half (it lands there) and count only while no other unit of its
side (but its own group's) is within 2 tiles and no enemy troop within 7.5, so collisions and fights stay out of it.
Groups (archers, minions): one of them.
"""
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from speed_check import CARD, tracks_of  # noqa: E402

TIMES = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
ALONE = 2.0
NO_ENEMY = 7.5  # nor any enemy troop it could see (sight ranges are 5.5-7.5): it would turn to it or stop and shoot


def solo_tracks(frames):
    """(cls, team, [(s since landing, x, y)]) for troops landing on their own half, cut where they stop being alone."""
    out = []
    by_t = {f["t"]: [(d[0], d[1], d[7], d[8]) for d in f.get("det", []) if d[2] >= 0.4] for f in frames}
    for cls in CARD:
        for team in ("blue", "red"):
            for tr in tracks_of(frames, cls, team):
                t0, x0, y0 = tr[0]
                own = 3.0 < y0 < 14.5 if team == "blue" else 17.5 < y0 < 29.0
                if not own or len(tr) < 15:
                    continue
                kept = []
                for t, x, y in tr:
                    crowd = any(math.dist((x, y), (ox, oy)) < (ALONE if s == team else NO_ENEMY)
                                for c, s, ox, oy in by_t.get(t, []) if not (c == cls and s == team))
                    if crowd or t - t0 > TIMES[-1] + 0.3:
                        break
                    kept.append((t - t0, x, y))
                if kept and kept[-1][0] >= 2.0:
                    out.append((cls, team, kept))
    return out


def landing(track):
    settle = [p for p in track if 0.3 <= p[0] <= 0.7] or track[:3]
    return statistics.median(p[1] for p in settle), statistics.median(p[2] for p in settle)


def sim_walk(data, card, team, pos):
    """Positions of the troop put down at pos, at TIMES after landing."""
    b = crsim.Battle(data, ce.DEFAULT_DECK, ce.DEFAULT_DECK, 0)
    side = crsim.Team.BLUE if team == "blue" else crsim.Team.RED
    uid = b.twin_spawn(side, card, crsim.Vec2(*pos))
    b.twin_set_hp(uid, b.find(uid).max_hp)
    deploy = data.card(card).unit.deploy_time_ms / 1000  # twin_spawn skips the deploy timer: count it in
    out, k = {}, 0
    for t in TIMES:
        while deploy + k * 0.05 < t:
            b.step()
            k += 1
        e = b.find(uid)
        out[t] = (e.pos.x, e.pos.y) if e is not None else None
    return out


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json")))
    data = ce.load_data()
    a = data.arena
    rows = defaultdict(lambda: defaultdict(list))  # (cls, region) -> t -> (forward, toward bridge, off)
    n_tracks = defaultdict(int)
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        for cls, team, tr in solo_tracks(frames):
            lx, ly = landing(tr)
            own_y = ly if team == "blue" else a.height - ly
            region = "back (y<7)" if own_y < 7 else "middle (7-11)" if own_y < 11 else "front (11+)"
            sim = sim_walk(data, CARD[cls], team, (lx, ly))
            fwd = 1 if team == "blue" else -1
            bx = min(a.bridge_x, key=lambda x: abs(x - lx))
            n_tracks[(cls, region)] += 1
            for t in TIMES:
                near = [p for p in tr if abs(p[0] - t) <= 0.15]
                if not near or sim[t] is None:
                    continue
                gx, gy = near[len(near) // 2][1:]
                sx, sy = sim[t]
                # + : the game's troop further forward / closer to the bridge's line than the simulator's
                rows[(cls, region)][t].append(((gy - sy) * fwd, abs(sx - bx) - abs(gx - bx), math.dist((gx, gy), (sx, sy))))
    print(f"{len(files)} recordings: alone troops from where they land, game vs simulator (median tiles; n tracks)")
    print("  forward + : the game's troop is further on its way;  side + : the game's is closer to the bridge's line")
    for key in sorted(rows):
        cols = []
        for t in TIMES:
            v = rows[key][t]
            if len(v) >= 3:
                cols.append(f"{t:.0f}s fwd {statistics.median(r[0] for r in v):+5.2f} side "
                            f"{statistics.median(r[1] for r in v):+5.2f}")
        if cols:
            print(f"  {key[0]:11s} {key[1]:14s} n={n_tracks[key]:3d}  " + " | ".join(cols))
    alls = defaultdict(list)
    for key, by_t in rows.items():
        for t, v in by_t.items():
            alls[(key[1], t)] += v
    print("all kinds:")
    for region in ("back (y<7)", "middle (7-11)", "front (11+)"):
        print(f"  {region:14s} " + " | ".join(
            f"{t:.0f}s fwd {statistics.median(r[0] for r in alls[(region, t)]):+5.2f} side "
            f"{statistics.median(r[1] for r in alls[(region, t)]):+5.2f} (n {len(alls[(region, t)])})"
            for t in TIMES if len(alls[(region, t)]) >= 5))


if __name__ == "__main__":
    main()
