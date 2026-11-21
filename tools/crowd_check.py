"""A mini pekka (or any troop) walking behind its own giant: how fast it gets along in the game vs the simulator.

The user (2026-11-21): in the game a mini pekka gets past its giant with more trouble than in the simulator.
From every recording: moments where a troop of --unit walks within --near tiles behind its own walking giant (behind:
on the side it comes from), no enemy troop within --clear tiles. Over the next second, in the game: how far the troop
got, and how far it got along the way relative to the giant (> 0: it is overtaking). The same two units are put at the
same spots in an empty simulated battle (walking, deploy over) and stepped one second.

    .venv\\Scripts\\python.exe tools\\crowd_check.py [--unit mini_pekka] [caches...]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from speed_check import tracks_of  # noqa: E402

CACHE = os.path.join(ROOT, "raw", "perception", "cache")
CARD = {"knight": "knight", "archer": "archers", "minion": "minions", "giant": "giant", "mini_pekka": "mini_pekka",
        "musketeer": "musketeer"}


def default_files():
    """Whole recordings (and their _red sides), parts only where there is no whole one; no clock caches."""
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CACHE, "video*.json")))
    names = [n for n in names if not n.endswith("_clocks")]
    out = []
    for n in names:
        m = re.fullmatch(r"(video\d+)_[bx]\d+(_red)?", n)
        if m and (m.group(1) + (m.group(2) or "")) in names:
            continue
        out.append(os.path.join(CACHE, n + ".json"))
    return out


def at(track, t, tol=0.12):
    best = min(track, key=lambda p: abs(p[0] - t))
    return best if abs(best[0] - t) <= tol else None


def cases(frames, unit, near, clear, horizon=1.0):
    out = []
    for team, way in (("blue", 1.0), ("red", -1.0)):
        giants = [g for g in tracks_of(frames, "giant", team) if len(g) > 10]
        troops = [p for p in tracks_of(frames, unit, team) if len(p) > 10]
        other = "red" if team == "blue" else "blue"
        enemies = {round(f["t"], 2): [(d[7], d[8]) for d in f.get("det", []) if d[1] == other and d[2] >= 0.5
                                      and d[0] in CARD] for f in frames}
        times = sorted(enemies)
        for p in troops:
            next_t = -1.0
            for t0, x0, y0 in p:
                if t0 < next_t:
                    continue
                p1 = at(p, t0 + horizon)
                if p1 is None:
                    continue
                for g in giants:
                    g0, g1 = at(g, t0), at(g, t0 + horizon)
                    if g0 is None or g1 is None:
                        continue
                    gap = math.dist((x0, y0), g0[1:])
                    behind = (g0[2] - y0) * way
                    if gap > near or behind < 0.3 or math.dist(g0[1:], g1[1:]) < 0.5 * horizon:
                        continue
                    # no enemy troop near either over the second
                    window = [enemies[k] for k in times if t0 <= k <= t0 + horizon]
                    if any(math.dist(e, q) < clear for w in window for e in w for q in ((x0, y0), g0[1:], g1[1:])):
                        continue
                    if min(y0, g0[2]) < 1.0 or max(y0, g0[2]) > 31.0:
                        continue
                    out.append(dict(team=team, way=way, p0=(x0, y0), p1=p1[1:], g0=g0[1:], g1=g1[1:],
                                    dt=p1[0] - t0, gdt=g1[0] - g0[0]))
                    next_t = t0 + horizon
                    break
    return out


def sim_case(data, deck, c, unit, seed):
    b = crsim.Battle(data, deck, deck, seed)
    team = crsim.Team.BLUE if c["team"] == "blue" else crsim.Team.RED
    gid = b.twin_spawn(team, "giant", crsim.Vec2(*c["g0"]))
    pid = b.twin_spawn(team, CARD[unit], crsim.Vec2(*c["p0"]))
    b.steps(int(round(c["dt"] * 1000 / data.rules.tick_ms)))
    p, g = b.find(pid), b.find(gid)
    if p is None or g is None:
        return None
    return (p.pos.x, p.pos.y), (g.pos.x, g.pos.y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="*")
    ap.add_argument("--unit", default="mini_pekka", choices=sorted(CARD))
    ap.add_argument("--near", type=float, default=2.0, help="tiles from the giant at most")
    ap.add_argument("--clear", type=float, default=4.0, help="no enemy troop this close")
    ap.add_argument("--horizon", type=float, default=1.0, help="s followed from each moment")
    args = ap.parse_args()
    data = ce.load_data()
    deck = ce.parse_deck(",".join(ce.DEFAULT_DECK), data)
    files = args.caches or default_files()
    rows = []
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        for c in cases(frames, args.unit, args.near, args.clear, args.horizon):
            s = sim_case(data, deck, c, args.unit, len(rows) + 1)
            if s is None:
                continue
            way = c["way"]
            game_v = math.dist(c["p0"], c["p1"]) / c["dt"]
            sim_v = math.dist(c["p0"], s[0]) / c["dt"]
            # along the way relative to the giant (> 0: catching up / overtaking)
            game_rel = ((c["p1"][1] - c["p0"][1]) - (c["g1"][1] - c["g0"][1])) * way
            sim_rel = ((s[0][1] - c["p0"][1]) - (s[1][1] - c["g0"][1])) * way
            game_ahead = (c["p1"][1] - c["g1"][1]) * way > 0
            sim_ahead = (s[0][1] - s[1][1]) * way > 0
            rows.append((os.path.basename(path)[:-5], c, game_v, sim_v, game_rel, sim_rel, game_ahead, sim_ahead))
    if not rows:
        print("no cases")
        return
    med = statistics.median
    print(f"{len(files)} recordings; {len(rows)} stretches of {args.horizon:g} s of a {args.unit} walking within {args.near:g} tiles behind its "
          f"own walking giant, no enemy troop within {args.clear:g} tiles")
    print(f"  its speed (tiles/s), median: game {med(r[2] for r in rows):.2f}, simulator {med(r[3] for r in rows):.2f} "
          f"(walking free: {data.card(CARD[args.unit]).unit.speed:.2f}; the giant {data.card('giant').unit.speed:.2f})")
    print(f"  gained on the giant along the way over the stretch (tiles), median: game {med(r[4] for r in rows):+.2f}, "
          f"simulator {med(r[5] for r in rows):+.2f}")
    print(f"  ahead of the giant at the end: game {sum(r[6] for r in rows) / len(rows):.0%}, "
          f"simulator {sum(r[7] for r in rows) / len(rows):.0%}")
    for lo, hi in ((0.0, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 1.75), (1.75, 2.1)):
        sel = [r for r in rows if lo <= math.dist(r[1]["p0"], r[1]["g0"]) < hi]
        if sel:
            print(f"    starting {lo:g}-{hi:g} tiles from the giant ({len(sel)}): speed game {med(r[2] for r in sel):.2f} "
                  f"sim {med(r[3] for r in sel):.2f}, gained game {med(r[4] for r in sel):+.2f} sim {med(r[5] for r in sel):+.2f}")
    print("  some of them (recording, team, troop -> giant at the start; game / sim after 1 s):")
    for name, c, gv, sv, gr, sr, ga, sa in rows[:12]:
        print(f"    {name} {c['team']}: {args.unit} ({c['p0'][0]:.1f},{c['p0'][1]:.1f}) giant ({c['g0'][0]:.1f},"
              f"{c['g0'][1]:.1f}); speed {gv:.2f} / {sv:.2f}, gained {gr:+.2f} / {sr:+.2f}")


if __name__ == "__main__":
    main()
