"""A walking test recording (each troop put in front of its own princess tower, walking alone to the enemy tower
and hitting it) against the same walk in the simulator: when it starts, where it is along the way, when it
arrives, when it first hits.

    .venv\\Scripts\\python.exe tools\\walk_test_check.py raw/perception/cache/video32.json raw/perception/cache/video32_red.json
Each recording shows its own side's troops at the bottom ("blue"): both are used, so both sides' walks count.
Times are from the first sight of the troop (it lands; the game draws it dropping in for ~0.3 s).
"""
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from first_hit_timeline import steady_drops  # noqa: E402
from speed_check import CARD, tracks_of  # noqa: E402


def episodes(frames):
    """(kind, track) of own troops that land on their own half and walk to the other one."""
    out = []
    for cls in CARD:
        for tr in tracks_of(frames, cls, "blue"):
            if len(tr) < 15 or tr[0][2] > 13.5:
                continue
            if max(p[2] for p in tr) < 18.0 or tr[-1][0] - tr[0][0] < 3.0:
                continue
            out.append((cls, tr))
    return out


def walked_at(points, t0, sx, sy, dist=0.3):
    return next((p[0] - t0 for p in points if p[0] - t0 > 0.5 and math.dist(p[1:], (sx, sy)) > dist), None)


def simulate(data, card, pos, seconds):
    """Positions (s since it landed, x, y) and the first tower damage time, of one troop played at pos."""
    b = crsim.Battle(data, ce.DEFAULT_DECK, ce.DEFAULT_DECK, 0)
    hp0 = {e.id: e.hp for e in b.entities if e.is_building}
    b.force_play(crsim.Team.BLUE, card, crsim.Vec2(*pos))
    land = None  # when it appears on the field (after the blueprint second): the game's first sight
    path, first_hit = [], None
    for k in range(int((seconds + 2.0) / 0.05)):
        b.step()
        units = [e for e in b.entities if not e.is_building and e.team == crsim.Team.BLUE]
        if units and land is None:
            land = b.time_ms / 1000.0
        if land is None:
            continue
        t = b.time_ms / 1000.0 - land
        if units:
            u = units[0]
            path.append((t, u.pos.x, u.pos.y))
        if first_hit is None and any(e.hp < hp0.get(e.id, e.hp) for e in b.entities if e.is_building):
            first_hit = t
    return path, first_hit


def main():
    data = ce.load_data()
    rows = []
    for path in sys.argv[1:]:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        drops = {tw: steady_drops(frames, tw) for tw in ("red_left", "red_right", "red_king")}
        for cls, tr in episodes(frames):
            t0 = tr[0][0]
            settle = [p for p in tr if 0.4 <= p[0] - t0 <= 0.8] or tr[:3]
            sx, sy = statistics.median(p[1] for p in settle), statistics.median(p[2] for p in settle)
            g_walk = walked_at(tr, t0, sx, sy)
            # arrival: first time it stays within 0.2 tiles for 0.6 s, on the enemy half
            g_arrive = None
            for k, p in enumerate(tr):
                nxt = [q for q in tr[k:] if q[0] - p[0] <= 0.6]
                if p[2] > 17 and nxt[-1][0] - p[0] >= 0.5 and max(math.dist(p[1:], q[1:]) for q in nxt) < 0.2:
                    g_arrive = p[0] - t0
                    break
            tower = "red_left" if sx < 9 else "red_right"
            g_hit = next((d[0] - t0 for d in drops[tower] if d[0] > t0 + 1.0 and d[0] - t0 < 25), None)
            spath, s_hit = simulate(data, CARD[cls], (sx, sy), (tr[-1][0] - t0) + 3)
            s_walk = walked_at([(t, x, y) for t, x, y in spath], 0.0, sx, sy)
            s_arrive = None
            for k, p in enumerate(spath):
                nxt = [q for q in spath[k:] if q[0] - p[0] <= 0.6]
                if p[2] > 17 and nxt[-1][0] - p[0] >= 0.5 and max(math.dist(p[1:], q[1:]) for q in nxt) < 0.2:
                    s_arrive = p[0]
                    break
            # along the way: y (the walk is up the lane) at the same time since it landed (a group's units push
            # each other apart while deploying, so "started walking" is not a clean mark for them)
            diffs = []
            for t, x, y in tr:
                tt = t - t0
                if 1.5 <= tt and (g_arrive is None or tt < g_arrive) and y < 25:
                    sp = min(spath, key=lambda p: abs(p[0] - tt))
                    if abs(sp[0] - tt) < 0.1:
                        diffs.append((y - sp[2], x - sp[1]))
            rows.append((os.path.basename(path), cls, (sx, sy), g_walk, s_walk, g_arrive, s_arrive, g_hit, s_hit, diffs))
    fmt = lambda v: f"{v:5.2f}" if v is not None else "  -  "  # noqa: E731
    print(f"{'rec':16s} {'troop':10s} {'landed':>11s} | start walk game/sim | arrive game/sim | 1st hit game/sim | "
          f"y game-sim (median, max)  x game-sim")
    for rec, cls, (sx, sy), gw, sw, ga, sa, gh, sh, diffs in rows:
        dy = f"{statistics.median(d[0] for d in diffs):+.2f} {max((d[0] for d in diffs), key=abs):+.2f}" if diffs else "-"
        dx = f"{statistics.median(d[1] for d in diffs):+.2f}" if diffs else "-"
        print(f"{rec:16s} {cls:10s} ({sx:4.1f},{sy:4.1f}) |   {fmt(gw)} / {fmt(sw)}    | {fmt(ga)} / {fmt(sa)}   | "
              f"{fmt(gh)} / {fmt(sh)}    | {dy:>14s}   {dx}")
    print("y game - sim > 0: the game's troop is further up its way than the simulator's, at the same time since "
          "it landed (start walk: s to 0.3 tiles from where it landed; groups push apart while deploying)")


if __name__ == "__main__":
    main()
