"""Measures how fast things move in the real game, from the perception caches, to check the simulator.

    .venv\\Scripts\\python.exe tools\\measure_speeds.py [--caches "raw/perception/cache/video?.json"]

Needs caches from tools/perceive_video.py (10 frames/s).

Walking speed: the tracker follows every unit; over each ~1 s stretch where a unit was seen at both
ends, its displacement in tiles gives a speed. Units stop to fight, so the typical walking speed is
the upper part of that distribution (75th / 90th percentile of the moving stretches).

Spell flight: our own spells are played from the hand (the moment the card leaves its slot) and seen
landing (the hit appears); (hit - play) = delay + distance / speed, so a line fit over spells cast at
different distances gives both the speed and the fixed delay (blueprint second etc.).
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402

SIM_CARD = {"knight": "knight", "archer": "archers", "minion": "minions", "giant": "giant",
            "mini_pekka": "mini_pekka", "musketeer": "musketeer"}


def walk_speeds(track, window=2.5, min_points=15, max_resid=0.25):
    """Speeds (tiles/s) of straight walks: a line fitted to x(t) and y(t) over each ~window-second
    stretch of seen positions. Two-point differences are biased up by the detector's jitter (a few
    tenths of a tile), most for slow units; a fit over many points is not. Stretches that are not
    straight (turning, pushed, fighting) are skipped."""
    h = track.history
    out = []
    for i in range(0, len(h), 5):
        pts = [p for p in h[i:] if p[0] - h[i][0] <= window]
        if len(pts) < min_points or pts[-1][0] - pts[0][0] < 0.8 * window:
            continue
        ts = [p[0] for p in pts]
        fx = statistics.linear_regression(ts, [p[1] for p in pts])
        fy = statistics.linear_regression(ts, [p[2] for p in pts])
        resid = [math.hypot(p[1] - (fx.intercept + fx.slope * p[0]), p[2] - (fy.intercept + fy.slope * p[0]))
                 for p in pts]
        if statistics.fmean(r * r for r in resid) ** 0.5 > max_resid:
            continue
        out.append(math.hypot(fx.slope, fy.slope))
    return out


def sim_numbers():
    with open(os.path.join(ROOT, "data", "cards.json"), encoding="utf-8") as f:
        cards = json.load(f)
    cards = cards.get("cards", cards)
    with open(os.path.join(ROOT, "data", "overrides.json"), encoding="utf-8") as f:
        over = json.load(f).get("cards", {})
    walk = {cls: cards[c]["unit"]["speed"] / 60.0 for cls, c in SIM_CARD.items()}
    fly = {s: over.get(s, {}).get("spell", {}).get("projectile_speed", cards[s]["spell"]["projectile_speed"])
           for s in twin.SPELL_CARDS}
    return walk, fly


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caches", default=os.path.join(ROOT, "raw", "perception", "cache", "video?.json"))
    args = p.parse_args()
    sim_walk, sim_fly = sim_numbers()

    speeds, flights = {}, {}
    for path in sorted(glob.glob(args.caches)):
        frames = twin.load_cache(path)["frames"]
        tracks = twin.run_tracker(frames)
        for tr in tracks:
            if tr.cls in sim_walk:
                speeds.setdefault((tr.cls, tr.team), []).extend(walk_speeds(tr))
        # our spells: played (hand) -> hit seen
        used = set()
        for play in twin.own_plays_from_hand(frames):
            if play.card not in twin.SPELL_CARDS:
                continue
            hits = [tr for tr in tracks if tr.team == "blue" and tr.cls == play.card + "_hit" and tr.id not in used
                    and play.t + 0.8 < tr.first_seen < play.t + 4.5]
            if not hits:
                continue
            hit = hits[0]
            used.add(hit.id)
            dist = math.dist(hit.history[0][1:], twin.KING["blue"])
            flights.setdefault(play.card, []).append((dist, hit.first_seen - play.t))
        print(f"{os.path.basename(path)}: {len(tracks)} tracks", flush=True)

    print("\nwalking speed, tiles/s     real p75   real p90   simulator   stretches")
    for (cls, team) in sorted(speeds):
        moving = sorted(v for v in speeds[(cls, team)] if v > 0.4)
        if len(moving) < 20:
            continue
        q = lambda f: moving[min(len(moving) - 1, int(f * len(moving)))]  # noqa: E731
        print(f"  {cls:11s} {team:4s}                  {q(0.5):6.2f}       {q(0.75):6.2f}      {sim_walk[cls]:5.2f}     {len(moving)}")

    print("\nour spells: hit time after playing the card vs distance from our king tower")
    for spell, rows in sorted(flights.items()):
        print(f"  {spell}: {len(rows)} casts, distances {min(r[0] for r in rows):.1f}..{max(r[0] for r in rows):.1f} tiles")
        if len(rows) >= 4 and statistics.pvariance([r[0] for r in rows]) > 1.0:
            slope, delay = statistics.linear_regression([r[0] for r in rows], [r[1] for r in rows])
            resid = [r[1] - (delay + slope * r[0]) for r in rows]
            print(f"    fit: speed {1 / slope:5.1f} tiles/s (simulator {sim_fly[spell]:.1f}), delay {delay:.2f}s, "
                  f"spread {statistics.pstdev(resid):.2f}s")
            sim_t = [1.0 + r[0] / sim_fly[spell] for r in rows]
            print(f"    median real {statistics.median(r[1] for r in rows):.2f}s vs simulator "
                  f"{statistics.median(sim_t):.2f}s (1 s blueprint + flight)")


if __name__ == "__main__":
    main()
