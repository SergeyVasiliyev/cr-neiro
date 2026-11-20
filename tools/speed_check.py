"""Walking speeds in the recordings vs the simulator's: per unit kind, the median distance covered per second
while a unit walks freely (tracks of one detection per frame, 1 s windows of steady movement).

    .venv\\Scripts\\python.exe tools\\speed_check.py [caches...]   (default: every raw/perception/cache/video*.json)
"""
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim_env as ce  # noqa: E402

CARD = {"knight": "knight", "archer": "archers", "minion": "minions", "giant": "giant", "mini_pekka": "mini_pekka",
        "musketeer": "musketeer"}


def tracks_of(frames, cls, team):
    """Simple tracks: each detection joins the track whose last position (within 0.4 s) is nearest, < 0.8 tiles."""
    tracks, live = [], []
    for f in frames:
        t = f["t"]
        dets = [(d[7], d[8]) for d in f.get("det", []) if d[0] == cls and d[1] == team and d[2] >= 0.6]
        live = [tr for tr in live if t - tr[-1][0] <= 0.4]
        used = set()
        for x, y in dets:
            best = min((tr for tr in live if id(tr) not in used), default=None,
                       key=lambda tr: math.dist(tr[-1][1:], (x, y)))
            if best is not None and math.dist(best[-1][1:], (x, y)) < 0.8:
                best.append((t, x, y))
                used.add(id(best))
            else:
                tr = [(t, x, y)]
                tracks.append(tr)
                live.append(tr)
                used.add(id(tr))
    return tracks


def speeds(track, window=1.0):
    out = []
    for k, (t0, x0, y0) in enumerate(track):
        later = [p for p in track[k + 1:] if p[0] - t0 >= window]
        if not later:
            break
        t1, x1, y1 = later[0]
        if t1 - t0 > window + 0.25:
            continue
        mid = [p for p in track[k:] if p[0] <= t1]
        path = sum(math.dist(mid[i][1:], mid[i + 1][1:]) for i in range(len(mid) - 1))
        straight = math.dist((x0, y0), (x1, y1))
        if straight > 0.3 and path < straight * 1.3:  # walking, not jittering on the spot or attacking
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            way = "vertical" if dy > 2 * dx else "sideways" if dx > 2 * dy else "diagonal"
            out.append((straight / (t1 - t0), way, (y0 + y1) / 2))
    return out


def main():
    files = sys.argv[1:] or sorted(p for p in glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json"))
                                    if not p.endswith("_clocks.json"))  # (deploy clock caches: another format)
    data = ce.load_data()
    found = {c: [] for c in CARD}
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        for cls in CARD:
            for team in ("blue", "red"):
                for tr in tracks_of(frames, cls, team):
                    if len(tr) >= 8:
                        found[cls] += speeds(tr)
    print(f"{len(files)} recordings; median tiles/s of 1 s walking windows (n)")
    print(f"{'unit':12s} {'simulator':>9s} {'all':>14s} {'vertical':>14s} {'sideways':>14s} {'own half':>14s} {'their half':>14s}")
    for cls, card in CARD.items():
        v = found[cls]
        sim = data.card(card).unit.speed

        def med(sel):
            s = [w[0] for w in v if sel(w)]
            return f"{statistics.median(s):6.3f} ({len(s):5d})" if len(s) >= 10 else f"{'-':>14s}"
        # (y is the recording's own frame: < 16 is the recording player's half)
        print(f"{cls:12s} {sim:9.3f} {med(lambda w: True)} {med(lambda w: w[1] == 'vertical')} "
              f"{med(lambda w: w[1] == 'sideways')} {med(lambda w: w[2] < 15)} {med(lambda w: w[2] > 17)}")
    print("Game speeds do not depend on direction or place: if vertical walks come out faster than sideways ones,"
          " or one half faster than the other, the screen-to-tiles mapping is stretched, not the speeds off.")


if __name__ == "__main__":
    main()
