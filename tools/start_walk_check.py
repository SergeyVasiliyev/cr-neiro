"""When a troop starts walking after it is put down, in the recordings vs the simulator: from its first sighting
(it lands: the deploy timer starts) to the first frame it is 0.3+ tiles from where it landed.

    .venv\\Scripts\\python.exe tools\\start_walk_check.py [caches...]
Only troops first seen in the open on their own half, alone (no other troop of that kind within 3 tiles).
"""
import glob
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
from speed_check import CARD, tracks_of  # noqa: E402

DROP_IN = {"giant": 0.5, "knight": 0.35, "minion": 0.35}  # drawn above where it lands at first (tracker.py)


def game_starts(files):
    out = {c: [] for c in CARD}
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        for cls in CARD:
            for team in ("blue", "red"):
                for tr in tracks_of(frames, cls, team):
                    t0, x0, y0 = tr[0]
                    own = y0 < 14 if team == "blue" else y0 > 18
                    if not own or len(tr) < 10:
                        continue
                    # where it stands once landed: the median of its positions 0.4-0.8 s after first sight
                    settle = [p for p in tr if 0.4 <= p[0] - t0 <= 0.8]
                    if len(settle) < 2:
                        continue
                    sx = statistics.median(p[1] for p in settle)
                    sy = statistics.median(p[2] for p in settle)
                    moved = next((p[0] for p in tr if p[0] - t0 > 0.8 and math.dist(p[1:], (sx, sy)) > 0.3), None)
                    if moved is not None and moved - t0 < 4.0:
                        out[cls].append(moved - t0)
    return out


def sim_start(data, card):
    b = crsim.Battle(data, ce.DEFAULT_DECK, ce.DEFAULT_DECK, 0)
    uid = b.twin_spawn(crsim.Team.BLUE, card, crsim.Vec2(9.0, 8.0))
    e = b.find(uid)
    b.twin_set_hp(uid, e.max_hp)
    # twin_spawn skips the deploy timer: count it in by hand
    deploy = data.card(card).unit.deploy_time_ms / 1000
    for k in range(400):
        b.step()
        e = b.find(uid)
        if e is not None and math.dist((e.pos.x, e.pos.y), (9.0, 8.0)) > 0.3:
            return deploy + (k + 1) * 0.05
    return None


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json")))
    data = ce.load_data()
    g = game_starts(files)
    print(f"{len(files)} recordings: s from first sight (landing) to 0.3 tiles walked")
    for cls, card in CARD.items():
        v = sorted(g[cls])
        s = sim_start(data, card)
        if len(v) >= 5:
            print(f"  {cls:11s} game {statistics.median(v):5.2f} (25%..75% {v[len(v) // 4]:.2f}..{v[len(v) * 3 // 4]:.2f}, "
                  f"n={len(v)})   simulator {s:.2f}")
    print("(the game's figure includes the walk to 0.3 tiles at the troop's speed, as the simulator's does)")


if __name__ == "__main__":
    main()
