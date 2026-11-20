"""How long after a troop stops at a tower its first hit lands, in the recordings vs the simulator.

A troop arrives: its track, within reach of the other side's princess tower, stops moving (< 0.2 tiles in the
next 0.5 s). Its first hit: the tower's HP number (read the same twice in a row) first drops after that, with no
other kind of the troop's side at the tower. Ranged hits include the projectile's flight.
The simulator's figure comes from the same thing played out in the simulator: one troop walked up to a tower.

    .venv\\Scripts\\python.exe tools\\first_hit_check.py [caches...]
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
from speed_check import tracks_of  # noqa: E402
from tower_hits_check import CARD, TOWER_POS, attackers  # noqa: E402


def game_delays(files, data):
    reach = {cls: data.card(card).unit.attack.range + data.card(card).unit.radius + 2.0 for cls, card in CARD.items()}
    out = {cls: [] for cls in CARD}
    for path in files:
        frames = json.load(open(path, encoding="utf-8"))["frames"]
        for tower, (tx, ty) in TOWER_POS.items():
            side = "blue" if tower.startswith("red") else "red"
            reads = []
            for f in frames:
                hp, exact = f.get("towers", {}).get(tower, [None, False])
                if exact and hp:
                    reads.append((f["t"], hp, f))
            stable = [r for k, r in enumerate(reads[1:], 1) if r[1] == reads[k - 1][1]]
            drops = [(t1, f1) for (t0, h0, _), (t1, h1, f1) in zip(stable, stable[1:]) if h1 < h0 and t1 - t0 < 0.6]
            for cls in CARD:
                for tr in tracks_of(frames, cls, side):
                    arrive = None
                    for k, (t, x, y) in enumerate(tr):
                        if math.dist((x, y), (tx, ty)) > reach[cls]:
                            continue
                        nxt = [p for p in tr[k:] if p[0] - t <= 0.5]
                        if nxt[-1][0] - t >= 0.4 and max(math.dist((x, y), p[1:]) for p in nxt) < 0.2:
                            arrive = t
                            break
                    if arrive is None:
                        continue
                    first = next(((t1, f1) for t1, f1 in drops if arrive - 0.2 <= t1 <= arrive + 4.0), None)
                    if first is None:
                        continue
                    kinds = attackers(first[1], tower, reach)
                    if set(kinds) == {cls}:  # nothing else of theirs at the tower: the drop is this troop's
                        out[cls].append(first[0] - arrive)
    return out


def sim_delay(data, card):
    """Seconds from a troop stopping at a tower to the tower's first damage, in the simulator."""
    b = crsim.Battle(data, ce.DEFAULT_DECK, ce.DEFAULT_DECK, 0)
    tower = b.find(b.princess_tower(crsim.Team.RED, 0))
    hp0 = tower.hp
    uid = b.twin_spawn(crsim.Team.BLUE, card, crsim.Vec2(3.5, 17.5))
    last, still_since = None, None
    for _ in range(2000):
        b.step()
        u = b.find(uid)
        if u is None:
            return None
        if last is not None and math.dist(last, (u.pos.x, u.pos.y)) < 1e-6 and u.deploy_ms_left == 0:
            if still_since is None:
                still_since = b.time_ms
        elif u.deploy_ms_left == 0:
            still_since = None
        last = (u.pos.x, u.pos.y)
        t = b.find(b.princess_tower(crsim.Team.RED, 0))
        if t is not None and t.hp < hp0 and still_since is not None:
            return (b.time_ms - still_since) / 1000.0
    return None


def main():
    files = sys.argv[1:] or sorted(p for p in glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json"))
                                    if not p.endswith("_clocks.json"))  # (deploy clock caches: another format)
    data = ce.load_data()
    g = game_delays(files, data)
    print(f"{len(files)} recordings: first hit on a tower, seconds after the troop stops next to it")
    print(f"{'unit':11s} {'game median':>12s} {'25%..75%':>14s} {'n':>4s}   {'simulator':>9s}")
    for cls, card in CARD.items():
        v = sorted(g[cls])
        s = sim_delay(data, card)
        sim = f"{s:9.2f}" if s is not None else f"{'-':>9s}"
        if len(v) >= 3:
            print(f"{cls:11s} {statistics.median(v):12.2f} {v[len(v) // 4]:6.2f}..{v[len(v) * 3 // 4]:5.2f} {len(v):4d}   {sim}")
        else:
            print(f"{cls:11s} {'-':>12s} {'':>14s} {len(v):4d}   {sim}")
    print("game times include ~0.1 s of reading lag (the tower number is read ~8 times a second)")


if __name__ == "__main__":
    main()
