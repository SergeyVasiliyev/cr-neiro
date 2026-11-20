"""After a kill, how soon does a troop hit its next target? The recordings vs the simulator.

In the simulator, losing a target unlocks the troop, and its next target waits only the first-hit wind-up
(first_hit_ms, e.g. 0.7 s for a mini pekka). That is sooner than its hit speed (1.6 s), even right after a hit. If
the game keeps the hit-speed rhythm instead, defences are too quick in the simulator.

The measure: a troop A has an enemy V1 in reach, V1 dies (gone from the screen for 1 s+), and another enemy V2 is
already in A's reach with no HP bar (full HP). The time from V1 going to V2's first bar is A's next hit (+ ~0.1 s of
reading lag, + the projectile's flight for archers and musketeers). Only one of each kind of A, V1 and V2 on the
screen (their bars are surely theirs), nothing else of A's side near V2 (no troop within 6.5 tiles, no tower within
its reach) and no spell near V2 meanwhile. The simulator's figure comes from the same scene played out in it:
A between two enemy knights standing in its reach.

    .venv\\Scripts\\python.exe tools\\retarget_check.py [caches...]   (default: every recorded battle once)

First run, 2026-11-16 (raw/retarget_check.txt): knights 13 times, median 1.40 s (simulator 0.70); mini pekkas 7 times,
0.40 (0.70 death to death, 5 times); musketeers 7 times, 1.13 (simulator 1.00). Mixed and few: not changed.
"""
import glob
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
import twin  # noqa: E402
from perception.unit_hp import assign  # noqa: E402
from perception.units import Detection  # noqa: E402

CARD = {"knight": "knight", "mini_pekka": "mini_pekka", "musketeer": "musketeer", "archer": "archers"}
VICTIMS = {"knight", "mini_pekka", "giant", "musketeer", "archer"}
TOWERS = {"blue": [(3.5, 6.5), (14.5, 6.5), (9.0, 3.0)], "red": [(3.5, 25.5), (14.5, 25.5), (9.0, 29.0)]}


def load(path):
    frames = json.load(open(path, encoding="utf-8"))["frames"]
    dets, bars = [], []
    for f in frames:
        ds = [Detection(d[0], d[1], d[2], tuple(d[3:7]), (d[7], d[8])) for d in f.get("det", [])]
        bs = twin.bars_from_cache(f)
        dets.append(ds)
        bars.append(assign(bs, ds) if bs else {})
    return frames, dets, bars


def only(ds, cls, team):
    """The index of the one detection of that kind and side, or None (none or several)."""
    same = [i for i, d in enumerate(ds) if d.cls == cls and d.team == team]
    return same[0] if len(same) == 1 else None


def game_delays(files, data):
    unit = {cls: data.card(card).unit for cls, card in CARD.items()}
    radius = {cls: data.card(card).unit.radius for cls, card in {**CARD, "giant": "giant"}.items()}
    out = {cls: [] for cls in CARD}
    for path in files:
        frames, dets, bars = load(path)
        last = {}
        for k in range(1, len(frames) - 1):
            t, ds = frames[k]["t"], dets[k]
            for ai, a in enumerate(ds):
                if a.cls not in CARD or a.team not in ("blue", "red") or only(ds, a.cls, a.team) is None:
                    continue
                enemy = "red" if a.team == "blue" else "blue"
                reach = unit[a.cls].attack.range + unit[a.cls].radius + 0.3

                def in_reach(d):
                    return math.dist(a.tile, d.tile) - radius.get(d.cls, 0.5) <= reach

                for v1i, v1 in enumerate(ds):
                    if v1.team != enemy or v1.cls not in VICTIMS or only(ds, v1.cls, v1.team) is None or not in_reach(v1):
                        continue
                    after = [j for j in range(k + 1, len(frames)) if frames[j]["t"] - t <= 1.0]
                    if not after or frames[after[-1]]["t"] - t < 0.8:
                        continue
                    if any(any(d.cls == v1.cls and d.team == enemy for d in dets[j]) for j in after):
                        continue  # still there: not dead
                    if v1i not in bars[k] and not any(only(dets[j], v1.cls, enemy) in bars[j]
                                                      for j in range(max(0, k - 5), k)):
                        continue  # no bar on it: nothing showed it was being hit (it may have walked out of sight)
                    t_dead = (t + frames[k + 1]["t"]) / 2
                    for v2 in ds:
                        if v2.team != enemy or v2.cls not in VICTIMS or v2.cls == v1.cls or not in_reach(v2):
                            continue
                        if only(ds, v2.cls, enemy) is None:
                            continue
                        key = (path, a.cls, a.team, v2.cls)
                        if t - last.get(key, -99.0) < 5.0:
                            continue
                        # full HP until V1 went: no bar on V2 for the second before
                        before = [j for j in range(max(0, k - 10), k + 1) if t - frames[j]["t"] <= 1.0]
                        if any(only(dets[j], v2.cls, enemy) in bars[j] for j in before
                               if only(dets[j], v2.cls, enemy) is not None):
                            continue
                        hit = None
                        for j in range(k + 1, len(frames)):
                            if frames[j]["t"] - t_dead > 3.0:
                                break
                            dj = dets[j]
                            vi = only(dj, v2.cls, enemy)
                            if vi is None:
                                continue
                            v = dj[vi]
                            # nothing else of A's side could have hit V2: no other troop near, no tower in reach
                            others = [d for d in dj if d.team == a.team and not d.is_spell
                                      and not (d.cls == a.cls and only(dj, a.cls, a.team) is not None)
                                      and math.dist(d.tile, v.tile) <= 6.5]
                            spells = [d for d in dj if d.is_spell and math.dist(d.tile, v.tile) <= 3.5]
                            towers = [p for p in TOWERS[a.team] if math.dist(p, v.tile) <= 8.0]
                            if others or spells or towers or only(dj, a.cls, a.team) is None:
                                hit = None
                                break
                            if vi in bars[j]:
                                hit = frames[j]["t"]
                                break
                        if hit is not None:
                            out[a.cls].append(hit - t_dead)
                            last[key] = t
    return out


def melee_retargets(files, data):
    """Knights and mini pekkas, wherever they fight: A's hit is told by the HP it takes. V1 dies (gone 1 s+) with
    no more HP left than one hit of A takes (A surely struck the last blow); V2, already in A's reach, then loses
    at least 0.7 of one A hit between two readings (no tower or archer hit is that big), or vanishes with no more
    left than one A hit (then both times are deaths seen on the screen: their lag cancels out).
    Returns {kind: [(delay s, "drop" | "death")]}."""
    hp = {cls: data.card(card).unit.hitpoints for cls, card in
          {"knight": "knight", "mini_pekka": "mini_pekka", "giant": "giant", "musketeer": "musketeer",
           "archer": "archers"}.items()}
    out = {"knight": [], "mini_pekka": []}
    for path in files:
        frames, dets, bars = load(path)
        last = {}

        def frac(j, cls, team):
            i = only(dets[j], cls, team)
            if i is None:
                return None
            b = bars[j].get(i)
            return b.fraction if b is not None else None

        for k in range(1, len(frames) - 1):
            t, ds = frames[k]["t"], dets[k]
            for a in ds:
                if a.cls not in out or a.team not in ("blue", "red") or only(ds, a.cls, a.team) is None:
                    continue
                enemy = "red" if a.team == "blue" else "blue"
                ua = data.card(CARD[a.cls]).unit
                dmg, reach = ua.attack.damage, ua.attack.range + ua.radius + 0.3
                for v1 in ds:
                    if v1.team != enemy or v1.cls not in hp or only(ds, v1.cls, enemy) is None:
                        continue
                    if math.dist(a.tile, v1.tile) - 0.5 > reach:
                        continue
                    after = [j for j in range(k + 1, len(frames)) if frames[j]["t"] - t <= 1.0]
                    if not after or frames[after[-1]]["t"] - t < 0.8 or \
                            any(any(d.cls == v1.cls and d.team == enemy for d in dets[j]) for j in after):
                        continue
                    left = [f for f in (frac(j, v1.cls, enemy) for j in range(max(0, k - 5), k + 1)) if f is not None]
                    if not left or left[-1] * hp[v1.cls] > dmg * 1.15 + 20:
                        continue  # more left than one of A's hits: someone else may have killed it
                    t_dead = (t + frames[k + 1]["t"]) / 2
                    for v2 in ds:
                        if v2.team != enemy or v2.cls not in hp or v2.cls == v1.cls or only(ds, v2.cls, enemy) is None:
                            continue
                        if math.dist(a.tile, v2.tile) - 0.5 > reach:
                            continue
                        key = (path, a.team, a.cls, v2.cls)
                        if t - last.get(key, -99.0) < 5.0:
                            continue
                        step = dmg / hp[v2.cls]
                        prev = frac(k, v2.cls, enemy)
                        if prev is None and not any(frac(j, v2.cls, enemy) is not None for j in range(max(0, k - 15), k)):
                            prev = 1.0  # no bar seen lately: full HP
                        found = None
                        for j in range(k + 1, len(frames)):
                            if frames[j]["t"] - t_dead > 3.0 or only(dets[j], a.cls, a.team) is None:
                                break
                            if only(dets[j], v2.cls, enemy) is None:
                                gone = [i for i in range(j, len(frames)) if frames[i]["t"] - frames[j]["t"] <= 1.0]
                                if not any(any(d.cls == v2.cls and d.team == enemy for d in dets[i]) for i in gone) \
                                        and prev is not None and prev * hp[v2.cls] <= dmg * 1.15 + 20:
                                    found = ((frames[j - 1]["t"] + frames[j]["t"]) / 2 - t_dead, "death")
                                break
                            f = frac(j, v2.cls, enemy)
                            if f is not None and prev is not None and prev - f >= 0.7 * step:
                                found = (frames[j]["t"] - t_dead, "drop")
                                break
                            if f is not None:
                                prev = f
                        if found is not None:
                            out[a.cls].append(found)
                            last[key] = t
    return out


def sim_delay(data, card):
    """The same in the simulator: A (blue) between two red knights in its reach, out of every tower's reach; the
    nearer knight has 1 HP, so A's first hit kills it. From that knight dying to the other's first lost HP."""
    b = crsim.Battle(data, ce.DEFAULT_DECK, ce.DEFAULT_DECK, 3)
    u, kn = data.card(card).unit, data.card("knight").unit
    reach = u.attack.range + u.radius + kn.radius
    ax, ay = 9.0, 13.0
    aid = b.twin_spawn(crsim.Team.BLUE, card, crsim.Vec2(ax, ay))
    v1 = b.twin_spawn(crsim.Team.RED, "knight", crsim.Vec2(ax, ay + reach - 0.2))
    s = math.sqrt(0.5)
    v2 = b.twin_spawn(crsim.Team.RED, "knight", crsim.Vec2(ax + (reach - 0.1) * s, ay + (reach - 0.1) * s))
    b.twin_set_hp(v1, 1)
    dead_at = None
    for _ in range(1200):
        b.step()
        k1, k2 = b.find(v1), b.find(v2)
        if b.find(aid) is None or k2 is None:
            return None
        if k1 is None and dead_at is None:
            dead_at = b.time_ms
        if k2.hp < k2.max_hp:
            return None if dead_at is None else (b.time_ms - dead_at) / 1000.0
    return None


def default_files():
    """Every cache once: not the deploy clock caches (another format), and not a battle cut out of a recording whose
    whole cache is there too (its fights would count twice)."""
    files = sorted(p for p in glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json"))
                   if not p.endswith("_clocks.json"))
    keep = []
    for p in files:
        m = re.fullmatch(r"(video\d+)_[bx]\d+(_red)?\.json", os.path.basename(p))  # a battle cut out of videoNN
        if m and os.path.exists(os.path.join(os.path.dirname(p), m.group(1) + (m.group(2) or "") + ".json")):
            continue
        keep.append(p)
    return keep


def main():
    files = sys.argv[1:] or default_files()
    data = ce.load_data()
    g = game_delays(files, data)
    print(f"{len(files)} recordings: from a troop's victim dying to its next victim's first HP bar (s)")
    print(f"{'attacker':11s} {'game median':>12s} {'25%..75%':>14s} {'n':>4s}   {'simulator':>9s}   "
          f"{'first hit':>9s} {'hit speed':>9s}")
    for cls, card in CARD.items():
        v = sorted(g[cls])
        s = sim_delay(data, card)
        a = data.card(card).unit.attack
        sim = f"{s:9.2f}" if s is not None else f"{'-':>9s}"
        row = f"{cls:11s} "
        if len(v) >= 3:
            row += f"{statistics.median(v):12.2f} {v[len(v) // 4]:6.2f}..{v[len(v) * 3 // 4]:5.2f} {len(v):4d}"
        else:
            row += f"{'-':>12s} {'':>14s} {len(v):4d}"
        print(row + f"   {sim}   {a.first_hit_ms / 1000:9.2f} {a.hit_speed_ms / 1000:9.2f}")
    print("game: + ~0.1 s of reading lag; a dying troop may stay on screen a frame or two (the death counts late);"
          " ranged: + the projectile's flight in both")
    m = melee_retargets(files, data)
    print("\nmelee, A's next hit told by the HP it takes (anywhere, towers and crowds too):")
    print(f"{'attacker':11s} {'how seen':>9s} {'game median':>12s} {'25%..75%':>14s} {'n':>4s}   "
          f"{'sim (first hit)':>15s} {'hit speed':>9s}")
    for cls, rows in m.items():
        a = data.card(CARD[cls]).unit.attack
        for how in ("death", "drop", "both"):
            v = sorted(d for d, h in rows if how in ("both", h))
            if len(v) >= 3:
                print(f"{cls:11s} {how:>9s} {statistics.median(v):12.2f} {v[len(v) // 4]:6.2f}..{v[len(v) * 3 // 4]:5.2f}"
                      f" {len(v):4d}   {a.first_hit_ms / 1000:15.2f} {a.hit_speed_ms / 1000:9.2f}")
            else:
                print(f"{cls:11s} {how:>9s} {'-':>12s} {'':>14s} {len(v):4d}")
    print("death: V2 vanished too (both deaths seen alike: no lag between); drop: V2's bar fell (the death of V1 is"
          " seen a little late, so this comes out a little short)")


if __name__ == "__main__":
    main()
