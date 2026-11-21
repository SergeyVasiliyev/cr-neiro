"""The simulator's numbers against the random-bot battles of the Classic Decks pool (tools/random_battles.py): every
drop of a tower's HP number in the video is one hit, and each card's hits take a value of their own off a tower.

    .venv\\Scripts\\python.exe tools\\classic_check.py              all battles in raw/classic with a plays log
    .venv\\Scripts\\python.exe tools\\classic_check.py c008 c009

1. Spells: every spell the bots cast within reach of a tower - the drop it made there, when it should have landed
   (blueprint second, then its flight from the king tower; area spells at once), against the simulator's damage on
   crown towers.
2. All drops: their values matched to what the decks' cards take off a tower in the simulator (troops, buildings,
   charge and dash hits, inferno stages, death and landing damage, the Electro Giant's zap); values nothing explains
   are listed with how often they came - a card whose real damage differs shows up there.
The tower HP comes from the blue recording (all six towers are on it), read like the live bot reads it; a reading
counts once two frames in a row agree. Card names: raw/classic/card_names.json via the card pictures.
"""
import argparse
import collections
import json
import math
import os
import re
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim_env as ce  # noqa: E402
from perception.reader import ScreenReader, edge_features, hand_art  # noqa: E402

OUT = os.path.join(ROOT, "raw", "classic")
TOWERS = {"blue_left": (3.5, 6.5), "blue_right": (14.5, 6.5), "blue_king": (9.0, 3.0),
          "red_left": (3.5, 25.5), "red_right": (14.5, 25.5), "red_king": (9.0, 29.0)}
TOWER_RADIUS = {"king": 2.0, "princess": 1.5}
TAP_LATENCY = 0.5   # s from a bot's tap to the game showing it: Zap and the projectile spells hit 0.5-0.7 s later than
                    # the blueprint second and their flight make it, all alike; a troop's own deploy clock shows 1.2-1.5 s
                    # after the tap, not 1.0
SPEED_SCALE = json.load(open(os.path.join(ROOT, "data", "rules.json"), encoding="utf-8")).get("speed_scale", 1.0)


def tower_damage(damage, pct):
    """What a hit of `damage` takes off a crown tower (battle.cpp apply_damage)."""
    return damage - damage * (100 - pct) // 100 if pct != 100 else damage


def card_namer():
    names = json.load(open(os.path.join(OUT, "card_names.json"), encoding="utf-8"))
    feats = {}
    for f in os.listdir(os.path.join(OUT, "cards")):
        m = re.match(r"(k\d+)\.png$", f)
        if m and names.get(m.group(1)):
            feats[m.group(1)] = edge_features(hand_art(cv2.imread(os.path.join(OUT, "cards", f))))

    def name(crop_path):
        f = edge_features(hand_art(cv2.imread(crop_path)))
        best = max(((float(v @ f) / len(f), k) for k, v in feats.items()), default=(0.0, None))
        return names[best[1]] if best[0] > 0.6 else None
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("battles", nargs="*")
    args = p.parse_args()
    data = ce.load_data()
    pool = json.load(open(os.path.join(ROOT, "data", "classic_decks.json"), encoding="utf-8"))
    raw_cards = json.load(open(os.path.join(ROOT, "data", "cards.json"), encoding="utf-8"))
    name = card_namer()
    battles = args.battles or sorted(f[:-12] for f in os.listdir(OUT) if re.match(r"c\d+_plays\.jsonl$", f))
    reader = ScreenReader()

    # what each card (and what it makes) takes off a crown tower in the simulator
    def values_of(cid, seen=None):
        seen = seen if seen is not None else set()
        if cid in seen:
            return []
        seen.add(cid)
        c = raw_cards[cid]
        out = []
        if c["type"] == "spell":
            s = c["spell"]
            pct = s.get("tower_damage_pct", 100)
            if s.get("damage"):
                out.append((s["tower_damage"] if s.get("tower_damage", -1) >= 0 else tower_damage(s["damage"], pct), f"{cid}"))
            if s.get("tick_damage"):
                tick = s.get("tower_tick_damage", -1)
                out.append((tick if tick >= 0 else tower_damage(s["tick_damage"], pct), f"{cid} tick"))
            if s.get("spawn"):
                out += values_of(s["spawn"]["unit"], seen)
            return out
        u = c["unit"]
        a = u["attack"]
        pct = a.get("tower_damage_pct", 100)
        if u.get("pellets"):
            out.append((u["pellets"]["count"] * u["pellets"]["damage"], f"{cid} (all pellets)"))
            out.append((u["pellets"]["count"] // 2 * u["pellets"]["damage"], f"{cid} (half, a tower at full range)"))
        elif a.get("damage"):
            out.append((tower_damage(a["damage"], pct) * u.get("multi_targets", 1), cid))  # (all bolts on a lone tower)
        if u.get("ramp"):
            out += [(u["ramp"]["damage2"], f"{cid} stage 2"), (u["ramp"]["damage3"], f"{cid} stage 3")]
        if u.get("charge"):
            out.append((u["charge"]["damage"], f"{cid} charge"))
        if u.get("dash"):
            out.append((u["dash"]["damage"], f"{cid} dash"))
        for key in ("death_damage", "deploy_damage", "reflect"):
            d = u.get(key)
            if d and d.get("damage"):
                td = d["tower_damage"] if d.get("tower_damage", -1) >= 0 else tower_damage(d["damage"], d.get("tower_damage_pct", 100))
                out.append((td, f"{cid} {key.replace('_', ' ')}"))
        for key in ("spawn", "death_spawn", "riders"):
            if u.get(key):
                out += values_of(u[key]["unit"], seen)
        if c.get("second"):
            out += values_of(c["second"]["unit"], seen)
        if (u.get("hit_buff") or {}).get("unit"):
            out += values_of(u["hit_buff"]["unit"], seen)
        return out

    spell_rows = collections.defaultdict(list)   # card -> [(expected, observed or None)]
    all_drops = collections.Counter()             # value -> count
    explained = collections.Counter()             # what -> count
    unexplained = collections.Counter()
    for b in battles:
        summary = json.load(open(os.path.join(OUT, f"{b}.json"), encoding="utf-8"))
        if not os.path.exists(os.path.join(OUT, summary["videos"]["blue"])):
            continue
        plays = [json.loads(line) for line in open(os.path.join(OUT, f"{b}_plays.jsonl"), encoding="utf-8")]
        for pl in plays:
            pl["name"] = name(os.path.join(OUT, pl["card"]))
        deck_cards = {pl["name"] for pl in plays if pl["name"]}
        table = collections.defaultdict(set)
        for cid in deck_cards:
            for value, what in values_of(cid):
                table[value].add(what)
        # tower HP over the video
        cap = cv2.VideoCapture(os.path.join(OUT, summary["videos"]["blue"]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        series = {t: [] for t in TOWERS}
        k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(f, (1080, 1920)) if f.shape[1] != 1080 else f
            for tname, (hp, exact) in reader.read_towers(f).items():
                if exact and hp:
                    series[tname].append((k / fps, hp))
            k += 1
        drops = []  # (t, tower, value)
        for tname, reads in series.items():
            stable = [r for i, r in enumerate(reads[1:], 1) if r[1] == reads[i - 1][1]]
            for (t0, h0), (t1, h1) in zip(stable, stable[1:]):
                if h1 < h0 and t1 - t0 < 1.0:
                    drops.append((t1, tname, h0 - h1))
        used = set()
        # 1. spells near a tower
        for pl in plays:
            cid = pl["name"]
            if not cid or raw_cards[cid]["type"] != "spell" or not pl.get("left_hand"):
                continue
            s = raw_cards[cid]["spell"]
            land = pl.get("landed_blue") or pl["tile_blue"]
            caster_king = TOWERS["blue_king" if pl["side"] == "blue" else "red_king"]
            if s.get("kind") in ("area", "roll"):
                fly = s.get("first_ms", 0) / 1000  # (an area's first tick or bolt)
            else:
                fly = math.dist(caster_king, land) / (s["projectile_speed"] * SPEED_SCALE)
            hit_t = pl["t"] + TAP_LATENCY + 1.0 + fly
            for tname, pos in TOWERS.items():
                if tname.split("_")[0] == pl["side"]:
                    continue  # (spells hit the other side only)
                r = TOWER_RADIUS["king" if tname.endswith("king") else "princess"]
                if s.get("kind") == "roll":
                    near = abs(pos[0] - land[0]) <= s["radius"] + r and 0 <= (pos[1] - land[1]) * (1 if pl["side"] == "blue" else -1) <= s["range"] + r
                else:
                    near = math.dist(pos, land) <= s["radius"] + r
                if not near:
                    continue
                if s.get("tower_damage", -1) >= 0:
                    expected = s["tower_damage"]
                elif not s.get("damage") and s.get("tower_tick_damage", -1) >= 0:
                    expected = s["tower_tick_damage"]
                else:
                    expected = tower_damage(s.get("damage") or s.get("tick_damage") or 0, s.get("tower_damage_pct", 100))
                window = (hit_t - 0.5, hit_t + (4.0 if s.get("kind") == "roll" else 0.6))
                cands = [d for d in drops if d[1] == tname and window[0] <= d[0] <= window[1] and d not in used]
                best = min(cands, key=lambda d: abs(d[2] - expected), default=None)
                if best:
                    used.add(best)
                spell_rows[cid].append((expected, best[2] if best else None, f"{b} {pl['side']} {pl['t']:.1f}s {tname}"))
        # 2. every drop
        for d in drops:
            all_drops[d[2]] += 1
            whats = [w for v, ws in table.items() if abs(v - d[2]) <= 1 for w in ws]
            if whats:
                for w in whats:
                    explained[w] += 1 / len(whats)
            else:
                unexplained[d[2]] += 1
        print(f"{b}: {len(drops)} tower hits, decks {sorted(deck_cards)}", flush=True)

    print("\n=== spells within reach of a tower: expected crown tower damage vs the drop seen")
    for cid, rows in sorted(spell_rows.items()):
        seen = [o for _, o, _ in rows if o is not None]
        exp = rows[0][0]
        match = sum(1 for o in seen if abs(o - exp) <= 1)
        print(f"  {cid:18s} expected {exp:4d}: {len(rows)} casts, drop found {len(seen)}, equal {match}; seen "
              + ", ".join(str(o) for o in seen[:12]))
    print("\n=== all tower hits explained by a card's value (share split when several fit)")
    for w, n in explained.most_common():
        print(f"  {w:34s} {n:6.1f}")
    print(f"\n=== values nothing explains ({sum(unexplained.values())} of {sum(all_drops.values())} hits), most common:")
    print("  " + ", ".join(f"{v}x{n}" for v, n in unexplained.most_common(40)))


if __name__ == "__main__":
    main()
