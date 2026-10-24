"""How far the live twin's idea of the opponent's elixir is off, and why: two-sided recordings, the twin following
the blue recording alone (as live), against the red player's own elixir bar.

    .venv\Scripts\python.exe tools\elixir_check.py video24 video26 ...

Compared with what the net is shown in training: the opponent's elixir with the cards still lying as blueprints
(the first second after a play, hidden in the game) not taken off yet. The screen shows whole elixir: +0.5.
Every moment is put in one of these:
  blueprint   - a red card lies as a blueprint (unseen in training too)
  late troop  - a red troop is on the field, but the twin has not taken it for a play yet
  late spell  - a red spell is cast (flying) or landed, but the twin has not taken it for a play yet
  steady      - the twin knows of every red play so far
"""
import math
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import twin  # noqa: E402
from compare_red_truth import true_plays  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import LiveMatch  # noqa: E402

RED = crsim.Team.RED


def run(name, data, rows, delays):
    cache = os.path.join(ROOT, "raw", "perception", "cache")
    bf = twin.load_cache(os.path.join(cache, f"{name}.json"))["frames"]
    rf = twin.load_cache(os.path.join(cache, f"{name}_red.json"))["frames"]
    rstart, _ = twin.battle_start(rf)
    real = [(f["t"] - rstart, f["elixir"]) for f in rf if f.get("elixir") is not None]
    truth = true_plays(rf, data)  # (battle s, card, pos)
    m = LiveMatch(data)
    est, accepted_g = [], {}
    for f in bf:
        m.update(f["t"], twin.seen_from_cache(f), twin.detections(f), twin.bars_from_cache(f) if "bars" in f else None)
        m.log.clear()
        g = m.game_time(f["t"])
        if g is None or m.battle is None:
            continue
        for p in m.plays:
            if p.side == "red" and p.on and p.id not in accepted_g:
                accepted_g[p.id] = g  # when the twin first counted it
        if g > 5.0:  # (the bar shows 0 during the countdown)
            est.append((g, m.battle.player(RED).elixir, m.enemy_elixir_seen(f["t"])))
    # each true play: when the twin knew of it (None: never)
    known = []
    red = [p for p in m.plays if p.side == "red" and p.on]
    for t, card, _ in truth:
        match = min((p for p in red if p.card == card and abs(p.g - t) < 2.5), key=lambda p: abs(p.g - t), default=None)
        if match is not None:
            red.remove(match)
            delays["spell" if card in twin.SPELL_CARDS else "troop"].append(accepted_g[match.id] - t)
        known.append((t, card, accepted_g[match.id] if match is not None else None))
    # The bar's number is read with slips now and then (a 10 read as 1.89 while a card leaves the hand): the
    # median of the readings within 0.35 s, its whole part + 0.5 (the screen shows whole elixir).
    j0 = 0
    for g, e, e_seen in est[::3]:
        while j0 < len(real) and real[j0][0] < g - 0.35:
            j0 += 1
        near = [r for tr, r in real[j0:j0 + 20] if abs(tr - g) <= 0.35]
        if len(near) < 3:
            continue
        shown = statistics.median(near)
        if shown == 0.0 and e > 2.0:
            continue  # the bar not shown (the match over) or misread
        hidden = sum(data.card(c).elixir for t, c, _ in known if t <= g < t + 1.0)
        ref = min(10.0, math.floor(shown + 0.02) + 0.5 + hidden)
        unknown = [(t, c) for t, c, k in known if t <= g and (k is None or g < k)]
        if any(g - t >= 1.0 and c in twin.SPELL_CARDS for t, c in unknown):
            kind = "late spell"
        elif any(g - t >= 1.0 for t, c in unknown):
            kind = "late troop"
        elif unknown:
            kind = "blueprint"
        else:
            kind = "steady"
        rows[kind].append((e - ref, e_seen - ref))
        rows["all"].append((e - ref, e_seen - ref))


def main():
    data = load_data()
    rows, delays = defaultdict(list), defaultdict(list)
    for name in sys.argv[1:]:
        run(name, data, rows, delays)
        print(f"{name} done", flush=True)
    for k, label in ((0, "the simulator's count (plays taken off once judged)"),
                     (1, "enemy_elixir_seen (troops and spells paid for when first seen)")):
        total = sum(abs(x[k]) for x in rows["all"])
        print(f"\n{label} - what training shows: share of the time, mean |error|, mean error, part of the error")
        for kind in ("all", "steady", "blueprint", "late troop", "late spell"):
            v = [x[k] for x in rows[kind]]
            if v:
                a = [abs(x) for x in v]
                print(f"  {kind:11s} {len(v) / len(rows['all']):5.0%} of the time   |err| {statistics.fmean(a):4.2f}"
                      f"   err {statistics.fmean(v):+5.2f}   {sum(a) / total:5.0%} of the error   "
                      f"within 1: {sum(x <= 1 for x in a) / len(a):4.0%}")
    for kind, v in delays.items():
        v = sorted(v)
        print(f"red {kind} plays: the twin counts them {statistics.median(v):.2f} s after they are put down "
              f"(25-75%: {v[len(v) // 4]:.2f}-{v[len(v) * 3 // 4]:.2f}, n {len(v)})")


if __name__ == "__main__":
    main()
