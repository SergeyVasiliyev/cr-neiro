"""How well the live twin follows the user's plays and his elixir, on the two-sided recordings.

The live twin (python/live/session.py LiveMatch, the screen only, as the bot plays) against the truth (the user's own
recording: his plays, so his elixir, exactly):
  - his elixir as the net sees it (LiveMatch.enemy_elixir_seen) vs the truth, every 0.5 s of battle;
  - his plays missed and the phantom plays added, by card, and for each missed play what the twin made of it:
    taken for another card (a found play of another card there and then), rejected (its log line), or nothing.
2026-11-14 (tools/push_live_view.py) the bot waited at the user's pushes partly because the twin had his elixir
too high: missed plays are never paid for.

    .venv\\Scripts\\python.exe tools\\twin_plays_check.py [names ...] [--show 30]
"""
import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
import human_starts  # noqa: E402
import twin  # noqa: E402
from clock_cache import load_clocks  # noqa: E402
from compare_red_truth import match_plays, true_plays  # noqa: E402
from live.session import LiveMatch  # noqa: E402
from push_defence_check import CACHE, two_sided_battles  # noqa: E402

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED


def observed(battle, red_elixir=None):
    """The user's elixir as the net's input has it (crsim.rl.encode: his blueprints still hidden count as
    unspent - in the simulator as live), with his elixir set to red_elixir first if given."""
    if red_elixir is not None:
        battle = battle.clone()
        battle.set_elixir(RED, red_elixir)
    return float(crsim.rl.encode(battle, BLUE)["global"][5]) * 10.0


def troops_diff(truth, live, data, reach=2.5):
    """The user's troops (elixir: each unit its share of its card) the truth twin has and the live twin has
    nowhere near (reach tiles, the same kind), and the live twin's ones the truth has nowhere near."""
    def red(b):
        return [(e.def_id, e.pos.x, e.pos.y) for e in b.entities if not e.is_building and e.team == RED and e.hp > 0]

    def value(card):
        cd = data.card(card)
        return cd.elixir / max(1, cd.count)

    tr, lv = red(truth), red(live)
    missing = sum(value(c) for c, x, y in tr if not any(c2 == c and math.dist((x, y), (x2, y2)) < reach
                                                         for c2, x2, y2 in lv))
    extra = sum(value(c) for c, x, y in lv if not any(c2 == c and math.dist((x, y), (x2, y2)) < reach
                                                       for c2, x2, y2 in tr))
    return missing, extra


def follow_battle(name, data, every=0.5, clocks=True):
    """Runs the live twin and the truth twin over one battle: (elixir rows [(g, seen, raw, truth)], the user's
    true plays, the live twin's red plays, its rejected candidates, its log). clocks: the live twin sees the deploy
    clocks (tools/clock_cache.py), as the live bot does."""
    frames, truth = human_starts.follow(name, f"{name}_red", data)
    live = LiveMatch(data)
    clock_rows = load_clocks(name) if clocks else None
    if clock_rows is not None and len(clock_rows) != len(frames):
        clock_rows = None
    rows, logs, last = [], [], -1e9
    b_start, _ = twin.battle_start(frames)
    end = max(fr["t"] - b_start for fr in frames if fr["timer"][0] is not None or fr["timer"][1])
    for k, fr in enumerate(frames):
        if fr["t"] - b_start > end:
            break  # the battle is over (the twins would go on on their own)
        args = (fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        truth.update(*args)
        truth.log.clear()
        live.update(*args, clocks=clock_rows[k] if clock_rows is not None else None)
        logs += live.log
        live.log.clear()
        g = truth.game_time(fr["t"])
        if truth.battle is None or live.battle is None or g is None or g < 0:
            continue
        if truth.battle.finished or live.battle.finished or g - last < every:
            continue
        last = g
        seen = live.enemy_elixir_seen(fr["t"])
        rows.append((g, seen, live.battle.player(RED).elixir, truth.battle.player(RED).elixir,
                     observed(live.battle, seen), observed(truth.battle), *troops_diff(truth.battle, live.battle, data)))
    red_frames = twin.load_cache(os.path.join(CACHE, f"{name}_red.json"))["frames"]
    # The user's elixir as his own screen shows it: the truth for the counts above, the twin's included.
    start, _ = twin.battle_start(red_frames)
    screen = sorted((fr["t"] - start, fr["elixir"]) for fr in red_frames if fr.get("elixir") is not None
                    and (fr["timer"][0] is not None or fr["timer"][1]))  # (the battle's HUD on screen)
    st = np.array([s[0] for s in screen]) if screen else np.zeros(0)
    out = []
    for g, seen, raw, tw, obs_live, obs_truth, miss_e, extra_e in rows:
        k = int(np.searchsorted(st, g))
        near = [j for j in (k - 1, k) if 0 <= j < len(st) and abs(st[j] - g) < 0.15]
        out.append((g, seen, raw, tw, screen[min(near, key=lambda j: abs(st[j] - g))][1] if near else np.nan,
                    obs_live, obs_truth, miss_e, extra_e))
    plays = true_plays(red_frames, data)
    found = [(g, card, pos) for g, card, pos, _ in live.red_plays]
    # When the twin took each of them (battle s): how late the bot learns of a play
    accepted = {(g, card, pos): acc for g, card, pos, acc in
                ((p.g, p.card, p.origin or p.pos, p.accepted_at - live.start) for p in live.plays
                 if p.side == "red" and p.on and p.source != "truth")}
    return out, plays, found, list(live.red_rejected), logs, accepted


def why_missed(play, found, rejected, logs, dt=2.5, dist=3.0):
    """What the twin made of a true play it missed."""
    t, card, pos = play
    near = [f for f in found if f[1] != card and abs(f[0] - t) < dt and
            (pos is None or f[2] is None or math.dist(pos, f[2]) < dist)]
    if near:
        return f"taken for {near[0][1]}"
    rej = [r for r in rejected if r[1] == card and -1.0 < r[0] - t < dt + 1.0 and
           (pos is None or r[2] is None or math.dist(pos, r[2]) < dist)]
    if rej:
        return f"rejected: {rej[0][3]}"
    unit = twin.UNIT_OF_CARD.get(card, card)
    said = [txt for g, txt in logs if -0.5 < g - t < dt + 2.0 and (f"red {card}" in txt or f"red {unit}" in txt)]
    if said:
        return "said: " + said[0].split(":", 1)[-1].strip()[:60]
    return "nothing seen" if pos is not None else "nothing seen (no position in the truth)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="battles (default: every two-sided one cached)")
    p.add_argument("--show", type=int, default=25, help="missed plays listed one by one")
    p.add_argument("--no-clocks", action="store_true", help="the live twin without the deploy clocks (as before "
                                                            "2026-11-16)")
    args = p.parse_args()
    data = ce.load_data()
    names = args.names or two_sided_battles()
    rows, tot, miss, phantom, reasons, listed = [], Counter(), Counter(), Counter(), Counter(), []
    phantom_kind = Counter()
    lags = defaultdict(list)  # troops / spells: s from the true play to the twin taking it
    t0 = time.time()
    for name in names:
        try:
            r, plays, found, rejected, logs, accepted = follow_battle(name, data, clocks=not args.no_clocks)
        except Exception as e:
            print(f"{name}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        rows += r
        hits, missed, extra = match_plays(plays, found)
        for t, card, pos, f in hits:
            if f in accepted:
                lags["spells" if card in twin.SPELL_CARDS else "troops"].append(accepted[f] - t)
        for _, c, _ in plays:
            tot[c] += 1
        for pl in missed:
            miss[pl[1]] += 1
            why = why_missed(pl, found, rejected, logs)
            reasons[why if not why.startswith("said") else "said: ..."] += 1
            listed.append((name, pl, why))
        for f in extra:
            phantom[f[1]] += 1
            near = [pl for pl in plays if pl[1] != f[1] and abs(pl[0] - f[0]) < 2.5 and
                    (pl[2] is None or f[2] is None or math.dist(pl[2], f[2]) < 3.0)]
            phantom_kind[f"{f[1]} (really {near[0][1]})" if near else f"{f[1]} (nothing played there)"] += 1
    print(f"{len(names)} battles in {time.time() - t0:.0f} s")
    r = np.array(rows, dtype=float)
    d = r[:, 5] - r[:, 6]
    print(f"\nthe user's elixir in the net's input, live twin minus the truth twin's ({len(r)} moments; his hidden "
          f"blueprints count as unspent in both, as in training):")
    print(f"  mean {d.mean():+.2f}, |diff| < 0.5 {np.mean(abs(d) < 0.5):.0%}, >= +2 {np.mean(d >= 2):.0%}, "
          f"<= -2 {np.mean(d <= -2):.0%}, > 4 off {np.mean(abs(d) > 4):.0%}")
    print(f"the user's troops on the field (elixir; the truth twin's vs the live twin's, same kind within 2.5 tiles):")
    print(f"  missing in the live twin: {r[:, 7].mean():.2f} a moment, >= 3 at {np.mean(r[:, 7] >= 3):.0%}; "
          f"extra in it: {r[:, 8].mean():.2f}, >= 3 at {np.mean(r[:, 8] >= 3):.0%}")
    ok = ~np.isnan(r[:, 4])
    r = r[ok]
    print(f"\nthe user's elixir, every 0.5 s ({len(r)} moments with his screen read), minus his own screen's:")
    print("  (the screen shows whole elixir: compared with the counts rounded down)")
    whole = np.floor(r[:, 4] + 0.005)
    for label, col in (("live twin, as the net sees it", 1), ("live twin, the simulator's count", 2),
                       ("the truth twin (his plays known)", 3)):
        d = np.floor(r[:, col] + 1e-6) - whole
        print(f"  {label}: mean {d.mean():+.2f}, exact {np.mean(d == 0):.0%}, >= +2 {np.mean(d >= 2):.0%}, "
              f"<= -2 {np.mean(d <= -2):.0%}, > 4 off {np.mean(abs(d) > 4):.0%}")
    n, m = sum(tot.values()), sum(miss.values())
    print(f"\nthe user's plays {n}: missed {m} ({m / max(1, n):.0%}), phantoms {sum(phantom.values())}")
    for kind, v in sorted(lags.items()):  # the bot can answer a play only from then on
        v = np.array(v)
        print(f"  taken by the twin after the play ({kind}, {len(v)}): median {np.median(v):.2f} s, "
              f"25% {np.percentile(v, 25):.2f}, 75% {np.percentile(v, 75):.2f}, 90% {np.percentile(v, 90):.2f}")
    for c in sorted(tot, key=lambda c: -miss[c]):
        print(f"  {c:11s} missed {miss[c]:3d}/{tot[c]:3d} ({miss[c] / tot[c]:.0%})  phantoms {phantom[c]:3d}")
    print("why missed:")
    for why, k in reasons.most_common():
        print(f"  {k:3d}  {why}")
    print("phantoms:")
    for what, k in phantom_kind.most_common(12):
        print(f"  {k:3d}  {what}")
    print(f"\nmissed plays (first {args.show}):")
    for name, (t, card, pos), why in listed[:args.show]:
        where = "?" if pos is None else f"({pos[0]:.1f},{pos[1]:.1f})"
        print(f"  {name} {t:6.1f}s {card:10s} {where:12s} {why}")


if __name__ == "__main__":
    main()
