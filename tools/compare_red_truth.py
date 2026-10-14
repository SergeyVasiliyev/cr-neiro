"""Checks the opponent's plays the twin finds against what they really played.

    .venv\\Scripts\\python.exe tools\\compare_red_truth.py --blue raw/perception/cache/video8.json --red raw/perception/cache/video8_red.json

Needs one battle recorded from both sides (two LDPlayer instances), each read with tools/perceive_video.py.
The red player's own recording shows its hand: that is the truth for its plays (card, moment, and -
from its own unit tracks - position, turned 180 degrees into the blue player's view). Both recordings are
put on the battle clock by their countdown timers. Then every true play is looked for among the plays
the live twin found from the blue recording alone: same card, within --dt seconds and --dist tiles.
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import LiveMatch  # noqa: E402

W, H = 18.0, 32.0


def own_plays_timed(frames, data=None, deploy_delay=1.0):
    """A player's own plays from its recording (its hand), [(battle s, card, pos in its own view)].
    The hand shows a card leave when the drag starts, a moment before the drop: a troop play's time is
    taken from its units instead (first seen = the drop + the blueprint second), a spell's from its hit
    (minus the flight from its king tower) - the moments the simulator has to match."""
    start, _ = twin.battle_start(frames)
    tracks = twin.run_tracker(frames)
    by_id = {tr.id: tr for tr in tracks}
    plays = twin.own_plays_from_hand(frames)
    twin.place_own_plays(plays, tracks)
    out = []
    for p in plays:
        t = p.t
        seen = [by_id[u] for u in p.units if u in by_id]
        if seen and p.card in twin.SPELL_CARDS and data is not None and p.pos is not None:
            speed = data.card(p.card).spell.projectile_speed
            t = min(tr.first_seen for tr in seen) - math.dist(p.pos, twin.KING["blue"]) / speed - deploy_delay
        elif seen:
            t = min(tr.first_seen for tr in seen) - deploy_delay
        out.append((t - start, p.card, p.pos))
    return out


def true_plays(red_frames, data=None):
    """The red player's plays from its own recording, in blue's battle time and blue's tiles."""
    return [(t, card, None if pos is None else (W - pos[0], H - pos[1]))  # its bottom is blue's top
            for t, card, pos in own_plays_timed(red_frames, data)]


def found_plays(blue_frames, data):
    """The opponent's plays the live twin accepts from the blue recording: (game s, card, pos)."""
    m = LiveMatch(data)
    for fr in blue_frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr),
                 twin.bars_from_cache(fr) if "bars" in fr else None)
        m.log.clear()
    # A unit play is recorded when its units are seen (about the blueprint second after the play);
    # a spell when its hit is seen. Put both back to the moment of the play, as in the truth.
    out = []
    for g, card, pos, _ in m.red_plays:
        out.append((g, card, pos))
    return out, m.red_rejected


def match_plays(truth, found, dt=2.5, dist=3.0):
    """Each true play to the closest-in-time found play of the same card (within dt s, and dist tiles when
    both positions are known). Returns (hits [(t, card, pos, found play)], missed, extra)."""
    used = set()
    hits, missed = [], []
    for t, card, pos in truth:
        best = None
        for k, (g, c, q) in enumerate(found):
            if k in used or c != card or abs(g - t) > dt + (3.0 if card in twin.SPELL_CARDS else 0.0):
                continue
            if pos is not None and q is not None and math.dist(pos, q) > dist:
                continue
            if best is None or abs(g - t) < abs(found[best][0] - t):
                best = k
        if best is None:
            missed.append((t, card, pos))
        else:
            used.add(best)
            hits.append((t, card, pos, found[best]))
    extra = [f for k, f in enumerate(found) if k not in used]
    return hits, missed, extra


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blue", required=True, help="perception cache of the blue player's recording")
    p.add_argument("--red", required=True, help="perception cache of the red player's recording")
    p.add_argument("--dt", type=float, default=2.5, help="seconds a found play may be off")
    p.add_argument("--dist", type=float, default=3.0, help="tiles a found play may be off")
    args = p.parse_args()

    data = load_data()
    truth = true_plays(twin.load_cache(args.red)["frames"])
    found, rejected = found_plays(twin.load_cache(args.blue)["frames"], data)
    hits, missed, extra = match_plays(truth, found, args.dt, args.dist)

    fmt = lambda q: "   ?      " if q is None else f"({q[0]:4.1f},{q[1]:4.1f})"  # noqa: E731
    print(f"red really played {len(truth)} cards; the twin found {len(found)}: "
          f"{len(hits)} right, {len(missed)} missed, {len(extra)} extra")
    if hits:
        dts = sorted(abs(h[3][0] - h[0]) for h in hits)
        dps = sorted(math.dist(h[2], h[3][2]) for h in hits if h[2] and h[3][2])
        print(f"  timing off by median {dts[len(dts) // 2]:.1f}s" +
              (f", position off by median {dps[len(dps) // 2]:.1f} tiles" if dps else ""))
    print("\nmissed (really played, not found) - and the closest thing the twin saw:")
    for t, card, pos in missed:
        near = [(abs(g - t), f"accepted {g:6.1f}s {fmt(q)}") for g, c, q in found if c == card and abs(g - t) < 8]
        near += [(abs(g - t), f"rejected {g:6.1f}s {fmt(q)} ({why})") for g, c, q, why in rejected
                 if c == card and abs(g - t) < 8]
        print(f"  {t:6.1f}s  {card:10s}  {fmt(pos)}   " + (min(near)[1] if near else "nothing like it"))
    print("\nextra (found, not really played):")
    for g, card, pos in extra:
        print(f"  {g:6.1f}s  {card:10s}  {fmt(pos)}")


if __name__ == "__main__":
    main()
