"""Where the simulated mechanics disagree with the game, from what the live twin had to correct.

    .venv\\Scripts\\python.exe tools\\sim_check.py --blue raw/perception/cache/video8.json --red raw/perception/cache/video8_red.json
    .venv\\Scripts\\python.exe tools\\sim_check.py --blue raw/perception/cache/video8.json --red ... --blue2 ... (several matches)

The twin follows the recorded match and pulls the simulator back to the screen every 0.1 s. Every pull
is a disagreement: a unit the simulator walked elsewhere (pathing, speed), HP it took away too fast or
too slow (damage, attack speed, targeting), a unit it killed that lives on in the game (added back) or
kept alive after the game lost it (dropped), tower HP. Per unit kind these add up to a picture of which
mechanic is off. With the red player's own recording (--red), their plays are the real ones, not guessed:
what is left is the simulator itself. With --both the twin also sees through the red recording (fewer
detector mistakes taken for the simulator's).
"""
import argparse
import dataclasses
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402
from auto_label import LIVE_CONF, merge_views, view_dets  # noqa: E402
from compare_red_truth import own_plays_timed, true_plays  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import LiveMatch  # noqa: E402
from perception.units import merge_spells  # noqa: E402


def both_views(blue_frame, red_frame):
    """The detections of both recordings at one moment, as one list in the blue recording's tiles (a unit
    hidden in one view is usually clear in the other; its kind and side by the more confident view)."""
    A = view_dets(blue_frame)
    merged = merge_views(A, view_dets(red_frame, flip=True))
    own = {d.box for d in A}
    # Units only the red recording saw keep no box here: HP bars are read on the blue recording.
    merged = [d if d.box in own else dataclasses.replace(d, box=(-1e4, -1e4, -1e4 + 1, -1e4 + 1)) for d in merged]
    return merge_spells([d for d in merged if d.conf >= LIVE_CONF])


def run(blue_path, red_path, data, both=False):
    frames = twin.load_cache(blue_path)["frames"]
    red_frames = twin.load_cache(red_path)["frames"] if red_path else None
    truth = true_plays(red_frames, data) if red_path else None
    both = both and red_frames is not None and len(red_frames) == len(frames)
    # Our own plays also on time (known in advance offline), so units appear in the simulator when they do
    # in the game and what is left to correct is the simulator's own doing.
    m = LiveMatch(data, red_truth=truth, blue_truth=own_plays_timed(frames, data), record_damage=True)
    for i, fr in enumerate(frames):
        dets = both_views(fr, red_frames[i]) if both else twin.detections(fr)
        m.update(fr["t"], twin.seen_from_cache(fr), dets, twin.bars_from_cache(fr) if "bars" in fr else None)
        m.log.clear()
    length = m.game_time(frames[-1]["t"]) or 0.0
    return m, length


def summarize(runs):
    moves, hps, drops, adds, towers = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), []
    events = []
    unit_time = 0.0
    for name, m, length in runs:
        unit_time += length
        for g, kind, cls, side, d in m.corrections:
            key = (cls, side)
            if kind == "moved":
                sx, sy = d["sim"]
                rx, ry = d["real"]
                ahead = (ry - sy) * (1 if side == "blue" else -1)  # + : the real unit is further on its way
                moves[key].append((ahead, rx - sx))
                dist = ((rx - sx) ** 2 + (ry - sy) ** 2) ** 0.5
                if dist > 2.0:
                    events.append((dist, f"{name} {g:6.1f}s  {side} {cls} {dist:.1f} tiles off "
                                         f"(sim {sx:.1f},{sy:.1f} game {rx:.1f},{ry:.1f})"))
            elif kind == "hp":
                hps[key].append(d["sim"] - d["real"])
                if abs(d["sim"] - d["real"]) > 0.4:
                    events.append((abs(d["sim"] - d["real"]) * 5,
                                   f"{name} {g:6.1f}s  {side} {cls} HP sim {d['sim']:.0%} game {d['real']:.0%}"))
            elif kind == "dropped":
                drops[key].append(d)
                events.append((3.0, f"{name} {g:6.1f}s  {side} {cls} alive in the sim, gone in the game "
                                    f"(at {d['pos'][0]:.1f},{d['pos'][1]:.1f})"))
            elif kind == "added":
                adds[key].append(d)
                events.append((3.0, f"{name} {g:6.1f}s  {side} {cls} alive in the game, not in the sim "
                                    f"(at {d['pos'][0]:.1f},{d['pos'][1]:.1f})"))
            elif kind == "tower":
                towers.append((side, d["sim"] - d["real"]))

    print("units: what the twin had to correct (per unit kind and side)")
    print("  unit         side   pulls  pulled ahead (median, tiles)  HP fixes  sim HP - game HP (median)"
          "   sim kept alive  sim lost")
    for key in sorted(set(moves) | set(hps) | set(drops) | set(adds)):
        mv, hp = moves.get(key, []), hps.get(key, [])
        ahead = f"{statistics.median(a for a, _ in mv):+.2f}" if mv else "  -  "
        hpd = f"{statistics.median(hp):+.0%}" if hp else "  -  "
        print(f"  {key[0]:11s}  {key[1]:4s}  {len(mv):6d}  {ahead:>10s}                    {len(hp):6d}  "
              f"{hpd:>8s}                    {len(drops.get(key, [])):6d}  {len(adds.get(key, [])):8d}")
    print("  pulled ahead > 0: the game's unit was further along its way than the simulator's (sim too slow /")
    print("  stopped too early); sim HP - game HP > 0: the simulator dealt that unit too little damage.")
    for side in ("blue", "red"):
        d = [x for s, x in towers if s == side]
        if d:
            print(f"towers ({side}): {len(d)} fixes, sim HP - game HP median {statistics.median(d):+.0f} "
                  f"(> 0: the simulator dealt too little damage to these towers)")
    for name, m, _ in runs:
        for side in ("blue", "red"):
            real = sum(v[0] for k, v in m.tower_damage.items() if k.startswith(side))
            sim = sum(v[1] for k, v in m.tower_damage.items() if k.startswith(side))
            if real:
                print(f"  {name} {side} towers: damage in the game {real}, by the simulator's own fighting {sim} "
                      f"({sim / real:.0%})")
    # Who did the damage when the simulator had taken a unit lower than the game shows.
    blame = defaultdict(lambda: defaultdict(int))
    for name, m, _ in runs:
        for g, kind, cls, side, d in m.corrections:
            if kind == "hp" and d["real"] - d["sim"] > 0.2:
                for source, amount in d.get("hit_by", []):
                    blame[cls][source] += amount
    print("\nwhen the simulator had a unit lower than the game: its damage of the 2 s before, by source")
    for cls, by in sorted(blame.items()):
        total = sum(by.values())
        print(f"  {cls:11s} " + ", ".join(f"{s} {a / total:.0%}" for s, a in sorted(by.items(), key=lambda kv: -kv[1])[:5]))
    print("\nbiggest single disagreements (to look at in the recording):")
    for _, text in sorted(events, reverse=True)[:25]:
        print("  " + text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blue", nargs="+", required=True, help="perception caches of blue recordings")
    p.add_argument("--red", nargs="*", default=[], help="the red recordings of the same matches, in order")
    p.add_argument("--both", action="store_true", help="follow from both recordings' detections (red needed)")
    args = p.parse_args()
    data = load_data()
    runs = []
    for i, blue in enumerate(args.blue):
        red = args.red[i] if i < len(args.red) else None
        m, length = run(blue, red, data, args.both)
        runs.append((os.path.splitext(os.path.basename(blue))[0], m, length))
    summarize(runs)


if __name__ == "__main__":
    main()
