"""Checks the simulator on the user's defences. From the moment the user put a troop down in their own half, the
simulator plays on by itself for a few seconds - both players' real plays from their own recordings, nothing pulled
back to the screen - and its units are compared with the twin's, which follows the screen. Where the two part, the
simulator plays that defence (a kite, a pull to the king, a block) differently from the game.

    .venv\\Scripts\\python.exe tools\\defense_check.py video44_b03_red [video44_b04_red ...] [--horizon 6]

Each name is the user's own recording of one battle (tools/split_battles.py parts); the other side's recording is
the same name without _red (or with it). For each defence it prints the video time (of the user's recording), the
card and tile, and for every troop near it the most the simulator's own play and the twin parted over the horizon,
with when, and troops that died in one of them only. "!" marks a part of more than --far tiles.
"""
import argparse
import math
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import twin  # noqa: E402
from compare_red_truth import own_plays_timed, true_plays  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import BLUE, TEAM, LiveMatch, Play  # noqa: E402

CACHE = os.path.join(ROOT, "raw", "perception", "cache")
# The fastest troops walk ~1.77 tiles/s: two copies of one can part at most twice that (tiles per s), plus a little
# for where the twin had it to start with.
JUMP_RATE, JUMP_SLACK = 3.6, 0.6


def troops(b, data):
    """{(team is the user's, play index, unit): (card, x, y, what it is after)} of the troops on the field."""
    out = {}
    for e in b.entities:
        if e.is_building or e.hp <= 0 or e.play < 0:
            continue
        card = data.card_ids[e.card_index] if e.card_index >= 0 else "?"
        t = b.find(e.target) if e.target >= 0 else None
        if t is None:
            after = "-"
        elif t.is_building:
            after = "tower"
        else:
            after = data.card_ids[t.card_index] if t.card_index >= 0 else "?"
        out[(e.team == BLUE, e.play, e.unit)] = (card, e.pos.x, e.pos.y, after)
    return out


def check(view, data, horizon, near, far, stats):
    other = view[:-4] if view.endswith("_red") else f"{view}_red"
    frames = twin.load_cache(os.path.join(CACHE, f"{view}.json"))["frames"]
    other_frames = twin.load_cache(os.path.join(CACHE, f"{other}.json"))["frames"]
    own = sorted(p for p in own_plays_timed(frames, data) if p[2] is not None)
    m = LiveMatch(data, red_truth=true_plays(other_frames, data), blue_truth=own)
    # the user's troops put down in their own half: the defences (spells are left out)
    pending = [p for p in own if p[1] not in twin.SPELL_CARDS and p[2][1] < 16.0]
    starts, closed = [], []  # (play, g0, video t, battle copy); (g, troops, out of sight) of the twin, by frame
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0:
            continue
        while pending and pending[0][0] <= m.now + 0.2:  # about to go down: the simulator goes on alone from here
            p = pending.pop(0)
            if p[0] >= m.now - 0.3:
                starts.append((p, m.now, fr["t"], m.battle.clone()))
        # troops the twin took out because the screen stopped showing them (not killed): no death to compare
        unseen = set()
        for (pid, unit), gone in m.gone.items():
            if gone[4] == "drop" and pid in m.sim_index:
                side, index = m.sim_index[pid]
                unseen.add((side == "blue", index, unit))
        closed.append((m.now, troops(m.battle, data), unseen))
    plays = sorted((ev for ev in m.journal if isinstance(ev, Play) and ev.on), key=lambda ev: ev.g)
    lines = []
    for (gp, card, pos), g0, t0, b in starts:
        # the enemy troops near the defence when it goes down, and the user's troops put down from then on
        at_start = troops(b, data)
        watch = {k for k, (c, x, y, _) in at_start.items() if not k[0] and math.hypot(x - pos[0], y - pos[1]) <= near}
        if not watch:
            continue  # nothing of theirs close: not a defence
        todo = [ev for ev in plays if ev.g > g0 + 1e-9]
        frames_ahead = [f for f in closed if g0 < f[0] <= g0 + horizon]
        worst = {}  # key -> (most apart while both have it, when, card, (x, y, after) simulator, the same twin)
        jumped = set()
        end = {}    # key -> [card, last seen alive in the twin, died in the twin (not just unseen), last alive alone]
        k = 0
        for g, real, unseen in frames_ahead:
            while b.time_ms / 1000.0 < g - 1e-9:
                now = b.time_ms / 1000.0
                while k < len(todo) and todo[k].g <= now + 1e-9:
                    ev = todo[k]
                    k += 1
                    b.force_play(TEAM[ev.side], ev.card, None if ev.pos is None else crsim.Vec2(*ev.pos),
                                 max(0, int(round((now - ev.g) * 1000))))
                b.step()
            sim = troops(b, data)
            for key in real:
                if key[0] and key not in at_start:
                    watch.add(key)
            for key in watch:
                e = end.setdefault(key, [None, None, False, None])
                if key in real:
                    e[0], e[1] = real[key][0], g - g0
                elif key in unseen:
                    e[2] = None  # out of sight: whether and when it died in the game is not known
                elif e[1] is not None and e[2] is False:
                    e[2] = True
                if key in sim:
                    e[0] = e[0] or sim[key][0]
                    e[3] = g - g0
                if key in real and key in sim and key not in jumped:
                    d = math.hypot(sim[key][1] - real[key][1], sim[key][2] - real[key][2])
                    if d > JUMP_RATE * (g - g0) + JUMP_SLACK:
                        # farther apart than two copies of a troop could walk: the twin moved its unit to where the
                        # screen showed it (a detector or tracking slip) - no measure of the simulator
                        jumped.add(key)
                        worst.pop(key, None)
                        stats["unit followed with a jump (left out)"] += 1
                    elif d > worst.get(key, (-1,))[0]:
                        worst[key] = (d, g - g0, real[key][0], sim[key][1:4], real[key][1:4])
        if not frames_ahead:
            continue
        span = frames_ahead[-1][0] - g0
        parts, flagged = [], False
        for key, (d, when, c, at_sim, at_real) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
            if not key[0]:
                stats["enemy troops"] += 1
                stats["enemy troops, tiles apart (sum of the most)"] += d
            far_off = d > far
            flagged |= far_off
            where = (f" (simulator ({at_sim[0]:.1f}, {at_sim[1]:.1f}) after {at_sim[2]}, game ({at_real[0]:.1f}, "
                     f"{at_real[1]:.1f}))" if far_off else "")
            parts.append(f"{c} ({'yours' if key[0] else 'theirs'}) {d:.1f} tiles at +{when:.1f} s{where}"
                         f"{' !' if far_off else ''}")
        for key, (c, real_last, real_died, sim_last) in end.items():
            if real_died is None or c is None or sim_last is None:
                continue  # out of sight in the game, or never on in the simulator's own play (a different play order)
            real_end = real_last if real_died else span  # (alive to the end of the horizon)
            sim_end = sim_last if sim_last is not None else 0.0
            if sim_last is not None and sim_last >= span - 1e-9:
                sim_end = span
            if abs(real_end - sim_end) > 1.5:
                flagged = True
                who = "yours" if key[0] else "theirs"
                if sim_end < real_end:
                    parts.append(f"{c} ({who}) dies at +{sim_end:.1f} s in the simulator, "
                                 f"{'at +%.1f s' % real_end if real_died else 'lives on'} in the game !")
                else:
                    parts.append(f"{c} ({who}) dies at +{real_end:.1f} s in the game, "
                                 f"{'at +%.1f s' % sim_end if sim_end < span else 'lives on'} in the simulator !")
        if not parts:
            continue
        stats["defences"] += 1
        stats["parted"] += int(flagged)
        mins, secs = divmod(t0, 60)
        lines.append(f"{'!' if flagged else ' '} {view} video {int(mins)}:{secs:04.1f} (game {gp:.0f} s): {card} at "
                     f"({pos[0]:.1f}, {pos[1]:.1f}) | " + "; ".join(parts))
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("views", nargs="+")
    p.add_argument("--horizon", type=float, default=6.0, help="s the simulator plays on by itself")
    p.add_argument("--near", type=float, default=6.0, help="tiles: enemy troops this close to the defence are watched")
    p.add_argument("--far", type=float, default=1.5, help="tiles apart that count as the simulator doing it otherwise")
    p.add_argument("--data", default=None, help="another game data directory (e.g. with changed overrides)")
    args = p.parse_args()
    data = load_data(args.data) if args.data else load_data()
    stats = Counter()
    for view in args.views:
        try:
            for line in check(view, data, args.horizon, args.near, args.far, stats):
                print(line, flush=True)
        except Exception as e:  # a battle the twin cannot follow
            print(f"{view}: skipped ({type(e).__name__}: {e})", flush=True)
    n = max(1, stats["defences"])
    if stats["unit followed with a jump (left out)"]:
        print(f"{stats['unit followed with a jump (left out)']} troops left out: the twin moved them farther than a "
              f"troop walks (a detector or tracking slip)")
    if stats["enemy troops"]:
        print(f"enemy troops near the defences: {stats['enemy troops']}, the simulator's own play and the game "
              f"at most {stats['enemy troops, tiles apart (sum of the most)'] / stats['enemy troops']:.2f} tiles apart "
              f"on average")
    print(f"{stats['defences']} defences: the simulator on its own parted from the game (> {args.far} tiles, or a "
          f"troop dead in one only) in {stats['parted']} ({stats['parted'] / n:.0%})")


if __name__ == "__main__":
    main()
