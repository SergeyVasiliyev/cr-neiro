"""Start positions for training, from the recorded battles: the digital twin follows each recording (with both
players' real plays where a battle was recorded from both sides) and every few seconds of the battle a copy of the
simulated battle is kept. They are moments of human play the policy's own games never reach - a push built behind
the king, a tower left low, an elixir lead - and crsim_env.VecEnv.set_starts starts episodes from them, so the
critics learn how such positions end and the policy what pays from there. Nothing tells it what to play.

    .venv\\Scripts\\python.exe python\\human_starts.py   (builds them and prints what they are like)
"""
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import crsim  # noqa: E402
import twin  # noqa: E402
from live.session import LiveMatch  # noqa: E402

CACHE = os.path.join(ROOT, "raw", "perception", "cache")
BLUE, RED = crsim.Team.BLUE, crsim.Team.RED


def recordings():
    """One recording per battle, (name, its red side's recording or None): of a two-sided pair the blue one, the
    red one giving red's real plays. Left out: old parts cut from a longer recording (name_1, name_2, name_v1_...),
    and a recording cut into battles by tools/split_battles.py (name_b01, name_b02 ...): its battles are used (not
    the ones kept apart as name_x03: tests, not play)."""
    names = {f[:-5] for f in os.listdir(CACHE) if f.endswith(".json")}
    split = {re.sub(r"_b\d+$", "", n) for n in names if re.search(r"_b\d+$", n)}
    return [(n, f"{n}_red" if f"{n}_red" in names else None) for n in sorted(names)
            if not n.endswith("_red") and not re.search(r"_\d+$|_v\d|_x\d+$", n) and n not in split]


def follow(name, red_name, data):
    """(frames, LiveMatch) for one recording."""
    frames = twin.load_cache(os.path.join(CACHE, f"{name}.json"))["frames"]
    if red_name is None:
        return frames, LiveMatch(data)
    from compare_red_truth import own_plays_timed, true_plays
    red_frames = twin.load_cache(os.path.join(CACHE, f"{red_name}.json"))["frames"]
    own = sorted(p for p in own_plays_timed(frames, data) if p[2] is not None)
    return frames, LiveMatch(data, red_truth=true_plays(red_frames, data), blue_truth=own)


def recording_starts(name, red_name, data, every=1.5, skip_first=6.0, skip_last=3.0, min_battle=20.0):
    """[(battle no, game s, Battle)] of one recording: a copy of the twin's battle every `every` s of game time,
    from skip_first s on (the twin syncs to the screen first) to skip_last s before the recording of that battle
    ends (its last frames are the least sure); battles shorter than min_battle s (clips) are left out."""
    frames, m = follow(name, red_name, data)
    out, cur = [], []
    no, last_g, next_g = 0, None, skip_first

    def close():
        if cur and cur[-1][1] - cur[0][1] >= min_battle:
            end = cur[-1][1]
            out.extend(c for c in cur if c[1] <= end - skip_last)

    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0 or m.battle.finished:
            continue
        if last_g is not None and g < last_g - 5:  # a new battle in the same recording
            close()
            cur, no, next_g = [], no + 1, skip_first
        last_g = g
        if g >= next_g:
            cur.append((no, g, m.battle.clone()))
            next_g = g + every
    close()
    return out


def build(data, every=1.5, names=None, log=print):
    """Start positions from every recording (or `names`): a list of Battles, built on `data` (the GameData the
    training envs use - it must outlive them), and a list of (recording, battle no, game s) beside it."""
    battles, where = [], []
    for name, red in recordings():
        if names and name not in names:
            continue
        try:
            got = recording_starts(name, red, data, every)
        except Exception as e:  # a recording the twin cannot follow costs only its positions
            log(f"human starts: {name} skipped ({type(e).__name__}: {e})")
            continue
        battles += [b for _, _, b in got]
        where += [(name, no, g) for no, g, _ in got]
    log(f"human starts: {len(battles)} positions from {len({(w[0], w[1]) for w in where})} battles "
        f"({len({w[0] for w in where})} recordings), one every {every} s")
    return battles, where


def describe(battles, data):
    """What the positions are like: game time, towers, elixir, pushes behind the king."""
    giant = data.cards["giant"].index
    c = Counter()
    for b in battles:
        c["all"] += 1
        c["2:00 on"] += b.time_ms >= 120_000
        c["overtime"] += b.time_ms >= data.rules.regulation_ms
        towers = [b.find(b.princess_tower(team, side)) for team in (BLUE, RED) for side in (0, 1)]
        c["a princess tower down"] += any(t is None for t in towers)
        c["a princess tower below 35%"] += any(t is not None and t.hp < 0.35 * t.max_hp for t in towers)
        e = b.player(BLUE).elixir - b.player(RED).elixir
        c["elixir 3+ apart"] += abs(e) >= 3
        back = any(x.card_index == giant and not x.is_building and
                   ((x.team == BLUE and x.pos.y < 3.0) or (x.team == RED and x.pos.y > 29.0)) for x in b.entities)
        c["a giant behind its king"] += back
    n = max(1, c.pop("all"))
    return f"{n} positions: " + ", ".join(f"{k} {v / n:.0%}" for k, v in c.items())


if __name__ == "__main__":
    from crsim_env import load_data
    d = load_data()
    bs, _ = build(d)
    print(describe(bs, d))
