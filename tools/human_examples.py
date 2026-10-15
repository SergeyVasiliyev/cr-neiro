"""Turns two-sided recordings into examples of human play for the policy: what the player saw, what they did.

    .venv\\Scripts\\python.exe tools\\human_examples.py [--matches video8 ...] [--out raw/bc/human.npz]

Each player of a battle recorded from both sides is a human whose every play is known from their own hand
(card, moment, spot). The twin follows the battle with both players' real plays, so at the moment a
player picked a card up (the last frame it was still in their hand) the simulated battle is what they
were looking at: that player's observation, exactly as the policy gets it (key spots on), is the example,
and the play is the action - the card's hand slot and the key spot nearest to where it went down. Moments
with no play around them are examples of waiting. Both recordings of a battle are used, each from its own
player's side (their own half at the bottom), so one battle gives two players' worth of examples.
"""
import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import twin  # noqa: E402
from compare_red_truth import own_plays_timed, true_plays  # noqa: E402
from crsim_env import OBS_KEYS, load_data  # noqa: E402
from live.session import BLUE, LiveMatch  # noqa: E402

CELLS = crsim.rl.CELLS
GRID_W = crsim.rl.GRID_W
WAIT_EVERY = 1.0   # s between "waiting" examples
WAIT_CLEAR = 1.5   # s from any play of that player


def examples(view, other, data, stats):
    """(obs rows, actions) of the player whose recording `view` is (their side at the bottom)."""
    cache = os.path.join(ROOT, "raw", "perception", "cache")
    frames = twin.load_cache(os.path.join(cache, f"{view}.json"))["frames"]
    other_frames = twin.load_cache(os.path.join(cache, f"{other}.json"))["frames"]
    own = sorted(p for p in own_plays_timed(frames, data) if p[2] is not None)
    m = LiveMatch(data, red_truth=true_plays(other_frames, data), blue_truth=own)
    history = []  # (g, obs, hand) of the last 2 s
    obs_rows, actions = [], []
    k, last_wait, prev_g = 0, -1e9, None
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0 or m.battle.finished:
            continue
        # Plays made since the last frame: the player decided when they picked the card up - the last moment
        # it was still in their hand.
        while k < len(own) and own[k][0] <= g:
            gp, card, pos = own[k]
            k += 1
            before = [h for h in history if h[0] <= gp and card in h[2]]
            if not before:
                stats["play: card not seen in the hand before it"] += 1
                continue
            _, obs, hand = before[-1]
            slot = hand.index(card)
            mask = obs["action_mask"][1 + slot * CELLS:1 + (slot + 1) * CELLS]
            cells = np.flatnonzero(mask)
            if len(cells) == 0:
                stats["play: no legal spot for it (elixir)"] += 1
                continue
            centers = np.stack([cells % GRID_W + 0.5, cells // GRID_W + 0.5], 1)
            d = np.hypot(centers[:, 0] - pos[0], centers[:, 1] - pos[1])
            best = int(np.argmin(d))
            stats["play"] += 1
            stats["play: spot within 1.5 tiles"] += int(d[best] <= 1.5)
            obs_rows.append(obs)
            actions.append(1 + slot * CELLS + int(cells[best]))
        hand = list(m.battle.player(BLUE).hand)
        obs = crsim.rl.encode(m.battle, BLUE, key_spots=True)
        obs = {key: np.array(obs[key]) for key in OBS_KEYS}
        history.append((g, obs, hand))
        while history and g - history[0][0] > 2.0:
            history.pop(0)
        quiet = all(abs(g - p[0]) > WAIT_CLEAR for p in own)
        if quiet and g - last_wait >= WAIT_EVERY and obs["action_mask"].sum() > 1:
            last_wait = g
            obs_rows.append(obs)
            actions.append(0)
            stats["wait"] += 1
        prev_g = g
    return obs_rows, actions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matches", nargs="+",
                   default=["video8", "video9", "video10", "video11", "video12", "video13", "video14", "video15"])
    p.add_argument("--players", nargs="*", default=None,
                   help="only these recordings' players are people (e.g. video22_red - its opponent was the bot); "
                        "default: both players of every --matches battle")
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "bc", "human.npz"))
    args = p.parse_args()
    data = load_data()
    rows, actions, stats = [], [], Counter()
    if args.players:
        pairs = [(v, v[:-4] if v.endswith("_red") else f"{v}_red") for v in args.players]
    else:
        pairs = [pair for m in args.matches for pair in ((m, f"{m}_red"), (f"{m}_red", m))]
    for view, other in pairs:
        r, a = examples(view, other, data, stats)
        rows += r
        actions += a
        print(f"{view}: {sum(1 for x in a if x)} plays, {sum(1 for x in a if not x)} waits", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, action=np.array(actions, np.int64),
                        **{key: np.stack([r[key] for r in rows]) for key in OBS_KEYS})
    print(dict(stats))
    print(f"{len(actions)} examples -> {args.out}")


if __name__ == "__main__":
    main()
