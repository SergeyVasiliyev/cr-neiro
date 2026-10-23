"""The user's play against the live bot as examples for the policy, from the bot's recording alone: the twin
follows the battles and finds the user's (red) plays; the simulated battle just before each one, seen from the
user's side, is what they decided on, and the play is the action. Moments with no play of theirs around are
examples of waiting.

    .venv\\Scripts\\python.exe tools\\user_examples.py video33 [more recordings...] [--out raw/bc/user.npz]

The user's hand and elixir are the twin's guesses (only the bot's own are read): the played card is put in their
hand and their elixir raised to its cost where the guess says otherwise. Positions go to the nearest key spot.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import twin  # noqa: E402
from crsim_env import DEFAULT_DECK, OBS_KEYS, load_data  # noqa: E402
from live.session import LiveMatch  # noqa: E402

RED = crsim.Team.RED
CELLS = crsim.rl.CELLS
GRID_W = crsim.rl.GRID_W
W, H = 18.0, 32.0
WAIT_EVERY = 1.0   # s between waiting examples
WAIT_CLEAR = 1.5   # s from any play of the user


def observe(b, data, card=None, stats=None):
    """The user's observation of battle b (key spots on), with card made playable if given; (obs, hand)."""
    p = b.player(RED)
    hand = list(p.hand)
    if card is not None:
        if card not in hand:
            stats["play: card put in their hand"] += 1
            slot = next((i for i, c in enumerate(hand) if not c), 0)
            hand[slot] = card
            rest = [c for c in DEFAULT_DECK if c not in hand]
            try:
                b.set_hand(RED, hand, rest[0])
            except ValueError:
                return None, None
        cost = data.card(card).elixir
        if p.elixir < cost:
            stats["play: elixir raised to its cost"] += 1
            b.set_elixir(RED, cost + 0.01)
    obs = crsim.rl.encode(b, RED, key_spots=True)
    return {key: np.array(obs[key]) for key in OBS_KEYS}, list(b.player(RED).hand)


def examples(name, data, stats):
    frames = twin.load_cache(os.path.join(ROOT, "raw", "perception", "cache", f"{name}.json"))["frames"]
    m = LiveMatch(data)
    history = []          # (game s, battle clone) of the last 6 s (plays are found a few seconds late)
    seen = 0              # red plays already turned into examples
    rows, actions, plays_g = [], [], []
    last_wait, last_g = -1e9, None
    waits = []            # (game s, obs) candidates, kept once it is clear no play was near
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0 or m.battle.finished:
            continue
        if last_g is not None and g < last_g - 5:  # a new battle
            history, seen, plays_g, last_wait = [], len(m.red_plays), [], -1e9
            m_waits, waits = waits, []
            for wg, wobs in m_waits:
                rows.append(wobs)
                actions.append(0)
                stats["wait"] += 1
        last_g = g
        while seen < len(m.red_plays):
            gp, card, pos, _ = m.red_plays[seen]
            seen += 1
            plays_g.append(gp)
            if pos is None:
                stats["play: no position"] += 1
                continue
            before = [h for h in history if h[0] <= gp]
            if not before:
                stats["play: no state from before it"] += 1
                continue
            b = before[-1][1].clone()
            obs, hand = observe(b, data, card, stats)
            if obs is None or card not in hand:
                stats["play: hand could not be set"] += 1
                continue
            slot = hand.index(card)
            mask = obs["action_mask"][1 + slot * CELLS:1 + (slot + 1) * CELLS]
            cells = np.flatnonzero(mask)
            if len(cells) == 0:
                stats["play: no legal spot"] += 1
                continue
            px, py = W - pos[0], H - pos[1]  # the user's own view: their half at the bottom
            centers = np.stack([cells % GRID_W + 0.5, cells // GRID_W + 0.5], 1)
            d = np.hypot(centers[:, 0] - px, centers[:, 1] - py)
            best = int(np.argmin(d))
            stats["play"] += 1
            stats["play: key spot within 1.5 tiles"] += int(d[best] <= 1.5)
            stats[f"card {card}"] += 1
            rows.append(obs)
            actions.append(1 + slot * CELLS + int(cells[best]))
        history.append((g, m.battle.clone()))
        while history and g - history[0][0] > 6.0:
            history.pop(0)
        if g - last_wait >= WAIT_EVERY:
            obs, _ = observe(m.battle.clone(), data)
            if obs["action_mask"].sum() > 1:
                last_wait = g
                waits.append((g, obs))
        # waits far enough from every play of theirs are kept (the plays are seen a while after they are made)
        keep = [w for w in waits if g - w[0] > 4.0]
        for wg, wobs in keep:
            if all(abs(wg - pg) > WAIT_CLEAR for pg in plays_g):
                rows.append(wobs)
                actions.append(0)
                stats["wait"] += 1
        waits = [w for w in waits if g - w[0] <= 4.0]
    return rows, actions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings", nargs="+")
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "bc", "user.npz"))
    args = p.parse_args()
    data = load_data()
    rows, actions, stats = [], [], Counter()
    for name in args.recordings:
        r, a = examples(name, data, stats)
        rows += r
        actions += a
        print(f"{name}: {sum(1 for x in a if x)} plays, {sum(1 for x in a if not x)} waits", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, action=np.array(actions, np.int64),
                        **{key: np.stack([r[key] for r in rows]) for key in OBS_KEYS})
    print(dict(sorted(stats.items())))
    print(f"{len(actions)} examples -> {args.out}")


if __name__ == "__main__":
    main()
