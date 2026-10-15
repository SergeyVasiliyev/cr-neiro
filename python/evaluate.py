"""Measures a trained policy (or a bot) against scripted bots or another checkpoint.

    .venv\\Scripts\\python.exe python\\evaluate.py --model runs\\first\\latest.pt --opponent heuristic --games 500
    .venv\\Scripts\\python.exe python\\evaluate.py --model runs\\a\\latest.pt --opponent runs\\b\\latest.pt
"""
import argparse
import os
import time

import numpy as np
import torch

import crsim_env as ce
from model import load_policy, obs_to_torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint, or a bot name (noop|random|heuristic)")
    p.add_argument("--opponent", default="heuristic", help="bot name or checkpoint path")
    p.add_argument("--deck", default=",".join(ce.DEFAULT_DECK))
    p.add_argument("--opp-deck", default=None)
    p.add_argument("--games", type=int, default=500)
    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    p.add_argument("--decision-ticks", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    data = ce.load_data()
    deck = ce.parse_deck(args.deck, data)
    opp_deck = ce.parse_deck(args.opp_deck, data) if args.opp_deck else deck
    players, key_spots = [], []
    for spec in (args.model, args.opponent):
        if spec in ce.BOT_KINDS:
            players.append(spec)
            key_spots.append(False)
        elif os.path.exists(spec):
            net, ckpt = load_policy(spec, args.device)
            players.append(net)
            key_spots.append(bool(ckpt.get("key_spots", False)))  # the action space each policy was trained on
        else:
            raise SystemExit(f"{spec!r} is neither a bot name nor a checkpoint file")

    bots = tuple(pl if isinstance(pl, str) else "noop" for pl in players)
    env = ce.VecEnv(data, [(deck, opp_deck)], num_envs=min(args.envs, args.games), bots=bots,
                    decision_ticks=args.decision_ticks, seed=args.seed, key_spots=tuple(key_spots))
    results, crowns, lengths = [], [], []
    started = env.num_envs
    active = np.ones(env.num_envs, dtype=bool)  # envs still allowed to finish an episode we count
    t0 = time.time()
    while len(results) < args.games:
        actions = np.full((env.num_envs, 2), ce.BOT_ACTION, dtype=np.int32)
        for side, pl in enumerate(players):
            if not isinstance(pl, str):
                obs = obs_to_torch(env.obs(side), args.device)
                actions[:, side] = pl.act(obs, greedy=args.greedy)[0].cpu().numpy()
        _, done = env.step(actions)
        for i in np.flatnonzero(done & active):
            w = int(env.winner[i])
            results.append(1.0 if w == 0 else (0.5 if w == 2 else 0.0))
            crowns.append((int(env.crowns[i, 0]), int(env.crowns[i, 1])))
            lengths.append(int(env.episode_steps[i]) * args.decision_ticks * 50 / 1000)
            if started >= args.games:
                active[i] = False
            started += 1

    results = np.array(results[:args.games])
    wins, draws = (results == 1).sum(), (results == 0.5).sum()
    c = np.array(crowns[:args.games])
    name = lambda s: os.path.basename(os.path.dirname(s)) + "/" + os.path.basename(s) if os.path.exists(s) else s
    print(f"{name(args.model)} (blue) vs {name(args.opponent)} (red), {len(results)} games, "
          f"{'greedy' if args.greedy else 'sampled'} actions, {time.time() - t0:.1f}s")
    print(f"  wins {wins} ({wins / len(results):.1%}), draws {draws}, losses {len(results) - wins - draws}")
    print(f"  avg crowns {c[:, 0].mean():.2f} : {c[:, 1].mean():.2f}, avg match length {np.mean(lengths):.0f}s")
    se = np.sqrt(results.mean() * (1 - results.mean()) / len(results))
    print(f"  score {results.mean():.3f} +- {1.96 * se:.3f} (95%)")


if __name__ == "__main__":
    main()
