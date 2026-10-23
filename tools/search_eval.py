"""Does lookahead make a policy stronger? Matches in the simulator: the policy with python/search.py's lookahead
(blue) against the same policy without it (red, sampled at the same temperature), both deciding at their trained
rhythm.

    .venv\\Scripts\\python.exe tools\\search_eval.py --games 40 [--model runs/v7e/top.pt] [--rollouts 6 --horizon 4]
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from model import load_policy  # noqa: E402
from search import SearchPlayer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runs/v7e/top.pt")
    p.add_argument("--opponent", default=None, help="checkpoint for red (default: --model without lookahead)")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--rollouts", type=int, default=6)
    p.add_argument("--horizon", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    data = ce.load_data()
    deck = ce.DEFAULT_DECK
    net, ckpt = load_policy(os.path.join(ROOT, args.model), "cuda")
    opp_net, opp_ck = (net, ckpt) if not args.opponent else load_policy(os.path.join(ROOT, args.opponent), "cuda")
    searcher = SearchPlayer(net, ckpt, candidates=args.candidates, rollouts=args.rollouts, horizon_s=args.horizon,
                            temperature=args.temperature)
    plain = SearchPlayer(opp_net, opp_ck, temperature=args.temperature, think_below_wait=-1.0)  # never searches
    blue, red = crsim.Team.BLUE, crsim.Team.RED
    wins = draws = 0
    t0 = time.time()
    for g in range(args.games):
        b = crsim.Battle(data, deck, deck, 1000 + g)
        k = 0
        while not b.finished:
            if k % searcher.ticks == 0:
                a = searcher.decide(b, blue)
            else:
                a = 0
            r = plain.decide(b, red) if k % plain.ticks == 0 else 0
            if a:
                crsim.rl.apply_action(b, blue, a)
            if r:
                crsim.rl.apply_action(b, red, r)
            b.step()
            k += 1
        wins += b.winner == blue
        draws += b.winner is None
        done = g + 1
        print(f"game {done}: {'win' if b.winner == blue else 'draw' if b.winner is None else 'loss'} | "
              f"so far {wins}/{done} ({wins / done:.0%}), {searcher.searches / done:.0f} searches a game, "
              f"{(time.time() - t0) / done:.0f} s a game", flush=True)
    print(f"lookahead {wins}/{args.games} wins ({wins / args.games:.0%}), {draws} draws")


if __name__ == "__main__":
    main()
