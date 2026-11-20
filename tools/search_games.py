"""Games for distilling the lookahead into the network (expert iteration): the policy with python/search.py's
lookahead plays itself in the simulator, both sides searching, and every position it searched is saved with the
move the search chose (and the candidates' scores); a share of the positions it did not search is saved too, so
training can keep the network as it was there. tools/distill_search.py then trains on the shards.

    .venv\\Scripts\\python.exe tools\\search_games.py --out raw\\search_games\\r1 --games 200 --seed 1 [--model runs/v7e/top.pt]

Several of these can run side by side (different --seed): the GPU is far from full with one.
--fair: each side searches only from what it can know (search.py), so the moves it finds are learnable from what the
net sees. --think-at-push: it searches also while an enemy troop comes (the moments the net learned to sit out).
--push-share: games starting at the user's recorded pushes, red putting down his real next cards first (--follow-user).
Shard per 20 games: <out>/shard_<seed>_<n>.npz with the observations (OBS_KEYS), "target" (the move to learn; -1
for a plain position), "searched", "team", "result" (+1 / 0 / -1 for that side at the end of the game), "cands"
(the moves the search compared, -1 = none) and "q" (their mean scores, the critic's view of the result).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from crsim_env import OBS_KEYS  # noqa: E402
from model import load_policy, obs_to_torch  # noqa: E402
from search import SearchPlayer  # noqa: E402

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
MAX_CANDS = 10  # waiting + up to 9 plays (the policy's best + --explore)


def save(path, rows):
    out = {k: np.stack([r["obs"][k] for r in rows]) for k in OBS_KEYS}
    for k in ("target", "searched", "team", "result", "cands", "q"):
        out[k] = np.array([r[k] for r in rows])
    np.savez_compressed(path, **out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runs/v7e/top.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--plain-share", type=float, default=0.1, help="share of the positions not searched to keep")
    p.add_argument("--rollouts", type=int, default=6)
    p.add_argument("--horizon", type=float, default=4.0)
    p.add_argument("--think", type=float, default=0.9, help="search when the policy's chance of waiting is below this")
    p.add_argument("--explore", type=int, default=0, help="more candidates: a random card and spot each")
    p.add_argument("--think-anyway", type=float, default=0.0,
                   help="share of the moments the policy is sure to wait that are searched anyway")
    p.add_argument("--per-shard", type=int, default=20)
    p.add_argument("--fair", action="store_true",
                   help="each side looks ahead only from what it can know (search.py fair): the other's blueprints "
                        "unseen, its elixir off like the live twin's - moves the net can learn from what it sees")
    p.add_argument("--think-at-push", action="store_true",
                   help="search also while an enemy troop is at the bridge or on our side (search.py)")
    p.add_argument("--push-share", type=float, default=0.0,
                   help="share of the games starting at the user's pushes (tools/push_defence_check.py positions of "
                        "the two-sided recordings: the bot blue, the user red with 5+ elixir of troops on blue's side)")
    p.add_argument("--follow-user", type=float, default=15.0,
                   help="with --push-share: red puts down the user's real next cards for this long (s) first")
    args = p.parse_args()
    out = os.path.join(ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = ce.load_data()
    CARD_ELIXIR.update({cid: card.elixir for cid, card in data.cards.items()})
    deck = ce.DEFAULT_DECK
    net, ckpt = load_policy(os.path.join(ROOT, args.model), "cuda")
    sp = SearchPlayer(net, ckpt, rollouts=args.rollouts, horizon_s=args.horizon, think_below_wait=args.think,
                      explore=args.explore, think_anyway=args.think_anyway, seed=args.seed, fair=args.fair,
                      think_at_push=args.think_at_push)
    starts, scripts = [], []
    if args.push_share > 0:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import push_defence_check as pdc
        starts = pdc.push_positions(data, log=lambda s: print(s, flush=True), follow=args.follow_user,
                                    scripts=scripts if args.follow_user > 0 else None)
    rows, shard, wins, t0 = [], 0, 0, time.time()
    made = dropped = from_push = 0
    for g in range(args.games):
        script = None
        if starts and rng.random() < args.push_share:
            k0 = int(rng.integers(len(starts)))
            b = starts[k0].clone()
            b.set_follow_real(False)  # (a copy of the twin's battle: it ends by itself here)
            if scripts and scripts[k0][2]:
                script = [scripts[k0][1], list(scripts[k0][2])]  # until_ms, the plays still to come
            from_push += 1
        else:
            b = crsim.Battle(data, deck, deck, args.seed * 100000 + g)
        game_rows = []
        k = 0
        while not b.finished:
            red_scripted = script is not None and b.time_ms < script[0]
            if script is not None:
                m, d = run_script(b, script)
                made, dropped = made + m, dropped + d
                if b.time_ms >= script[0]:
                    dropped += len(script[1])
                    script = None
            if k % sp.ticks == 0:
                obs = [sp._encode(b, team) for team in (BLUE, RED)]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    logits, _ = net(obs_to_torch({key: np.stack([o[key] for o in obs]) for key in OBS_KEYS}, "cuda"))
                logits = logits.float()
                moves = []
                for i, team in enumerate((BLUE, RED)):
                    if team == RED and red_scripted:
                        continue  # the user's real cards for now, not a move to learn
                    a = sp.decide(b, team, logits[i])
                    moves.append((team, a))
                    cands, q = np.full(MAX_CANDS, -1, np.int32), np.full(MAX_CANDS, np.nan, np.float32)
                    if sp.last is not None:
                        cands[:len(sp.last[0])], q[:len(sp.last[1])] = sp.last[0], sp.last[1]
                        game_rows.append(dict(obs=obs[i], target=a, searched=True, team=int(i), cands=cands, q=q))
                    elif rng.random() < args.plain_share:
                        game_rows.append(dict(obs=obs[i], target=-1, searched=False, team=int(i), cands=cands, q=q))
                for team, a in moves:
                    if a:
                        crsim.rl.apply_action(b, team, a)
            b.step()
            k += 1
        winner = None if b.winner is None else (0 if b.winner == BLUE else 1)
        wins += winner == 0
        for r in game_rows:
            r["result"] = 0 if winner is None else (1 if r["team"] == winner else -1)
        rows += game_rows
        if (g + 1) % args.per_shard == 0 or g + 1 == args.games:
            save(os.path.join(out, f"shard_{args.seed}_{shard}.npz"), rows)
            searched = sum(r["searched"] for r in rows)
            print(f"games {g + 1}/{args.games}: shard {shard} with {len(rows)} positions ({searched} searched), "
                  f"{(time.time() - t0) / (g + 1):.1f} s a game, blue won {wins}"
                  + (f"; {from_push} from the user's pushes, his cards put down {made}, left out {dropped}"
                     if starts else ""), flush=True)
            rows, shard = [], shard + 1
    print("=== search games done", flush=True)


def run_script(b, script):
    """Puts down red's scripted (the user's real) cards that are due, like BatchEnv::run_script: in order, each
    when its card is in hand and paid for; a spot just off where it may go (read off the screen) moves to the
    nearest one within 2 tiles. Returns (made, dropped) this tick."""
    made = 0
    plays = script[1]
    while plays and plays[0][0] <= b.time_ms:
        _, card, x, y = plays[0]
        p = b.player(RED)
        hand = list(p.hand)
        if card not in hand:
            break  # not back in hand yet: wait (what is still due when the script ends is left out)
        slot = hand.index(card)
        if b.play(RED, slot, crsim.Vec2(x, y)):
            plays.pop(0)
            made += 1
            continue
        if p.elixir + 1e-9 < CARD_ELIXIR[card]:
            break  # not paid for yet
        best = None
        for dx in (0.0, -1.0, 1.0, -2.0, 2.0):
            for dy in (0.0, -1.0, 1.0, -2.0, 2.0):
                q = crsim.Vec2(x + dx, y + dy)
                if dx * dx + dy * dy <= 4.0 and not b.check_play(RED, slot, q):
                    if best is None or dx * dx + dy * dy < best[0]:
                        best = (dx * dx + dy * dy, q)
        if best is not None and b.play(RED, slot, best[1]):
            plays.pop(0)
            made += 1
            continue
        plays.pop(0)  # nowhere near to put it
        return made, 1
    return made, 0


CARD_ELIXIR = {}


if __name__ == "__main__":
    main()
