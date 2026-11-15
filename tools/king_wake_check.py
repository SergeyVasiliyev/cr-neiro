"""Does the net wake the sleeping enemy king with its spells, and what does that cost it?

Live on 2026-11-11 h1 cycle 4 threw arrows and fireballs at the user's king tower while both princess towers stood
(the user's archers often go behind the king); a hit wakes the king, who then shoots for the whole game. The simulator
wakes him too (core/src/battle.cpp: any damage to a sleeping king). This plays --model (blue) against --opponent (red)
as python/evaluate.py does and counts, for each side, the games where its spell (cast at most --within s before) woke
the other king while both his princess towers stood; and blue's score in those games and in the others.

    .venv\\Scripts\\python.exe tools\\king_wake_check.py --model runs/h1/final_cycle4.pt --opponent runs/h1/final_cycle3.pt
    .venv\\Scripts\\python.exe tools\\king_wake_check.py --model runs/h1/final_cycle4.pt --back-starts
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim_env as ce  # noqa: E402
from model import load_policy, obs_to_torch  # noqa: E402

# global features (core/src/rl.cpp encode): 8.. own king hp, own princess left/right, own king awake; 12.. the enemy's
OWN_PRINCESS, OWN_AWAKE, ENEMY_PRINCESS, ENEMY_AWAKE = (9, 10), 11, (13, 14), 15


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--opponent", default=None, help="red's net (default: the model itself)")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--temp", type=float, default=0.5, help="both sample at this temperature (as live)")
    p.add_argument("--within", type=float, default=3.0, help="s from a side's spell to the wake it is blamed for")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--back-starts", action="store_true",
                   help="every game starts at a moment of the recorded two-sided battles where the user (red) has troops "
                        "behind his king or a princess tower and his king sleeps (tools/push_defence_check.py "
                        "back_positions; train_v7.py --back-starts): where h1 cycle 4 spelled the king live")
    args = p.parse_args()
    args.opponent = args.opponent or args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    data = ce.load_data()
    deck = ce.parse_deck(",".join(ce.DEFAULT_DECK), data)
    nets, spots, rhythm = [], [], []
    for spec in (args.model, args.opponent):
        net, ckpt = load_policy(spec, device)
        nets.append(net)
        spots.append(int(ckpt.get("key_spots", 0)))
        rhythm.append(int(ckpt.get("args", {}).get("decision_ticks", 4)))
    step_ticks = math.gcd(*rhythm)
    every = [r // step_ticks for r in rhythm]
    step_s = step_ticks * 0.05
    env = ce.VecEnv(data, [(deck, deck)], num_envs=min(args.envs, args.games), bots=("noop", "noop"),
                    decision_ticks=step_ticks, seed=args.seed, key_spots=tuple(spots))
    if args.back_starts:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import push_defence_check
        starts = push_defence_check.back_positions(data, log=lambda m: print(m, flush=True))
        print(f"every game from one of {env.set_starts(starts, 1.0)} back positions (blue: the bot's side)", flush=True)
        env.reset()
    spell_token = np.zeros(ce.crsim.rl.vocab_size(data), dtype=bool)
    for card in data.cards.values():
        spell_token[ce.crsim.rl.FIRST_CARD_TOKEN + card.index] = card.is_spell

    n = env.num_envs
    last_spell = np.full((n, 2), -1e9)  # step of each side's last spell
    woke = np.zeros((n, 2), dtype=bool)  # this game: side s's spell woke the other king, both his princesses up
    spells_at_sleeping = np.zeros(2)
    results, wakes = [], []
    started, active, step_no = n, np.ones(n, dtype=bool), 0
    t0 = time.time()
    while len(results) < args.games:
        actions = np.zeros((n, 2), dtype=np.int32)
        for side in (0, 1):
            if step_no % every[side]:
                continue
            obs = obs_to_torch(env.obs(side), device)
            with torch.no_grad():
                logits = nets[side](obs)[0].float() / args.temp
            actions[:, side] = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        g = env.obs(0)["global"].copy()  # blue's view: own = blue, enemy = red
        hands = [env.obs(side)["hand"].copy() for side in (0, 1)]
        _, done = env.step(actions)
        done = np.asarray(done).astype(bool)
        step_no += 1
        g2 = env.obs(0)["global"]
        for side in (0, 1):
            taken = env.env.taken[:, side]
            for i in np.flatnonzero(taken > 0):
                if spell_token[hands[side][i, (taken[i] - 1) // ce.CELLS]]:
                    last_spell[i, side] = step_no
                    asleep = g[i, ENEMY_AWAKE if side == 0 else OWN_AWAKE] < 0.5
                    spells_at_sleeping[side] += asleep and active[i]
        for side, (awake, princess) in enumerate(((ENEMY_AWAKE, ENEMY_PRINCESS), (OWN_AWAKE, OWN_PRINCESS))):
            # both princess towers still up after the step: not a wake from a tower that just fell
            wake = (~done) & (g[:, awake] < 0.5) & (g2[:, awake] > 0.5) & (g2[:, princess[0]] > 0) & (g2[:, princess[1]] > 0)
            wake &= step_no - last_spell[:, side] <= args.within / step_s
            woke[wake, side] = True
        for i in np.flatnonzero(done & active):
            w = int(env.winner[i])
            results.append(1.0 if w == 0 else (0.5 if w == 2 else 0.0))
            wakes.append(woke[i].copy())
            if started >= args.games:
                active[i] = False
            started += 1
        woke[done] = False
        last_spell[done] = -1e9

    r = np.array(results[:args.games])
    wk = np.array(wakes[:args.games])
    name = lambda s: os.path.basename(os.path.dirname(s)) + "/" + os.path.basename(s)
    print(f"{name(args.model)} (blue) vs {name(args.opponent)} (red), {len(r)} games at temp {args.temp}, "
          f"{time.time() - t0:.0f} s: blue's score {r.mean():.3f}")
    for side, who in enumerate(("blue", "red")):
        sel = wk[:, side]
        print(f"  {who}'s spell woke the other king (both princess towers up) in {sel.mean():.1%} of the games "
              f"({sel.sum()})")
        if sel.sum() >= 5 and (~sel).sum() >= 5:
            print(f"    blue's score in those games {r[sel].mean():.3f}, in the others {r[~sel].mean():.3f}")
    print(f"  spells cast while the other king slept (not only at him): blue {spells_at_sleeping[0] / len(r):.1f}, "
          f"red {spells_at_sleeping[1] / len(r):.1f} a game")


if __name__ == "__main__":
    main()
