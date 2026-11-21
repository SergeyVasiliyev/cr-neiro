"""Measures a trained policy (or a bot) against scripted bots or another checkpoint.

    .venv\\Scripts\\python.exe python\\evaluate.py --model runs\\first\\latest.pt --opponent heuristic --games 500
    .venv\\Scripts\\python.exe python\\evaluate.py --model runs\\a\\latest.pt --opponent runs\\b\\latest.pt
"""
import argparse
import math
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
    p.add_argument("--model-temp", type=float, default=None,
                   help="the model (blue) samples its moves at this temperature (1 = as trained, < 1 sharper) while "
                        "the opponent plays as --greedy says")
    p.add_argument("--opp-temp", type=float, default=None,
                   help="the opponent (a checkpoint) samples at this temperature; with --model-temp 0.5 both play as live")
    p.add_argument("--native-rhythm", action="store_true",
                   help="each policy decides at the rhythm it was trained at (its checkpoint's decision_ticks), "
                        "the match stepping at their greatest common divisor; --decision-ticks is then ignored")
    p.add_argument("--human-starts", type=float, default=0.0,
                   help="this share of the games starts at a moment of a recorded human battle (python/human_starts.py). "
                        "Those positions are not even for blue and red: compare a net from both sides")
    p.add_argument("--user-starts", choices=("push", "back", "both"), default=None,
                   help="every game starts at a moment of the recorded two-sided battles - the user's pushes on blue's "
                        "side (push), his troops behind his king (back), or both - and red (his side) puts down his real "
                        "next cards for --follow-user s, then the opponent plays on. Blue (the model) is the bot's side: "
                        "how it copes with what he really did, not with a net's follow-up. Compare nets as blue against "
                        "the same opponent")
    p.add_argument("--follow-user", type=float, default=15.0, help="with --user-starts: s of his real cards (0: none)")
    p.add_argument("--drill", type=int, default=0,
                   help="every game starts at one of this many defence drills (python/defence_starts.py): red begins "
                        "a hard push on blue's side (its cards put down by script, then the opponent plays on), mostly "
                        "in double and triple elixir, blue often behind on elixir. Blue (the model) defends; also "
                        "prints where its first troop goes after the push starts. Compare nets as blue against the "
                        "same opponent")
    p.add_argument("--drill-seed", type=int, default=2026, help="(train_v7.py --drill-starts builds its own from seed+31)")
    p.add_argument("--live-view", type=float, nargs=5, default=(0.0, 0.0, 0.0, 0.0, 0.0),
                   help="both sides see the other like the live bot's twin does (train_v7.py --live-view; measured "
                        "2026-11-15: 0.035 0.03 0.9 0.05 0.3); all 0: exactly")
    args = p.parse_args()

    data = ce.load_data()
    deck = ce.parse_deck(args.deck, data)
    opp_deck = ce.parse_deck(args.opp_deck, data) if args.opp_deck else deck
    players, key_spots, rhythm = [], [], []
    for spec in (args.model, args.opponent):
        if spec in ce.BOT_KINDS:
            players.append(spec)
            key_spots.append(False)
            rhythm.append(args.decision_ticks)
        elif os.path.exists(spec):
            net, ckpt = load_policy(spec, args.device)
            players.append(net)
            key_spots.append(int(ckpt.get("key_spots", 0)))  # the action space each policy was trained on
            rhythm.append(int(ckpt.get("args", {}).get("decision_ticks", args.decision_ticks)))
        else:
            raise SystemExit(f"{spec!r} is neither a bot name nor a checkpoint file")
    step_ticks = math.gcd(*rhythm) if args.native_rhythm else args.decision_ticks
    every = [r // step_ticks if args.native_rhythm else 1 for r in rhythm]  # a player decides every n-th step

    bots = tuple(pl if isinstance(pl, str) else "noop" for pl in players)
    env = ce.VecEnv(data, [(deck, opp_deck)], num_envs=min(args.envs, args.games), bots=bots,
                    decision_ticks=step_ticks, seed=args.seed, key_spots=tuple(key_spots), live_view=tuple(args.live_view))
    if args.human_starts > 0:
        import human_starts
        env.set_starts(human_starts.build(data, log=lambda _: None)[0], args.human_starts)
        env.reset()
    if args.user_starts:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
        import push_defence_check as pdc
        positions, scripts = [], [] if args.follow_user > 0 else None
        if args.user_starts in ("push", "both"):
            positions += pdc.push_positions(data, log=lambda _: None, follow=args.follow_user, scripts=scripts)
        if args.user_starts in ("back", "both"):
            positions += pdc.back_positions(data, log=lambda _: None, follow=args.follow_user, scripts=scripts)
        kept = env.set_starts(positions, 1.0, scripts)
        env.reset()
        print(f"user starts ({args.user_starts}): {kept} positions"
              + (f", red puts down the user's real cards for {args.follow_user:g} s" if scripts is not None else ""))
    drill = None
    if args.drill > 0:
        import defence_starts
        battles, scripts, lanes, kinds = defence_starts.generate(data, deck, args.drill, seed=args.drill_seed)
        kept = env.set_starts(battles, 1.0, scripts)
        env.reset()
        if kept != len(battles):
            print(f"warning: {len(battles) - kept} drills not kept, the lane stats below are off")
        drill = dict(lanes=lanes, kinds=kinds, push_ms=[sc[2][0][0] for sc in scripts], first=[None] * env.num_envs,
                     where=[], reaction=[], by_kind={})
    step_no = 0
    results, crowns, lengths = [], [], []
    started = env.num_envs
    active = np.ones(env.num_envs, dtype=bool)  # envs still allowed to finish an episode we count
    # How the model plays (blue): elixir it had when it put each card down, and its spells.
    spell_token = np.zeros(ce.crsim.rl.vocab_size(data), dtype=bool)
    for cid, card in data.cards.items():
        spell_token[ce.crsim.rl.FIRST_CARD_TOKEN + card.index] = card.is_spell
    play_elixir, spells = [], 0
    giant_token = ce.crsim.rl.FIRST_CARD_TOKEN + data.cards["giant"].index
    giants, giants_back = 0, 0  # its giants, and those put behind its king (the back two rows: a push built from there)
    troop_x = {}  # card token -> [troops it put down, of them between the princess towers' lanes (x 6-12)]
    t0 = time.time()
    while len(results) < args.games:
        actions = np.full((env.num_envs, 2), ce.BOT_ACTION, dtype=np.int32)
        for side, pl in enumerate(players):
            if not isinstance(pl, str):
                if step_no % every[side]:  # not this player's moment to decide: it waits
                    actions[:, side] = 0
                    continue
                obs = obs_to_torch(env.obs(side), args.device)
                with torch.no_grad():
                    temp = args.model_temp if side == 0 else args.opp_temp
                    if temp is not None:
                        logits = pl(obs)[0].float() / max(temp, 1e-3)
                        actions[:, side] = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
                    else:
                        actions[:, side] = pl.act(obs, greedy=args.greedy)[0].cpu().numpy()
        step_no += 1
        elixir = env.obs(0)["global"][:, 4] * 10.0
        hand = env.obs(0)["hand"].copy()
        _, done = env.step(actions)
        taken = env.env.taken[:, 0]
        for i in np.flatnonzero((taken > 0) & active):
            play_elixir.append(elixir[i])
            card = hand[i, (taken[i] - 1) // ce.CELLS]
            spells += int(spell_token[card])
            if card == giant_token:
                giants += 1
                giants_back += int((taken[i] - 1) % ce.CELLS // ce.GRID_W < 2)
            if not spell_token[card]:
                tx = troop_x.setdefault(int(card), [0, 0])
                tx[0] += 1
                tx[1] += int(6.0 <= (taken[i] - 1) % ce.CELLS % ce.GRID_W + 0.5 <= 12.0)
            if drill is not None and not spell_token[card] and drill["first"][i] is None:
                cell = (taken[i] - 1) % ce.CELLS  # blue's own frame is the arena's
                drill["first"][i] = (cell % ce.GRID_W + 0.5, cell // ce.GRID_W + 0.5, env.battle(i).time_ms)
        for i in np.flatnonzero(done & active):
            if drill is not None:
                n = int(env.episode_start[i])
                if n >= 0:
                    lane, push_ms, first = drill["lanes"][n], drill["push_ms"][n], drill["first"][i]
                    if first is None or first[2] - push_ms > 15000:
                        where = "none"
                    elif 6.0 <= first[0] <= 12.0:
                        where = "middle"
                    elif abs(first[0] - lane) <= 2.5:
                        where = "push lane"
                    else:
                        where = "other lane"
                    drill["where"].append(where)
                    drill["by_kind"].setdefault(drill["kinds"][n], []).append(where)
                    if first is not None and where != "none":
                        drill["reaction"].append(max(0.0, (first[2] - push_ms) / 1000.0))
        if drill is not None:
            for i in np.flatnonzero(done):
                drill["first"][i] = None
        for i in np.flatnonzero(done & active):
            w = int(env.winner[i])
            results.append(1.0 if w == 0 else (0.5 if w == 2 else 0.0))
            crowns.append((int(env.crowns[i, 0]), int(env.crowns[i, 1])))
            lengths.append(int(env.episode_steps[i]) * step_ticks * 50 / 1000)
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
    if args.user_starts and args.follow_user > 0:
        print(f"  the user's scripted cards put down {env.env.script_made}, left out {env.env.script_dropped}")
    if drill is not None:
        print(f"  red's scripted push cards put down {env.env.script_made}, left out {env.env.script_dropped}")
        share = lambda ws: ", ".join(f"{k} {ws.count(k) / max(1, len(ws)):.0%}"  # noqa: E731
                                     for k in ("middle", "push lane", "other lane", "none"))
        rt = np.array(drill["reaction"]) if drill["reaction"] else np.array([np.nan])
        print(f"  blue's first troop after the push starts (within 15 s; middle: x 6-12, between the towers): "
              f"{share(drill['where'])}; it comes {np.median(rt):.1f} s after red's first card (median)")
        for kind, ws in sorted(drill["by_kind"].items()):
            print(f"    {kind}: {share(ws)} ({len(ws)} games)")
    if play_elixir:
        pe, n = np.array(play_elixir), max(1, len(crowns))  # (all finished games, a few past --games)
        print(f"  blue's style: {len(pe) / n:.1f} cards per game, elixir when playing {pe.mean():.1f} "
              f"(from 8+: {np.mean(pe >= 8):.0%}), spells per game {spells / n:.1f}, "
              f"giants behind the king {giants_back / max(1, giants):.0%} of {giants / n:.1f} a game")
        names = {ce.crsim.rl.FIRST_CARD_TOKEN + c.index: cid for cid, c in data.cards.items()}
        print("  blue's troops put down between the lanes (x 6-12): " + ", ".join(
            f"{names.get(k, k)} {v[1] / max(1, v[0]):.0%} of {v[0]}" for k, v in sorted(troop_x.items(), key=lambda kv: -kv[1][0])))


if __name__ == "__main__":
    main()
