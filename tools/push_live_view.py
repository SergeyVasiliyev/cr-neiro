"""Why do the user's pushes cost the bot more live than in the simulator?

tools/push_follow_check.py (2026-11-14): after the user's pushes of the two-sided recordings the bot of that day lost
~1.5x the tower HP the same net loses in the simulator from the same positions (20 s on: ~2250 vs ~1400-1650). Here,
at the same pushes:
  - what the bot saw: the push (elixir of red troops on blue's side) in the live twin (the screen only, as the bot
    plays) next to the two-sided twin (the user's own plays known);
  - what the bot did: its first play after the push in the real battle (its own recording) next to the first play
    of the same net in the simulator (copy A of push_defence_check, red the net), and how much elixir it held.

    .venv\\Scripts\\python.exe tools\\push_live_view.py --model runs/h1/cycle2_stopped.pt video46_b01 ...
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
import twin  # noqa: E402
from back_giant_check import Players  # noqa: E402
from compare_red_truth import own_plays_timed  # noqa: E402
from live.session import LiveMatch  # noqa: E402
from push_defence_check import CACHE, moments, push_value  # noqa: E402

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
HORIZON_S = 15.0


def red_troops(b, data):
    """(elixir of red troops anywhere on the field, how many of them) - each unit its share of its card."""
    value, n = 0.0, 0
    for e in b.entities:
        if e.is_building or e.team != RED or e.hp <= 0 or e.card_index < 0:
            continue
        cd = data.card(data.card_ids[e.card_index])
        value += cd.elixir / max(1, cd.count)
        n += 1
    return value, n


def live_view(name, data):
    """[(game s, push value, red elixir on the field, blue elixir)] of the live twin (the screen only)."""
    frames = twin.load_cache(os.path.join(CACHE, f"{name}.json"))["frames"]
    m, out = LiveMatch(data), []
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0 or m.battle.finished:
            continue
        out.append((g, push_value(m.battle, data)[0], red_troops(m.battle, data)[0], m.battle.player(BLUE).elixir))
    return out


def first_plays_sim(players, data, deck, ms, args):
    """The net's first play after each push in the simulator (copy A only, red the net): per push the delays (s) of
    its games, HORIZON_S + 1 where it played nothing in HORIZON_S."""
    starts = [m[0] for m in ms]
    env = players.env(data, deck, args.envs, args.seed + 1)
    kept = env.set_starts(starts, 1.0)
    assert kept == len(starts)
    env.reset()
    n = env.num_envs
    t_start = np.array([s.time_ms for s in starts])
    base = np.array([len(env.battle(i).player(BLUE).plays) for i in range(n)])
    delays = [[] for _ in starts]
    want, got, step_no, t0 = len(starts) * args.rollouts, 0, 0, time.time()
    while got < want:
        actions, _ = players.actions(env, step_no)
        env.step(actions)
        step_no += 1
        cur = np.array(env.env.start)
        done = np.array(env.env.done)
        for i in range(n):
            if done[i]:  # the game ended before blue played: nothing within the horizon
                delays[int(env.env.episode_start[i])].append(HORIZON_S + 1)
                got += 1
            else:
                b = env.battle(i)
                t = (b.time_ms - t_start[cur[i]]) / 1000.0
                if len(b.player(BLUE).plays) > base[i]:
                    delays[cur[i]].append(t)
                elif t > HORIZON_S:
                    delays[cur[i]].append(HORIZON_S + 1)
                else:
                    continue
                got += 1
                env.env.reset_env(i)  # its first play is known: the next game
            base[i] = len(env.battle(i).player(BLUE).plays)
    print(f"  {got} first plays from {len(starts)} pushes in {time.time() - t0:.0f} s", flush=True)
    return delays


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runs/h1/cycle2_stopped.pt", help="the net that played these battles")
    p.add_argument("names", nargs="*", help="battles (default: every two-sided one cached)")
    p.add_argument("--every", type=float, default=0.5)
    p.add_argument("--gap", type=float, default=5.0)
    p.add_argument("--push", type=float, default=5.0)
    p.add_argument("--alone", type=float, default=0.0)
    p.add_argument("--elixir", type=float, default=5.0)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    args.opponent, args.b_net = args.model, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    data = ce.load_data()
    deck = ce.parse_deck(",".join(ce.DEFAULT_DECK), data)
    players = Players(args, device)
    print(f"{args.model}: the pushes of {len(args.names) or 'all'} battles", flush=True)
    ms = [m for m in moments(players, data, args) if m[6] == 0]

    views, real_first, rows = {}, {}, []
    for name in sorted({m[5] for m in ms}):
        views[name] = live_view(name, data)
        frames = twin.load_cache(os.path.join(CACHE, f"{name}.json"))["frames"]
        real_first[name] = sorted(t for t, _, pos in own_plays_timed(frames, data))
    for m in ms:
        t0 = m[0].time_ms / 1000.0
        view = views[m[5]]
        near = min(view, key=lambda r: abs(r[0] - t0)) if view else None
        truth_push = push_value(m[0], data)[0]
        truth_all = red_troops(m[0], data)[0]
        nxt = [t for t in real_first[m[5]] if t > t0 + 1e-6]
        real_delay = (nxt[0] - t0) if nxt and nxt[0] - t0 <= HORIZON_S else HORIZON_S + 1
        rows.append((truth_push, truth_all, near[1] if near else np.nan, near[2] if near else np.nan,
                     m[0].player(BLUE).elixir, near[3] if near else np.nan, real_delay, m[4]))
    r = np.array([x[:7] for x in rows], dtype=float)
    print(f"\n{len(rows)} pushes")
    print("what the bot saw at the push (elixir of the user's troops):")
    print(f"  on blue's side: two-sided twin {r[:, 0].mean():.1f}, live twin {np.nanmean(r[:, 2]):.1f}; the live "
          f"twin saw less than half of it at {np.nanmean(r[:, 2] < 0.5 * r[:, 0]):.0%} of the pushes")
    print(f"  on the whole field: two-sided twin {r[:, 1].mean():.1f}, live twin {np.nanmean(r[:, 3]):.1f}")
    print(f"  blue's own elixir: two-sided twin {r[:, 4].mean():.1f}, live twin {np.nanmean(r[:, 5]):.1f}")

    sim = first_plays_sim(players, data, deck, ms, args)
    sim_med = np.array([np.median(d) if d else np.nan for d in sim])
    real = r[:, 6]
    print("what the bot did: its first play after the push (s; 'none' = nothing in 15 s)")
    for name, d in (("real battle", real), ("simulator, same net (median of its games)", sim_med)):
        print(f"  {name}: median {np.nanmedian(d):.1f} s, within 2 s {np.nanmean(d <= 2):.0%}, within 5 s "
              f"{np.nanmean(d <= 5):.0%}, none {np.nanmean(d > HORIZON_S):.0%}")
    print("  push by push (real / simulator median, s; live twin's push / the truth):")
    for k in np.argsort(-(real - sim_med)):
        x = rows[k]
        print(f"    {x[7][:60]:60s}  {real[k]:5.1f} / {sim_med[k]:5.1f}   saw {x[2]:4.1f} / {x[0]:4.1f}")


if __name__ == "__main__":
    main()
