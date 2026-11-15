"""Is waiting punished when the push is played on the way the user played it?

tools/push_defence_check.py found (h1 cycles 2-8) that at the user's real pushes the nets want to wait and that the
simulator hardly punishes it: B - A (defending at once vs the net's own play) is ~0. There red goes on as a net from
the push on. Live the user supports the push himself - this measures whether that is the difference.

The same pushes (the twin's battle at the user's pushes in the two-sided recordings) are played on two ways:
  net:  red is the net from the moment on (as push_defence_check);
  user: for the first --follow s red plays exactly the user's real plays of that battle (card, moment, tile, from his
        own recording; his net stays silent), then the net takes over.
Each with the two copies of push_defence_check (A: the net's own play; B: its best move other than waiting first),
about --rollouts games per copy. Both ways use the same pushes. The user's plays were his answers to the bot of that
battle, not to these copies (his spells may miss here): --no-spells leaves his spells out of the script.

    .venv\\Scripts\\python.exe tools\\push_follow_check.py --model runs/h1/final_cycle8.pt [--follow 15]
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
import twin  # noqa: E402
from back_giant_check import WINDOWS_S, Players, tower_hp  # noqa: E402
from compare_red_truth import true_plays  # noqa: E402
from push_defence_check import CACHE, moments  # noqa: E402

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED


def user_plays(name, data):
    """The user's (red's) real plays of a two-sided battle: [(battle s, card, pos in blue's tiles)], as the twin
    took them (python/human_starts.py follow)."""
    frames = twin.load_cache(os.path.join(CACHE, f"{name}_red.json"))["frames"]
    return sorted((t, card, pos) for t, card, pos in true_plays(frames, data) if pos is not None)


def scripts_for(ms, data, follow_s, spells=True):
    """Per moment: the user's plays after it (the twin's copy has the ones up to its time), up to follow_s on."""
    cache, out = {}, []
    for m in ms:
        name, no = m[5], m[6]
        if name not in cache:
            cache[name] = user_plays(name, data)
        now = m[0].time_ms / 1000.0
        out.append([] if no else  # (a second battle of one recording: its times are not these)
                   [(t, c, p) for t, c, p in cache[name] if now + 1e-9 < t <= now + follow_s and
                    (spells or c not in twin.SPELL_CARDS)])
    return out


def real_losses(ms, data, every):
    """What the real battle did after each push (the twin follows the towers' HP on the screen): tower HP lost by
    blue and red after each of WINDOWS_S, (pushes, windows, 2); a battle over (or its recording) before a window
    counts as it last stood. Also how far each recorded battle went past the push: s."""
    import human_starts
    states = {}
    for name in sorted({m[5] for m in ms}):
        states[name] = [(no, b.time_ms, tower_hp(b, BLUE), tower_hp(b, RED))
                        for no, _, b in human_starts.recording_starts(name, f"{name}_red", data, every=every)]
    out = np.zeros((len(ms), len(WINDOWS_S), 2))
    for k, m in enumerate(ms):
        rows = [r for r in states[m[5]] if r[0] == m[6]]
        t0 = m[0].time_ms
        here = min(rows, key=lambda r: abs(r[1] - t0))
        for j, w in enumerate(WINDOWS_S):
            later = [r for r in rows if r[1] <= t0 + w * 1000]
            last = max(later, key=lambda r: r[1])
            out[k, j] = here[2] - last[2], here[3] - last[3]
    return out


def rollouts(players, data, deck, ms, scripts, follow_s, args):
    """Like back_giant_check.rollouts, red following each start's script for follow_s s (its net silent there)."""
    starts = [x for m in ms for x in m[:2]]
    script_of = [scripts[k // 2] for k in range(len(starts))]
    env = players.env(data, deck, args.envs, args.seed + 1)
    kept = env.set_starts(starts, 1.0)
    assert kept == len(starts), f"{kept} of {len(starts)} starts kept"
    env.reset()
    k, n = len(starts), env.num_envs
    hp0 = np.array([[tower_hp(s, BLUE), tower_hp(s, RED)] for s in starts])
    t_start = np.array([s.time_ms for s in starts])
    games, score = np.zeros(k), np.zeros(k)
    lost = np.zeros((k, len(WINDOWS_S), 2))
    pending, last_lost = np.zeros(n, dtype=int), np.zeros((n, 2))
    ptr = np.zeros(n, dtype=int)  # the next play of the script in each env's current game
    played, short = 0, []
    want, done_games, step_no, t0 = k * args.rollouts, 0, 0, time.time()
    active = np.ones(n, dtype=bool)
    started = n
    follow_ms = follow_s * 1000.0
    while done_games < want:
        actions, _ = players.actions(env, step_no)
        cur = np.array(env.env.start)
        if follow_ms > 0:
            for i in range(n):
                if env.battle(i).time_ms < t_start[cur[i]] + follow_ms:
                    actions[i, 1] = 0  # red is the user's script here
        env.step(actions)
        step_no += 1
        cur = np.array(env.env.start)
        done = np.array(env.env.done)
        for i in range(n):
            if done[i]:
                ptr[i] = 0  # (a new game from its start)
                continue
            b, sc = env.battle(i), script_of[cur[i]]
            while ptr[i] < len(sc) and sc[ptr[i]][0] * 1000.0 <= b.time_ms:
                t, card, pos = sc[ptr[i]]
                ptr[i] += 1
                try:
                    miss = b.force_play(RED, card, crsim.Vec2(*pos), int(b.time_ms - t * 1000.0))
                except Exception:  # (a card the simulated deck does not hold)
                    continue
                played += 1
                short.append(miss)
        for i in np.flatnonzero(active & (pending < len(WINDOWS_S))):
            if done[i]:
                continue
            s, b = cur[i], env.battle(i)
            last_lost[i] = hp0[s] - [tower_hp(b, BLUE), tower_hp(b, RED)]
            while pending[i] < len(WINDOWS_S) and b.time_ms >= t_start[s] + WINDOWS_S[pending[i]] * 1000:
                lost[s, pending[i]] += last_lost[i]
                pending[i] += 1
        for i in np.flatnonzero(done & active):
            s, w = int(env.env.episode_start[i]), int(env.winner[i])
            while pending[i] < len(WINDOWS_S):  # the game ended first: as it last stood
                lost[s, pending[i]] += last_lost[i]
                pending[i] += 1
            games[s] += 1
            score[s] += 1.0 if w == 0 else (0.5 if w == 2 else 0.0)
            done_games += 1
            pending[i], last_lost[i] = 0, 0.0
            if started >= want:
                active[i] = False
            started += 1
    if follow_ms > 0:
        print(f"  the user's plays put in: {played} ({np.mean(np.array(short) > 0.05):.0%} of them with less elixir "
              f"than the card in the copy)", flush=True)
    print(f"  {done_games} games from {k} starts in {time.time() - t0:.0f} s", flush=True)
    return games, score, lost


def report(title, ms, games, score, lost):
    a, b = slice(0, None, 2), slice(1, None, 2)
    both = (games[a] > 0) & (games[b] > 0)
    sa, sb = score[a][both] / games[a][both], score[b][both] / games[b][both]
    d = sb - sa
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f"{title}: {both.sum()} pushes")
    print(f"  A (the net's own play) {sa.mean():.3f}, B (its best other move than waiting first) {sb.mean():.3f}, "
          f"B - A {d.mean():+.3f} +- {1.96 * se:.3f} (95%)")
    pw = np.array([m[2] for m in ms])[both]
    for lo, hi in ((0.9, 1.01), (0.0, 0.5)):
        sel = (pw >= lo) & (pw < hi)
        if sel.sum() > 5:
            dd = d[sel]
            print(f"    where it wants to wait {lo:.0%}-{min(hi, 1):.0%}: {sel.sum()} pushes, B - A {dd.mean():+.3f} +- "
                  f"{1.96 * dd.std(ddof=1) / math.sqrt(len(dd)):.3f}")
    print("  tower HP lost (average per game), blue (the net) / red (the user's side):")
    for j, w in enumerate(WINDOWS_S):
        la = lost[a][both, j] / games[a][both, None]
        lb = lost[b][both, j] / games[b][both, None]
        print(f"    after {w:2d} s: A {la[:, 0].mean():6.0f} / {la[:, 1].mean():6.0f}   "
              f"B {lb[:, 0].mean():6.0f} / {lb[:, 1].mean():6.0f}")
    return sa, sb


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runs/h1/final_cycle8.pt")
    p.add_argument("--opponent", default=None, help="red's net (default: the model itself)")
    p.add_argument("names", nargs="*", help="battles (default: every two-sided one cached)")
    p.add_argument("--follow", type=float, default=15.0, help="s red plays the user's real plays after a push")
    p.add_argument("--no-spells", action="store_true", help="leave the user's spells out of the script")
    p.add_argument("--no-rollouts", action="store_true", help="only the pushes and what the real battles did")
    p.add_argument("--every", type=float, default=0.5)
    p.add_argument("--gap", type=float, default=5.0)
    p.add_argument("--push", type=float, default=5.0)
    p.add_argument("--alone", type=float, default=0.0)
    p.add_argument("--elixir", type=float, default=5.0)
    p.add_argument("--b-net", default=None)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    args.opponent = args.opponent or args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    data = ce.load_data()
    deck = ce.parse_deck(",".join(ce.DEFAULT_DECK), data)
    players = Players(args, device)
    print(f"{args.model} (blue) vs {args.opponent} (red), temp {args.temp}", flush=True)
    ms = moments(players, data, args)
    scripts = scripts_for(ms, data, args.follow, spells=not args.no_spells)
    counts = [len(s) for s in scripts]
    print(f"the user's plays in the {args.follow:g} s after each push: {np.mean(counts):.1f} on average "
          f"(none at {np.mean(np.array(counts) == 0):.0%} of the pushes)", flush=True)
    real = real_losses(ms, data, args.every)
    print("\n--- the real battles after these pushes (the bot of that day; tower HP from the screen)")
    print("  tower HP lost, blue (the bot) / red (the user):")
    for j, w in enumerate(WINDOWS_S):
        print(f"    after {w:2d} s: {real[:, j, 0].mean():6.0f} / {real[:, j, 1].mean():6.0f}")
    if args.no_rollouts:
        return
    print("\n--- red: the net from the push on")
    g1 = rollouts(players, data, deck, ms, [[] for _ in ms], 0.0, args)
    net = report("red the net", ms, *g1)
    print(f"\n--- red: the user's real plays for {args.follow:g} s, then the net")
    g2 = rollouts(players, data, deck, ms, scripts, args.follow, args)
    user = report("red the user's plays", ms, *g2)
    print(f"\nthe user's follow-up instead of the net's: A {user[0].mean() - net[0].mean():+.3f}, "
          f"B {user[1].mean() - net[1].mean():+.3f} (blue's score; same pushes)")


if __name__ == "__main__":
    main()
