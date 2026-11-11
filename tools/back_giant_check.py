"""Is a giant behind the king worth it in the simulator, for the nets that play there?

The user puts 40% of their giants right behind the king (one of the two tiles beside the middle of the back row,
(8, 0) or (9, 0) of their own side), always from 8+ elixir (9.4 on average), at any time of the battle; the nets put
none there. Here, at moments of a net's own games where the user would (it holds --elixir or more, the giant is in
its hand, no enemy troop is on its side of the river), the battle is copied twice:
  A - the net goes on as it would (it waits, or plays what it likes);
  B - the giant goes behind its king first (of the two tiles, the one the net scores higher), then the net goes on.
Both copies are played to the end about --rollouts times each by the same two nets (the model as blue, the opponent
as red) at --temp. The difference in wins is what the move is worth to them. Tower HP lost by each side over the
next seconds says why: B losing more in the first 10-20 s is the enemy punishing the 5 elixir spent; B taking no
more from the enemy by 35 s (a back giant reaches the bridge after ~20 s) is a push that does not pay.

    .venv\\Scripts\\python.exe tools\\back_giant_check.py --model runs/h1/final.pt [--opponent runs/sx1/best.pt]
        [--lead 2 3]   (blue ahead by 2-3 elixir at each moment: red's elixir cut to match, in both copies)
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
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from model import load_policy, obs_to_torch  # noqa: E402

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
WINDOWS_S = (10, 20, 35)
BACK_CELLS = ((9, 0), (8, 0))  # own frame (col, row): where the user puts them
# A BatchEnv keeps its own copy of the GameData and its battles (and copies of them) refer to that copy: the envs the
# moments were found in must outlive the copies, wherever they go (another env's starts copy them as they are).
_FOUND_IN = []


def tower_hp(b, team):
    """HP left in a team's three towers."""
    ids = [b.king_tower(team), b.princess_tower(team, 0), b.princess_tower(team, 1)]
    return sum(t.hp for t in (b.find(i) for i in ids) if t is not None)


class Players:
    """The two nets (blue: the model, red: the opponent) sampling at a temperature, each at its own rhythm."""

    def __init__(self, args, device):
        self.nets, self.key_spots, rhythm = [], [], []
        for path in (args.model, args.opponent):
            net, ckpt = load_policy(path, device)
            self.nets.append(net)
            self.key_spots.append(int(ckpt.get("key_spots", 0)))
            rhythm.append(int(ckpt.get("args", {}).get("decision_ticks", 4)))
        self.step_ticks = math.gcd(*rhythm)
        self.every = [r // self.step_ticks for r in rhythm]
        self.temp, self.device = args.temp, device

    def env(self, data, deck, envs, seed):
        return ce.VecEnv(data, [(deck, deck)], num_envs=envs, bots=("noop", "noop"), decision_ticks=self.step_ticks,
                         seed=seed, key_spots=tuple(self.key_spots))

    def actions(self, env, step_no):
        """(actions (N, 2), blue's logits or None when it does not decide at this step)."""
        actions = np.zeros((env.num_envs, 2), dtype=np.int32)
        blue_logits = None
        for side in (0, 1):
            if step_no % self.every[side]:
                continue  # not this player's moment: it waits
            with torch.no_grad():
                logits = self.nets[side](obs_to_torch(env.obs(side), self.device))[0].float()
            actions[:, side] = torch.distributions.Categorical(logits=logits / self.temp).sample().cpu().numpy()
            if side == 0:
                blue_logits = logits.cpu().numpy()
        return actions, blue_logits


def collect(players, data, deck, args):
    """[(A battle, B battle, what the net itself did there)] at the moments a user would put a giant behind the king."""
    env = players.env(data, deck, args.envs, args.seed)
    _FOUND_IN.append(env)  # (the copies below refer to its GameData)
    lead = getattr(args, "lead", None) or None
    if lead:
        lead = (lead[0], lead[-1])  # one number: that lead; two: drawn evenly between them
    rng = np.random.default_rng(args.seed)
    last = np.full(env.num_envs, -1e9)  # game ms of each env's last moment taken
    moments, step_no, t0 = [], 0, time.time()
    own = {"wait": 0, "giant behind the king": 0, "giant elsewhere": 0, "another card": 0}
    while len(moments) < args.moments:
        actions, logits = players.actions(env, step_no)
        if logits is not None:
            mask = env.obs(0)["action_mask"]
            for i in range(env.num_envs):
                b = env.battle(i)
                if b.time_ms < last[i]:
                    last[i] = -1e9  # a new battle in this env
                p = b.player(BLUE)
                if b.finished or p.elixir < args.elixir or "giant" not in list(p.hand):
                    continue
                if b.time_ms - last[i] < args.gap * 1000:
                    continue
                if any(x.team == RED and not x.is_building and x.pos.y < 16.0 for x in b.entities):
                    continue  # an enemy troop on its side: a defence comes first
                slot = list(p.hand).index("giant")
                options = [1 + slot * ce.CELLS + row * ce.GRID_W + col for col, row in BACK_CELLS]
                options = [a for a in options if mask[i, a]]
                if not options:
                    continue
                back = max(options, key=lambda a: logits[i, a])
                a_battle, b_battle = b.clone(), b.clone()
                if lead:  # an elixir lead for blue: red gets at most blue's elixir minus it, in both copies
                    gap_now = rng.uniform(*lead)
                    red_elixir = min(b.player(RED).elixir, max(0.0, p.elixir - gap_now))
                    for copy in (a_battle, b_battle):
                        copy.set_elixir(RED, red_elixir)
                if not crsim.rl.apply_action(b_battle, BLUE, back):
                    continue
                took = int(actions[i, 0])
                if took == 0:
                    did = "wait"
                elif (took - 1) // ce.CELLS == slot:
                    did = "giant behind the king" if (took - 1) % ce.CELLS // ce.GRID_W < 2 else "giant elsewhere"
                else:
                    did = "another card"
                own[did] += 1
                moments.append((a_battle, b_battle, did, b.time_ms))
                last[i] = b.time_ms
                if len(moments) >= args.moments:
                    break
        env.step(actions)
        step_no += 1
    n = len(moments)
    if lead:
        print(f"red's elixir cut to at most blue's minus {lead[0]:g}-{lead[1]:g} at each moment (both copies)")
    print(f"{n} moments ({args.elixir}+ elixir, the giant in hand, no enemy troop on its side) in "
          f"{time.time() - t0:.0f} s; there the net itself: " + ", ".join(f"{k} {v / n:.0%}" for k, v in own.items()),
          flush=True)
    return moments


def rollouts(players, data, deck, moments, args):
    """Per start (A0, B0, A1, B1 ...): games, score, and tower HP each side lost after each window."""
    starts = [x for m in moments for x in m[:2]]
    env = players.env(data, deck, args.envs, args.seed + 1)
    kept = env.set_starts(starts, 1.0)
    assert kept == len(starts), f"{kept} of {len(starts)} starts kept"
    env.reset()
    k = len(starts)
    hp0 = np.array([[tower_hp(s, BLUE), tower_hp(s, RED)] for s in starts])
    t_start = np.array([s.time_ms for s in starts])
    games, score = np.zeros(k), np.zeros(k)
    lost = np.zeros((k, len(WINDOWS_S), 2))  # summed over games: HP lost by blue, by red
    pending = np.zeros(env.num_envs, dtype=int)  # windows already recorded in each env's current game
    last_lost = np.zeros((env.num_envs, 2))
    want, done_games, step_no, t0 = k * args.rollouts, 0, 0, time.time()
    active = np.ones(env.num_envs, dtype=bool)
    started = env.num_envs
    while done_games < want:
        actions, _ = players.actions(env, step_no)
        env.step(actions)
        step_no += 1
        cur = np.array(env.env.start)
        for i in np.flatnonzero(active & (pending < len(WINDOWS_S))):
            if env.env.done[i]:
                continue
            s = cur[i]
            b = env.battle(i)
            last_lost[i] = hp0[s] - [tower_hp(b, BLUE), tower_hp(b, RED)]
            while pending[i] < len(WINDOWS_S) and b.time_ms >= t_start[s] + WINDOWS_S[pending[i]] * 1000:
                lost[s, pending[i]] += last_lost[i]
                pending[i] += 1
        for i in np.flatnonzero(np.array(env.env.done) & active):
            s = int(env.env.episode_start[i])
            w = int(env.winner[i])
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
    print(f"{done_games} games from {k} starts in {time.time() - t0:.0f} s", flush=True)
    return games, score, lost


def report(moments, games, score, lost, args):
    a, b = slice(0, None, 2), slice(1, None, 2)
    both = (games[a] > 0) & (games[b] > 0)
    sa, sb = score[a][both] / games[a][both], score[b][both] / games[b][both]
    d = sb - sa
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f"\n{both.sum()} moments with games from both copies ({games.sum():.0f} games, "
          f"{games[a].mean():.1f} / {games[b].mean():.1f} per copy)")
    print(f"  A (the net's own play): score {sa.mean():.3f}")
    print(f"  B (giant behind the king first): score {sb.mean():.3f}")
    print(f"  B - A: {d.mean():+.3f} +- {1.96 * se:.3f} (95%)")
    ms = np.array([m[3] for m in moments])[both]
    for name, sel in (("before 2:00", ms < 120_000), ("2:00 on (double elixir)", ms >= 120_000)):
        if sel.sum() > 5:
            dd = d[sel]
            print(f"    {name}: {sel.sum()} moments, B - A {dd.mean():+.3f} +- "
                  f"{1.96 * dd.std(ddof=1) / math.sqrt(len(dd)):.3f}")
    did = np.array([m[2] for m in moments])[both]
    for kind in ("wait", "giant elsewhere", "another card"):
        sel = did == kind
        if sel.sum() > 5:
            dd = d[sel]
            print(f"    where the net itself played '{kind}': {sel.sum()} moments, B - A {dd.mean():+.3f} +- "
                  f"{1.96 * dd.std(ddof=1) / math.sqrt(len(dd)):.3f}")
    print("  tower HP lost (average per game), blue (the net) / red (the enemy):")
    for j, w in enumerate(WINDOWS_S):
        la = lost[a][both, j] / games[a][both, None]
        lb = lost[b][both, j] / games[b][both, None]
        print(f"    after {w:2d} s: A {la[:, 0].mean():6.0f} / {la[:, 1].mean():6.0f}   "
              f"B {lb[:, 0].mean():6.0f} / {lb[:, 1].mean():6.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runs/h1/final.pt")
    p.add_argument("--opponent", default=None, help="red's net (default: the model itself)")
    p.add_argument("--moments", type=int, default=400)
    p.add_argument("--rollouts", type=int, default=8, help="games per copy (on average)")
    p.add_argument("--elixir", type=float, default=9.0)
    p.add_argument("--gap", type=float, default=20.0, help="s between two moments of the same game")
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--envs", type=int, default=100)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--lead", type=float, nargs="+", default=None,
                   help="elixir lead for blue at each moment: red keeps at most blue's elixir minus it (one number, or "
                        "two to draw it evenly between); the user's idea: a back giant pays with an elixir lead")
    args = p.parse_args()
    args.opponent = args.opponent or args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    data = ce.load_data()
    deck = ce.parse_deck(",".join(ce.DEFAULT_DECK), data)
    players = Players(args, device)
    print(f"{args.model} (blue) vs {args.opponent} (red), temp {args.temp}", flush=True)
    moments = collect(players, data, deck, args)
    games, score, lost = rollouts(players, data, deck, moments, args)
    report(moments, games, score, lost, args)


if __name__ == "__main__":
    main()
