"""Does waiting pay when a push comes? Live on 2026-11-10 h1 waited 15 s at 10 elixir while the user's giant, mini
pekka and musketeer took a tower; in the simulator both h1 cycles want to wait there (92-100%), v7e top and si2 defend.

At the user's real pushes in the recorded two-sided battles (the twin's battle every --every s, from the bot's
recording: the bot is blue, the user red), where red has --push elixir or more of troops on blue's side, no blue troop
within --alone tiles of them, and blue holds --elixir or more, the battle is copied twice:
  A - the net goes on as it would (mostly it waits);
  B - its best move other than waiting (the highest-scored playable move) first, then it goes on.
Both copies are played to the end about --rollouts times each, the model as blue, --opponent as red, at --temp.
B - A in wins is what not waiting is worth to them there. If B wins clearly more, the net waits because it never
learnt these positions (training on such pushes, a user-like sparring partner, would fix it); if not, the
simulator itself does not punish the wait.

    .venv\\Scripts\\python.exe tools\\push_defence_check.py --model runs/h1/cycle2_stopped.pt [--opponent ...]
"""
import argparse
import math
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
import human_starts  # noqa: E402
from back_giant_check import Players, rollouts  # noqa: E402
from model import load_policy, obs_to_torch  # noqa: E402

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
CACHE = os.path.join(ROOT, "raw", "perception", "cache")


def two_sided_battles():
    names = sorted(f[:-5] for f in os.listdir(CACHE) if re.fullmatch(r"video\d+_b\d\d\.json", f))
    return [n for n in names if os.path.exists(os.path.join(CACHE, f"{n}_red.json"))]


def push_value(b, data):
    """(elixir of red's troops on blue's side, their centre) - each unit its share of its card."""
    value, xs, ys = 0.0, [], []
    for e in b.entities:
        if e.is_building or e.team != RED or e.hp <= 0 or e.card_index < 0 or e.pos.y >= 15.0:
            continue
        cd = data.card(data.card_ids[e.card_index])
        value += cd.elixir / max(1, cd.count)
        xs.append(e.pos.x)
        ys.append(e.pos.y)
    return value, (np.mean(xs), np.mean(ys)) if xs else None


def push_positions(data, names=None, every=0.5, push=5.0, log=print, follow=0.0, scripts=None):
    """Training starts (train_v7.py --push-starts): the twin's battle every `every` s of the recorded two-sided
    battles (from the bot's recording: the bot blue, the user red) where red has `push` elixir or more of troops on
    blue's side. Built on `data` (it must outlive the envs that start from them). follow > 0 and a list `scripts`:
    one script per position is appended to it - the user's real plays for `follow` s from there
    (human_starts.red_script, for BatchEnv.set_starts)."""
    out, battles = [], 0
    for name in names or two_sided_battles():
        try:
            got = human_starts.recording_starts(name, f"{name}_red", data, every=every,
                                                follow_s=follow if scripts is not None else 0.0)
        except Exception as e:  # a battle the twin cannot follow costs only its positions
            log(f"push starts: {name} skipped ({type(e).__name__}: {e})")
            continue
        battles += 1
        for item in got:
            if push_value(item[2], data)[0] >= push:
                out.append(item[2])
                if scripts is not None:
                    scripts.append(item[3] if len(item) > 3 else (1, 0, []))
    log(f"push starts: {len(out)} positions from {battles} battles")
    return out


def back_value(b, data, y_min=27.5):
    """Elixir of red's troops behind its princess towers' line (y >= y_min: behind its king or a princess tower)
    while its king sleeps with both princess towers up, else 0 - each unit its share of its card."""
    asleep, princesses = False, 0
    for e in b.entities:
        if e.is_building and e.team == RED and e.hp > 0:
            if e.kind == crsim.EntityKind.KING_TOWER:
                asleep = not e.active
            else:
                princesses += 1
    if not asleep or princesses < 2:
        return 0.0
    value = 0.0
    for e in b.entities:
        if e.is_building or e.team != RED or e.hp <= 0 or e.card_index < 0 or e.pos.y < y_min:
            continue
        cd = data.card(data.card_ids[e.card_index])
        value += cd.elixir / max(1, cd.count)
    return value


def back_positions(data, names=None, every=0.5, y_min=27.5, log=print, follow=0.0, scripts=None):
    """Training starts (train_v7.py --back-starts): the twin's battle every `every` s of the recorded two-sided
    battles (the bot blue, the user red) where the user has troops behind his king or a princess tower (y >= y_min)
    and his king still sleeps. Live on 2026-11-11 h1 cycle 4 threw arrows / a fireball at the user's sleeping king
    there (waking him for the rest of the game), and its own giants all went to the bridge; its games against nets
    never reach such openings. Built on `data` (it must outlive the envs that start from them). follow and scripts:
    as push_positions."""
    out, battles = [], 0
    for name in names or two_sided_battles():
        try:
            got = human_starts.recording_starts(name, f"{name}_red", data, every=every,
                                                follow_s=follow if scripts is not None else 0.0)
        except Exception as e:  # a battle the twin cannot follow costs only its positions
            log(f"back starts: {name} skipped ({type(e).__name__}: {e})")
            continue
        battles += 1
        for item in got:
            if back_value(item[2], data, y_min) > 0:
                out.append(item[2])
                if scripts is not None:
                    scripts.append(item[3] if len(item) > 3 else (1, 0, []))
    log(f"back starts: {len(out)} positions from {battles} battles")
    return out


def moments(players, data, args):
    """[(A battle, B battle, how much the net wants to wait there, game ms, what it is, recording, battle no)] at
    the user's pushes."""
    net, device = players.nets[0], players.device
    b_net, b_spots = None, 0
    if args.b_net:
        b_net, bck = load_policy(args.b_net, device)
        b_spots = int(bck.get("key_spots", 0))
    out = []
    t0 = time.time()
    for name in args.names or two_sided_battles():
        try:
            got = human_starts.recording_starts(name, f"{name}_red", data, every=args.every)
        except Exception as e:  # a battle the twin cannot follow
            print(f"{name}: skipped ({type(e).__name__}: {e})", flush=True)
            continue
        last = defaultdict(lambda: -1e9)
        for no, g, b in got:
            p = b.player(BLUE)
            if p.elixir < args.elixir or g - last[no] < args.gap:
                continue
            value, centre = push_value(b, data)
            if value < args.push or centre is None:
                continue
            if any(e.team == BLUE and not e.is_building and e.hp > 0 and
                   math.hypot(e.pos.x - centre[0], e.pos.y - centre[1]) <= args.alone for e in b.entities):
                continue  # blue is defending already
            obs = crsim.rl.encode(b, BLUE, key_spots=players.key_spots[0])
            mask = torch.as_tensor(obs["action_mask"], device=device).bool()
            if mask[1:].sum() == 0:
                continue
            with torch.no_grad():
                logits = net(obs_to_torch({k: v[None] for k, v in obs.items()}, device))[0].float()[0]
            logits = logits.masked_fill(~mask, -1e9)
            p_wait = float(torch.softmax(logits / args.temp, 0)[0])
            best = int(torch.argmax(logits[1:])) + 1
            if b_net is not None:  # B's first move: the other net's best move other than waiting, from its own view
                bobs = crsim.rl.encode(b, BLUE, key_spots=b_spots)
                bmask = torch.as_tensor(bobs["action_mask"], device=device).bool()
                if bmask[1:].sum() == 0:
                    continue
                with torch.no_grad():
                    bl = b_net(obs_to_torch({k: v[None] for k, v in bobs.items()}, device))[0].float()[0]
                best = int(torch.argmax(bl.masked_fill(~bmask, -1e9)[1:])) + 1
            a_battle, b_battle = b.clone(), b.clone()
            if not crsim.rl.apply_action(b_battle, BLUE, best):
                continue
            slot, cell = divmod(best - 1, crsim.rl.CELLS)
            pos = crsim.rl.cell_position(b, BLUE, cell)
            out.append((a_battle, b_battle, p_wait, b.time_ms, f"{name} {g:.0f}s: push {value:.1f} elixir at "
                        f"({centre[0]:.1f},{centre[1]:.1f}), blue {p.elixir:.1f} elixir, waits {p_wait:.0%}, "
                        f"best other: {list(p.hand)[slot]} at ({pos.x:.1f},{pos.y:.1f})", name, no))
            last[no] = g
    print(f"{len(out)} pushes in {time.time() - t0:.0f} s; the net wants to wait there: median "
          f"{np.median([m[2] for m in out]):.0%}, >= 90% at {np.mean([m[2] >= 0.9 for m in out]):.0%} of them",
          flush=True)
    return out


def report(ms, games, score, lost):
    a, b = slice(0, None, 2), slice(1, None, 2)
    both = (games[a] > 0) & (games[b] > 0)
    sa, sb = score[a][both] / games[a][both], score[b][both] / games[b][both]
    d = sb - sa
    se = d.std(ddof=1) / math.sqrt(len(d))
    print(f"\n{both.sum()} pushes with games from both copies ({games.sum():.0f} games)")
    print(f"  A (the net's own play): score {sa.mean():.3f}")
    print(f"  B (the best move other than waiting, first): score {sb.mean():.3f}")
    print(f"  B - A: {d.mean():+.3f} +- {1.96 * se:.3f} (95%)")
    pw = np.array([m[2] for m in ms])[both]
    for lo, hi in ((0.9, 1.01), (0.5, 0.9), (0.0, 0.5)):
        sel = (pw >= lo) & (pw < hi)
        if sel.sum() > 5:
            dd = d[sel]
            print(f"    where it wants to wait {lo:.0%}-{min(hi, 1):.0%}: {sel.sum()} pushes, B - A {dd.mean():+.3f} +- "
                  f"{1.96 * dd.std(ddof=1) / math.sqrt(len(dd)):.3f}")
    order = np.argsort(-pw)
    print("  the pushes where it wants to wait most:")
    idx = np.flatnonzero(both)
    for k in order[:8]:
        i = idx[k]
        print(f"    {ms[i][4]}   B - A {d[k]:+.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runs/h1/cycle2_stopped.pt")
    p.add_argument("--opponent", default=None, help="red's net (default: the model itself)")
    p.add_argument("names", nargs="*", help="battles (default: every two-sided one cached)")
    p.add_argument("--every", type=float, default=0.5, help="s of a battle between positions looked at")
    p.add_argument("--gap", type=float, default=5.0, help="s between two pushes of the same battle")
    p.add_argument("--push", type=float, default=5.0, help="elixir of red troops on blue's side that make a push")
    p.add_argument("--alone", type=float, default=0.0, help="tiles: no blue troop this close to the push yet (0: any)")
    p.add_argument("--elixir", type=float, default=5.0, help="blue holds at least this much")
    p.add_argument("--b-net", default=None, help="B's first move from this net (its best move other than waiting) "
                   "instead of the model's own: e.g. runs/v7e/top.pt, which defends at once where h1 waits")
    p.add_argument("--rollouts", type=int, default=8, help="games per copy (on average)")
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
    games, score, lost = rollouts(players, data, deck, [m[:2] for m in ms], args)
    report(ms, games, score, lost)


if __name__ == "__main__":
    main()
