"""v7: PPO from scratch with three critics, self-play on both sides and opponents picked by win rate.

    .venv\\Scripts\\python.exe python\\train_v7.py --run v7 --hours 7.5

- Critics: one per reward part - the result (+-1, gamma 0.999: the whole match), the shaping signals (towers,
  crowns, elixir trades, leak; gamma 0.99) and the style one (the elixir bank: patience). A move's advantage is
  result + w_shaping * shaping + w_style * style, and the weights fade (1 -> 0.1 and 1 -> 0.3 over the first
  --w-until of the run): our hints first, the policy's own judgement of what wins later.
- Self-play: the current policy on both sides of --self-envs matches, learning from both. Pool: --pool-envs
  matches against scripted bots, earlier policies and its own snapshots, each picked by how close the win
  rate against it is to 50% (p * (1 - p)): beaten bots fade out, stronger nets come in as it catches up.
- Learning rate: adapted to keep the policy change per iteration near --target-kl, then decays to ~0 over the
  last --anneal of the run (a policy stopped at a high rate plays clearly worse). Entropy: two targets the
  bonus weights adapt to - "which card where" when it plays (--ent-place) and "play now or wait" (--ent-time).
- Like the live game: moves land --delay ticks after the decision, and the observations get the screen
  reader's kind of noise (--noise).
The schedule runs on the clock: progress = time spent / --hours, so it ends annealed when the time is up.
Progress: runs/<run>/train.log; ">>> eval" lines: greedy games against the benchmark policies.
"""
import argparse
import collections
import copy
import math
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch

import crsim_env as ce
from model import TORCH_DTYPES, PolicyNet, load_policy, obs_to_torch, save_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = ("result", "shaping", "style")
POOL = ["heuristic", "pusher", "runs/bc_v5/latest.pt", "runs/ppo_v5/latest.pt", "runs/ppo_v5c/latest.pt",
        "runs/ppo_v5d/latest.pt", "runs/ppo_v6/chunk_1.pt", "runs/ppo_v6/best.pt"]
BENCH = ["heuristic", "pusher", "runs/ppo_v5d/latest.pt", "runs/ppo_v6/best.pt"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="v7")
    p.add_argument("--hours", type=float, default=7.5, help="time budget (the schedule runs on the clock)")
    p.add_argument("--resume", default=None, help="checkpoint of this kind of run; --hours is then the time left")
    p.add_argument("--progress", type=float, default=None,
                   help="with --resume: where in the schedule to go on from instead of the checkpoint's own (a run "
                        "made longer: back from inside the anneal, to train at the full rate again before it)")
    p.add_argument("--init", default=None, help="start from these weights (a distilled net), schedule from zero")
    p.add_argument("--distill", default=None,
                   help="teacher checkpoint: instead of PPO, the teacher plays and the net learns to choose like it "
                        "(KL to its move probabilities); the critics learn the teacher's games")
    p.add_argument("--anchor", default=None, help="KL pull towards this policy at first (fading, see --anchor-coef)")
    p.add_argument("--anchor-coef", type=float, nargs=2, default=(0.05, 0.0))
    p.add_argument("--anchor-until", type=float, default=0.3, help="share of the run the anchor fades over")
    p.add_argument("--deck", default=",".join(ce.DEFAULT_DECK))
    p.add_argument("--decision-ticks", type=int, default=6)
    p.add_argument("--spots", type=int, default=2,
                   help="placements: 1 = key spots, 2 = key spots and the tactical ones next to the troops")
    p.add_argument("--self-envs", type=int, default=48)
    p.add_argument("--pool-envs", type=int, default=40)
    p.add_argument("--steps", type=int, default=128, help="rollout length per iteration")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--forced-keep", type=float, default=0.2,
                   help="share of the wait-only steps kept for the critics' update (they carry no policy gradient)")
    p.add_argument("--size", default="160,3,4,48,320", help="d_model,layers,heads,grid channels,trunk")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-range", type=float, nargs=2, default=(1e-5, 5e-4))
    p.add_argument("--target-kl", type=float, default=0.01)
    p.add_argument("--anneal", type=float, default=0.15, help="last share of the run with the rate decaying")
    p.add_argument("--max-grad", type=float, default=1.0)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--gamma", type=float, nargs=3, default=(0.999, 0.99, 0.99), help="per critic")
    p.add_argument("--value-scale", type=float, nargs=3, default=(1.0, 1.0, 30.0),
                   help="each critic learns its returns times this, so all three are about the same size")
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--w-shaping", type=float, nargs=2, default=(1.0, 0.1))
    p.add_argument("--w-style", type=float, nargs=2, default=(1.0, 0.3))
    p.add_argument("--w-until", type=float, default=0.6)
    p.add_argument("--ent-place", type=float, nargs=2, default=(1.2, 0.7), help="target, start and end (nats)")
    p.add_argument("--card-floor", type=float, nargs=2, default=(0.0, 0.0),
                   help="when it plays, every card in hand keeps at least this share of the play (start, end): a card "
                        "never tried is never learned (v10 dropped the spells within 5M steps)")
    p.add_argument("--card-floor-coef", type=float, default=0.1, help="weight of the --card-floor penalty")
    p.add_argument("--ent-time", type=float, nargs=2, default=(0.0, 0.0),
                   help="0 = no bonus on play-now-or-wait (a target there left greedy play hardly ever playing)")
    p.add_argument("--reward-win", type=float, default=1.0)
    p.add_argument("--reward-tower", type=float, default=1.0)
    p.add_argument("--reward-crown", type=float, default=0.2)
    p.add_argument("--reward-trade", type=float, default=0.03)
    p.add_argument("--trade-floor", type=float, default=0.25)
    p.add_argument("--reward-leak", type=float, default=0.005)
    p.add_argument("--leak-relative", action="store_true",
                   help="leak penalty only for the leak the opponent doesn't match (both waiting at 10 is fine)")
    p.add_argument("--elixir-bank", type=float, default=0.0005, help="per elixir held per second")
    p.add_argument("--bank-cap", type=float, default=9.0)
    p.add_argument("--delay", type=int, nargs=2, default=(2, 5), help="ticks (50 ms) a move lands after the decision")
    p.add_argument("--noise", type=float, nargs=2, default=(0.03, 0.3), help="miss chance, position jitter (tiles)")
    p.add_argument("--elixir-noise", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                   help="enemy elixir seen as live: per-match offset mean and spread, jitter per step (elixir)")
    p.add_argument("--live-view", type=float, nargs=5, default=(0.0, 0.0, 0.0, 0.0, 0.0),
                   help="the opponent seen like the live twin sees it (rl::LiveView, measured 2026-11-15 with "
                        "tools/twin_plays_check.py, after the second-box fix: 0.035 0.03 0.9 0.05 0.3): chance a play of theirs is missed (its "
                        "troops never seen, its elixir never taken off), chance per play of a phantom one besides, "
                        "seconds a play is made out late at most, chance a troop's HP is seen off, by this spread")
    p.add_argument("--endgame", type=float, default=0.0,
                   help="share of episodes starting late with towers at random HP (finishing, defending, overtime)")
    p.add_argument("--human-starts", type=float, default=0.0,
                   help="share of episodes starting at a moment of a recorded human battle (python/human_starts.py): "
                        "positions its own games never reach - a push from behind the king, a low tower, an "
                        "elixir lead - to learn how they end")
    p.add_argument("--human-every", type=float, default=1.5, help="s of a recorded battle between its start positions")
    p.add_argument("--giant-starts", type=float, default=0.0,
                   help="share of episodes starting right after blue put a giant behind its king at a moment the user "
                        "would (9+ elixir, the giant in hand, no enemy troop on its side; tools/back_giant_check.py), "
                        "from the start net's games against itself: practice in following such a push up")
    p.add_argument("--giant-moments", type=int, default=2000, help="how many such positions")
    p.add_argument("--push-starts", type=float, default=0.0,
                   help="share of episodes starting at a moment of the user's pushes in the recorded two-sided battles "
                        "(tools/push_defence_check.py: red troops worth --push-value elixir or more on blue's side; blue "
                        "is the bot's side): live, h1 waited at 10 elixir while such a push took a tower (2026-11-10)")
    p.add_argument("--push-every", type=float, default=0.5, help="s of a recorded battle between push positions")
    p.add_argument("--push-value", type=float, default=5.0, help="elixir of red troops on blue's side that make a push")
    p.add_argument("--back-starts", type=float, default=0.0,
                   help="share of episodes starting at a moment of the recorded two-sided battles where the user (red) "
                        "has troops behind his king or a princess tower and his king still sleeps "
                        "(tools/push_defence_check.py back_positions): live, h1 cycle 4 spelled the sleeping king there "
                        "and its games against nets never reach such openings (2026-11-11)")
    p.add_argument("--back-every", type=float, default=0.5, help="s of a recorded battle between back positions")
    p.add_argument("--back-y", type=float, default=27.5,
                   help="red troops at this y or more are behind its princess towers' line (red's king is at 29)")
    p.add_argument("--giant-lead", type=float, nargs="+", default=None,
                   help="with --giant-starts: blue is ahead by this much elixir when it puts the giant down (red keeps "
                        "at most blue's elixir minus it; two numbers: drawn evenly between) - the user's idea "
                        "(2026-11-10): a push from behind the king pays with an elixir lead, so practise building one")
    p.add_argument("--pool", nargs="*", default=POOL)
    p.add_argument("--bench", nargs="*", default=BENCH, help="opponents of the greedy evaluations")
    p.add_argument("--snapshot-every", type=float, default=2e6, help="learner steps between own snapshots")
    p.add_argument("--max-snapshots", type=int, default=6)
    p.add_argument("--fixed-share", type=float, default=0.25,
                   help="share of pool games spread evenly over the bots and earlier nets whatever the win rates")
    p.add_argument("--bots-early", type=float, default=0.0,
                   help="share of the run (from the start) in which pool games are against the scripted bots only: "
                        "a net from scratch learns the basics against them before meeting stronger nets")
    p.add_argument("--bot-share", type=float, default=0.0,
                   help="share of the pool games against the scripted bots, whatever else: self-play drifts away "
                        "from beating them (v10 after its bots-only start: 92%% -> 18%% against the heuristic in 3 h)")
    p.add_argument("--strong", nargs="*", default=[],
                   help="pool nets with a share of their own (--strong-share): ones it rarely beats would get few "
                        "games by win rate alone")
    p.add_argument("--strong-share", type=float, default=0.0, help="share of the pool games against --strong")
    p.add_argument("--strong-weights", type=float, nargs="*", default=None,
                   help="how often each --strong net is picked, relative (default: all alike)")
    p.add_argument("--active-nets", type=int, default=0,
                   help="at most this many opponent nets in the pool games at a time (0: no limit), chosen afresh "
                        "every --active-every iterations: each net is a GPU pass of its own per step (v10 with ~18 "
                        "of them: 1000 steps/s instead of 3700)")
    p.add_argument("--active-every", type=int, default=10)
    p.add_argument("--eval-minutes", type=float, default=50)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--final-games", type=int, default=400)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


class Opponent:
    def __init__(self, name, bot=None, net=None, spots=1):
        self.name, self.bot, self.net, self.spots = name, bot, net, spots  # spots: the placements it knows
        self.results = collections.deque(maxlen=100)

    def win_rate(self):  # ours against it, starting from one win and one loss
        return (sum(self.results) + 1.0) / (len(self.results) + 2.0)

    def weight(self):
        p = self.win_rate()
        return max(p * (1.0 - p), 0.03 if self.bot else 0.02)


def lerp(a, b, t):
    return a + (b - a) * min(1.0, max(0.0, t))


def evaluate(model, opponent, args, games):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "python", "evaluate.py"), "--model", model,
                          "--opponent", opponent, "--deck", args.deck, "--games", str(games), "--envs", "100",
                          "--decision-ticks", str(args.decision_ticks), "--greedy"],
                         cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    m = re.search(r"wins \d+ \(([\d.]+)%\)", out)
    style = re.search(r"blue's style: (.*)", out)
    return (float(m.group(1)) if m else float("nan")), (style.group(1) if style else "")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda"
    amp = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)  # noqa: E731
    scaler = torch.amp.GradScaler(enabled=use_amp)
    run_dir = os.path.join(ROOT, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")

    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    data = ce.load_data()
    deck = ce.parse_deck(args.deck, data)
    reward = ce.make_reward(args.reward_win, args.reward_crown, args.reward_tower, args.reward_leak,
                            args.reward_trade, args.trade_floor, True, args.elixir_bank, args.bank_cap,
                            leak_relative=args.leak_relative)
    common = dict(decision_ticks=args.decision_ticks, reward=reward, num_threads=args.threads, key_spots=args.spots,
                  delay_ticks=tuple(args.delay), obs_noise=tuple(args.noise), endgame_share=args.endgame,
                  elixir_noise=tuple(args.elixir_noise), live_view=tuple(args.live_view))
    S, P = args.self_envs, args.pool_envs
    self_env = ce.VecEnv(data, [(deck, deck)], num_envs=S, bots=("noop", "noop"), seed=args.seed * 100 + 1, **common)
    pool_env = ce.VecEnv(data, [(deck, deck)], num_envs=P, bots=("noop", "noop"), seed=args.seed * 100 + 2, **common)
    L = 2 * S + P  # learner rows: self-play blue, self-play red, pool blue
    groups = []  # (what, positions, share of the episodes): starts other than a fresh match, mixed in one list
    if args.human_starts > 0:
        import human_starts
        starts, _ = human_starts.build(data, args.human_every, log=log)  # (built on `data`, which outlives the envs)
        log(f"human starts for {args.human_starts:.0%} of the episodes: {human_starts.describe(starts, data)}")
        groups.append(("human", starts, args.human_starts))
    if args.giant_starts > 0:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import back_giant_check
        src = args.resume or args.init
        if not src:
            raise SystemExit("--giant-starts needs a net to find the moments with (--resume or --init)")
        gargs = argparse.Namespace(model=src, opponent=src, temp=0.5, envs=100, seed=args.seed + 11,
                                   moments=args.giant_moments, elixir=9.0, gap=20.0, lead=args.giant_lead)
        moments = back_giant_check.collect(back_giant_check.Players(gargs, device), data, deck, gargs)
        starts = [m[1] for m in moments]  # the copies with the giant put behind blue's king
        lead = f", blue {'-'.join(f'{x:g}' for x in args.giant_lead)} elixir ahead" if args.giant_lead else ""
        log(f"giant starts for {args.giant_starts:.0%} of the episodes: {len(starts)} positions with a giant just put "
            f"behind blue's king{lead} (from {src})")
        groups.append(("giant", starts, args.giant_starts))
        del moments
        torch.cuda.empty_cache()
    if args.push_starts > 0:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import push_defence_check
        starts = push_defence_check.push_positions(data, every=args.push_every, push=args.push_value, log=log)
        log(f"push starts for {args.push_starts:.0%} of the episodes: {len(starts)} positions of the user's pushes on "
            f"blue's side in the recorded battles (blue: the bot's side, to defend)")
        groups.append(("push", starts, args.push_starts))
    if args.back_starts > 0:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import push_defence_check
        starts = push_defence_check.back_positions(data, every=args.back_every, y_min=args.back_y, log=log)
        log(f"back starts for {args.back_starts:.0%} of the episodes: {len(starts)} positions with the user's troops "
            f"behind his king or a princess tower while his king sleeps (red: the user)")
        groups.append(("back", starts, args.back_starts))
    if groups:
        total = sum(share for _, _, share in groups)
        if total >= 1.0:
            raise SystemExit("the start shares add up to 100% or more: no fresh matches left")
        # one list for set_starts: each kind drawn (with repeats) in proportion to its share
        rng = random.Random(args.seed + 21)
        combined = []
        for what, starts, share in groups:
            combined += rng.choices(starts, k=max(1, round(4000 * share / total))) if starts else []
        kept = [env.set_starts(combined, total) for env in (self_env, pool_env)]
        log(f"starts: {total:.0%} of the episodes from {kept[0]} of {len(combined)} positions (" +
            ", ".join(f"{what} {share:.0%}" for what, _, share in groups) + ")")
        del combined, groups

    state = {}
    if args.resume:
        net, ckpt = load_policy(args.resume, device)
        state = ckpt.get("v7", {})
        global_step = int(ckpt.get("global_step", 0))
    elif args.init:
        net, _ = load_policy(args.init, device)
        ckpt, global_step = None, 0
    else:
        d, layers, heads, grid, trunk = (int(x) for x in args.size.split(","))
        net = PolicyNet(ce.crsim.rl.vocab_size(data), d_model=d, n_layers=layers, n_heads=heads, grid_channels=grid,
                        trunk=trunk, n_values=len(PARTS)).to(device)
        ckpt, global_step = None, 0
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    if ckpt and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    lr = state.get("lr", args.lr)
    log_alpha = {"place": state.get("log_alpha_place", math.log(0.01)), "time": state.get("log_alpha_time", math.log(0.01))}
    progress0 = state.get("progress", 0.0) if args.progress is None else args.progress

    def frozen(path):
        fnet, fck = load_policy(path, device)
        for prm in fnet.parameters():
            prm.requires_grad_(False)
        return fnet, fck

    teacher = frozen(args.distill)[0] if args.distill else None
    anchor = frozen(args.anchor)[0] if args.anchor else None

    pool = []
    for spec in args.pool:
        if spec in ce.BOT_KINDS:
            pool.append(Opponent(spec, bot=spec))
            continue
        path = os.path.join(ROOT, spec)
        if not os.path.exists(path):
            log(f"pool: {spec} not found, skipped")
            continue
        onet, ock = load_policy(path, device)
        if "key_spots" not in ock:  # (nets from before key spots: old and weak; an all-tiles net says key_spots 0)
            log(f"pool: {spec} acts on every tile, not key spots: skipped")
            continue
        for prm in onet.parameters():
            prm.requires_grad_(False)
        pool.append(Opponent(os.path.relpath(path, os.path.join(ROOT, "runs")).replace("\\", "/"), net=onet,
                             spots=int(ock.get("key_spots", 0))))
    snapshots = []
    if args.resume:  # the latest step checkpoints of this run as its earlier selves, as before the restart
        steps = sorted(int(m.group(1)) for m in (re.fullmatch(r"step_(\d+)M\.pt", f) for f in os.listdir(run_dir))
                       if m and int(m.group(1)) * 1e6 < global_step)
        for n in steps[-args.max_snapshots:]:
            snapshots.append(Opponent(f"self@{n}M", net=frozen(os.path.join(run_dir, f"step_{n}M.pt"))[0].eval(),
                                      spots=args.spots))
        if snapshots:
            log(f"snapshots from the run's checkpoints: {', '.join(o.name for o in snapshots)}")
    strong_names = {os.path.relpath(os.path.join(ROOT, s), os.path.join(ROOT, "runs")).replace("\\", "/")
                    for s in args.strong}
    strong = [o for o in pool if o.name in strong_names]
    if args.strong and len(strong) < len(args.strong):
        log(f"strong: only {', '.join(o.name for o in strong)} found in the pool")
    weight_of = {}
    if args.strong_weights:
        if len(args.strong_weights) != len(args.strong):
            raise SystemExit("--strong-weights: one per --strong net")
        weight_of = {os.path.relpath(os.path.join(ROOT, s), os.path.join(ROOT, "runs")).replace("\\", "/"): w
                     for s, w in zip(args.strong, args.strong_weights)}
    strong_weights = [weight_of.get(o.name, 1.0) for o in strong]

    progress = progress0  # (pick() reads it; the loop below advances it)

    active = []  # --active-nets: the opponent nets in play now (refresh_active)

    def refresh_active():
        # half by win rate (the closest games), the rest at random among the earlier nets and snapshots
        nets = [o for o in pool + snapshots if o.net is not None]
        chosen = []
        while nets and len(chosen) < args.active_nets:
            if len(chosen) < (args.active_nets + 1) // 2:
                o = random.choices(nets, weights=[o.weight() for o in nets])[0]
            else:
                o = random.choice(nets)
            chosen.append(o)
            nets.remove(o)
        active[:] = chosen

    def pick():
        # at first only the bots (--bots-early) ...
        bots = [o for o in pool if o.bot]
        if bots and (progress < args.bots_early or random.random() < args.bot_share):
            return random.choice(bots)
        if strong and random.random() < args.strong_share:
            return random.choices(strong, weights=strong_weights)[0]
        # ... then mostly by win rate, but a fixed share spread evenly over the bots and earlier nets, so they
        # never drop out (v7's first night: self-play only, it went round in circles and lost to the bots)
        if args.active_nets > 0:
            if not active:
                refresh_active()
            if args.bot_share > 0:
                bots = []  # (the bots play through --bot-share only)
            fixed, choice = bots + [o for o in active if o in pool], bots + active
        else:
            fixed, choice = pool, pool + snapshots
        if random.random() < args.fixed_share and fixed:
            return random.choice(fixed)
        return random.choices(choice, weights=[o.weight() for o in choice])[0]

    cur = [pick() for _ in range(P)]
    for i, o in enumerate(cur):
        pool_env.set_bot(i, 1, o.bot or "noop")
        pool_env.set_key_spots(i, 1, o.spots)
    pool_env.reset()
    nxt = [pick() for _ in range(P)]
    for i, o in enumerate(nxt):
        pool_env.set_bot(i, 1, o.bot or "noop")

    log(f"run dir: {run_dir}, device {device}")
    log(f"learner rows: {L} = self-play {S} x 2 sides + pool {P}; pool: " + ", ".join(o.name for o in pool))
    log(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M, config {net.config}")
    log(f"args: {vars(args)}")

    keys = list(self_env.obs(0).keys())
    shapes = {k: self_env.obs(0)[k].shape[1:] for k in keys}
    T = args.steps
    buf = {k: torch.zeros((T, L) + tuple(s), dtype=TORCH_DTYPES[k], device=device) for k, s in shapes.items()}
    actions_buf = torch.zeros((T, L), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, L), device=device)
    values_buf = torch.zeros((T, L, len(PARTS)), device=device)
    rewards_buf = torch.zeros((T, L, len(PARTS)), device=device)
    dones_buf = torch.zeros((T, L), device=device)
    gammas = torch.tensor(args.gamma, device=device)
    scales = torch.tensor(args.value_scale, device=device)
    spell_token = torch.zeros(ce.crsim.rl.vocab_size(data), dtype=torch.bool, device=device)
    for card in data.cards.values():
        spell_token[ce.crsim.rl.FIRST_CARD_TOKEN + card.index] = bool(card.is_spell)

    def learner_obs():
        a, b, c = self_env.obs(0), self_env.obs(1), pool_env.obs(0)
        return {k: np.concatenate([a[k], b[k], c[k]]) for k in keys}

    started = time.time()
    budget = args.hours * 3600.0
    progress = progress0
    last_eval = time.time()
    next_snapshot = global_step + args.snapshot_every
    lr_at_anneal = None
    iteration = 0
    game_lengths = collections.deque(maxlen=300)
    eval_no = 0

    def save(path):
        meta = dict(global_step=global_step, card_ids=list(data.card_ids), args=vars(args), key_spots=args.spots,
                    v7=dict(progress=progress, lr=lr, log_alpha_place=log_alpha["place"],
                            log_alpha_time=log_alpha["time"]))
        save_checkpoint(path, net, optimizer, **meta)

    def run_eval(games, tag):
        latest = os.path.join(run_dir, "latest.pt")
        save(latest)
        parts = []
        for opp in args.bench:
            wr, style = evaluate(latest, os.path.join(ROOT, opp) if opp not in ce.BOT_KINDS else opp, args, games)
            parts.append(f"{opp.replace('runs/', '').replace('/latest.pt', '').replace('.pt', '')} {wr:.0f}%")
        fin = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "finisher_test.py"), latest], cwd=ROOT,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
        m = re.search(r"finishes (\d+/\d+)", fin)
        log(f">>> eval {tag} at {global_step / 1e6:.1f}M ({time.strftime('%H:%M')}): " + ", ".join(parts)
            + f" | overtime finish {m.group(1) if m else '?'} | style: {style}")

    while progress < 1.0:
        iteration += 1
        t0 = time.time()
        progress = progress0 + (1.0 - progress0) * (time.time() - started) / budget
        w_shaping = lerp(*args.w_shaping, progress / args.w_until)
        w_style = lerp(*args.w_style, progress / args.w_until)
        ent_frac = progress / (1.0 - args.anneal)
        target_place = lerp(*args.ent_place, ent_frac)
        card_floor = lerp(*args.card_floor, ent_frac)
        target_time = lerp(*args.ent_time, ent_frac)
        if progress >= 1.0 - args.anneal:
            if lr_at_anneal is None:
                lr_at_anneal = lr
                log(f"--- annealing from lr {lr:.2e}")
            lr = lr_at_anneal * max(0.03, (1.0 - progress) / args.anneal)
        optimizer.param_groups[0]["lr"] = lr

        # --- rollout ---
        net.eval()
        for t in range(T):
            obs = obs_to_torch(learner_obs(), device)
            with torch.no_grad(), amp():
                logits, value = net(obs)
                if teacher is not None:  # distilling: the teacher plays, the net's critics follow its games
                    logits = teacher(obs)[0]
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            for k in keys:
                buf[k][t] = obs[k]
            actions_buf[t], logp_buf[t], values_buf[t] = action, dist.log_prob(action), value
            a = action.cpu().numpy()
            self_acts = np.stack([a[:S], a[S:2 * S]], axis=1)
            pool_acts = np.full((P, 2), ce.BOT_ACTION, dtype=np.int32)
            pool_acts[:, 0] = a[2 * S:]
            red = pool_env.obs(1)
            by_net = collections.defaultdict(list)
            for i, o in enumerate(cur):
                if o.net is not None:
                    by_net[id(o)].append(i)
            for idx in by_net.values():
                # padded to a multiple of 8: a few batch shapes only (cuDNN tunes itself for every new one)
                rows = idx + [idx[-1]] * (min(P, -(-len(idx) // 8) * 8) - len(idx))
                with torch.no_grad(), amp():
                    out = cur[idx[0]].net.act(obs_to_torch({k: red[k][rows] for k in keys}, device))[0]
                pool_acts[idx, 1] = out[:len(idx)].cpu().numpy()
            _, done_s = self_env.step(self_acts)
            parts_s = torch.as_tensor(self_env.reward_parts, device=device) * scales
            rewards_buf[t, :S], rewards_buf[t, S:2 * S] = parts_s[:, 0], parts_s[:, 1]
            ds = torch.as_tensor(done_s, device=device, dtype=torch.float32)
            dones_buf[t, :S], dones_buf[t, S:2 * S] = ds, ds
            for i in np.flatnonzero(done_s):
                game_lengths.append(int(self_env.episode_steps[i]))
            _, done_p = pool_env.step(pool_acts)
            rewards_buf[t, 2 * S:] = torch.as_tensor(pool_env.reward_parts[:, 0], device=device) * scales
            dones_buf[t, 2 * S:] = torch.as_tensor(done_p, device=device, dtype=torch.float32)
            for i in np.flatnonzero(done_p):
                w = int(pool_env.winner[i])
                cur[i].results.append(1.0 if w == 0 else (0.5 if w == 2 else 0.0))
                game_lengths.append(int(pool_env.episode_steps[i]))
                cur[i] = nxt[i]
                pool_env.set_key_spots(i, 1, cur[i].spots)
                nxt[i] = pick()
                pool_env.set_bot(i, 1, nxt[i].bot or "noop")
        global_step += T * L
        rollout_s = time.time() - t0

        # --- advantages, one per critic ---
        with torch.no_grad():
            with amp():
                _, next_value = net(obs_to_torch(learner_obs(), device))
            next_value = next_value.float()
            adv = torch.zeros_like(rewards_buf)
            last = torch.zeros((L, len(PARTS)), device=device)
            for t in reversed(range(T)):
                nv = next_value if t == T - 1 else values_buf[t + 1]
                nonterminal = (1.0 - dones_buf[t]).unsqueeze(-1)
                delta = rewards_buf[t] + gammas * nv * nonterminal - values_buf[t]
                last = delta + gammas * args.lam * nonterminal * last
                adv[t] = last
            returns = adv + values_buf
            mixed = adv[..., 0] / scales[0] + w_shaping * adv[..., 1] / scales[1] + w_style * adv[..., 2] / scales[2]
            ev = [(1.0 - (returns[..., k] - values_buf[..., k]).var() / returns[..., k].var().clamp(min=1e-8)).item()
                  for k in range(len(PARTS))]
            # how it plays: elixir when putting a card down, spells
            played = actions_buf > 0
            elixir = buf["global"][..., 4] * 10.0
            slot = ((actions_buf - 1).clamp(min=0) // ce.CELLS)
            card_tok = buf["hand"].gather(-1, slot.unsqueeze(-1)).squeeze(-1)
            n_play = int(played.sum())
            play_elixir = float(elixir[played].mean()) if n_play else float("nan")
            from_8 = float((elixir[played] >= 8).float().mean()) if n_play else float("nan")
            spell_share = float(spell_token[card_tok[played]].float().mean()) if n_play else float("nan")

        # --- update ---
        net.train()
        flat = {k: v.reshape((T * L,) + v.shape[2:]) for k, v in buf.items()}
        b_act, b_logp, b_adv = actions_buf.reshape(-1), logp_buf.reshape(-1), mixed.reshape(-1)
        b_ret = returns.reshape(-1, len(PARTS))
        choice_all = flat["action_mask"].sum(-1) > 1
        keep = choice_all | (torch.rand(T * L, device=device) < args.forced_keep)
        rows = torch.nonzero(keep).squeeze(-1)
        # whole minibatches only (one batch shape for cuDNN); the rest is a random few left out each epoch
        n_use = (len(rows) // args.minibatch) * args.minibatch or len(rows)
        stats = collections.defaultdict(list)
        stop = False
        backup = {k: v.detach().clone() for k, v in net.state_dict().items()}  # to step back from a blow-up
        anchor_coef = lerp(*args.anchor_coef, progress / args.anchor_until) if anchor is not None else 0.0
        skipped = 0
        for _ in range(args.epochs):
            perm = rows[torch.randperm(len(rows), device=device)[:n_use]]
            for start in range(0, len(perm), args.minibatch):
                idx = perm[start:start + args.minibatch]
                mb = {k: v[idx] for k, v in flat.items()}
                with amp():
                    logits, value = net(mb)
                logp_all = torch.log_softmax(logits, -1)
                choice = choice_all[idx].float()
                n_choice = choice.sum().clamp(min=1.0)
                v_loss = ((value.float() - b_ret[idx]) ** 2).mean(0).sum()
                if teacher is not None:  # distilling: choose like the teacher (KL to its move probabilities)
                    with torch.no_grad(), amp():
                        t_logits = teacher(mb)[0]
                    t_logp = torch.log_softmax(t_logits.float(), -1)
                    kl_t = (t_logp.exp() * (t_logp - logp_all)).sum(-1)
                    distill_loss = (kl_t * choice).sum() / n_choice
                    loss = distill_loss + args.vf * v_loss
                    if not torch.isfinite(loss):
                        skipped += 1
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    stats["kl"].append(distill_loss.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["grad"].append(grad_norm.item())
                    with torch.no_grad():
                        stats["agree"].append(((logits.argmax(-1) == t_logits.argmax(-1)).float() * choice).sum().item()
                                              / n_choice.item())
                        logq = torch.log_softmax(logits[:, 1:], -1)
                        p_wait = logp_all[:, 0].exp().clamp(1e-6, 1 - 1e-6)
                        w_play = choice * (1 - p_wait)
                        stats["h_place"].append(((-(logq.exp() * logq).sum(-1) * w_play).sum()
                                                 / w_play.sum().clamp(min=1e-6)).item())
                        stats["h_time"].append(0.0)
                        stats["clip"].append(0.0)
                        stats["pg_loss"].append(0.0)
                    continue
                logp = logp_all.gather(-1, b_act[idx].unsqueeze(-1)).squeeze(-1)
                ratio = (logp - b_logp[idx]).exp()
                a = b_adv[idx]
                a_mean = (a * choice).sum() / n_choice
                a_std = (((a - a_mean) ** 2 * choice).sum() / n_choice).sqrt()
                a = (a - a_mean) / (a_std + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip))
                pg_loss = (pg * choice).sum() / n_choice
                # entropy in two parts: play now or wait, and which card where when it plays
                p_wait = logp_all[:, 0].exp().clamp(1e-6, 1 - 1e-6)
                h_time = -(p_wait * p_wait.log() + (1 - p_wait) * (1 - p_wait).log())
                logq = torch.log_softmax(logits[:, 1:], -1)
                h_place = -(logq.exp() * logq).sum(-1)
                w_play = choice * (1 - p_wait).detach()
                h_place_mean = (h_place * w_play).sum() / w_play.sum().clamp(min=1e-6)
                if card_floor > 0:
                    # each card in hand (legal now) gets at least card_floor of the play: penalty on the shortfall
                    slot_logits = logits[:, 1:].float().reshape(logits.shape[0], ce.crsim.rl.HAND_SIZE, -1).logsumexp(-1)
                    slot_legal = slot_logits > -1e8
                    slot_logp = torch.log_softmax(slot_logits.masked_fill(~slot_legal, -1e9), -1)
                    shortfall = (math.log(card_floor) - slot_logp).clamp(min=0.0, max=5.0) * slot_legal
                    floor_loss = (shortfall.sum(-1) * w_play).sum() / w_play.sum().clamp(min=1e-6)
                else:
                    floor_loss = torch.zeros((), device=logits.device)
                h_time_mean = (h_time * choice).sum() / n_choice
                alpha_time = math.exp(log_alpha["time"]) if args.ent_time[0] > 0 else 0.0
                loss = (pg_loss + args.vf * v_loss - math.exp(log_alpha["place"]) * h_place_mean
                        - alpha_time * h_time_mean + args.card_floor_coef * floor_loss)
                if anchor_coef > 0:
                    with torch.no_grad(), amp():
                        a_logp = torch.log_softmax(anchor(mb)[0].float(), -1)
                    kl_a = (logp_all.exp() * (logp_all - a_logp)).sum(-1)
                    kl_anchor = (kl_a * choice).sum() / n_choice
                    loss = loss + anchor_coef * kl_anchor
                    stats["kl_anchor"].append(kl_anchor.item())
                if not torch.isfinite(loss):  # never step on a broken loss
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    kl = ((((ratio - 1) - ratio.log()) * choice).sum() / n_choice).item()
                    stats["kl"].append(kl)
                    stats["clip"].append(((((ratio - 1).abs() > args.clip).float() * choice).sum() / n_choice).item())
                    stats["h_place"].append(h_place_mean.item())
                    stats["floor"].append(floor_loss.item())
                    stats["h_time"].append(h_time_mean.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["pg_loss"].append(pg_loss.item())
                    stats["grad"].append(grad_norm.item())
                if kl > 3 * args.target_kl or not math.isfinite(kl):  # too far from the policy that played: stop here
                    stop = True
                    break
            if stop:
                break

        # a blow-up (non-finite weights, or nothing it could learn from): back to the weights before, half the rate
        broken = not all(torch.isfinite(v).all() for v in net.state_dict().values() if v.is_floating_point())
        if broken or not stats["kl"]:
            net.load_state_dict(backup)
            lr = max(lr / 2, args.lr_range[0])
            if lr_at_anneal is not None:
                lr_at_anneal = max(lr_at_anneal / 2, args.lr_range[0])
            log(f"!!! it {iteration}: update broke the net ({skipped} broken minibatches): stepped back, lr {lr:.1e}")
            continue
        # --- adapt: learning rate to the target change, entropy weights to their targets ---
        kl = float(np.mean(stats["kl"]))
        if lr_at_anneal is None and teacher is None:
            if kl > 2 * args.target_kl:
                lr = max(lr / 1.5, args.lr_range[0])
            elif kl < args.target_kl / 2:
                lr = min(lr * 1.5, args.lr_range[1])
        h_place, h_time = float(np.mean(stats["h_place"])), float(np.mean(stats["h_time"]))
        log_alpha["place"] = min(math.log(0.3), max(math.log(1e-4), log_alpha["place"] + 0.3 * (target_place - h_place)))
        log_alpha["time"] = min(math.log(0.3), max(math.log(1e-4), log_alpha["time"] + 0.3 * (target_time - h_time)))

        if args.active_nets > 0 and iteration % args.active_every == 0:
            refresh_active()

        # --- snapshots, logging, checkpoints, evaluation ---
        if global_step >= next_snapshot:
            frozen = copy.deepcopy(net).eval()
            for prm in frozen.parameters():
                prm.requires_grad_(False)
            snapshots.append(Opponent(f"self@{global_step / 1e6:.0f}M", net=frozen, spots=args.spots))
            if len(snapshots) > args.max_snapshots:
                snapshots.pop(0)
            next_snapshot += args.snapshot_every
        dt = time.time() - t0
        log(f"it {iteration:4d} | {global_step / 1e6:6.2f}M | {T * L / dt:5.0f} sps ({rollout_s / dt:.0%} play) | "
            f"{progress:5.1%} | lr {lr:.1e} kl {kl:.4f} clip {np.mean(stats['clip']):.2f} "
            f"grad {np.mean(stats['grad']):.2f} pg {np.mean(stats['pg_loss']):.3f} v {np.mean(stats['v_loss']):.2f}"
            + (" (stopped early)" if stop else "") + (f" skipped {skipped}" if skipped else "")
            + (f" | same move as the teacher {np.mean(stats['agree']):.0%}" if stats["agree"] else "")
            + (f" | kl to anchor {np.mean(stats['kl_anchor']):.3f} (x{anchor_coef:.3f})" if stats["kl_anchor"] else "")
            + (f" | card floor {card_floor:.3f} short {np.mean(stats['floor']):.2f}" if card_floor > 0 else "")
            + f" | ent place {h_place:.2f}/{target_place:.2f} time {h_time:.2f}/{target_time:.2f} "
            f"(x{math.exp(log_alpha['place']):.3f}, x{math.exp(log_alpha['time']):.3f}) | "
            f"w {w_shaping:.2f}/{w_style:.2f} | EV " + "/".join(f"{e:.2f}" for e in ev)
            + f" | play at {play_elixir:.1f} elixir, 8+ {from_8:.0%}, spells {spell_share:.0%}, "
            f"game {np.mean(game_lengths) * args.decision_ticks * 0.05 if game_lengths else 0:.0f}s")
        if iteration % 10 == 0:
            ranked = sorted(pool + snapshots, key=lambda o: -o.weight())
            log("    pool: " + ", ".join(f"{o.name} {o.win_rate():.0%}/{len(o.results)}" for o in ranked))
            save(os.path.join(run_dir, "latest.pt"))
        if time.time() - last_eval > args.eval_minutes * 60 and progress < 0.97:
            eval_no += 1
            run_eval(args.eval_games, f"#{eval_no}")
            save(os.path.join(run_dir, f"step_{global_step // 1_000_000}M.pt"))
            last_eval = time.time()

    save(os.path.join(run_dir, "latest.pt"))
    save(os.path.join(run_dir, "final.pt"))
    log(f"--- done ({time.strftime('%Y-%m-%d %H:%M')}), final evaluation")
    run_eval(args.final_games, "final")
    log("=== all done")


if __name__ == "__main__":
    main()
