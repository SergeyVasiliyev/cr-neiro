"""Behavior cloning: teaches the policy network to play like a scripted bot and like the human players of
the two-sided recordings - a start for PPO.

    .venv\\Scripts\\python.exe python\\train_bc.py --run bc_pusher [--minutes 60] [--key-spots]
    .venv\\Scripts\\python.exe python\\train_ppo.py --run ppo_from_bc --resume runs\\bc_pusher\\latest.pt ...

Matches of the teacher bot run in the C++ batch env, against itself, the heuristic bot and a random bot
(states it would not reach against itself); every decision of a teacher side is an example: what it saw
(the network's observation) and the move it made (as the network's action - with --key-spots, the legal
key spot nearest to where the bot put the card). The policy learns to pick that move (cross-entropy over
the legal actions); the value head learns the outcome from there (the same discounted reward PPO uses), so
PPO starts from both. Waiting is most of the teacher's decisions: a share of those is kept, weighted back
up. --human mixes in examples of people playing (tools/human_examples.py), a --human-share of each batch.
Progress: runs/<name>/train.log.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import crsim_env as ce
from model import TORCH_DTYPES, PolicyNet, obs_to_torch, save_checkpoint

ROOT = ce.ROOT
NOOP = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="bc_pusher")
    p.add_argument("--deck", default=",".join(ce.DEFAULT_DECK))
    p.add_argument("--teacher", default="pusher")
    p.add_argument("--key-spots", action="store_true", help="act on key spots only (see crsim rl.key_spot_cells)")
    p.add_argument("--human", default="",
                   help="examples of human play (tools/human_examples.py) - only from games played for real")
    p.add_argument("--human-share", type=float, default=0.25, help="share of each batch from the human examples")
    p.add_argument("--human-wait-weight", type=float, default=3.0,
                   help="weight of a human 'wait' example (they are taken once a second, decisions are 0.3 s)")
    p.add_argument("--envs", type=int, default=128, help="matches at once (a quarter against a random bot)")
    p.add_argument("--decision-ticks", type=int, default=6)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--pool", type=int, default=200_000, help="examples kept for training (the newest)")
    p.add_argument("--keep-wait", type=float, default=0.1, help="share of 'wait' decisions kept")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--reward-win", type=float, default=1.0)
    p.add_argument("--reward-tower", type=float, default=0.5)
    p.add_argument("--reward-crown", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


class Pool:
    """A ring buffer of examples: observation arrays, action, return, weight."""

    def __init__(self, size, shapes, dtypes):
        self.size, self.n, self.at = size, 0, 0
        self.obs = {k: np.zeros((size,) + s, dtype=dtypes[k]) for k, s in shapes.items()}
        self.action = np.zeros(size, np.int64)
        self.ret = np.zeros(size, np.float32)
        self.weight = np.zeros(size, np.float32)

    def add(self, obs_rows, action, ret, weight):
        for j in range(len(action)):
            i = self.at
            for k, v in obs_rows.items():
                self.obs[k][i] = v[j]
            self.action[i], self.ret[i], self.weight[i] = action[j], ret[j], weight[j]
            self.at = (self.at + 1) % self.size
            self.n = min(self.n + 1, self.size)

    def sample(self, b, rng):
        idx = rng.integers(0, self.n, b)
        return {k: v[idx] for k, v in self.obs.items()}, self.action[idx], self.ret[idx], self.weight[idx]


class Episodes:
    """Per env and teacher side: the decisions of the match running there, until it ends."""

    def __init__(self, n_envs, sides):
        self.sides = sides
        self.steps = {(i, s): [] for i in range(n_envs) for s in sides}

    def record(self, i, side, obs_row, action, reward):
        self.steps[(i, side)].append((obs_row, action, reward))

    def finish(self, i, side, gamma, keep_wait, rng):
        """The ended match's examples: (obs rows, actions, returns, weights), waits thinned out."""
        steps, self.steps[(i, side)] = self.steps[(i, side)], []
        g, rets = 0.0, []
        for _, _, r in reversed(steps):
            g = r + gamma * g
            rets.append(g)
        rets.reverse()
        keep = [j for j, (_, a, _) in enumerate(steps) if a != NOOP or rng.random() < keep_wait]
        if not keep:
            return None
        obs = {k: np.stack([steps[j][0][k] for j in keep]) for k in steps[0][0]}
        actions = np.array([steps[j][1] for j in keep])
        weights = np.where(actions == NOOP, 1.0 / keep_wait, 1.0).astype(np.float32)
        return obs, actions, np.array([rets[j] for j in keep], np.float32), weights


def nearest_legal(action, mask):
    """The legal action of the same hand slot whose cell is nearest to this action's cell (NOOP if none)."""
    cells_n = ce.crsim.rl.CELLS
    grid_w = ce.crsim.rl.GRID_W
    slot, cell = divmod(action - 1, cells_n)
    legal = np.flatnonzero(mask[1 + slot * cells_n:1 + (slot + 1) * cells_n])
    if len(legal) == 0:
        return NOOP
    d = (legal % grid_w - cell % grid_w) ** 2 + (legal // grid_w - cell // grid_w) ** 2
    return 1 + slot * cells_n + int(legal[np.argmin(d)])


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    run_dir = os.path.join(ROOT, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")

    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    data = ce.load_data()
    deck = ce.parse_deck(args.deck, data)
    reward = ce.make_reward(args.reward_win, args.reward_crown, args.reward_tower)
    n_rand = max(1, args.envs // 8)
    n_heur = max(1, args.envs // 4)
    vec = lambda n, bots, seed: ce.VecEnv(data, [(deck, deck)], num_envs=n, bots=bots,  # noqa: E731
                                          decision_ticks=args.decision_ticks, seed=seed, reward=reward,
                                          key_spots=args.key_spots)
    groups = [  # (env, sides the teacher plays)
        (vec(args.envs - n_rand - n_heur, (args.teacher, args.teacher), args.seed), (0, 1)),
        (vec(n_heur, (args.teacher, "heuristic"), args.seed + 1), (0,)),
        (vec(n_rand, (args.teacher, "random"), args.seed + 2), (0,)),
    ]
    human = None
    if args.human and os.path.exists(args.human):
        z = np.load(args.human)
        human = {k: z[k] for k in z.files}
        human_w = np.where(human["action"] == NOOP, args.human_wait_weight, 1.0).astype(np.float32)
    shapes = {k: v.shape[1:] for k, v in groups[0][0].obs(0).items()}
    np_dtypes = {k: v.dtype for k, v in groups[0][0].obs(0).items()}
    pool = Pool(args.pool, shapes, np_dtypes)
    episodes = [Episodes(env.num_envs, sides) for env, sides in groups]

    net = PolicyNet(ce.crsim.rl.vocab_size(data)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    log(f"behavior cloning of {args.teacher}: {args.envs} matches at once, deck {','.join(deck)}, "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters, {device}")

    start = time.time()
    env_steps = updates = skipped = 0
    stats = {"loss": [], "acc_move": [], "acc_wait": [], "vloss": []}
    last_log = start
    while time.time() - start < args.minutes * 60:
        for (env, sides), eps in zip(groups, episodes):
            before = {s: {k: v.copy() for k, v in env.obs(s).items()} for s in sides}
            rew, done = env.step(np.full((env.num_envs, 2), ce.BOT_ACTION, dtype=np.int32))
            taken = env.env.taken.copy()
            rew, done = rew.copy(), done.copy()
            for s in sides:
                for i in range(env.num_envs):
                    a = int(taken[i, s])
                    if a != NOOP and not before[s]["action_mask"][i, a]:
                        a = nearest_legal(a, before[s]["action_mask"][i])  # the key spot nearest to the bot's
                        if a == NOOP:
                            skipped += 1  # nowhere the network could put it: not an example
                    eps.record(i, s, {k: v[i] for k, v in before[s].items()}, a, float(rew[i, s]))
                    if done[i]:
                        out = eps.finish(i, s, args.gamma, args.keep_wait, rng)
                        if out is not None:
                            pool.add(*out)
            env_steps += env.num_envs
        if pool.n < 5 * args.batch:
            continue
        net.train()
        for _ in range(args.updates_per_step):
            obs, action, ret, weight = pool.sample(args.batch, rng)
            if human is not None:  # a share of the batch: people playing (no return known: value not trained)
                n_h = int(args.batch * args.human_share)
                idx = rng.integers(0, len(human["action"]), n_h)
                obs = {k: np.concatenate([v[:args.batch - n_h], human[k][idx]]) for k, v in obs.items()}
                action = np.concatenate([action[:args.batch - n_h], human["action"][idx]])
                ret = np.concatenate([ret[:args.batch - n_h], np.full(n_h, np.nan, np.float32)])
                weight = np.concatenate([weight[:args.batch - n_h], human_w[idx]])
            obs_t = obs_to_torch(obs, device)
            action_t = torch.as_tensor(action, device=device)
            ret_t = torch.as_tensor(ret, device=device)
            w_t = torch.as_tensor(weight, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, value = net(obs_t)
            ce_loss = F.cross_entropy(logits, action_t, reduction="none")
            loss_pi = (ce_loss * w_t).sum() / w_t.sum()
            known = ~torch.isnan(ret_t)
            loss_v = F.mse_loss(value[known], ret_t[known])
            loss = loss_pi + args.vf * loss_v
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            updates += 1
            with torch.no_grad():
                pred = logits.argmax(-1)
                moves = action_t != NOOP
                stats["loss"].append(loss_pi.item())
                stats["vloss"].append(loss_v.item())
                if moves.any():
                    stats["acc_move"].append((pred[moves] == action_t[moves]).float().mean().item())
                if (~moves).any():
                    stats["acc_wait"].append((pred[~moves] == NOOP).float().mean().item())
        if time.time() - last_log > 60:
            last_log = time.time()
            m = {k: float(np.mean(v[-200:])) if v else float("nan") for k, v in stats.items()}
            log(f"{(time.time() - start) / 60:5.1f} min | env steps {env_steps / 1e6:6.2f}M | examples {pool.n} | "
                f"updates {updates} | loss {m['loss']:.3f} value {m['vloss']:.3f} | right move {m['acc_move']:.1%} "
                f"right wait {m['acc_wait']:.1%} | bot spots off the grid {skipped}")
            save_checkpoint(os.path.join(run_dir, "latest.pt"), net, opt, global_step=0,
                            card_ids=list(data.card_ids), teacher=args.teacher, key_spots=args.key_spots)
    save_checkpoint(os.path.join(run_dir, "latest.pt"), net, opt, global_step=0, card_ids=list(data.card_ids),
                    teacher=args.teacher, key_spots=args.key_spots)
    log("=== done")


if __name__ == "__main__":
    main()
