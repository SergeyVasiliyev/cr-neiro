"""PPO with action masking on the batched crsim environment.

    .venv\\Scripts\\python.exe python\\train_ppo.py --run first --opponent heuristic
    .venv\\Scripts\\python.exe python\\train_ppo.py --run mix --opponent random:0.3,heuristic:0.4,selfplay:0.3
    .venv\\Scripts\\python.exe python\\train_ppo.py --run cont --resume runs\\first\\latest.pt

The agent always plays blue (side 0); thanks to the own-frame observations the trained policy
can play either side. Opponents: noop | random | heuristic (scripted C++ bots) and selfplay
(frozen snapshots of the agent itself). Progress: TensorBoard logs in runs/<name>
(`.venv\\Scripts\\tensorboard.exe --logdir runs`), checkpoints runs/<name>/latest.pt.
"""
import argparse
import collections
import copy
import itertools
import os
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import crsim_env as ce
from model import TORCH_DTYPES, PolicyNet, load_policy, obs_to_torch, save_checkpoint

ROOT = ce.ROOT


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=time.strftime("run_%Y%m%d_%H%M%S"), help="run name (folder in runs/)")
    p.add_argument("--data", default=ce.DATA_DIR)
    p.add_argument("--deck", default=",".join(ce.DEFAULT_DECK), help="agent deck, 8 card ids")
    p.add_argument("--opp-deck", default=None, help="opponent deck (default: same as --deck)")
    p.add_argument("--random-decks", type=int, default=0,
                   help="instead of fixed decks: N random decks of fully supported cards, every pairing")
    p.add_argument("--opponent", default="heuristic",
                   help="noop|random|heuristic|selfplay or a mix like random:0.3,heuristic:0.7")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--total-steps", type=float, default=5e6, help="agent decisions to train for")
    p.add_argument("--decision-ticks", type=int, default=10, help="ticks (50 ms) between decisions")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--reward-win", type=float, default=1.0)
    p.add_argument("--reward-tower", type=float, default=0.5, help="per full side of tower HP dealt/taken")
    p.add_argument("--reward-crown", type=float, default=0.1)
    p.add_argument("--reward-leak", type=float, default=0.0, help="penalty per wasted elixir")
    p.add_argument("--reward-trade", type=float, default=0.0,
                   help="per elixir of the opponent's troops lost (and spells cast) minus our own")
    p.add_argument("--trade-floor", type=float, default=0.25,
                   help="share of a troop's worth that only its death takes (damage takes the rest)")
    p.add_argument("--snapshot-every", type=int, default=25, help="iterations between self-play snapshots")
    p.add_argument("--save-every", type=int, default=20, help="iterations between checkpoints")
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--key-spots", action="store_true",
                   help="act on key spots only (on by itself when resuming a checkpoint trained so)")
    p.add_argument("--anchor", default=None,
                   help="a frozen policy to stay close to (e.g. the behavior-cloned one): KL penalty towards it")
    p.add_argument("--anchor-coef", type=float, default=0.1, help="KL penalty weight at the start ...")
    p.add_argument("--anchor-coef-end", type=float, default=0.01, help="... falling linearly to this at the end")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0, help="C++ simulation threads (0 = all cores)")
    p.add_argument("--no-amp", action="store_true", help="disable fp16 mixed precision on the GPU")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def parse_opponents(spec, total_envs):
    """'random:0.3,heuristic:0.7' -> [('random', 19), ('heuristic', 45)]"""
    parts = []
    for item in spec.split(","):
        name, _, weight = item.partition(":")
        name = name.strip()
        if name not in ce.BOT_KINDS + ("selfplay",):
            raise SystemExit(f"unknown opponent {name!r}")
        parts.append((name, float(weight) if weight else 1.0))
    total_w = sum(w for _, w in parts)
    counts = [max(1, round(total_envs * w / total_w)) for _, w in parts]
    counts[-1] = max(1, total_envs - sum(counts[:-1]))
    return [(name, c) for (name, _), c in zip(parts, counts)]


def build_matchups(args, data):
    if args.random_decks:
        decks = ce.random_decks(data, args.random_decks, seed=args.seed)
        return [(a, b) for a, b in itertools.product(decks, decks)], decks
    deck = ce.parse_deck(args.deck, data)
    opp = ce.parse_deck(args.opp_deck, data) if args.opp_deck else deck
    return [(deck, opp)], [deck, opp]


class Group:
    """A VecEnv whose red side is one kind of opponent."""

    def __init__(self, name, n, data, matchups, args, reward, seed):
        bot = name if name != "selfplay" else "noop"
        self.name, self.n = name, n
        self.env = ce.VecEnv(data, matchups, num_envs=n, bots=("noop", bot), decision_ticks=args.decision_ticks,
                             seed=seed, reward=reward, num_threads=args.threads, key_spots=args.key_spots)
        self.results = collections.deque(maxlen=300)
        self.lengths = collections.deque(maxlen=300)
        self.crowns = collections.deque(maxlen=300)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    # Mixed precision + cuDNN autotuning: ~10x faster updates on a Turing GPU (RTX 20xx).
    torch.backends.cudnn.benchmark = True
    use_amp = device.type == "cuda" and not args.no_amp
    amp = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=use_amp)  # noqa: E731
    scaler = torch.amp.GradScaler(enabled=use_amp)
    run_dir = os.path.join(ROOT, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    log_file = open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8")

    def log(line):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    data = ce.load_data(args.data)
    matchups, decks = build_matchups(args, data)
    reward = ce.make_reward(args.reward_win, args.reward_crown, args.reward_tower, args.reward_leak,
                            args.reward_trade, args.trade_floor)
    if args.resume and not args.key_spots:
        args.key_spots = bool(torch.load(args.resume, map_location="cpu", weights_only=False).get("key_spots", False))
    groups = [Group(name, n, data, matchups, args, reward, args.seed * 100 + i)
              for i, (name, n) in enumerate(parse_opponents(args.opponent, args.envs))]
    n_envs = sum(g.n for g in groups)

    if args.resume:
        net, ckpt = load_policy(args.resume, device)
        if ckpt.get("card_ids") != list(data.card_ids):
            log("WARNING: card list differs from the checkpoint; card embeddings may be misaligned")
        net.train()
        start_step = ckpt.get("global_step", 0)
    else:
        net, ckpt, start_step = PolicyNet(ce.crsim.rl.vocab_size(data)).to(device), None, 0
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    if ckpt and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    snapshots = collections.deque([copy.deepcopy(net).eval()], maxlen=10)
    anchor = None
    if args.anchor:  # stays as it is: only a reference the policy is pulled towards, less and less
        anchor, _ = load_policy(args.anchor, device)
        for prm in anchor.parameters():
            prm.requires_grad_(False)

    writer = SummaryWriter(run_dir)
    writer.add_text("args", str(vars(args)))
    log(f"run dir: {run_dir}")
    log(f"device: {device}, envs: {n_envs} = " + ", ".join(f"{g.name} x{g.n}" for g in groups))
    log(f"matchups: {len(matchups)}, decks: " + " | ".join(",".join(d) for d in decks[:4]))
    log(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    T = args.steps
    shapes = {k: v.shape[1:] for k, v in groups[0].env.obs(0).items()}
    buf = {k: torch.zeros((T, n_envs) + tuple(s), dtype=TORCH_DTYPES[k], device=device) for k, s in shapes.items()}
    actions_buf = torch.zeros((T, n_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, n_envs), device=device)
    values_buf = torch.zeros((T, n_envs), device=device)
    rewards_buf = torch.zeros((T, n_envs), device=device)
    dones_buf = torch.zeros((T, n_envs), device=device)

    def gather(side):
        return {k: np.concatenate([g.env.obs(side)[k] for g in groups]) for k in shapes}

    global_step = start_step
    iteration = 0
    total = int(args.total_steps)
    while global_step < total:
        iteration += 1
        t0 = time.time()
        frac = 1.0 - (global_step - start_step) / max(1, total - start_step)
        optimizer.param_groups[0]["lr"] = args.lr * max(frac, 0.05)
        anchor_coef = args.anchor_coef_end + (args.anchor_coef - args.anchor_coef_end) * frac
        opponent_net = random.choice(snapshots)

        net.eval()
        for t in range(T):
            obs = obs_to_torch(gather(0), device)
            with torch.no_grad(), amp():
                logits, value = net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            for k in shapes:
                buf[k][t] = obs[k]
            actions_buf[t], logp_buf[t], values_buf[t] = action, dist.log_prob(action), value

            act_np = action.cpu().numpy()
            offset = 0
            for g in groups:
                acts = np.full((g.n, 2), ce.BOT_ACTION, dtype=np.int32)
                acts[:, 0] = act_np[offset:offset + g.n]
                if g.name == "selfplay":
                    red = obs_to_torch(g.env.obs(1), device)
                    with amp():
                        acts[:, 1] = opponent_net.act(red)[0].cpu().numpy()
                rew, done = g.env.step(acts)
                rewards_buf[t, offset:offset + g.n] = torch.as_tensor(rew[:, 0], device=device)
                dones_buf[t, offset:offset + g.n] = torch.as_tensor(done, device=device, dtype=torch.float32)
                for i in np.flatnonzero(done):
                    w = int(g.env.winner[i])
                    g.results.append(1.0 if w == 0 else (0.5 if w == 2 else 0.0))
                    g.lengths.append(int(g.env.episode_steps[i]))
                    g.crowns.append(int(g.env.crowns[i, 0]) - int(g.env.crowns[i, 1]))
                offset += g.n
        global_step += T * n_envs
        sim_time = time.time() - t0

        # --- GAE ---
        with torch.no_grad(), amp():
            _, next_value = net(obs_to_torch(gather(0), device))
        adv = torch.zeros_like(rewards_buf)
        last = torch.zeros(n_envs, device=device)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else values_buf[t + 1]
            nonterminal = 1.0 - dones_buf[t]
            delta = rewards_buf[t] + args.gamma * nv * nonterminal - values_buf[t]
            last = delta + args.gamma * args.lam * nonterminal * last
            adv[t] = last
        returns = adv + values_buf

        # --- PPO update ---
        net.train()
        flat = {k: v.reshape((T * n_envs,) + v.shape[2:]) for k, v in buf.items()}
        b_act, b_logp = actions_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret, b_val = adv.reshape(-1), returns.reshape(-1), values_buf.reshape(-1)
        n = T * n_envs
        # Steps where waiting was the only legal action carry no policy gradient; averaging over
        # them would dilute the update ~5-10x (most of the time elixir is below every card's cost).
        b_choice = flat["action_mask"].sum(-1) > 1
        stats = collections.defaultdict(list)
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, args.minibatch):
                idx = perm[start:start + args.minibatch]
                with amp():
                    logits, value = net({k: v[idx] for k, v in flat.items()})
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                choice = b_choice[idx].float()
                n_choice = choice.sum().clamp(min=1.0)
                a = b_adv[idx]
                a_mean = (a * choice).sum() / n_choice
                a_std = (((a - a_mean) ** 2 * choice).sum() / n_choice).sqrt()
                a = (a - a_mean) / (a_std + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip))
                pg_loss = (pg * choice).sum() / n_choice
                v_clipped = b_val[idx] + (value - b_val[idx]).clamp(-args.clip, args.clip)
                v_loss = 0.5 * torch.max((value - b_ret[idx]) ** 2, (v_clipped - b_ret[idx]) ** 2).mean()
                entropy = (dist.entropy() * choice).sum() / n_choice
                loss = pg_loss + args.vf * v_loss - args.ent * entropy
                if anchor is not None:
                    with torch.no_grad(), amp():
                        anchor_logits, _ = anchor({k: v[idx] for k, v in flat.items()})
                    logp_all = torch.log_softmax(logits.float(), -1)
                    logq_all = torch.log_softmax(anchor_logits.float(), -1)
                    kl_anchor = ((logp_all.exp() * (logp_all - logq_all)).sum(-1) * choice).sum() / n_choice
                    loss = loss + anchor_coef * kl_anchor
                    stats["kl_anchor"].append(kl_anchor.item())
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad)
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    stats["pg_loss"].append(pg_loss.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(entropy.item())
                    stats["approx_kl"].append(((((ratio - 1) - ratio.log()) * choice).sum() / n_choice).item())
                    stats["clip_frac"].append(((((ratio - 1).abs() > args.clip).float() * choice).sum() / n_choice).item())
                    stats["grad_norm"].append(grad_norm.item())
                    stats["amp_scale"].append(scaler.get_scale() if use_amp else 1.0)
                    stats["decision_frac"].append(choice.mean().item())

        # --- logging ---
        dt = time.time() - t0
        sps = T * n_envs / dt
        for k, v in stats.items():
            writer.add_scalar(f"loss/{k}", np.mean(v), global_step)
        writer.add_scalar("perf/steps_per_sec", sps, global_step)
        writer.add_scalar("perf/sim_fraction", sim_time / dt, global_step)
        writer.add_scalar("train/reward_per_step", rewards_buf.mean().item(), global_step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
        line = [f"it {iteration:4d}", f"steps {global_step / 1e6:6.2f}M", f"{sps:6.0f} sps"]
        for g in groups:
            if g.results:
                wr = float(np.mean(g.results))
                writer.add_scalar(f"winrate/{g.name}", wr, global_step)
                writer.add_scalar(f"episode/length_{g.name}", np.mean(g.lengths), global_step)
                writer.add_scalar(f"episode/crown_diff_{g.name}", np.mean(g.crowns), global_step)
                line.append(f"vs {g.name}: win {wr:5.1%} ({len(g.results)})")
        line.append(f"ent {np.mean(stats['entropy']):.2f} kl {np.mean(stats['approx_kl']):.4f}")
        if anchor is not None:
            line.append(f"kl to anchor {np.mean(stats['kl_anchor']):.3f} (x{anchor_coef:.3f})")
        log(" | ".join(line))

        if iteration % args.snapshot_every == 0:
            snapshots.append(copy.deepcopy(net).eval())
        if iteration % args.save_every == 0 or global_step >= total:
            meta = dict(global_step=global_step, card_ids=list(data.card_ids), args=vars(args),
                        key_spots=args.key_spots)
            save_checkpoint(os.path.join(run_dir, "latest.pt"), net, optimizer, **meta)
            if iteration % (args.save_every * 10) == 0:
                save_checkpoint(os.path.join(run_dir, f"step_{global_step}.pt"), net, optimizer, **meta)
    writer.close()
    log(f"done, checkpoint: {os.path.join(run_dir, 'latest.pt')}")
    log_file.close()


if __name__ == "__main__":
    main()
