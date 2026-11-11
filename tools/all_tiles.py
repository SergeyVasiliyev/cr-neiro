"""Turns a policy trained on the key + tactical spots into one that places on every tile and still plays as it did.

Opened up as it is, such a policy puts much of its weight on tiles it was never trained on (their scores mean
nothing). Here a copy learns, on states of the policy's own games, to give the original's answer - the original asked
with its spots' mask, the copy answering with every tile open - so it leaves the other tiles alone wherever the
original never had them. The result starts a copy of human play (tools/clone_user.py, which then moves placements
to the tiles people really use) or training on every tile.

    .venv\\Scripts\\python.exe tools\\all_tiles.py --start runs/sx1/best.pt --out runs/sx1/all_tiles.pt

It prints how much of the copy's weight is off the spots on held-out states (the original's: --check), and how
often their most likely moves agree.
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from model import load_policy, obs_to_torch, save_checkpoint  # noqa: E402

TEAMS = (crsim.Team.BLUE, crsim.Team.RED)


def collect(net, data, args, rhythm, device):
    """States where a play is possible, from the policy's games against itself: obs with every tile open, and the
    spots' mask beside."""
    deck = ce.parse_deck(",".join(ce.DEFAULT_DECK), data)
    env = ce.VecEnv(data, [(deck, deck)], num_envs=args.envs, bots=("noop", "noop"), decision_ticks=rhythm,
                    seed=args.seed, key_spots=(2, 2))
    rng = np.random.default_rng(args.seed)
    keys = list(ce.OBS_KEYS)
    store = {k: [] for k in keys + ["spot_mask"]}
    n, steps, t0 = 0, 0, time.time()
    while n < args.states:
        actions = np.zeros((env.num_envs, 2), dtype=np.int32)
        for side in (0, 1):
            obs_np = env.obs(side)
            with torch.no_grad():
                logits = net(obs_to_torch(obs_np, device))[0].float() / args.temp
                actions[:, side] = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            can_play = obs_np["action_mask"].sum(1) > 1
            for i in np.flatnonzero(can_play & (rng.random(env.num_envs) < args.keep)):
                if n >= args.states:
                    break
                full = crsim.rl.encode(env.battle(i), TEAMS[side], key_spots=0)["action_mask"]
                for k in keys:
                    store[k].append(np.array(obs_np[k][i]))
                store["spot_mask"].append(np.array(obs_np["action_mask"][i], dtype=bool))
                store["action_mask"][-1] = np.array(full, dtype=bool)
                n += 1
        env.step(actions)
        steps += 1
    print(f"{n} states from {steps} steps of {env.num_envs} games in {time.time() - t0:.0f} s", flush=True)
    return {k: np.stack(v) for k, v in store.items()}


def batches(states, idx, size, device):
    for k in range(0, len(idx), size):
        b = idx[k:k + size]
        obs = obs_to_torch({key: states[key][b] for key in ce.OBS_KEYS}, device)
        ref_obs = dict(obs)
        ref_obs["action_mask"] = torch.as_tensor(states["spot_mask"][b], device=device).bool()
        yield obs, ref_obs


def check(student, teacher, states, idx, device):
    """(weight off the spots, argmax agreement with the teacher, KL teacher -> student) over idx."""
    off = agree = kl = 0.0
    with torch.no_grad():
        for obs, ref_obs in batches(states, idx, 512, device):
            lp = torch.log_softmax(student(obs)[0].float(), -1)
            ref_lp = torch.log_softmax(teacher(ref_obs)[0].float(), -1)
            p = lp.exp()
            off += float((p * ~ref_obs["action_mask"]).sum())
            agree += float((lp.argmax(-1) == ref_lp.argmax(-1)).sum())
            kl += float((ref_lp.exp() * (ref_lp - lp)).sum())
    n = max(1, len(idx))
    return off / n, agree / n, kl / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="runs/sx1/best.pt")
    p.add_argument("--out", default="runs/sx1/all_tiles.pt")
    p.add_argument("--states", type=int, default=60000)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--keep", type=float, default=0.5, help="share of the possible-play states kept (spread them out)")
    p.add_argument("--temp", type=float, default=1.0, help="the games are played at this temperature")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    teacher, ckpt = load_policy(os.path.join(ROOT, args.start), device)
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    rhythm = int(ckpt.get("args", {}).get("decision_ticks", 4))
    data = ce.load_data()
    states = collect(teacher, data, args, rhythm, device)
    n = len(states["global"])
    order = np.random.default_rng(args.seed).permutation(n)
    hold, train = order[:int(n * args.holdout)], order[int(n * args.holdout):]
    student = copy.deepcopy(teacher)
    for prm in student.parameters():
        prm.requires_grad_(True)
    off, agree, kl = check(student, teacher, states, hold, device)
    print(f"start: weight off the spots {off:.2f}, same most likely move {agree:.0%}, KL {kl:.3f}", flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        student.train()
        rng = np.random.default_rng(epoch)
        losses = []
        for obs, ref_obs in batches(states, rng.permutation(train), 256, device):
            lp = torch.log_softmax(student(obs)[0].float(), -1)
            with torch.no_grad():
                ref_lp = torch.log_softmax(teacher(ref_obs)[0].float(), -1)
            loss = (ref_lp.exp() * (ref_lp - lp)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        student.eval()
        off, agree, kl = check(student, teacher, states, hold, device)
        print(f"epoch {epoch}: train KL {np.mean(losses):.3f} | held-out: weight off the spots {off:.3f}, "
              f"same most likely move {agree:.0%}, KL {kl:.3f}", flush=True)
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "config")}
    meta["key_spots"] = 0  # it places on every tile
    meta["all_tiles_from"] = args.start
    save_checkpoint(out, student, None, **meta)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
