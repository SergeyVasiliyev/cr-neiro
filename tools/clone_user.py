"""A copy of the user: a policy fine-tuned to choose as the user did (tools/user_examples.py, tools/human_examples.py),
kept close to where it started everywhere else (KL to it on the same states), so a handful of games moves its style,
not its skill.

    .venv\\Scripts\\python.exe tools\\clone_user.py --start runs/v7e/top.pt --data raw/bc/user.npz --out runs/clone/user_v1.pt
    .venv\\Scripts\\python.exe tools\\clone_user.py --start runs/sx1/best.pt --data raw/bc/user_v2.npz --kl 0.05 ^
        --out runs/clone/user_v2.pt

Examples with a "game" array are held out by whole games (every --holdout-every-th game), else the last --holdout
share of them. It prints how often the copy's most likely move on held-out plays is the user's card, the user's card
on the user's tile, and on a tile within 1.5 of it. --train-share keeps only that share of the training games (to see
whether more games would still help).

Examples made with every tile allowed (human_examples.py --spots 0) carry the key + tactical spots' mask beside: a
start policy trained on the spots is asked about its own moves with its own mask, and the copy is saved as a policy
that places on every tile (key_spots 0). Better: start from its all-tiles version (tools/all_tiles.py), which gives
no weight to tiles it was never trained on.
"""
import argparse
import copy
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from crsim_env import OBS_KEYS  # noqa: E402
from model import load_policy, obs_to_torch, save_checkpoint  # noqa: E402

CELLS = 576
GRID_W = 18


def batches(data, idx, size, device, ref_spots=False):
    """(obs, the start policy's obs, actions): the start policy is asked with the spots' mask if it was trained on
    them (ref_spots) and the examples have every tile open."""
    for k in range(0, len(idx), size):
        b = idx[k:k + size]
        obs = obs_to_torch({key: data[key][b] for key in OBS_KEYS}, device)
        ref_obs = obs
        if ref_spots and "spot_mask" in data:
            ref_obs = dict(obs)
            ref_obs["action_mask"] = torch.as_tensor(data["spot_mask"][b], device=device).bool()
        yield obs, ref_obs, torch.as_tensor(data["action"][b], device=device)


def evaluate(net, data, idx, device):
    plays = np.array([i for i in idx if data["action"][i] > 0])
    card = exact = near = 0
    with torch.no_grad():
        for obs, _, act in batches(data, plays, 256, device):
            logits = net(obs)[0].float()
            logits[:, 0] = -1e9  # which play, given that they played
            pick = logits.argmax(-1)
            exact += int((pick == act).sum())
            same_card = ((pick - 1) // CELLS) == ((act - 1) // CELLS)
            card += int(same_card.sum())
            pc, ac = (pick - 1) % CELLS, (act - 1) % CELLS
            d = torch.hypot((pc % GRID_W - ac % GRID_W).float(), (pc // GRID_W - ac // GRID_W).float())
            near += int((same_card & (d <= 1.5)).sum())
    n = max(1, len(plays))
    return card / n, exact / n, near / n, len(plays)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="runs/v7e/top.pt")
    p.add_argument("--data", default="raw/bc/user.npz")
    p.add_argument("--out", default="runs/clone/user_v1.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--kl", type=float, default=0.5, help="weight of the pull to the start policy")
    p.add_argument("--wait-weight", type=float, default=0.3)
    p.add_argument("--holdout", type=float, default=0.15, help="(examples without games) the last share held out")
    p.add_argument("--holdout-every", type=int, default=6, help="(examples by game) every n-th game is held out")
    p.add_argument("--holdout-from", default="",
                   help="(examples by game) held-out games only among those whose name matches this regex (the "
                        "user's own games, when other people's games are among the examples too)")
    p.add_argument("--train-share", type=float, default=1.0, help="train on only this share of the training games")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ckpt = load_policy(os.path.join(ROOT, args.start), device)
    ref = copy.deepcopy(net).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    ref_spots = int(ckpt.get("key_spots", 0)) > 0  # (an all-tiles start, tools/all_tiles.py, has 0)
    raw = np.load(os.path.join(ROOT, args.data))
    keys = list(OBS_KEYS) + ["action"] + [k for k in ("spot_mask", "game") if k in raw.files]
    data = {key: raw[key] for key in keys}
    n = len(data["action"])
    if "game" in data:
        games = list(dict.fromkeys(data["game"].tolist()))  # in the order they were made
        pool = [g for g in games if re.search(args.holdout_from, g)]  # the games a held-out one may be
        held = set(pool[args.holdout_every - 1::args.holdout_every])
        train_games = [g for g in games if g not in held]
        if args.train_share < 1.0:
            rng = np.random.default_rng(args.seed)
            keep = max(1, round(len(train_games) * args.train_share))
            train_games = [train_games[i] for i in sorted(rng.choice(len(train_games), keep, replace=False))]
        train_set = set(train_games)
        train_idx = np.array([i for i in range(n) if data["game"][i] in train_set])
        hold_idx = np.array([i for i in range(n) if data["game"][i] in held])
        print(f"{len(train_games)} games to learn from ({len(train_idx)} examples), {len(held)} held out "
              f"({len(hold_idx)} examples)", flush=True)
    else:
        split = int(n * (1 - args.holdout))
        train_idx, hold_idx = np.arange(split), np.arange(split, n)
    c0, e0, n0, n_hold = evaluate(net, data, hold_idx, device)
    print(f"start: held-out plays ({n_hold}): user's card {c0:.0%}, card and tile {e0:.0%}, within 1.5 tiles {n0:.0%}",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_score = None, -1.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        rng = np.random.default_rng(epoch + 1000 * args.seed)
        losses = []
        for obs, ref_obs, act in batches(data, rng.permutation(train_idx), 128, device, ref_spots):
            logits = net(obs)[0].float()
            logp = torch.log_softmax(logits, -1)
            w = torch.where(act > 0, 1.0, args.wait_weight)
            ce = -(logp.gather(-1, act.unsqueeze(-1)).squeeze(-1) * w).sum() / w.sum()
            with torch.no_grad():
                ref_logp = torch.log_softmax(ref(ref_obs)[0].float(), -1)
            ref_p = ref_logp.exp()  # (zero outside the start policy's own mask)
            kl = (ref_p * (ref_logp - logp)).sum(-1).mean()
            loss = ce + args.kl * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append((ce.item(), kl.item()))
        net.eval()
        c, e, nr, _ = evaluate(net, data, hold_idx, device)
        ce_m, kl_m = np.mean(losses, 0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}: train ce {ce_m:.3f} kl {kl_m:.3f} | held-out: user's card {c:.0%}, "
                  f"card and tile {e:.0%}, within 1.5 tiles {nr:.0%}", flush=True)
        score = c + nr
        if score > best_score:
            best_score, best = score, copy.deepcopy(net.state_dict())
    net.load_state_dict(best)
    c, e, nr, _ = evaluate(net, data, hold_idx, device)
    print(f"kept: held-out user's card {c:.0%} (start {c0:.0%}), card and tile {e:.0%} (start {e0:.0%}), "
          f"within 1.5 tiles {nr:.0%} (start {n0:.0%})")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "config")}
    meta["cloned_from"] = args.start
    if "spot_mask" in data:
        meta["key_spots"] = 0  # it places on every tile
    save_checkpoint(out, net, None, **meta)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
