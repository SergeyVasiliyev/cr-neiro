"""A copy of the user: a policy fine-tuned to choose as the user did (tools/user_examples.py), kept close to where it
started everywhere else (KL to it on the same states), so a handful of games moves its style, not its skill.

    .venv\\Scripts\\python.exe tools\\clone_user.py --start runs/v7e/top.pt --data raw/bc/user.npz --out runs/clone/user_v1.pt

The last --holdout share of the examples (the latest games) is kept out to check it learns the user, not the noise:
it prints how often the copy's most likely move on held-out plays is the user's card / the user's card and spot.
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from crsim_env import OBS_KEYS  # noqa: E402
from model import load_policy, obs_to_torch, save_checkpoint  # noqa: E402

CELLS = 576


def batches(data, idx, size, device):
    for k in range(0, len(idx), size):
        b = idx[k:k + size]
        yield obs_to_torch({key: data[key][b] for key in OBS_KEYS}, device), torch.as_tensor(data["action"][b], device=device)


def evaluate(net, data, idx, device):
    plays = [i for i in idx if data["action"][i] > 0]
    card = exact = 0
    with torch.no_grad():
        for obs, act in batches(data, np.array(plays), 256, device):
            logits = net(obs)[0].float()
            logits[:, 0] = -1e9  # which play, given that they played
            pick = logits.argmax(-1)
            exact += int((pick == act).sum())
            card += int((((pick - 1) // CELLS) == ((act - 1) // CELLS)).sum())
    return card / max(1, len(plays)), exact / max(1, len(plays)), len(plays)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="runs/v7e/top.pt")
    p.add_argument("--data", default="raw/bc/user.npz")
    p.add_argument("--out", default="runs/clone/user_v1.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--kl", type=float, default=0.5, help="weight of the pull to the start policy")
    p.add_argument("--wait-weight", type=float, default=0.3)
    p.add_argument("--holdout", type=float, default=0.15)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ckpt = load_policy(os.path.join(ROOT, args.start), device)
    ref = copy.deepcopy(net).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    raw = np.load(os.path.join(ROOT, args.data))
    data = {key: raw[key] for key in list(OBS_KEYS) + ["action"]}
    n = len(data["action"])
    split = int(n * (1 - args.holdout))
    train_idx, hold_idx = np.arange(split), np.arange(split, n)
    c0, e0, n_hold = evaluate(net, data, hold_idx, device)
    print(f"start: held-out plays ({n_hold}): user's card {c0:.0%}, card and spot {e0:.0%}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best, best_score = None, -1.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        rng = np.random.default_rng(epoch)
        losses = []
        for obs, act in batches(data, rng.permutation(train_idx), 128, device):
            logits = net(obs)[0].float()
            logp = torch.log_softmax(logits, -1)
            w = torch.where(act > 0, 1.0, args.wait_weight)
            ce = -(logp.gather(-1, act.unsqueeze(-1)).squeeze(-1) * w).sum() / w.sum()
            with torch.no_grad():
                ref_logp = torch.log_softmax(ref(obs)[0].float(), -1)
            kl = (ref_logp.exp() * (ref_logp - logp)).sum(-1).mean()
            loss = ce + args.kl * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append((ce.item(), kl.item()))
        net.eval()
        c, e, _ = evaluate(net, data, hold_idx, device)
        ce_m, kl_m = np.mean(losses, 0)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}: train ce {ce_m:.3f} kl {kl_m:.3f} | held-out: user's card {c:.0%}, card and spot {e:.0%}",
                  flush=True)
        score = c + e
        if score > best_score:
            best_score, best = score, copy.deepcopy(net.state_dict())
    net.load_state_dict(best)
    c, e, _ = evaluate(net, data, hold_idx, device)
    print(f"kept: held-out user's card {c:.0%} (start {c0:.0%}), card and spot {e:.0%} (start {e0:.0%})")
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "config")}
    meta["cloned_from"] = args.start
    save_checkpoint(out, net, None, **meta)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
