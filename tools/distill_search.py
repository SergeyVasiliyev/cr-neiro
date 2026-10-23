"""Teaches the network what its lookahead found (expert iteration, step 2): on the positions tools/search_games.py
saved, the net learns the move the search chose where it searched, and stays as it was (KL to the start net) where
it did not; its critics are kept as they were (they score the search's positions). A low learning rate: a settled
policy is easily spoiled.

    .venv\\Scripts\\python.exe tools\\distill_search.py --data raw\\search_games\\r1 --start runs/v7e/top.pt --run sd1

Then compare: python\\evaluate.py --model runs\\sd1\\final.pt --opponent runs\\v7e\\top.pt --native-rhythm ...
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from crsim_env import OBS_KEYS  # noqa: E402
from model import TORCH_DTYPES, load_policy, save_checkpoint  # noqa: E402


def load_shards(dirs):
    parts = [np.load(f) for d in dirs for f in sorted(glob.glob(os.path.join(ROOT, d, "*.npz")))]
    if not parts:
        sys.exit(f"no shards in {dirs}")
    keys = list(OBS_KEYS) + ["target", "searched", "cands", "q"]
    return {k: torch.from_numpy(np.concatenate([q[k] for q in parts])) for k in keys}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", required=True, help="folders of shards")
    p.add_argument("--start", default="runs/v7e/top.pt", help="the net to train (and the teacher for KL)")
    p.add_argument("--run", required=True)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--kl-search", type=float, default=0.1, help="--target hard: KL to the start net on searched positions too")
    p.add_argument("--target", default="soft", choices=("soft", "hard"),
                   help="soft: the start net's move probabilities, the compared moves reweighted by exp(beta x how much "
                        "better than average the search scored them); hard: the search's chosen move")
    p.add_argument("--beta", type=float, default=50.0, help="--target soft: how strongly the scores reweight")
    p.add_argument("--vf", type=float, default=0.5, help="weight of keeping the critics as the start net's")
    p.add_argument("--fresh-lr-mult", type=float, default=10.0,
                   help="steps this many times larger for unit features the start net never read (widened in, weights "
                        "at zero): a not promoted round starts them from zero again, so they must be learned in one")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    data = load_shards(args.data)
    n = len(data["target"])
    searched = data["searched"].bool()
    print(f"{n} positions, {int(searched.sum())} searched; search chose to wait in "
          f"{float((data['target'][searched] == 0).float().mean()):.0%} of them", flush=True)

    net, ckpt = load_policy(os.path.join(ROOT, args.start), device)
    net.train()
    teacher, _ = load_policy(os.path.join(ROOT, args.start), device)
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    fresh = net.entity_in.weight.detach().abs().sum(0) == 0  # inputs it never read yet
    if fresh.any():
        print(f"unit features {fresh.nonzero().flatten().tolist()} are new to it: steps x{args.fresh_lr_mult:g} there",
              flush=True)
    scaler = torch.amp.GradScaler("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.float16)  # noqa: E731
    run_dir = os.path.join(ROOT, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        stats = {"ce": [], "agree": [], "kl_plain": [], "v": []}
        for start in range(0, n - args.minibatch + 1, args.minibatch):  # whole minibatches: one shape for cuDNN
            idx = perm[start:start + args.minibatch]
            mb = {k: data[k][idx].to(device, TORCH_DTYPES[k], non_blocking=True) for k in OBS_KEYS}
            target = data["target"][idx].to(device).long()
            srch = searched[idx].to(device).float()
            cands = data["cands"][idx].to(device).long()
            q = data["q"][idx].to(device).float()
            with torch.no_grad(), amp():
                t_logits, t_value = teacher(mb)
            with amp():
                logits, value = net(mb)
            logp = torch.log_softmax(logits.float(), -1)
            t_logp = torch.log_softmax(t_logits.float(), -1)
            kl = (t_logp.exp() * (t_logp - logp)).sum(-1)
            ce = -logp.gather(1, target.clamp(min=0)[:, None])[:, 0]
            n_s, n_p = srch.sum().clamp(min=1), (1 - srch).sum().clamp(min=1)
            if args.target == "soft":
                valid = cands >= 0
                adv = torch.where(valid, q - torch.nan_to_num(q.nanmean(1, keepdim=True)), torch.zeros_like(q))
                adv = torch.nan_to_num(adv)
                improved = torch.log_softmax(t_logp.scatter_add(1, cands.clamp(min=0), args.beta * adv), -1)
                kl_s = (improved.exp() * (improved - logp)).sum(-1)
                loss_s = (kl_s * srch).sum() / n_s
            else:
                loss_s = ((ce + args.kl_search * kl) * srch).sum() / n_s
            loss_p = (kl * (1 - srch)).sum() / n_p
            v_loss = ((value.float() - t_value.float()) ** 2).mean()
            loss = loss_s + loss_p + args.vf * v_loss
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            before = net.entity_in.weight.detach()[:, fresh].clone() if fresh.any() else None
            scaler.step(optimizer)
            scaler.update()
            if before is not None:  # the step there, made larger
                with torch.no_grad():
                    w = net.entity_in.weight
                    w[:, fresh] += (args.fresh_lr_mult - 1) * (w[:, fresh] - before)
            with torch.no_grad():
                stats["ce"].append(float((ce * srch).sum() / n_s))
                stats["agree"].append(float(((logits.argmax(-1) == target).float() * srch).sum() / n_s))
                stats["kl_plain"].append(float(loss_p))
                stats["v"].append(float(v_loss))
        print(f"epoch {epoch + 1}/{args.epochs}: CE to the search's move {np.mean(stats['ce']):.3f}, "
              f"its move is the net's top one in {np.mean(stats['agree']):.0%}, KL where not searched "
              f"{np.mean(stats['kl_plain']):.4f}, critic drift {np.mean(stats['v']):.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model", "config", "optimizer")}
    meta["search_distill"] = {"start": args.start, "data": args.data, "positions": n}
    save_checkpoint(os.path.join(run_dir, "final.pt"), net, **meta)
    print(f"saved runs/{args.run}/final.pt\n=== distill done", flush=True)


if __name__ == "__main__":
    main()
