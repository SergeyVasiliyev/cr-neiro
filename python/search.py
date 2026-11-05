"""Lookahead for a trained policy: before a move it is thinking of, play each of its best few moves (and waiting)
forward in copies of the battle for a few seconds - both sides played by the policy itself, sampled, several times
each - and take the move whose positions come out best on average (the critic's view of the result, or the result
itself if the match ends). The policy is the intuition, the lookahead the calculation.

    from search import SearchPlayer
    player = SearchPlayer(net, ckpt)
    action = player.decide(battle, team)

Hidden information (the opponent's hand and elixir) is taken as the battle has it: exact in the simulator; in live
play the twin's guesses.

Speed: both sides of every copy go through the network in one batch, and on the GPU that batch runs as a captured
CUDA graph (one launch instead of hundreds of small ones) - a search is ~0.1-0.15 s instead of ~0.3 s.
"""
import numpy as np
import torch

import crsim
from crsim_env import OBS_KEYS
from model import TORCH_DTYPES, obs_to_torch


class _Runner:
    """The network on a fixed batch size: obs rows in, (sampled action, result value) per row out. On CUDA the
    forward + sampling is a captured graph fed through pinned buffers; elsewhere a plain forward."""

    def __init__(self, net, size, example, device):
        self.net, self.size, self.device = net, size, device
        self.graph = None
        self.host = {k: torch.zeros((size,) + example[k].shape, dtype=torch.from_numpy(example[k]).dtype,
                                    pin_memory=device == "cuda") for k in OBS_KEYS}
        self.host_np = {k: v.numpy() for k, v in self.host.items()}
        if device != "cuda":
            return
        self.static = {k: v.to(device) for k, v in self.host.items()}
        self.temp = torch.ones((), device=device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):  # warm up (cuDNN picks its kernels) before capturing
                self._run(self.static, self.temp)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out_action, self.out_value = self._run(self.static, self.temp)

    @torch.no_grad()
    def _run(self, raw, temp):
        obs = {k: v.to(TORCH_DTYPES[k]) for k, v in raw.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            logits, value = self.net(obs)
        value = value.float()
        value = value[:, 0] if value.dim() == 2 else value  # the result critic
        gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-9, 1.0)))
        return (logits / temp + gumbel).argmax(-1), value  # a sample at temperature temp

    def __call__(self, rows, temp):
        n = len(rows)
        for k in OBS_KEYS:
            np.stack([r[k] for r in rows], out=self.host_np[k][:n])
        if self.graph is None:
            return [t[:n].cpu().numpy() for t in self._run(obs_to_torch(self.host_np, self.device), temp)]
        for k in OBS_KEYS:
            self.static[k].copy_(self.host[k], non_blocking=True)
        self.temp.fill_(temp)
        self.graph.replay()
        return self.out_action[:n].cpu().numpy(), self.out_value[:n].cpu().numpy()


class SearchPlayer:
    def __init__(self, net, ckpt, device="cuda", candidates=5, rollouts=6, horizon_s=4.0, temperature=0.5,
                 rollout_temperature=1.0, think_below_wait=0.9, opponent_net=None, explore=0, think_anyway=0.0,
                 seed=0):
        """candidates: the policy's most likely plays tried (plus waiting); rollouts: copies played out per
        candidate; horizon_s: how far ahead; think_below_wait: search only when the policy's chance of waiting is
        below this (it wants to play something) - otherwise it just waits.
        For games to learn from (expert iteration) the search can also look where the policy would not:
        explore: this many more candidates, each a random card in hand and a random spot it may go (a play the
        policy gives ~0 chance is never tried, so never learned - a spell finishing a tower was one: the net 0/96,
        the search 39/48); think_anyway: the share of the moments the policy is sure to wait that are searched
        anyway."""
        self.net, self.device = net, device
        self.opp = opponent_net or net
        self.key_spots = int(ckpt.get("key_spots", 0))
        self.ticks = int(ckpt.get("args", {}).get("decision_ticks", 6))
        self.k, self.r, self.h = candidates, rollouts, horizon_s
        self.temp, self.rtemp, self.think = temperature, rollout_temperature, think_below_wait
        self.explore, self.think_anyway = explore, think_anyway
        self.rng = np.random.default_rng(seed)
        self.searches = 0
        self.last = None  # the latest search: (candidates, their mean scores, the move chosen)
        self._runners = {}

    def _encode(self, battle, team):
        return crsim.rl.encode(battle, team, key_spots=self.key_spots)

    def _runner(self, net, size, row):
        key = (id(net), size)
        if key not in self._runners:
            self._runners[key] = _Runner(net, size, row, self.device)
        return self._runners[key]

    @torch.no_grad()
    def _policy(self, battle, team):
        obs = obs_to_torch({k: v[None] for k, v in self._encode(battle, team).items()}, self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            logits, _ = self.net(obs)
        return logits[0].float()

    @torch.no_grad()
    def decide(self, battle, team, logits=None):
        """logits: the policy's for this position, if already at hand (e.g. both sides in one batch)."""
        self.last = None
        logits = self._policy(battle, team) if logits is None else logits
        p = torch.softmax(logits / self.temp, -1)
        legal = logits > -1e8
        if int(legal.sum()) <= 1 or (p[0] >= self.think and not (self.think_anyway and self.rng.random() < self.think_anyway)):
            return int(torch.distributions.Categorical(p).sample())  # nothing to weigh: as the policy says
        self.searches += 1
        top = [int(a) for a in torch.topk(p[1:], min(self.k, int((p[1:] > 0).sum()))).indices + 1]
        cands = [0] + top
        if self.explore:
            plays = np.flatnonzero(legal[1:].cpu().numpy()) + 1
            plays = plays[~np.isin(plays, cands)]
            for _ in range(self.explore):
                if not len(plays):
                    break
                slots = (plays - 1) // crsim.rl.CELLS
                slot = self.rng.choice(np.unique(slots))  # a card first, then where: each card as likely
                a = int(self.rng.choice(plays[slots == slot]))
                cands.append(a)
                plays = plays[plays != a]
        other = crsim.Team.RED if team == crsim.Team.BLUE else crsim.Team.BLUE
        games, owner = [], []
        for ci, a in enumerate(cands):
            for _ in range(self.r):
                b = battle.clone()
                if a:
                    crsim.rl.apply_action(b, team, a)
                games.append(b)
                owner.append(ci)
        n = len(games)
        size = 2 * (self.k + 1 + self.explore) * self.r  # fixed, so one captured graph serves every search
        joint = self.opp is self.net  # both sides in one batch
        steps = max(1, int(round(self.h / (self.ticks * 0.05))))
        for step in range(steps):
            live = [g for g in games if not g.finished]
            if not live:
                break
            ours = [self._encode(g, team) for g in live]
            theirs = [self._encode(g, other) for g in live]
            if joint:
                acts, _ = self._runner(self.net, size, ours[0])(ours + theirs, self.rtemp)
                a_ours, a_theirs = acts[:len(live)], acts[len(live):]
            else:
                a_ours, _ = self._runner(self.net, size, ours[0])(ours, self.rtemp)
                a_theirs, _ = self._runner(self.opp, size, theirs[0])(theirs, self.rtemp)
            for g, mine, yours in zip(live, a_ours, a_theirs):
                if mine and step > 0:  # on the first step our move is the candidate already played
                    crsim.rl.apply_action(g, team, int(mine))
                if yours:
                    crsim.rl.apply_action(g, other, int(yours))
                g.steps(self.ticks)
        # score: the result if over, else the critic's estimate, from our side
        score = np.zeros(n)
        open_idx = [i for i, g in enumerate(games) if not g.finished]
        for i, g in enumerate(games):
            if g.finished:
                score[i] = 0.0 if g.winner is None else (1.0 if g.winner == team else -1.0)
        if open_idx:
            rows = [self._encode(games[i], team) for i in open_idx]
            _, v = self._runner(self.net, size, rows[0])(rows, self.rtemp)
            score[open_idx] = v
        owner = np.array(owner)
        means = [score[owner == ci].mean() for ci in range(len(cands))]
        best = cands[int(np.argmax(means))]
        self.last = (cands, means, best)
        return best
