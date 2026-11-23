"""Policy/value network for crsim observations.

Entities (towers + troops) go through a small transformer, then get scattered onto the
18x32 deploy grid; a CNN over that grid scores every tile, and each hand card's query
vector picks a tile: logits[slot, cell] = query[slot] . cell_features[cell]. This keeps the
2305-way action space spatially aware (e.g. "place it right in front of that Giant").
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crsim

GRID_W, GRID_H = crsim.rl.GRID_W, crsim.rl.GRID_H
HAND_SIZE = crsim.rl.HAND_SIZE
TORCH_DTYPES = {
    "global": torch.float32,
    "entities": torch.float32,
    "entity_token": torch.long,
    "entity_mask": torch.bool,
    "hand": torch.long,
    "enemy_history": torch.long,
    "enemy_history_age": torch.float32,
    "action_mask": torch.bool,
    "enemy_deck": torch.long,
    "enemy_deck_info": torch.float32,
}


def obs_to_torch(obs, device):
    """numpy obs dict (leading batch axis) -> torch tensors on device (copies)."""
    return {k: torch.as_tensor(np.ascontiguousarray(v)).to(device=device, dtype=TORCH_DTYPES[k], non_blocking=True)
            for k, v in obs.items()}


class PolicyNet(nn.Module):
    """n_values > 1: one value head per reward part (result, shaping, style), value is then (B, n_values)."""

    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4, grid_channels=32, trunk=256, n_values=1,
                 entity_features=None, enemy_deck=False):
        """entity_features: how many per-unit features it reads (default: all the encoder gives). Newer features
        are appended, so a net trained on fewer reads the first ones and ignores the rest.
        enemy_deck: read the opponent's deck as known (obs enemy_deck / enemy_deck_info: the cards seen, the pool
        deck they fit, the cycle); nets saved before it existed do not."""
        super().__init__()
        entity_features = entity_features or crsim.rl.ENTITY_FEATURES
        self.config = dict(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           grid_channels=grid_channels, trunk=trunk, n_values=n_values,
                           entity_features=entity_features, enemy_deck=enemy_deck)
        d, c, w = d_model, grid_channels, trunk
        self.n_values = n_values
        self.token_emb = nn.Embedding(vocab_size, d)
        self.entity_in = nn.Linear(entity_features, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.global_in = nn.Sequential(nn.Linear(crsim.rl.GLOBAL_FEATURES, d), nn.ReLU())
        self.hand_slot_emb = nn.Parameter(torch.zeros(crsim.rl.HAND_TOKENS, d))
        self.hand_in = nn.Sequential(nn.Linear(crsim.rl.HAND_TOKENS * d, d), nn.ReLU())
        # Opponent's recent plays (cycle tracking): card + how long ago, in order.
        self.history_age = nn.Linear(1, d)
        self.history_in = nn.Sequential(nn.Linear(crsim.rl.HISTORY * d, d), nn.ReLU())
        # Opponent's deck as known (a set: the order of its cards means nothing): card + what is known of it,
        # pooled (mean and max over the cards there).
        self.enemy_deck = enemy_deck
        if enemy_deck:
            self.deck_info = nn.Linear(crsim.rl.ENEMY_DECK_FEATURES, d)
            self.deck_card = nn.Sequential(nn.Linear(d, d), nn.ReLU())
            self.deck_in = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear((6 if enemy_deck else 5) * d, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU())

        # Spatial branch: entity features painted on the grid + coordinate channels.
        self.to_grid = nn.Linear(d, c)
        self.cnn = nn.Sequential(
            nn.Conv2d(c + 2 + 1, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
        )
        self.grid_film = nn.Linear(w, c)  # trunk context modulates the grid features
        self.slot_query = nn.Sequential(nn.Linear(w + d, 128), nn.ReLU(), nn.Linear(128, c))
        self.noop_head = nn.Linear(w, 1)
        if n_values == 1:
            self.value_head = nn.Sequential(nn.Linear(w, 128), nn.ReLU(), nn.Linear(128, 1))
        else:  # a critic of its own per reward part, on a wider layer of their own
            self.value_head = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, w // 2), nn.ReLU(),
                                            nn.Linear(w // 2, n_values))

        ys, xs = torch.meshgrid(torch.linspace(0, 1, GRID_H), torch.linspace(0, 1, GRID_W), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None], persistent=False)

        # Start close to "uniform over legal tiles, wait about half of the time" so early
        # exploration covers both placements and saving elixir.
        nn.init.normal_(self.slot_query[-1].weight, std=0.01)
        nn.init.zeros_(self.slot_query[-1].bias)
        nn.init.zeros_(self.noop_head.weight)
        nn.init.constant_(self.noop_head.bias, 6.0)

    def forward(self, obs):
        ent_mask = obs["entity_mask"]
        b, n = ent_mask.shape
        x = self.token_emb(obs["entity_token"]) + self.entity_in(obs["entities"][..., :self.entity_in.in_features])
        x = self.encoder(x, src_key_padding_mask=~ent_mask)
        m = ent_mask.unsqueeze(-1).float()
        pooled_mean = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        pooled_max = x.masked_fill(~ent_mask.unsqueeze(-1), -1e4).max(1).values

        hand = self.token_emb(obs["hand"]) + self.hand_slot_emb
        g = self.global_in(obs["global"])
        history = self.token_emb(obs["enemy_history"]) + self.history_age(obs["enemy_history_age"].unsqueeze(-1))
        history = history * (obs["enemy_history"] > 0).unsqueeze(-1)
        parts = [pooled_mean, pooled_max, g, self.hand_in(hand.flatten(1)), self.history_in(history.flatten(1))]
        if self.enemy_deck:
            there = (obs["enemy_deck"] > 0).unsqueeze(-1)
            cards = self.deck_card(self.token_emb(obs["enemy_deck"]) + self.deck_info(obs["enemy_deck_info"]))
            n_cards = there.float().sum(1).clamp(min=1.0)
            deck_mean = (cards * there).sum(1) / n_cards
            deck_max = cards.masked_fill(~there, 0.0).max(1).values  # (ReLU outputs: 0 is their floor)
            parts.append(self.deck_in(torch.cat([deck_mean, deck_max], dim=-1)))
        h = self.trunk(torch.cat(parts, dim=-1))

        # Scatter entity features onto their tiles (own frame; features 1/2 are x/H-normalized y).
        col = (obs["entities"][..., 1] * GRID_W).long().clamp(0, GRID_W - 1)
        row = (obs["entities"][..., 2] * GRID_H).long().clamp(0, GRID_H - 1)
        cell = row * GRID_W + col
        feats = self.to_grid(x) * m
        grid = torch.zeros(b, GRID_H * GRID_W, feats.shape[-1], device=feats.device, dtype=feats.dtype)
        grid.scatter_add_(1, cell.unsqueeze(-1).expand_as(feats), feats)
        grid = grid.transpose(1, 2).reshape(b, -1, GRID_H, GRID_W)

        # Deployable tiles for this player (any card) as an extra channel.
        legal = obs["action_mask"][:, 1:].reshape(b, HAND_SIZE, GRID_H, GRID_W).any(1, keepdim=True).float()
        grid = torch.cat([grid, self.coords.expand(b, -1, -1, -1), legal], dim=1)
        grid = self.cnn(grid) * (1 + self.grid_film(h))[:, :, None, None]

        q = self.slot_query(torch.cat([h.unsqueeze(1).expand(-1, HAND_SIZE, -1), hand[:, :HAND_SIZE]], dim=-1))
        place = torch.einsum("bsc,bchw->bshw", q, grid).reshape(b, -1)
        # float32 outside autocast: -1e9 masking and the softmax over 2305 actions need the range.
        logits = torch.cat([self.noop_head(h), place], dim=-1).float()
        logits = logits.masked_fill(~obs["action_mask"], -1e9)
        value = self.value_head(h).float()
        return logits, (value.squeeze(-1) if self.n_values == 1 else value)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        logits, value = self(obs)
        if greedy:
            return logits.argmax(-1), value
        return torch.distributions.Categorical(logits=logits).sample(), value


def deck_pool_of(ckpt):
    """The decks the net's opponents had in training (its run's --decks pool file, else its --deck mirror), for the
    enemy deck block outside training: crsim.rl.encode(..., deck_pool=...). None if the checkpoint does not say."""
    import json
    import os
    args = (ckpt or {}).get("args") or {}
    path = args.get("decks")
    if path:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = path if os.path.isabs(path) else os.path.join(root, path)
        if os.path.exists(path):
            return [list(cards) for cards in json.load(open(path, encoding="utf-8"))["decks"].values()]
    deck = args.get("deck")
    if deck:
        return [[c.strip() for c in deck.split(",") if c.strip()]]
    return None


def save_checkpoint(path, net, optimizer=None, **meta):
    torch.save({"model": net.state_dict(), "config": net.config,
                "optimizer": optimizer.state_dict() if optimizer else None, **meta}, path)


def load_policy(path, device="cpu", entity_features=None, enemy_deck=False):
    """entity_features: widen an older net to read this many per-unit features, the new ones starting at zero
    weight (it plays exactly as before until trained).
    enemy_deck: give a net saved without the enemy deck block one; its way into the trunk starts at zero weight, so
    it too plays exactly as before until trained."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    trained = ckpt["model"]["entity_in.weight"].shape[1]  # nets saved before the setting existed
    config["entity_features"] = max(trained, entity_features or 0)
    had_deck = bool(config.get("enemy_deck", False))
    config["enemy_deck"] = had_deck or enemy_deck
    net = PolicyNet(**config).to(device)
    state = dict(ckpt["model"])
    if config["entity_features"] > trained:
        w = torch.zeros_like(net.entity_in.weight)
        w[:, :trained] = state["entity_in.weight"]
        state["entity_in.weight"] = w
    if config["enemy_deck"] and not had_deck:
        old = state["trunk.0.weight"]
        w = torch.zeros_like(net.trunk[0].weight)
        w[:, :old.shape[1]] = old
        state["trunk.0.weight"] = w
        for k, v in net.state_dict().items():
            if k.startswith(("deck_info.", "deck_card.", "deck_in.")):
                state[k] = v
    net.load_state_dict(state)
    net.eval()
    return net, ckpt


class PolicyPlayer:
    """Wraps a trained network as a callable(obs_dict) -> action for a single match."""

    def __init__(self, net, device="cpu", greedy=False, temperature=None):
        """temperature: sample at it (< 1 sharper than trained) instead of greedy / plain sampling."""
        self.net, self.device, self.greedy, self.temperature = net, device, greedy, temperature

    def __call__(self, obs):
        batch = obs_to_torch({k: v[None] for k, v in obs.items()}, self.device)
        if self.temperature is not None:
            with torch.no_grad():
                logits = self.net(batch)[0].float() / max(self.temperature, 1e-3)
            return int(torch.distributions.Categorical(logits=logits).sample()[0])
        with torch.no_grad():
            action, _ = self.net.act(batch, greedy=self.greedy)
        return int(action[0])
