"""Training environments on top of the crsim C++ core.

Two entry points:
  * VecEnv   - N matches stepped in parallel in C++ (OpenMP). Use this for training.
  * CrEnv    - a standard single-agent gymnasium.Env (one match vs a bot or a policy),
               compatible with sb3-contrib MaskablePPO via action_masks().

Observations are dicts of numpy arrays, always from the acting player's own perspective
(their side at the bottom), so one policy can play either side:
  global        float32 [GLOBAL_FEATURES]               time, elixir, crowns, tower HP, hand costs
  entities      float32 [MAX_ENTITIES, ENTITY_FEATURES]  towers, troops and spells in flight (own/enemy, x, y, hp, ...)
  entity_token  int32   [MAX_ENTITIES]                   0 pad, 1 princess, 2 king, 3+ card index
  entity_mask   uint8   [MAX_ENTITIES]                   1 = slot holds a real entity
  hand          int32   [HAND_TOKENS]                    own cycle: 4 cards in hand, then the queue in order
  enemy_history int32   [HISTORY]                        opponent's last plays, newest first (0 = none yet)
  enemy_history_age float32 [HISTORY]                    seconds since each of those plays / 10
  action_mask   uint8   [NUM_ACTIONS]                    1 = legal action
Actions: 0 = wait, 1 + slot * 576 + row * 18 + col = play hand card `slot` on tile (col, row).
"""
import os
import random
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import crsim  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
OBS_KEYS = ("global", "entities", "entity_token", "entity_mask", "hand", "enemy_history", "enemy_history_age",
            "action_mask")
# The starter deck every new player gets.
DEFAULT_DECK = ["knight", "archers", "minions", "arrows", "fireball", "giant", "mini_pekka", "musketeer"]
BOT_KINDS = ("noop", "random", "heuristic", "pusher")

NUM_ACTIONS = crsim.rl.NUM_ACTIONS
CELLS = crsim.rl.CELLS
GRID_W = crsim.rl.GRID_W
BOT_ACTION = crsim.rl.BOT_ACTION


def load_data(path=DATA_DIR):
    return crsim.GameData.load(path)


def supported_cards(data):
    """Card ids whose mechanics the engine fully simulates."""
    return sorted(cid for cid, card in data.cards.items() if card.fully_supported)


def parse_deck(text, data):
    deck = [c.strip() for c in text.split(",") if c.strip()]
    if len(deck) != 8:
        raise ValueError(f"a deck needs 8 cards, got {len(deck)}: {deck}")
    unknown = [c for c in deck if c not in data.cards]
    if unknown:
        raise ValueError(f"unknown cards {unknown}; available: {', '.join(sorted(data.cards))}")
    return deck


def random_decks(data, count, seed=0, supported_only=True):
    pool = supported_cards(data) if supported_only else sorted(data.cards)
    rng = random.Random(seed)
    return [rng.sample(pool, 8) for _ in range(count)]


def make_reward(win=1.0, crown=0.0, tower_damage=0.0, elixir_leak=0.0, elixir_trade=0.0, trade_floor=0.25,
                spell_full_worth=False, elixir_bank=0.0, bank_cap=9.0):
    r = crsim.RewardConfig()
    r.win, r.crown, r.tower_damage, r.elixir_leak = win, crown, tower_damage, elixir_leak
    r.elixir_trade, r.trade_floor = elixir_trade, trade_floor
    r.spell_full_worth, r.elixir_bank, r.bank_cap = spell_full_worth, elixir_bank, bank_cap
    return r


def observation_space(data):
    vocab = crsim.rl.vocab_size(data)
    e, f = crsim.rl.MAX_ENTITIES, crsim.rl.ENTITY_FEATURES
    return spaces.Dict({
        "global": spaces.Box(0.0, 10.0, (crsim.rl.GLOBAL_FEATURES,), np.float32),
        "entities": spaces.Box(0.0, 10.0, (e, f), np.float32),
        "entity_token": spaces.Box(0, vocab - 1, (e,), np.int32),
        "entity_mask": spaces.Box(0, 1, (e,), np.uint8),
        "hand": spaces.Box(0, vocab - 1, (crsim.rl.HAND_TOKENS,), np.int32),
        "enemy_history": spaces.Box(0, vocab - 1, (crsim.rl.HISTORY,), np.int32),
        "enemy_history_age": spaces.Box(0.0, 100.0, (crsim.rl.HISTORY,), np.float32),
        "action_mask": spaces.Box(0, 1, (NUM_ACTIONS,), np.uint8),
    })


class VecEnv:
    """N parallel matches. Both sides take actions; BOT_ACTION (-1) lets the built-in bot play.

    matchups: list of (blue_deck, red_deck); one is drawn at random for every new episode.
    Buffers returned by obs()/rewards() are views into C++ memory that the next step()
    overwrites: copy them if you keep them.
    """

    def __init__(self, data, matchups, num_envs=64, bots=("heuristic", "heuristic"), decision_ticks=10,
                 seed=0, reward=None, observe_enemy_elixir=True, num_threads=0, auto_reset=True, key_spots=False,
                 delay_ticks=(0, 0), obs_noise=(0.0, 0.0)):
        self.data = data
        self.matchups = [(list(b), list(r)) for b, r in matchups]
        self.env = crsim.BatchEnv(
            data, self.matchups, num_envs=num_envs, decision_ticks=decision_ticks, seed=seed,
            blue_bot=bots[0], red_bot=bots[1], observe_enemy_elixir=observe_enemy_elixir,
            reward=reward or make_reward(), num_threads=num_threads, auto_reset=auto_reset, key_spots=key_spots,
            delay_ticks=tuple(delay_ticks), obs_noise=tuple(obs_noise))
        self.key_spots = key_spots
        self.num_envs = num_envs
        self.decision_ticks = decision_ticks

    def obs(self, side):
        """Observation dict for side 0 (blue) or 1 (red), each array with a leading env axis."""
        return {k: getattr(self.env, k)[:, side] for k in OBS_KEYS}

    def step(self, actions):
        """actions: int array (num_envs, 2). Returns (rewards (N, 2), dones (N,))."""
        self.env.step(np.ascontiguousarray(actions, dtype=np.int32))
        return self.env.reward, self.env.done

    def reset(self):
        self.env.reset()

    def set_bot(self, env, side, kind):
        """The scripted bot ("heuristic", "pusher", ...) of one side of one env, from its next episode on."""
        self.env.set_bot(env, side, kind)

    @property
    def reward_parts(self):
        """(N, 2, 3) rewards of the last step in parts: result, shaping, style (reward is their sum)."""
        return self.env.reward_parts

    # Results of episodes that ended in the last step (where done = 1).
    @property
    def winner(self):
        return self.env.winner

    @property
    def crowns(self):
        return self.env.crowns

    @property
    def episode_steps(self):
        return self.env.episode_steps

    def battle(self, i):
        return self.env.battle(i)


class CrEnv(gym.Env):
    """One match, the agent plays `side` against a built-in bot or any policy callable.

    opponent: "noop" | "random" | "heuristic", or a callable(obs_dict) -> action for the other side
              (e.g. a frozen copy of the agent for self-play).
    """

    metadata = {"render_modes": []}

    def __init__(self, data=None, deck=None, opponent_deck=None, opponent="heuristic", side=0,
                 decision_ticks=10, reward=None, observe_enemy_elixir=True, seed=0):
        super().__init__()
        self.data = data or load_data()
        deck = list(deck or DEFAULT_DECK)
        opponent_deck = list(opponent_deck or DEFAULT_DECK)
        self.side = side
        self.opponent = opponent
        bot = opponent if isinstance(opponent, str) else "noop"
        matchup = (deck, opponent_deck) if side == 0 else (opponent_deck, deck)
        bots = ("noop", bot) if side == 0 else (bot, "noop")
        self._make_vec = lambda s: VecEnv(
            self.data, [matchup], num_envs=1, bots=bots, decision_ticks=decision_ticks, seed=s, reward=reward,
            observe_enemy_elixir=observe_enemy_elixir, num_threads=1, auto_reset=False)
        self._vec = self._make_vec(seed)
        self.observation_space = observation_space(self.data)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def _obs(self, side):
        return {k: v[0].copy() for k, v in self._vec.obs(side).items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._vec = self._make_vec(seed)  # same seed -> same deal and bot behaviour
        else:
            self._vec.reset()
        return self._obs(self.side), {}

    def step(self, action):
        actions = np.full((1, 2), BOT_ACTION, dtype=np.int32)
        actions[0, self.side] = int(action)
        if callable(self.opponent):
            actions[0, 1 - self.side] = int(self.opponent(self._obs(1 - self.side)))
        rewards, dones = self._vec.step(actions)
        terminated = bool(dones[0])
        info = {}
        if terminated:
            w = int(self._vec.winner[0])
            info["result"] = "draw" if w == 2 else ("win" if w == self.side else "loss")
            info["crowns"] = tuple(int(c) for c in self._vec.crowns[0])
        return self._obs(self.side), float(rewards[0, self.side]), terminated, False, info

    def action_masks(self):
        """For sb3-contrib MaskablePPO."""
        return self._vec.obs(self.side)["action_mask"][0].astype(bool)

    @property
    def battle(self):
        return self._vec.battle(0)
