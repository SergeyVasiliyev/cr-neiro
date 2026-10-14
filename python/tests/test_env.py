"""Environment tests:  .venv\\Scripts\\python.exe -m pytest python\\tests -q"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crsim_env as ce  # noqa: E402
from crsim_env import crsim  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return ce.load_data()


def test_gymnasium_api(data):
    from gymnasium.utils.env_checker import check_env

    env = ce.CrEnv(data, opponent="random", seed=3)
    check_env(env, skip_render_check=True)


def test_own_frame_is_symmetric(data):
    # Same deck on both sides: at t=0 both players must see exactly the same board.
    env = ce.VecEnv(data, [(ce.DEFAULT_DECK, ce.DEFAULT_DECK)], num_envs=1, bots=("noop", "noop"))
    blue, red = env.obs(0), env.obs(1)
    for key in ("entities", "entity_token", "entity_mask"):
        np.testing.assert_array_equal(blue[key], red[key])
    np.testing.assert_array_equal(blue["global"][:, :16], red["global"][:, :16])
    # The legal deploy area is the same in both frames (hand cards may differ).
    np.testing.assert_array_equal(blue["action_mask"][:, 1:577], red["action_mask"][:, 1:577])


def test_action_places_card_on_the_chosen_tile(data):
    env = ce.VecEnv(data, [(["knight"] * 8, ["knight"] * 8)], num_envs=1, bots=("noop", "noop"))
    col, row = 3, 10
    action = 1 + 0 * ce.CELLS + row * ce.GRID_W + col
    assert env.obs(0)["action_mask"][0, action] and env.obs(1)["action_mask"][0, action]
    env.step(np.array([[action, action]]))
    for _ in range(2):  # the cards sit as blueprints for the deploy delay first
        env.step(np.array([[0, 0]]))
    troops = [e for e in env.battle(0).entities if not e.is_building]
    assert len(troops) == 2
    by_team = {e.team: e.pos for e in troops}
    # Units may have moved for 0.5 s; check they spawned on the right side and lane.
    blue, red = by_team[crsim.Team.BLUE], by_team[crsim.Team.RED]
    assert abs(blue.x - (col + 0.5)) < 1.0 and blue.y < 15
    assert abs(red.x - (18 - col - 0.5)) < 1.0 and red.y > 17


def test_masks_forbid_enemy_half_and_unaffordable_cards(data):
    env = ce.VecEnv(data, [(["giant"] * 8, ["giant"] * 8)], num_envs=1, bots=("noop", "noop"))
    mask = env.obs(0)["action_mask"][0]
    cells = mask[1:1 + ce.CELLS].reshape(32, 18)
    assert cells[:15].any() and not cells[15:].any()  # own half only (rows 0..14)
    env.step(np.array([[1 + 10 * 18 + 9, ce.BOT_ACTION]]))  # giant costs 5 of the 5.x elixir
    assert env.obs(0)["action_mask"][0, 1:].sum() == 0  # nothing affordable right after


def test_spell_mask_covers_the_whole_arena(data):
    env = ce.VecEnv(data, [(["fireball"] * 8, ["fireball"] * 8)], num_envs=1, bots=("noop", "noop"))
    cells = env.obs(0)["action_mask"][0, 1:1 + ce.CELLS].reshape(32, 18)
    assert cells.all()  # spells can be cast anywhere, enemy half and river included


def test_cycle_and_enemy_history(data):
    env = ce.VecEnv(data, [(ce.DEFAULT_DECK, ce.DEFAULT_DECK)], num_envs=1, bots=("noop", "noop"))
    first = crsim.rl.FIRST_CARD_TOKEN
    own_cycle = env.obs(0)["hand"][0]
    assert sorted(own_cycle - first) == sorted(data.card(c).index for c in ce.DEFAULT_DECK)  # all 8 cards
    assert (env.obs(0)["enemy_history"][0] == 0).all()
    # Red plays its first hand card on its own half; Blue now sees it at the top of the history.
    played = env.battle(0).player(crsim.Team.RED).hand[0]
    mask = env.obs(1)["action_mask"][0]
    action = int(np.flatnonzero(mask[1:1 + ce.CELLS])[0]) + 1
    env.step(np.array([[0, action]]))
    assert (env.obs(0)["enemy_history"][0] == 0).all()  # still a blueprint Blue cannot see
    env.step(np.array([[0, 0]]))
    env.step(np.array([[0, 0]]))
    history = env.obs(0)["enemy_history"][0]
    assert history[0] == first + data.card(played).index and (history[1:] == 0).all()
    assert 0 <= env.obs(0)["enemy_history_age"][0, 0] < 0.1  # revealed under 1 s ago, in units of 10 s


def test_blueprints_are_hidden_from_the_opponent(data):
    env = ce.VecEnv(data, [(["knight"] * 8, ["knight"] * 8)], num_envs=1, bots=("noop", "noop"))
    elixir_seen_by_red = env.obs(1)["global"][0, 5]
    action = 1 + 10 * ce.GRID_W + 9
    env.step(np.array([[action, 0]]))  # Blue plays a knight: 0.5 s later it is still a blueprint
    pending_col = crsim.rl.ENTITY_FEATURES - 1
    blue_sees = env.obs(0)["entities"][0][env.obs(0)["entity_mask"][0] == 1]
    red_sees = env.obs(1)["entities"][0][env.obs(1)["entity_mask"][0] == 1]
    assert (blue_sees[:, pending_col] == 1).sum() == 1
    assert (red_sees[:, pending_col] == 1).sum() == 0
    # Red's estimate of Blue's elixir has not dropped: as far as Red knows, nothing was played.
    assert env.obs(1)["global"][0, 5] >= elixir_seen_by_red


def test_full_episodes_with_random_masked_actions(data):
    rng = np.random.default_rng(0)
    env = ce.VecEnv(data, [(ce.DEFAULT_DECK, ce.DEFAULT_DECK)], num_envs=8, bots=("noop", "heuristic"),
                    reward=ce.make_reward(win=1.0))
    finished, total = 0, np.zeros(8)
    for _ in range(800):
        mask = env.obs(0)["action_mask"]
        acts = np.full((8, 2), ce.BOT_ACTION, dtype=np.int32)
        for i in range(8):
            legal = np.flatnonzero(mask[i])
            acts[i, 0] = rng.choice(legal) if rng.random() < 0.3 else 0
        rew, done = env.step(acts)
        total += rew[:, 0]
        for i in np.flatnonzero(done):
            assert env.winner[i] in (0, 1, 2)
            assert total[i] in (-1.0, 0.0, 1.0)  # only the terminal win/loss reward is enabled
            total[i] = 0
            finished += 1
    assert finished >= 8


def test_reward_shaping_is_zero_sum(data):
    env = ce.VecEnv(data, [(ce.DEFAULT_DECK, ce.DEFAULT_DECK)], num_envs=4, bots=("heuristic", "heuristic"),
                    reward=ce.make_reward(win=1.0, crown=0.2, tower_damage=0.5))
    for _ in range(400):
        rew, _ = env.step(np.full((4, 2), ce.BOT_ACTION, dtype=np.int32))
        np.testing.assert_allclose(rew[:, 0], -rew[:, 1], atol=1e-6)


def test_policy_network_forward(data):
    torch = pytest.importorskip("torch")
    from model import PolicyNet, obs_to_torch

    env = ce.VecEnv(data, [(ce.DEFAULT_DECK, ce.DEFAULT_DECK)], num_envs=3, bots=("noop", "noop"))
    net = PolicyNet(crsim.rl.vocab_size(data))
    logits, value = net(obs_to_torch(env.obs(0), "cpu"))
    assert logits.shape == (3, ce.NUM_ACTIONS) and value.shape == (3,)
    illegal = ~torch.as_tensor(env.obs(0)["action_mask"]).bool()
    assert (logits[illegal] < -1e8).all()
    action, _ = net.act(obs_to_torch(env.obs(0), "cpu"))
    assert all(env.obs(0)["action_mask"][i, a] for i, a in enumerate(action.tolist()))


def test_elixir_trade_reward(data):
    """Trades are zero-sum, and a troop's worth goes partly with damage, the rest when it dies."""
    deck = ce.DEFAULT_DECK
    env = ce.VecEnv(data, [(deck, deck)], num_envs=8, bots=("heuristic", "heuristic"), decision_ticks=6, seed=5,
                    reward=ce.make_reward(win=0.0, elixir_trade=1.0, trade_floor=0.25))
    acts = np.full((8, 2), ce.BOT_ACTION, dtype=np.int32)
    moved = 0.0
    for _ in range(400):
        r, _ = env.step(acts)
        assert np.allclose(r[:, 0], -r[:, 1])
        moved += float(np.abs(r).sum())
    assert moved > 1.0
