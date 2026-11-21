"""Defence drills (the user's idea, 2026-11-21): start positions where red begins a hard push on blue's side.
Pushes like a giant with a mini pekka and a musketeer behind it, a back giant with support, or two lanes at once.
They come in single, double and triple elixir, with blue (the bot's side) behind or ahead on elixir.

The push is red's start script (BatchEnv.set_starts): its cards go down when and where the template says (as soon as
red has the elixir). In 60% of the drills red's net waits until the last of them is down, then plays on as in any
game (more troops, spells on the defenders); in the rest it plays throughout, the push's cards put down besides its
own moves (some are then left out). Either way the defence is net against net from a hard push. Both sides know each other's cycle as in a real
game: before the drill each side has really played its 8 cards (their troops taken off at once, the spells cast on
the empty river), so the play history the nets see says which 4 cards the other holds and which comes next.

Live the user's big pushes break the bot: it waits, then puts its troops in front of the tower instead of pulling
them to the middle. Its own games rarely make pushes like these. Nothing tells it how to defend: it learns what
pays from these positions.

    .venv\\Scripts\\python.exe python\\defence_starts.py [n]   (builds n, checks them and prints a few)
"""
import random
import sys
from collections import Counter

import crsim

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
LANES = (3.5, 14.5)
BRIDGE_Y = 17.5  # red's first row past the river (it may deploy at y > 17)
TICK_S = 0.05
HISTORY_S = 30.0  # the 8 plays of each side's cycle are made in this many s before the drill
EXCLUSIVE = 0.6  # share of the drills where red's net waits until the push is down (else it plays throughout)
# (weight, from s, to s) of battle time: x1, x2, overtime x2, overtime x3 (overtime: the next tower wins)
PHASES = ((0.25, 40.0, 115.0), (0.35, 121.0, 176.0), (0.15, 182.0, 236.0), (0.25, 242.0, 278.0))
# (weight, name): what red puts down, see _template
KINDS = ((0.15, "giant_pekka_musk"), (0.09, "giant_musk_archers"), (0.09, "giant_minions_pekka"),
         (0.08, "giant_knight_musk"), (0.11, "back_giant"), (0.11, "split_giant_pekka"), (0.08, "split_troops"),
         (0.09, "pekka_musk"), (0.08, "knight_musk_minions"), (0.06, "pekka_minions"), (0.06, "knight_archers"))


def _template(r, kind, lane, other):
    """[(s after the start it is meant to go down, card, x, y)] in arena tiles on red's side."""
    inner = lambda x: x + (2.0 if x < 9 else -2.0)  # beside red's princess tower, middle side  # noqa: E731
    bridge = lambda x: (x + r.uniform(-0.5, 0.5), BRIDGE_Y + r.uniform(0.0, 1.0))  # noqa: E731
    behind = lambda x, lo=1.5, hi=3.0: (x + r.uniform(-1.0, 1.0), BRIDGE_Y + r.uniform(lo, hi))  # noqa: E731
    gap = lambda lo=1.2, hi=2.5: r.uniform(lo, hi)  # noqa: E731
    t, out = r.uniform(0.2, 1.0), []

    def add(card, pos, after=0.0):
        nonlocal t
        t += after
        out.append((t, card, pos[0], pos[1]))

    if kind == "giant_pekka_musk":
        add("giant", bridge(lane)); add("mini_pekka", behind(lane), gap()); add("musketeer", behind(lane, 2.5, 4.0), gap())
    elif kind == "giant_musk_archers":
        add("giant", bridge(lane)); add("musketeer", behind(lane), gap()); add("archers", behind(lane, 2.0, 3.5), gap())
    elif kind == "giant_minions_pekka":
        add("giant", bridge(lane)); add("minions", behind(lane), gap()); add("mini_pekka", behind(lane), gap(0.8, 2.0))
    elif kind == "giant_knight_musk":
        add("giant", bridge(lane)); add("knight", behind(lane), gap(1.0, 2.0)); add("musketeer", behind(lane, 2.5, 4.0), gap())
    elif kind == "back_giant":  # behind the king, then support as it passes the princess tower
        add("giant", (9.0 + (-1.0 if lane < 9 else 1.0) + r.uniform(-0.5, 0.5), 31.5))
        add("musketeer", (lane + r.uniform(-0.5, 0.5), 27.5 + r.uniform(0.0, 1.0)), r.uniform(7.0, 10.0))
        add("mini_pekka", (inner(lane), 24.5 + r.uniform(0.0, 1.5)), gap(1.5, 3.0))
    elif kind == "split_giant_pekka":
        add("giant", bridge(lane)); add("mini_pekka", bridge(other), r.uniform(0.0, 1.2))
        add("musketeer", behind(lane), gap(1.5, 2.5))
    elif kind == "split_troops":
        add("knight", bridge(lane)); add("archers", behind(lane), r.uniform(0.8, 1.5))
        add("mini_pekka", bridge(other), r.uniform(0.5, 1.5)); add("musketeer", behind(other), gap(1.0, 2.0))
    elif kind == "pekka_musk":
        add("mini_pekka", bridge(lane)); add("musketeer", behind(lane), gap(1.0, 2.0))
    elif kind == "knight_musk_minions":
        add("knight", bridge(lane)); add("musketeer", behind(lane), gap(1.0, 2.0)); add("minions", behind(lane), gap(1.0, 2.0))
    elif kind == "pekka_minions":
        add("mini_pekka", bridge(lane)); add("minions", behind(lane), gap(0.8, 1.8))
    elif kind == "knight_archers":
        add("knight", bridge(lane)); add("archers", behind(lane), gap(0.8, 1.5))
    else:
        raise ValueError(kind)
    return out


def _rate(t_s):
    """Elixir per second at this battle time (a full bar of 10 takes 28 s at x1)."""
    return (1.0 if t_s < 120.0 else (2.0 if t_s < 240.0 else 3.0)) / 2.8


def _timed(plays, start_s, elixir, costs):
    """The plays at the first moments red can pay for them (its elixir from `elixir` at start_s, capped at 10)."""
    out, e, now = [], elixir, start_s
    for t, card, x, y in plays:
        want = start_s + t
        while now < want or e + 1e-9 < costs[card]:
            e = min(10.0, e + _rate(now) * TICK_S)
            now += TICK_S
        e -= costs[card]
        out.append((now + TICK_S, card, x, y))  # a tick of slack: the env plays it once battle time reaches it
    return out


def _clear(b):
    """Takes every troop off (the history's units, at once after they show up)."""
    for e in b.entities:
        if not e.is_building and e.hp > 0:
            b.twin_remove(e.id)


def _history(b, r, data, plans, until_s, tick_ms):
    """Each team really plays its 8 cards (plans[team]: in this order, each in hand by then), alternating at random,
    from until_s - HISTORY_S to ~4 s before until_s: their troops go down in the back corner and are taken off as
    they show up, the spells land on the empty river. Then the last 4 a team played are its queue in that order and
    the other 4 its hand - what the other side can tell from the history it sees."""
    order = [BLUE] * len(plans[BLUE]) + [RED] * len(plans[RED])
    r.shuffle(order)
    left = {BLUE: list(plans[BLUE]), RED: list(plans[RED])}
    t0, spacing = until_s - HISTORY_S, (HISTORY_S - 4.5) / len(order)  # (each play is followed by 1.1 s of steps)
    times = [t0 + k * spacing + r.uniform(0.0, 0.4) for k in range(len(order))]
    for t_s, team in zip(times, order):
        card = left[team].pop(0)
        target = int(round(t_s * 1000 / tick_ms))
        b.steps(max(0, target - b.time_ms // tick_ms))
        _clear(b)
        p = b.player(team)
        slot = next((k for k in range(4) if p.hand[k] == card), -1)
        if slot < 0:  # not back in hand yet (its slot still refilling): wait for it
            for _ in range(60):
                b.steps(1)
                _clear(b)
                slot = next((k for k in range(4) if b.player(team).hand[k] == card), -1)
                if slot >= 0:
                    break
        if slot < 0:
            raise RuntimeError(f"history: {card} never came to hand")
        b.set_elixir(team, 10.0)
        if data.card(card).is_spell:
            pos = (9.0, 16.0)
        else:
            pos = (r.choice((0.5, 17.5)), 0.5 if team == BLUE else 31.5)
        if not b.play(team, slot, crsim.Vec2(*pos)):
            raise RuntimeError(f"history: {card} could not be played at {pos}")
        b.steps(int(round(1100 / tick_ms)))  # its troops show up after the 1 s blueprint
        _clear(b)
    target = int(round(until_s * 1000 / tick_ms))
    b.steps(max(0, target - b.time_ms // tick_ms))
    _clear(b)
    b.steps(1)  # (taken off on the next step)
    for _ in range(80):  # the last cards played: their slots refilled, so each hand is whole when the drill starts
        if all(c for team in (BLUE, RED) for c in b.player(team).hand):
            break
        b.steps(1)


def generate(data, deck, n, seed=31, log=print, check=False):
    """n drill positions: (battles, scripts, lanes, kinds). battles are built on `data` (it must outlive the envs that
    start from them); scripts[k] is red's push as BatchEnv.set_starts takes it, (1, until_ms, [(t_ms, card, x, y)],
    exclusive) - exclusive False: red's net plays on meanwhile; lanes[k] the x of blue's princess tower it goes for first,
    kinds[k] the template's name. check: verify each side's hand and queue after its history."""
    r = random.Random(seed)
    deck = list(deck)
    costs = {cid: data.card(cid).elixir for cid in deck}
    tick_ms = data.rules.tick_ms
    battles, scripts, lanes, kinds = [], [], [], []
    skipped = Counter()
    for _ in range(n):
        lo, hi = r.choices([(a, b) for _, a, b in PHASES], weights=[w for w, _, _ in PHASES])[0]
        start_s = r.uniform(lo, hi)
        overtime = start_s >= data.rules.regulation_ms / 1000.0
        b = crsim.Battle(data, deck, deck, r.getrandbits(48))
        # towers down: before the battle runs, so the crowns are counted and overtime is reached with them even
        down = {BLUE: None, RED: None}
        if overtime:
            if r.random() < 0.2:
                down = {BLUE: r.randrange(2), RED: r.randrange(2)}  # 1 : 1, or the battle would be over
        else:
            for team in (BLUE, RED):
                if r.random() < 0.12:
                    down[team] = r.randrange(2)
            if down[BLUE] is not None and down[RED] is not None and r.random() < 0.5:
                down[RED] = None
        for team, side in down.items():
            if side is not None:
                b.twin_set_hp(b.princess_tower(team, side), 0)
        # the push, and the cycles: red holds the push's cards when it starts
        lane = r.choice(LANES)
        other = LANES[1] if lane == LANES[0] else LANES[0]
        kind = r.choices([k for _, k in KINDS], weights=[w for w, _ in KINDS])[0]
        push = _template(r, kind, lane, other)
        cards = [c for _, c, _, _ in push]
        plans, want = {}, {}
        for team in (BLUE, RED):
            rest = [c for c in deck if c not in cards] if team == RED else list(deck)
            r.shuffle(rest)
            hand = (cards + rest)[:4] if team == RED else rest[:4]
            queue = [c for c in deck if c not in hand]
            r.shuffle(queue)
            first = list(hand)
            r.shuffle(first)
            plans[team] = first + queue  # the hand's cards first, then the queue's in the order they will come back
            want[team] = (sorted(hand), queue)
            b.set_hand(team, hand, queue[0])  # (the hand's cards come back after the queue's: what the 8 plays need)
        b.steps(max(0, int(round((start_s - HISTORY_S) * 1000 / tick_ms)) - b.time_ms // tick_ms))
        try:
            _history(b, r, data, plans, start_s, tick_ms)
        except RuntimeError as e:
            skipped[str(e).split(":")[0]] += 1
            continue
        if b.finished:
            skipped["finished"] += 1
            continue
        if check:
            for team in (BLUE, RED):
                p = b.player(team)
                got = (sorted(c for c in p.hand if c), list(p.queue))
                if got != want[team]:
                    raise AssertionError(f"{team}: hand/queue {got}, wanted {want[team]}")
        # tower HP and elixir
        for team in (BLUE, RED):
            for side in (0, 1):
                t = b.find(b.princess_tower(team, side))
                if t is not None and r.random() < 0.45:
                    b.twin_set_hp(t.id, int(t.max_hp * r.uniform(0.3, 0.95)))
            k = b.find(b.king_tower(team))
            if k is not None and down[team] is not None and r.random() < 0.3:
                b.twin_set_hp(k.id, int(k.max_hp * r.uniform(0.6, 0.95)))
        red_e = r.uniform(6.5, 10.0)
        u = r.random()  # red's elixir minus blue's: blue ahead, about even, behind
        d = r.uniform(-3.0, -0.5) if u < 0.3 else (r.uniform(-0.5, 0.5) if u < 0.55 else r.uniform(0.5, 5.0))
        b.set_elixir(RED, red_e)
        b.set_elixir(BLUE, min(10.0, max(0.0, red_e - d)))
        plays = _timed(push, b.time_ms / 1000.0, red_e, costs)
        # Exclusive: red's net waits until the push's last card is down (0.5 s after it is due), then plays on - the
        # whole push. Not: it plays throughout, the cards only put down besides its own moves (a third or more are
        # then left out: it spends the elixir on its own plays) - net against net from the push's first card.
        exclusive = r.random() < EXCLUSIVE
        until = plays[-1][0] + (0.5 if exclusive else 3.0)
        battles.append(b)
        scripts.append((1, int(round(until * 1000)),
                        [(int(round(t * 1000)), c, float(x), float(y)) for t, c, x, y in plays], exclusive))
        lanes.append(lane)
        kinds.append(kind)
    log(f"defence drills: {len(battles)} positions ({describe(battles, kinds, data)})"
        + (f"; left out: {dict(skipped)}" if skipped else ""))
    return battles, scripts, lanes, kinds


def describe(battles, kinds, data):
    c = Counter()
    for b in battles:
        c["x1"] += b.time_ms < 120_000
        c["x2"] += 120_000 <= b.time_ms < 240_000
        c["x3"] += b.time_ms >= 240_000
        c["overtime"] += b.time_ms >= data.rules.regulation_ms
        gap = b.player(RED).elixir - b.player(BLUE).elixir
        c["blue 2+ elixir behind"] += gap >= 2.0
        c["blue 1+ ahead"] += gap <= -1.0
        c["a princess tower down"] += any(b.find(b.princess_tower(t, s)) is None for t in (BLUE, RED) for s in (0, 1))
    n = max(1, len(battles))
    k = Counter(kinds)
    return (", ".join(f"{key} {v / n:.0%}" for key, v in c.items()) + "; " +
            ", ".join(f"{key} {v / n:.0%}" for key, v in k.most_common()))


if __name__ == "__main__":
    import time
    import crsim_env as ce
    d = ce.load_data()
    dk = ce.parse_deck(",".join(ce.DEFAULT_DECK), d)
    t = time.time()
    bs, sc, ln, kd = generate(d, dk, int(sys.argv[1]) if len(sys.argv) > 1 else 200, check=True)
    print(f"{time.time() - t:.1f} s")
    for b, s, lane, kind in list(zip(bs, sc, ln, kd))[:5]:
        hist = {team: [d.card_ids[p.card_index] for p in b.player(team).plays] for team in (BLUE, RED)}
        print(f"  {b.time_ms / 1000:.1f}s {kind} lane {lane}: red {b.player(RED).elixir:.1f} blue "
              f"{b.player(BLUE).elixir:.1f}; red hand {sorted(c for c in b.player(RED).hand)} queue "
              f"{list(b.player(RED).queue)}, red's plays {hist[RED]}; units {len([e for e in b.entities if not e.is_building])}; "
              f"push until {s[1] / 1000:.1f}s: " + ", ".join(f"{c}@{t / 1000:.1f}s({x:.1f},{y:.1f})" for t, c, x, y, in s[2]))
