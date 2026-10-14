"""Live twin (python/live/session.py) on made-up screens:  .venv\\Scripts\\python.exe -m pytest python\\tests -q"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crsim  # noqa: E402
import twin  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import LiveMatch  # noqa: E402
from perception.units import Detection  # noqa: E402

DT = 0.1
DATA = load_data()


def screen(t, start):
    """What the screen shows at video time t of a battle that started at video time start."""
    left = 180 - (t - start)
    frame = {"hand": ["?"] * 4, "next": None, "elixir": None, "timer": [math.ceil(left), False], "mult": 1,
             "towers": {}}
    return twin.seen_from_cache(frame)


def red_knight(x, y):
    tile = (x, y)
    return Detection("knight", "red", 0.9, (0, 0, 0, 0), tile)


def test_clock_drift_does_not_reset_the_match():
    # Frames get lost in a recording: after a while the timer runs a second ahead of the video clock. The
    # twin follows the timer; the match and what it found (an opponent's play) stay.
    m = LiveMatch(DATA)
    t = 0.0
    while t < 30.0:
        dets = [red_knight(4.0, 20.0 - 0.5 * max(0.0, t - 12.0))] if t >= 10.0 else []
        m.update(t, screen(t, 0.0), dets)
        t += DT
    assert [p[1] for p in m.red_plays] == ["knight"]
    while t < 50.0:
        m.update(t, screen(t, -1.2), [])  # the timer now says the battle started 1.2 s earlier
        t += DT
    assert [p[1] for p in m.red_plays] == ["knight"]
    assert abs(m.start - (-1.2)) < 0.6


def test_unit_the_simulator_just_killed_is_not_added_back():
    # The simulator kills a red knight; the game still shows it dying for a few frames. The twin must not
    # bring it back (or take it for a new play) for those frames.
    m = LiveMatch(DATA)
    t = 0.0
    while t < 2.0:
        m.update(t, screen(t, 0.0), [])
        t += DT
    play = m.add_play("red", "knight", m.now, (9.0, 20.0), "seen")
    while t < 5.0:  # seen standing there, matched to the simulated knight
        m.update(t, screen(t, 0.0), [red_knight(9.0, 20.0)])
        t += DT
    knight = m._entity((play.id, 0))
    assert knight is not None
    m.battle.twin_set_hp(knight.id, 0)  # killed in the simulator
    end = t + 0.8
    while t < end:  # ... still on screen, dying
        m.update(t, screen(t, 0.0), [red_knight(9.0, 20.0)])
        t += DT
    assert not [c for c in m.corrections if c[1] in ("added", "late play")]
    assert len([p for p in m.plays if p.on]) == 1


def test_every_unit_belongs_to_a_play_and_a_late_play_is_replayed_from_its_moment():
    # A red knight noticed only 2 s after it appeared: the twin goes back and plays it from when it was
    # put down, so the simulated knight has walked as far as the real one, not started late.
    m = LiveMatch(DATA)
    t = 0.0
    while t < 3.0:
        m.update(t, screen(t, 0.0), [])
        t += DT
    play = m.add_play("red", "knight", m.now - 2.0, (4.0, 22.0), "seen")
    assert m.rewinds >= 1
    knight = m._entity((play.id, 0))
    assert knight is not None and knight.deploy_ms_left == 0
    for e in m.battle.entities:
        if not e.is_building:
            assert m._key(e) is not None


def test_phantom_taken_back_undoes_what_it_did():
    # A red mini pekka play that turns out to be a phantom: taken back, the simulator replays without it -
    # no mini pekka, and the tower it was hitting has its HP back.
    m = LiveMatch(DATA)
    t = 0.0
    while t < 2.0:
        m.update(t, screen(t, 0.0), [])
        t += DT
    play = m.add_play("red", "mini_pekka", m.now, (3.5, 9.0), "seen")  # right next to our left tower
    while t < 12.0:
        m.update(t, screen(t, 0.0), [])
        t += DT
    tower = m.battle.find(m.battle.princess_tower(crsim.Team.BLUE, 0))
    hit = tower.hp
    m.take_back(play, "test")
    tower = m.battle.find(m.battle.princess_tower(crsim.Team.BLUE, 0))
    assert tower.hp > hit and tower.hp == tower.max_hp
    assert not [e for e in m.battle.entities if not e.is_building and e.team == crsim.Team.RED]
