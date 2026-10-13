"""Tracker tests on made-up detections:  .venv\\Scripts\\python.exe -m pytest python\\tests -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.tracker import Tracker  # noqa: E402
from perception.units import Detection  # noqa: E402

DT = 0.1  # 10 frames per second


def det(cls, team, x, y):
    return Detection(cls, team, 0.9, (0, 0, 0, 0), (x, y))


def ids(tracker, t, cls):
    return {s["id"]: s for s in tracker.state(t) if s["cls"] == cls}


def test_walking_unit_keeps_its_id_and_speed():
    tr = Tracker()
    for k in range(20):
        tr.update(k * DT, [det("knight", "blue", 3.5, 8 + k * DT)])  # 1 tile/s up the lane
    (s,) = tr.state(19 * DT)
    assert s["id"] == 1 and s["seen"]
    assert abs(tr.tracks[0].vy - 1.0) < 0.05


def test_unit_lost_in_the_open_is_predicted_then_dropped():
    tr = Tracker(keep_unseen=1.5)
    for k in range(10):
        tr.update(k * DT, [det("knight", "blue", 3.5, 8 + k * DT)])
    t = 0.9
    for _ in range(10):  # 1 s unseen: still there, moved on by its speed
        t += DT
        tr.update(t, [])
    (s,) = tr.state(t)
    assert not s["seen"] and abs(s["y"] - (8.9 + 1.0)) < 0.2
    for _ in range(10):  # 2 s unseen: gone
        t += DT
        tr.update(t, [])
    assert tr.state(t) == []


def test_unit_behind_a_giant_is_remembered_longer():
    tr = Tracker(keep_unseen=1.5, keep_hidden=4.0)
    t = 0.0
    for _ in range(10):  # a mini pekka walking right behind (above on screen) a giant
        tr.update(t, [det("giant", "red", 14.5, 20.0), det("mini_pekka", "red", 14.5, 21.0)])
        t += DT
    for _ in range(30):  # 3 s: only the giant is visible
        tr.update(t, [det("giant", "red", 14.5, 20.0)])
        t += DT
    assert len(ids(tr, t, "mini_pekka")) == 1
    tr.update(t, [det("giant", "red", 14.5, 20.0), det("mini_pekka", "red", 14.5, 21.1)])
    assert list(ids(tr, t, "mini_pekka")) == [2]  # the same unit, not a new one


def test_two_units_passing_keep_their_ids():
    tr = Tracker()
    for k in range(30):  # two blue archers: one walks right, the other stands
        x = 6 + k * 0.1
        tr.update(k * DT, [det("archer", "blue", x, 10.0), det("archer", "blue", 8.0, 11.5)])
    walking = [s for s in tr.state(29 * DT) if s["x"] > 8.5]
    assert len(walking) == 1 and walking[0]["id"] == 1


def test_single_frame_false_detection_is_not_reported():
    tr = Tracker()
    tr.update(0.0, [det("musketeer", "red", 9, 25)])
    assert tr.state(0.0) == []  # needs two sightings
    tr.update(DT, [])
    tr.update(4 * DT, [])
    assert tr.tracks == []


def test_drop_in_is_not_walking():
    # A giant drops in from 2.3 tiles above its spot in 0.3 s, then stands in its deploy timer.
    tr = Tracker()
    for k in range(12):
        t = k * DT
        y = 10.0 + max(0.0, 2.3 * (1 - t / 0.3))
        tr.update(t, [det("giant", "blue", 9.0, y)])
    (track,) = tr.tracks
    assert abs(track.landing_pos()[1] - 10.0) < 0.05
    assert abs(track.vy) < 0.05  # the fall did not become a speed


def test_arrows_in_pieces_are_one_volley():
    from perception.units import merge_spells
    pieces = [det("arrows", "blue", 9.0, 12.0), det("arrows", "blue", 9.5, 15.0), det("arrows", "blue", 8.6, 17.5),
              det("arrows", "red", 4.0, 20.0), det("knight", "blue", 3.5, 8.0)]
    merged = merge_spells(pieces)
    blue = [d for d in merged if d.cls == "arrows" and d.team == "blue"]
    assert len(blue) == 1 and abs(blue[0].tile[1] - 14.83) < 0.01
    assert len(merged) == 3  # one blue volley, one red, the knight
