"""Labels frames of two-sided recordings by itself, for training the unit detector.

    .venv\\Scripts\\python.exe tools\\auto_label.py --check            (how good: against frames labelled by hand)
    .venv\\Scripts\\python.exe tools\\auto_label.py --write [--every 1.0] [--review 40] [--matches video13 ...]
    .venv\\Scripts\\python.exe tools\\auto_label.py --score --matches video15 [--cache raw/perception/cache_v5]

A battle recorded from both sides (raw/videos/<name>.mp4 + <name>_red.mp4, read into raw/perception/cache)
tells more than the detector sees in either recording:
  - both hands are on screen, so every play of both players is known: card, side, moment, spot. The
    simulator twin, fed those plays, knows which unit is which wherever it walks - a mini pekka the
    detector calls a knight for a moment, or gives the other side's colour, is still that mini pekka;
  - a unit hidden behind a giant in one recording stands in front of it in the other (the view is turned).
The twin follows the game from the detections of both recordings together (each unit's kind and side by
the more confident view), and each frame of each recording is labelled from it: the detector's boxes, with
the kind and side of the twin unit each box is on; boxes on nothing the twin has are left out (false
detections) unless sure. Frames with something left unexplained - a sure detection the twin can't
account for, a unit the other recording sees clearly that this one misses in the open, a spell nobody
cast - are not labelled automatically: they are left out, or (--review N) a spread-out N of them go to
scripts\\label.bat for a person, pre-labelled. Frames already waiting there that turn out to have no
problem are taken over by the automatic labels.

Automatic labels go to raw/labels/auto_labels.json and raw/labels/auto_frames (build_detector_dataset.py
--auto takes them in); frames for a person to raw/labels/labels.json and raw/labels/frames, as usual.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import twin  # noqa: E402
from compare_red_truth import own_plays_timed  # noqa: E402
from crsim_env import load_data  # noqa: E402
from label_units import FRAMES, load_labels, save_labels  # noqa: E402
from live.session import LiveMatch  # noqa: E402
from perception.layout import Layout  # noqa: E402
from perception.units import SPELL_HITS, SPELLS_IN_FLIGHT, Detection, UnitDetector, merge_spells  # noqa: E402

W, H = 18.0, 32.0
LAYOUT = Layout(scale=1.0)
AUTO_FRAMES = os.path.join(ROOT, "raw", "labels", "auto_frames")
AUTO_LABELS = os.path.join(ROOT, "raw", "labels", "auto_labels.json")

LIVE_CONF = 0.35  # the detector's threshold in play
SURE = 0.6        # a detection this confident is a unit for sure
WEAK = 0.2         # weaker detections are labelled only when the twin and the other recording back them
PAIR_DIST = 1.0   # tiles: the same unit in the two recordings
GATE = 1.2        # tiles: a detection on a twin unit
FAR_GATE = 2.2    # ... of the same kind and side, when nothing closer is left
KIND_COST, SIDE_COST = 0.6, 0.4  # a detection of another kind / side is a worse match, not impossible
DROP_IN = 0.5     # s after appearing: the unit still falls in from above (drawn higher than it stands)


def flip_side(team):
    return {"blue": "red", "red": "blue"}.get(team, team)


def view_dets(frame, flip=False, min_conf=0.15):
    """A cache frame's detections; flip=True puts them into the other recording's tiles and sides."""
    out = []
    for d in frame["det"]:
        if d[2] < min_conf:
            continue
        tile = UnitDetector.box_to_tile(d[0], d[3:7], LAYOUT)
        if flip:
            out.append(Detection(d[0], flip_side(d[1]), d[2], tuple(d[3:7]), (W - tile[0], H - tile[1])))
        else:
            out.append(Detection(d[0], d[1], d[2], tuple(d[3:7]), tile))
    return out


def greedy(pairs):
    pairs.sort()
    used_a, used_b, out = set(), set(), {}
    for _, a, b in pairs:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        out[a] = b
    return out


def merge_views(A, B):
    """Troop detections of both recordings at one moment, in A's tiles (B already flipped): a unit seen in
    both is one detection, of the kind and side with more confidence over both views, and its position
    halfway (each view's error is mirrored in the other). Spells come from A alone."""
    troops_a = [d for d in A if not d.is_spell]
    troops_b = [d for d in B if not d.is_spell]
    pairs = [(math.dist(a.tile, b.tile), i, j) for i, a in enumerate(troops_a) for j, b in enumerate(troops_b)
             if math.dist(a.tile, b.tile) <= PAIR_DIST and (a.cls == "minion") == (b.cls == "minion")]
    paired = greedy(pairs)
    out = [d for d in A if d.is_spell]
    for i, a in enumerate(troops_a):
        if i not in paired:
            out.append(a)
            continue
        b = troops_b[paired[i]]
        votes = defaultdict(float)
        for d in (a, b):
            votes[(d.cls, d.team)] += d.conf
        cls, team = max(votes, key=votes.get)
        agree = [d.conf for d in (a, b) if (d.cls, d.team) == (cls, team)]
        conf = 1 - math.prod(1 - c for c in agree) if len(agree) == 2 else agree[0]
        tile = ((a.tile[0] + b.tile[0]) / 2, (a.tile[1] + b.tile[1]) / 2)
        out.append(Detection(cls, team, conf, a.box, tile))
    used_b = set(paired.values())
    out += [b for j, b in enumerate(troops_b) if j not in used_b]
    return out


def follow(name, data, both=True, cache=None):
    """Runs the twin over a two-sided recording with both players' real plays.
    Returns the frames of both recordings, per frame index the twin's units (blue recording's tiles), the
    plays [(game s, card, pos, side)], the battle start and the tower positions."""
    cache = cache or os.path.join(ROOT, "raw", "perception", "cache")
    blue = twin.load_cache(os.path.join(cache, f"{name}.json"))["frames"]
    red = twin.load_cache(os.path.join(cache, f"{name}_red.json"))["frames"]
    if len(blue) != len(red):
        raise SystemExit(f"{name}: the two recordings have {len(blue)} and {len(red)} frames (not recorded together?)")
    blue_plays = own_plays_timed(blue, data)
    red_plays = [(t, c, None if p is None else (W - p[0], H - p[1])) for t, c, p in own_plays_timed(red, data)]
    m = LiveMatch(data, red_truth=red_plays, blue_truth=blue_plays)
    states, born, from_screen = [], {}, set()
    for i, fr in enumerate(blue):
        A, B = view_dets(fr), view_dets(red[i], flip=True)
        dets = merge_views(A, B) if both else A
        dets = merge_spells([d for d in dets if d.conf >= LIVE_CONF])
        before = {e.id for e in m.battle.entities} if m.battle else set()
        n_corr = len(m.corrections)
        m.update(fr["t"], twin.seen_from_cache(fr), dets, None)
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0:
            states.append(None)
            continue
        snap = twin.snapshot(m.battle)
        added = [c for c in m.corrections[n_corr:] if c[1] == "late play"]
        for u in snap["units"]:
            if u["id"] not in before and u["id"] not in born:
                born[u["id"]] = g
                # a play the twin made up from a unit no known play explained: its kind is the detector's guess
                # (a unit of a known play brought back keeps that play's kind: trusted)
                if any(c[2] == u["cls"] and c[3] == u["team"] and math.dist(c[4]["pos"], (u["x"], u["y"])) < 0.6
                       for c in added):
                    from_screen.add(u["id"])
            u["age"] = g - born.get(u["id"], g)
            u["trusted"] = u["id"] not in from_screen
        snap["finished"] = m.battle.finished
        states.append(snap)
    towers = []
    b = m.battle
    for team in (crsim.Team.BLUE, crsim.Team.RED):
        towers.append(b.find(b.king_tower(team)))
        towers += [b.find(b.princess_tower(team, s)) for s in (0, 1)]
    # (x, y, half width): the king's tower is 4 tiles wide, a princess's 3
    tower_pos = [(e.pos.x, e.pos.y, 2.0 if k % 3 == 0 else 1.5) for k, e in enumerate(towers) if e is not None]
    plays = [(t, c, p, "blue") for t, c, p in blue_plays] + [(t, c, p, "red") for t, c, p in red_plays]
    start, _ = twin.battle_start(blue)
    return blue, red, states, plays, start, tower_pos


def in_view(units, flip):
    """Twin units in one recording's tiles and sides."""
    if not flip:
        return units
    return [dict(u, x=W - u["x"], y=H - u["y"], team=flip_side(u["team"])) for u in units]


def match(units, dets):
    """{unit index: detection index}: each twin unit to the detection on it (troops only)."""
    pairs = []
    for i, u in enumerate(units):
        for j, d in enumerate(dets):
            if d.is_spell or (d.cls == "minion") != (u["cls"] == "minion"):
                continue
            dx, dy = d.tile[0] - u["x"], d.tile[1] - u["y"]
            if u["age"] < DROP_IN:  # falling in: drawn above (further up the screen) where it will stand
                dy = max(0.0, dy - 2.5) if dy > 0 else dy
            dist = math.hypot(dx, dy)
            if dist > GATE:
                continue
            cost = dist + KIND_COST * (d.cls != u["cls"]) + SIDE_COST * (d.team != u["team"]) - 0.3 * d.conf
            pairs.append((cost, i, j))
    out = greedy(pairs)
    # The twin's position can be further off (a unit it walked a bit differently, a minion of three): what is
    # left of the same kind and side is matched within FAR_GATE.
    used = set(out.values())
    pairs = [(math.dist((u["x"], u["y"]), d.tile), i, j) for i, u in enumerate(units) if i not in out
             for j, d in enumerate(dets) if j not in used and not d.is_spell and d.cls == u["cls"] and d.team == u["team"]
             and math.dist((u["x"], u["y"]), d.tile) <= FAR_GATE]
    out.update(greedy(pairs))
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def hidden(u, dets, towers):
    """Could this unit be out of sight in this recording: behind a giant or a tower (above it on the screen:
    towers are drawn tall), or under the top bar? Beside or in front of a tower it is in plain view - left
    unlabelled there, it would teach the detector not to see units around towers."""
    px, py = LAYOUT.tile_to_px(u["x"], u["y"])
    for d in dets:
        if d.cls == "giant" and d.conf >= LIVE_CONF and d.box[0] - 10 <= px <= d.box[2] + 10 \
                and d.box[1] - 10 <= py <= d.box[3] + 30:
            return True
    if any(abs(u["x"] - x) < half + 0.2 and -0.5 < u["y"] - y < 3.0 for x, y, half in towers):
        return True
    return u["y"] > 31.0


def spell_side(cls, g, plays, data):
    """Sides that have this spell in the air at game time g (blue recording's sides)."""
    sides = set()
    for t, card, pos, side in plays:
        if card != cls or pos is None:
            continue
        flight = math.dist(twin.KING[side], pos) / data.card(card).spell.projectile_speed
        if t <= g <= t + 1.0 + flight + 0.6:
            sides.add(side)
    return sides


def label_frame(units, dets, other_dets, towers, g, plays, data, flip):
    """(objects, problems) for one frame of one recording. units: twin units in this recording's tiles;
    dets / other_dets: this recording's detections and the other's (flipped into this one's tiles)."""
    here = match(units, dets)
    there = match(units, other_dets)
    objects, problems = [], []
    taken = set()
    for i, j in here.items():
        u, d = units[i], dets[j]
        same = d.cls == u["cls"] and d.team == u["team"]
        seen_there = i in there and other_dets[there[i]].conf >= SURE
        if not u["trusted"] and not same and not (seen_there and (other_dets[there[i]].cls, other_dets[there[i]].team)
                                                  == (u["cls"], u["team"])):
            problems.append(f"{u['team']} {u['cls']}? detector says {d.team} {d.cls}")
            taken.add(j)
            continue
        if d.conf >= LIVE_CONF or seen_there or (same and d.conf >= WEAK):
            objects.append({"cls": u["cls"], "team": u["team"], "box": [round(v, 1) for v in d.box],
                            "src": "det" if same else f"twin (detector: {d.team} {d.cls})"})
            taken.add(j)
    boxes = [o["box"] for o in objects]
    for j, d in enumerate(dets):
        if j in taken or d.is_spell or d.conf < LIVE_CONF:
            continue
        if any(iou(d.box, b) > 0.45 for b in boxes):
            continue  # a second box on a unit already labelled
        backed = any(math.dist(o.tile, d.tile) < PAIR_DIST and o.conf >= LIVE_CONF and not o.is_spell
                     for o in other_dets)
        if d.conf >= SURE or backed:
            problems.append(f"{d.team} {d.cls} {d.conf:.2f} at ({d.tile[0]:.1f},{d.tile[1]:.1f}): the twin has nothing there")
    for i, u in enumerate(units):
        if i in here or u.get("deploying") and u["age"] < 0.2:
            continue
        if i in there and other_dets[there[i]].conf >= SURE and not hidden(u, dets, towers):
            problems.append(f"{u['team']} {u['cls']} at ({u['x']:.1f},{u['y']:.1f}): clear in the other recording, missed here")
    for d in merge_spells([d for d in dets if d.is_spell and d.conf >= LIVE_CONF]):
        if d.cls in SPELL_HITS:
            objects.append({"cls": d.cls, "team": "blue", "box": [round(v, 1) for v in d.box], "src": "det"})
            continue
        sides = {flip_side(s) if flip else s for s in spell_side(d.cls, g, plays, data)}
        if len(sides) == 1 or d.team in sides:
            team = d.team if d.team in sides else sides.pop()
            objects.append({"cls": d.cls, "team": team, "box": [round(v, 1) for v in d.box],
                            "src": "det" if team == d.team else f"play (detector: {d.team})"})
        else:
            problems.append(f"{d.team} {d.cls} {d.conf:.2f}: nobody has this spell in the air")
    return objects, problems


def detector_objects(dets):
    """What the bare detector says at the live threshold, as label objects."""
    out = []
    for d in merge_spells([d for d in dets if d.conf >= LIVE_CONF]):
        out.append({"cls": d.cls, "team": "blue" if d.cls in SPELL_HITS else d.team, "box": list(d.box)})
    return out


def compare(objects, truth):
    """(right, wrong kind or side, missed, extra) of objects against hand labels."""
    pairs = [(-iou(o["box"], t["box"]), i, j) for i, o in enumerate(objects) for j, t in enumerate(truth)
             if iou(o["box"], t["box"]) >= 0.4]
    m = greedy(pairs)
    right = wrong = 0
    for i, j in m.items():
        o, t = objects[i], truth[j]
        if o["cls"] == t["cls"] and (o["team"] == t["team"] or o["cls"] in SPELL_HITS):
            right += 1
        else:
            wrong += 1
    return right, wrong, len(truth) - len(m), len(objects) - len(m)


def run(name, data, both=True, cache=None):
    blue, red, states, plays, start, towers = follow(name, data, both, cache)
    towers_red = [(W - x, H - y, half) for x, y, half in towers]
    out = {}
    for i, st in enumerate(states):
        if st is None or st["finished"]:
            continue
        A, B = view_dets(blue[i]), view_dets(red[i])
        Bf = [Detection(d.cls, flip_side(d.team), d.conf, d.box, (W - d.tile[0], H - d.tile[1])) for d in B]
        Af = [Detection(d.cls, flip_side(d.team), d.conf, d.box, (W - d.tile[0], H - d.tile[1])) for d in A]
        units = st["units"]
        out[(name, i)] = (blue[i]["t"], st["g"], A) + label_frame(in_view(units, False), A, Bf, towers, st["g"], plays, data, False)
        out[(f"{name}_red", i)] = (red[i]["t"], st["g"], B) + label_frame(in_view(units, True), B, Af, towers_red, st["g"], plays, data, True)
    return out


def check(matches, data, both):
    doc = load_labels()
    by_rec = defaultdict(dict)
    for fname, entry in doc["frames"].items():
        if entry.get("status") != "labeled":
            continue
        rec, t = fname[:-4].rsplit("_", 1)
        if rec.split("_")[0] in matches:
            by_rec[rec][float(t)] = entry["objects"]
    totals = {"auto": Counter(), "auto (frames not sent to a person)": Counter(), "detector": Counter()}
    flagged = n = 0
    for name in matches:
        if not by_rec.get(name) and not by_rec.get(f"{name}_red"):
            continue
        res = run(name, data, both)
        for (rec, i), (t, g, dets, objects, problems) in res.items():
            truth = next((v for u, v in by_rec.get(rec, {}).items() if abs(u - t) < 0.07), None)
            if truth is None:
                continue
            n += 1
            a = compare(objects, truth)
            d = compare(detector_objects(dets), truth)
            totals["auto"].update(dict(zip(("right", "wrong", "missed", "extra"), a)))
            totals["detector"].update(dict(zip(("right", "wrong", "missed", "extra"), d)))
            if problems:
                flagged += 1
            else:
                totals["auto (frames not sent to a person)"].update(dict(zip(("right", "wrong", "missed", "extra"), a)))
            if a[1] or a[2] or a[3]:
                print(f"  {rec} {t:6.1f}s  auto: {a[0]} right, {a[1]} wrong, {a[2]} missed, {a[3]} extra"
                      + (f"   [to a person: {problems[0]}]" if problems else ""))
    print(f"\n{n} frames labelled by hand; {flagged} of them auto_label would send to a person")
    for k, c in totals.items():
        tot = c["right"] + c["wrong"] + c["missed"]
        if tot:
            print(f"  {k:36s} right {c['right'] / tot:5.1%}  wrong kind/side {c['wrong'] / tot:5.1%}  "
                  f"missed {c['missed'] / tot:5.1%}  extra {c['extra']}  (of {tot} hand-labelled objects)")


def to_person(objects, dets):
    """Pre-labels for a person: the automatic labels plus the detector's other boxes."""
    objs = [dict({k: o[k] for k in ("cls", "team", "box")}, auto=True, conf=0.99) for o in objects]
    for d in dets:
        if d.conf >= WEAK and not d.is_spell and not any(iou(d.box, o["box"]) > 0.45 for o in objects):
            objs.append({"cls": d.cls, "team": d.team, "box": [round(v, 1) for v in d.box], "auto": True,
                         "conf": round(d.conf, 2)})
    return objs


def read_frame(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    if ok and img.shape[1] != 1080:
        img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_LINEAR)
    return img if ok else None


def write(matches, data, every, review, both):
    """Automatic labels every `every` s of each recording; up to `review` frames with problems (spread out)
    for a person; frames already waiting for a person (pre-labelled) that turn out to have no problems are
    taken over by the automatic labels."""
    doc = load_labels()
    manual = defaultdict(list)
    for fname, entry in doc["frames"].items():
        rec, t = fname[:-4].rsplit("_", 1)
        manual[rec].append(float(t))
    auto = {"frames": {}}
    if os.path.exists(AUTO_LABELS):
        with open(AUTO_LABELS, encoding="utf-8") as f:
            auto = json.load(f)
    os.makedirs(AUTO_FRAMES, exist_ok=True)
    stats = Counter()
    flagged = []  # (recording, t, objects, problems, dets)
    for name in matches:
        res = run(name, data, both)
        by_rec = defaultdict(list)
        for (rec, i), (t, g, dets, objects, problems) in res.items():
            by_rec[rec].append((t, objects, problems, dets))
        for rec, items in by_rec.items():
            items.sort(key=lambda it: it[0])
            cap = cv2.VideoCapture(os.path.join(ROOT, "raw", "videos", f"{rec}.mp4"))
            # frames waiting for a person
            for fname, entry in list(doc["frames"].items()):
                if entry.get("status") != "prelabeled" or fname[:-4].rsplit("_", 1)[0] != rec:
                    continue
                t0 = float(fname[:-4].rsplit("_", 1)[1])
                t, objects, problems, dets = min(items, key=lambda it: abs(it[0] - t0))
                if abs(t - t0) > 0.07:
                    continue
                if problems:
                    entry["note"] = "; ".join(problems)
                    stats["still for a person"] += 1
                    continue
                os.replace(os.path.join(FRAMES, fname), os.path.join(AUTO_FRAMES, fname))
                auto["frames"][fname] = {"objects": [{k: o[k] for k in ("cls", "team", "box")} for o in objects],
                                         "fixed": sum(1 for o in objects if o["src"] != "det")}
                del doc["frames"][fname]
                stats["taken over from the queue"] += 1
            # automatic frames from an earlier run: labelled again (the rules may have changed)
            done = []
            for fname in [f for f in auto["frames"] if f[:-4].rsplit("_", 1)[0] == rec]:
                t0 = float(fname[:-4].rsplit("_", 1)[1])
                t, objects, problems, dets = min(items, key=lambda it: abs(it[0] - t0))
                if abs(t - t0) > 0.07:
                    continue
                if problems:
                    del auto["frames"][fname]
                    stats["earlier automatic frames now unsure, left out"] += 1
                    continue
                auto["frames"][fname] = {"objects": [{k: o[k] for k in ("cls", "team", "box")} for o in objects],
                                         "fixed": sum(1 for o in objects if o["src"] != "det")}
                done.append(t0)
            last = -1e9
            for t, objects, problems, dets in items:
                if not objects and not problems:
                    continue
                if t - last < every or any(abs(t - u) < 1.0 for u in manual[rec]) or any(abs(t - u) < 0.5 for u in done):
                    continue
                last = t
                if problems:
                    flagged.append((rec, t, objects, problems, dets))
                    continue
                fname = f"{rec}_{t:06.1f}.jpg"
                img = read_frame(cap, t)
                if img is None:
                    continue
                cv2.imwrite(os.path.join(AUTO_FRAMES, fname), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                auto["frames"][fname] = {"objects": [{k: o[k] for k in ("cls", "team", "box")} for o in objects],
                                         "fixed": sum(1 for o in objects if o["src"] != "det")}
                stats["automatic"] += 1
                stats["boxes the twin corrected"] += auto["frames"][fname]["fixed"]
        print(f"{name}: {dict(stats)}", flush=True)
    # The frames with problems are hard ones (crowds, a unit missed): a spread-out sample for a person.
    picked = []
    for item in sorted(flagged, key=lambda it: -len(it[3])):
        if len(picked) >= review:
            break
        if all(p[0] != item[0] or abs(p[1] - item[1]) > 4.0 for p in picked):
            picked.append(item)
    caps = {}
    for rec, t, objects, problems, dets in picked:
        cap = caps.setdefault(rec, cv2.VideoCapture(os.path.join(ROOT, "raw", "videos", f"{rec}.mp4")))
        img = read_frame(cap, t)
        if img is None:
            continue
        fname = f"{rec}_{t:06.1f}.jpg"
        cv2.imwrite(os.path.join(FRAMES, fname), img)
        doc["frames"][fname] = {"status": "prelabeled", "objects": to_person(objects, dets), "note": "; ".join(problems)}
    stats["new for a person"] = len(picked)
    stats["with problems, left out"] = len(flagged) - len(picked)
    save_labels(doc)
    with open(AUTO_LABELS + ".tmp", "w", encoding="utf-8") as f:
        json.dump(auto, f, ensure_ascii=False)
    os.replace(AUTO_LABELS + ".tmp", AUTO_LABELS)
    print(f"{dict(stats)}\nautomatic: {AUTO_LABELS} ({len(auto['frames'])} frames); for a person: scripts\\label.bat")


def score(matches, data, cache, both):
    """How often the detector (the one that read the cache) is wrong, judged by the automatic labels: per
    second of each recording, its boxes at the live threshold against the labels. For comparing detectors
    on a battle neither was trained on (the labels lean on the detector's own boxes: a relative measure)."""
    tot = Counter()
    for name in matches:
        res = run(name, data, both, cache)
        c = Counter()
        last = {}
        for (rec, i), (t, g, dets, objects, problems) in sorted(res.items(), key=lambda kv: kv[1][0]):
            if (not objects and not problems) or t - last.get(rec, -1e9) < 1.0:
                continue
            last[rec] = t
            c["frames"] += 1
            if problems:
                c["frames with problems"] += 1
                continue
            r, w, m, x = compare(detector_objects(dets), objects)
            c.update({"objects": r + w + m, "wrong kind/side": w, "missed": m, "extra": x})
        tot.update(c)
        print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    n = max(1, tot["objects"])
    print(f"  all: {tot['frames']} frames, {tot['frames with problems'] / max(1, tot['frames']):.1%} with problems; "
          f"of {tot['objects']} objects: wrong kind/side {tot['wrong kind/side'] / n:.1%}, missed {tot['missed'] / n:.1%}, "
          f"extra {tot['extra'] / n:.1%}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matches", nargs="+", default=["video8", "video9", "video10", "video11", "video12"])
    p.add_argument("--check", action="store_true", help="compare with the frames labelled by hand")
    p.add_argument("--write", action="store_true")
    p.add_argument("--every", type=float, default=1.0, help="s between frames taken from one recording")
    p.add_argument("--review", type=int, default=0, help="frames with problems to put in the queue for a person")
    p.add_argument("--one-view", action="store_true", help="twin from the blue recording's detections only")
    p.add_argument("--score", action="store_true", help="how often the detector is wrong (see score())")
    p.add_argument("--cache", default=None, help="perception caches to use (default raw/perception/cache)")
    args = p.parse_args()
    data = load_data()
    if args.check:
        check(args.matches, data, not args.one_view)
    if args.write:
        write(args.matches, data, args.every, args.review, not args.one_view)
    if args.score:
        score(args.matches, data, args.cache, not args.one_view)


if __name__ == "__main__":
    main()
