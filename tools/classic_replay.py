"""The random-bot battles of the Classic Decks pool (tools/random_battles.py) replayed in the simulator.

Both bots log every play: the card's picture, the tile, the video second. Fed those plays, the simulator twin knows
every unit of the battle frame by frame, the ones nobody played too: the bats a Night Witch makes, the brawler out
of a Goblin Cage, golemites, lava pups. tools/classic_labels.py names the detector's training boxes with it.

    .venv\\Scripts\\python.exe tools\\classic_replay.py c008                   start, plays, how the deploy clocks fit
    .venv\\Scripts\\python.exe tools\\classic_replay.py c008 --sheet 10 30 50  frames with the simulated units drawn

Time: the battle's start in video seconds comes from the timer on screen (every change of the seconds shown).
A play reaches the game a moment after the bot's tap (the video second in the log): TROOP_LATENCY for troops and
buildings, TAP_LATENCY for spells; the simulator adds the blueprint second itself. Positions: troops at the tile their deploy clock showed, spells where they were aimed.
"""
import argparse
import json
import math
import os
import statistics
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from perception.layout import Layout  # noqa: E402
from perception.reader import ScreenReader  # noqa: E402

OUT = os.path.join(ROOT, "raw", "classic")
TAP_LATENCY = 0.5        # s from a bot's tap to a spell's blueprint second starting (tools/classic_check.py)
TROOP_LATENCY = 0.32     # s from a tap to a troop's blueprint second: deploy clocks show 1.32 s after the tap (median of
                         # 161 plays in c008-c010, both views, every card alike); the troop walks off ~1 s later, while
                         # its clock stays on screen 0.67 s longer than the deploy time
FLYING_OFFSET = 1.15     # tiles: a flying unit is drawn this far above the ground it flies over (perception/units.py)
REGULATION_S, OVERTIME_S = 180, 120


def mirror(x, y):
    """A tile in blue's view <-> the same spot in red's view (each player sees his own side at the bottom)."""
    return 18.0 - x, 32.0 - y


def team_name(team):
    return "blue" if team == crsim.Team.BLUE else "red"


# --- time -------------------------------------------------------------------------------------------------
def read_timer(video):
    """[(video second, seconds shown, overtime)] for every frame where the timer reads."""
    reader = ScreenReader()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    out, k = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        s, overtime = reader.numbers.read_timer(cv2.resize(f, (1080, 1920)))
        if s is not None:
            out.append((k / fps, s, overtime))
        k += 1
    return out, fps


def game_second(shown, overtime):
    """Battle seconds elapsed when the timer starts to show `shown` (it shows S while the time left is in (S-1, S])."""
    return REGULATION_S + OVERTIME_S - shown if overtime else REGULATION_S - shown


def battle_start(reads, fps):
    """The video second of battle second 0, from every change of the seconds shown (the change happened between
    the frame before and the first frame with the new number). Returns (start, spread of the estimates)."""
    est = []
    for (t0, s0, o0), (t1, s1, o1) in zip(reads, reads[1:]):
        if o0 == o1 and s0 - s1 == 1 and t1 - t0 < 1.5 / fps:
            est.append(t1 - 0.5 / fps - game_second(s1, o1))
    if not est:
        raise ValueError("no timer changes read")
    start = statistics.median(est)
    return start, statistics.pstdev(est)


def load_timeline(name):
    """Start and fps of a battle, cached in raw/classic/<name>_time.json (reading the timer takes a while)."""
    path = os.path.join(OUT, f"{name}_time.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    reads, fps = read_timer(os.path.join(OUT, f"{name}.mp4"))
    start, spread = battle_start(reads, fps)
    reads_red, _ = read_timer(os.path.join(OUT, f"{name}_red.mp4"))
    start_red, spread_red = battle_start(reads_red, fps)
    t = {"start": start, "spread": spread, "start_red": start_red, "spread_red": spread_red, "fps": fps,
         "last_read": reads[-1][0] if reads else None}
    json.dump(t, open(path, "w", encoding="utf-8"), indent=1)
    return t


# --- plays ------------------------------------------------------------------------------------------------
def named_plays(name, namer=None):
    """The battle's plays that left the hand, each with its card id ("name", None when the picture is unknown)."""
    if namer is None:
        from classic_check import card_namer
        namer = card_namer()
    plays = [json.loads(line) for line in open(os.path.join(OUT, f"{name}_plays.jsonl"), encoding="utf-8")]
    out = []
    for pl in plays:
        if not pl.get("left_hand"):
            continue
        pl["name"] = namer(os.path.join(OUT, pl["card"]))
        out.append(pl)
    return out


def decks_of(plays):
    """Each side's deck: the pool deck holding every card the side played."""
    pool = json.load(open(os.path.join(ROOT, "data", "classic_decks.json"), encoding="utf-8"))["decks"]
    out = {}
    for side in ("blue", "red"):
        seen = {pl["name"] for pl in plays if pl["side"] == side and pl["name"]}
        fits = [k for k, v in pool.items() if seen <= set(v)]
        if len(fits) != 1:
            raise ValueError(f"{side}: cards {sorted(seen)} fit decks {fits}")
        out[side] = (fits[0], pool[fits[0]])
    return out


def play_spot(pl, data):
    """Where a play went, in blue's tiles: a troop's deploy clock (the tile it really took), else the aimed tile."""
    card = data.card(pl["name"])
    if pl.get("clock") and not card.is_spell:
        x, y = pl["clock"]
        y += 0.25  # (the clock is drawn ~0.25 tile below the tile centre)
        if pl["side"] == "red":
            x, y = mirror(x, y)
        return x, y
    return tuple(pl.get("landed_blue") or pl["tile_blue"])


# --- the replay -------------------------------------------------------------------------------------------
def snapshot(battle, data):
    """Everything on the field now, in blue's tiles."""
    units, spells = [], []
    for e in battle.entities:
        if e.is_tower:
            continue
        units.append({"id": e.id, "unit": e.def_id, "team": team_name(e.team), "x": e.pos.x, "y": e.pos.y,
                      "air": e.air, "building": e.is_building, "hp": e.hp, "max_hp": e.max_hp,
                      "deploying": e.deploy_ms_left > 0, "rider": e.host >= 0, "radius": e.radius,
                      "spawned": e.play < 0, "play": e.play})
    ids = data.card_ids  # (card_ids[index] == id)
    for p in battle.projectiles:
        if p.is_spell and p.delay_ms <= 0:
            spells.append({"card": ids[p.card_index] if 0 <= p.card_index < len(ids) else None,
                           "team": team_name(p.team), "x": p.pos.x, "y": p.pos.y, "kind": "flight"})
    for a in battle.areas:
        if a.card_index >= 0:
            spells.append({"card": ids[a.card_index], "team": team_name(a.team), "x": a.pos.x, "y": a.pos.y,
                           "radius": a.radius, "kind": "area", "waiting": a.delay_ms > 0})
    for r in battle.rollers:
        spells.append({"card": ids[r.card_index] if r.card_index >= 0 else None, "team": team_name(r.team),
                       "x": r.pos.x, "y": r.pos.y, "half_width": r.half_width, "kind": "roll"})
    return {"g": battle.time_ms / 1000.0, "units": units, "spells": spells}


class Replay:
    """A battle replayed step by step: advance(t) runs the simulator up to blue video second t, forcing the logged
    plays on the way; the battle stays open for corrections (Battle.twin_move) between steps."""

    def __init__(self, name, namer=None, latency=TAP_LATENCY):
        self.data = ce.load_data()
        self.tl = load_timeline(name)
        plays = named_plays(name, namer)
        self.decks = decks_of(plays)
        self.battle = crsim.Battle(self.data, self.decks["blue"][1], self.decks["red"][1], 0)
        self.tick = self.data.rules.tick_ms / 1000.0
        self.todo = [pl for pl in plays if pl["name"]]
        self.unknown = [pl for pl in plays if not pl["name"]]
        self.latency = latency
        for pl in self.todo:  # (when each play reaches the game, in battle seconds)
            lat = latency if self.data.card(pl["name"]).is_spell else TROOP_LATENCY + (latency - TAP_LATENCY)
            pl["g"] = pl["t"] + lat - self.tl["start"]
        self.todo.sort(key=lambda pl: pl["g"])
        self.k = 0
        self.missing = []

    def game_time(self, t):
        return t - self.tl["start"]

    def advance(self, t):
        """Runs the battle up to blue video second t. Returns False before the battle start."""
        g = self.game_time(t)
        b = self.battle
        while b.time_ms / 1000.0 + self.tick <= g + 1e-9 and not b.finished:
            now = b.time_ms / 1000.0
            while self.k < len(self.todo) and self.todo[self.k]["g"] <= now + 1e-9:
                pl = self.todo[self.k]
                self.k += 1
                team = crsim.Team.BLUE if pl["side"] == "blue" else crsim.Team.RED
                ago = int(round((now - pl["g"]) * 1000))
                miss = b.force_play(team, pl["name"], crsim.Vec2(*play_spot(pl, self.data)), ago)
                if miss > 0.5:
                    self.missing.append((round(pl["t"], 2), pl["side"], pl["name"], round(miss, 1)))
            b.step()
        return g >= 0

    def state(self):
        return snapshot(self.battle, self.data)

    def info(self):
        return {"timeline": self.tl, "decks": {s: d[0] for s, d in self.decks.items()}, "plays": len(self.todo),
                "unknown_cards": len(self.unknown), "elixir_short": self.missing}


def replay(name, times=None, namer=None, latency=TAP_LATENCY):
    """Replays battle `name`. Returns (info, states): states[i] is the snapshot at blue video second times[i]
    (every frame of the blue video when times is None), None before the battle start."""
    r = Replay(name, namer, latency)
    if times is None:
        n = int(cv2.VideoCapture(os.path.join(OUT, f"{name}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))
        times = [k / r.tl["fps"] for k in range(n)]
    states = [r.state() if r.advance(t) else None for t in times]
    return r.info(), states


# --- drawing ----------------------------------------------------------------------------------------------
def screen_xy(x, y, view, air, layout):
    """Pixel of a simulated spot (blue's tiles) in a view's frame; flying units are drawn above their ground."""
    if view == "red":
        x, y = mirror(x, y)
    if air:
        y += FLYING_OFFSET
    return layout.tile_to_px(x, y)


def draw(frame, state, view):
    layout = Layout.for_frame(frame)
    out = frame.copy()
    for u in state["units"]:
        px, py = screen_xy(u["x"], u["y"], view, u["air"], layout)
        col = (255, 120, 0) if (u["team"] == "blue") == (view == "blue") else (0, 0, 255)
        r = max(4, int(u["radius"] * layout.TILE_W * layout.scale))
        cv2.circle(out, (int(px), int(py)), r, col, 1 if u["deploying"] else 2)
        cv2.putText(out, u["unit"][:10], (int(px) - 20, int(py) - r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    for s in state["spells"]:
        px, py = screen_xy(s["x"], s["y"], view, False, layout)
        rad = int(s.get("radius", 0.5) * layout.TILE_W * layout.scale)
        cv2.circle(out, (int(px), int(py)), max(rad, 5), (0, 255, 255), 1)
        cv2.putText(out, str(s["card"]), (int(px) - 20, int(py)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def frame_at(video, t):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cap.read()
    return f if ok else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("battle")
    p.add_argument("--sheet", type=float, nargs="*", help="video seconds to draw (both views)")
    p.add_argument("--latency", type=float, default=TAP_LATENCY)
    p.add_argument("--out", default=None, help="where the sheet goes (default raw/classic/<battle>_replay.jpg)")
    args = p.parse_args()
    times = args.sheet or []
    tl = load_timeline(args.battle)
    # the red video starts at its own moment: the same battle second sits at another red video second
    red_shift = tl["start_red"] - tl["start"]
    info, states = replay(args.battle, times=times or None, latency=args.latency)
    print(json.dumps({k: v for k, v in info.items()}, ensure_ascii=False))
    if times:
        rows = []
        for t, st in zip(times, states):
            if st is None:
                continue
            fb = frame_at(os.path.join(OUT, f"{args.battle}.mp4"), t)
            fr = frame_at(os.path.join(OUT, f"{args.battle}_red.mp4"), t + red_shift)
            if fb is None or fr is None:
                continue
            pair = np.hstack([draw(fb, st, "blue"), draw(fr, st, "red")])
            cv2.putText(pair, f"{t:.1f}s g={st['g']:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            rows.append(pair)
        if rows:
            out = args.out or os.path.join(OUT, f"{args.battle}_replay.jpg")
            cv2.imwrite(out, np.vstack(rows))
            print("sheet:", out)


if __name__ == "__main__":
    main()
