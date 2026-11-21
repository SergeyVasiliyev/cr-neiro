"""Random bots on two LDPlayer instances, every battle recorded from both windows: data for the Classic Decks
stage (detector labels, sim checks) without a human playing both sides.

    .venv\\Scripts\\python.exe tools\\random_battles.py                 bots on LDPlayer 0 and 1, stops after 10 battles
    .venv\\Scripts\\python.exe tools\\random_battles.py --battles 3
    .venv\\Scripts\\python.exe tools\\random_battles.py --check         one reading of each screen, no touches

You start each friendly battle yourself (Classic Decks: one guest invites in the clan chat, the other accepts).
A bot sees a battle by its timer and plays a lit card (enough elixir) at a random spot, at random moments, never
sitting at 10. Card names are not known. A card's first play goes on the bot's own half: if a deploy clock shows up
there, it is a troop or a building (always our half), if not, a spell (then mostly aimed at the enemy's half).

Each battle: raw/classic/cNNN.mp4 (LDPlayer 0's window, "blue") + cNNN_red.mp4 (LDPlayer 1's), every touch in
cNNN_plays.jsonl (video seconds, side, slot, the tile aimed at in its own view and in blue's, elixir, timer,
whether the card left the hand, troop or spell, the deploy clock's tile, where it came down in blue's tiles) with
the card's picture in cNNN_cards/, and cNNN.json (summary). Recording starts 2 s before the
timer first shows and stops 8 s after it is gone from both screens (the result screen is in). It also stops when
less than 1.5 GB is left on the disk. Stop with Ctrl+C (closing the window also finishes the files).

Only guest accounts, only friendly battles.
"""
import argparse
import collections
import json
import math
import os
import queue
import random
import re
import shutil
import sys
import threading
import time
import traceback

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from live.device import Device  # noqa: E402
from live.window import WindowScreen  # noqa: E402
from perception.clock import find_clocks  # noqa: E402
from perception.layout import Layout  # noqa: E402
from perception.reader import ScreenReader, edge_features, hand_art, saturation  # noqa: E402
from record_windows import on_window_close  # noqa: E402

OUT = os.path.join(ROOT, "raw", "classic")
SIDES = {0: "blue", 1: "red"}
SUFFIX = {0: "", 1: "_red"}
W, H = 18.0, 32.0  # arena tiles; the other player's view is turned 180 degrees: (W - x, H - y)
PREROLL = 2.0      # s of video kept from before the timer first shows
POSTROLL = 8.0     # s without a timer on any screen = the battle is over
MIN_FREE_GB = 1.5


def full_size(frame):
    """Perception is calibrated on 1080x1920."""
    if frame.shape[1] != 1080:
        frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)
    return frame


def free_gb():
    return shutil.disk_usage(OUT).free / 1e9


OWN_TOWERS = ((9.0, 3.0, 2.0), (3.5, 6.5, 1.5), (14.5, 6.5, 1.5))  # x, y, half side of the no-deploy square
                                                                   # (data/towers.json footprint)


def own_tile(rng):
    """A random tile centre on our half a troop may take. The game puts a troop on the centre of the tile it is aimed
    at, or on the nearest free tile when that one is not free (on our tower, at the river row's closed ends)."""
    while True:
        x, y = math.floor(rng.uniform(0.0, W)) + 0.5, math.floor(rng.uniform(0.0, 15.0)) + 0.5
        if any(abs(x - tx) < r and abs(y - ty) < r for tx, ty, r in OWN_TOWERS):
            continue
        if y > 14.0 and (x < 1.0 or x > W - 1.0):
            continue
        return x, y


def clock_tile(tile):
    """The tile a deploy clock stands for: troops come down on tile centres and the clock is drawn ~0.25 tile below
    (c003: clock x always k + 0.5, y k + 0.2..0.3)."""
    return math.floor(tile[0]) + 0.5, math.floor(tile[1] + 0.25) + 0.5


def plain(value):
    """numpy numbers (and anything else odd) for json."""
    return value.item() if hasattr(value, "item") else str(value)


def next_name():
    numbers = [int(m.group(1)) for f in os.listdir(OUT) if (m := re.match(r"c(\d+)\.mp4$", f))]
    return f"c{max(numbers, default=0) + 1:03d}"


class Log:
    """Console + a log file in raw/classic (what happened stays when the window is closed)."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def __call__(self, text):
        line = time.strftime("%H:%M:%S ") + text
        with self.lock:
            print(line, flush=True)
            self.file.write(line + "\n")


class Capture:
    """Grabs every instance's window on a fixed clock. Keeps the latest frame of each for the bots and the last
    PREROLL seconds; while recording, writes one frame per tick to each video (the last one again when a grab fails
    or comes late), so video time = wall time since the recording's first frame."""

    def __init__(self, instances, fps):
        self.screens = {i: WindowScreen(index=i) for i in instances}
        self.fps = fps
        self.lock = threading.Lock()
        self.latest = {}  # instance -> (frame, monotonic time of the grab)
        self.preroll = collections.deque(maxlen=max(1, int(PREROLL * fps)))  # (tick time, {instance: frame})
        self.request = None  # ("begin", paths) or ("end", None): done by the capture thread at its next tick
        self.writers = self.paths = None
        self.sizes = {}
        self.t0 = None  # monotonic time of the recording's first frame
        self.written = 0
        self.ended = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="capture")
        self.thread.start()

    # --- for the other threads ---------------------------------------------------------------------------------
    def frame(self, i):
        with self.lock:
            return self.latest.get(i)

    def begin(self, paths):
        with self.lock:
            self.request, self.t0 = ("begin", paths), None

    def end(self):
        self.ended.clear()
        with self.lock:
            self.request = ("end", None)
        self.ended.wait(10.0)

    def start_time(self):
        with self.lock:
            return self.t0

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10.0)
        for s in self.screens.values():
            s.close()

    # --- the capture thread ------------------------------------------------------------------------------------
    def _run(self):
        start, tick, last = time.monotonic(), 0, {}
        try:
            while not self.stop.is_set():
                wait = start + tick / self.fps - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                for i, screen in self.screens.items():
                    try:
                        f = screen.screenshot()
                    except RuntimeError:
                        continue  # minimised or being resized: its last frame again
                    h, w = f.shape[:2]
                    last[i] = f[:h - h % 2, :w - w % 2]  # (even size for the encoder)
                    with self.lock:
                        self.latest[i] = (last[i], time.monotonic())
                due = int((time.monotonic() - start) * self.fps) + 1  # ticks due by now: more than one when late
                for k in range(tick, max(due, tick + 1)):
                    self._tick(start + k / self.fps, dict(last))
                tick = max(due, tick + 1)
        finally:
            self._close()
            self.ended.set()

    def _tick(self, t, frames):
        with self.lock:
            request, self.request = self.request, None
        if request is not None:
            if request[0] == "begin":
                self._open(request[1])
            else:
                self._close()
                self.ended.set()
        if self.writers is None:
            self.preroll.append((t, frames))
        else:
            self._write(t, frames)

    def _open(self, paths):
        self._close()
        self.paths, self.writers, self.written = paths, {}, 0
        buffered = list(self.preroll)
        self.preroll.clear()
        for t, frames in buffered:
            self._write(t, frames)

    def _write(self, t, frames):
        if self.written == 0:
            with self.lock:
                self.t0 = t
        for i, path in self.paths.items():
            f = frames.get(i)
            if f is None:
                continue
            if i not in self.writers:
                self.sizes[i] = (f.shape[1], f.shape[0])
                self.writers[i] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.sizes[i])
            if (f.shape[1], f.shape[0]) != self.sizes[i]:
                f = cv2.resize(f, self.sizes[i])  # the window was resized
            self.writers[i].write(f)
        self.written += 1

    def _close(self):
        for w in (self.writers or {}).values():
            w.release()
        self.writers = None


class RandomBot(threading.Thread):
    """Plays one LDPlayer instance at random while its screen shows a battle timer."""

    TICK = 0.25        # s between looks at the screen
    EMPTY_EDGE = 15.0  # mean gradient of a slot's picture: an empty slot (or the intro over the hand) 3-10, a card 20+
                       # (c001, c003). The hand is dealt one card at a time at the start and the next card comes 1 s
                       # after a play; an empty slot is blue, so it passed for a lit card
    STABLE = 0.85      # edge correlation with the look before: a card still sliding in is not tapped
    SAME_CARD = 0.7    # ... above which a slot still holds the card that was played (raw/videos/video49_red: a changed
                       # slot scored above it 1 time in 421, the same card 1.1 s later mostly 0.9+)
    LOOK_FROM = 0.9    # s after the tap: the hand is looked at
    CLOCK_TO = 1.9     # s after the tap: our troop's deploy clock showed up 1.20-1.53 s after it (c003)
    ENEMY_HALF = 0.7   # share of a known spell's plays aimed at the enemy's half. Never a troop's: the game puts a
                       # troop aimed there on the nearest tile it may go (the river's edge, a pocket's). An unknown
                       # card goes on our half, where a deploy clock (or none) tells a troop from a spell

    def __init__(self, index, capture, plays, stop, log, seed=None):
        super().__init__(daemon=True, name=f"bot {index}")
        self.index, self.side = index, SIDES.get(index, str(index))
        self.capture, self.plays, self.stop, self.log = capture, plays, stop, log
        self.reader = ScreenReader()
        self.rng = random.Random(None if seed is None else seed * 100 + index)
        self.device = Device(f"emulator-{5554 + 2 * index}")
        self.last_timer = -1e9  # monotonic time the battle timer was last seen here
        self.timer = self.elixir = None
        self.played = self.left = 0  # this battle's plays, and how many of them left the hand
        self._new_battle()

    def _new_battle(self):
        self.target = self._draw_target()
        self.avoid = None  # a slot whose play did not leave the hand: maybe still selected (a tap would unselect it)
        self.cards = []    # [edge features, "troop" / "spell" / None (not played yet)] of this battle's deck
        self.prev = [None] * 4  # each slot's edge features at the look before
        self.played = self.left = 0

    def _draw_target(self):
        """The elixir to wait for before the next play."""
        r = self.rng.random()
        if r < 0.25:
            return 0.0  # the next lit card at once (quick follow-ups)
        if r < 0.75:
            return self.rng.uniform(3.0, 7.0)
        return self.rng.uniform(7.0, 10.0)  # save up for a push

    def run(self):
        while not self.stop.is_set():
            try:
                self._step()
            except Exception:  # noqa: BLE001 - one bad frame must not stop the bot in the middle of a battle
                self.log(f"{self.side}: error, carrying on\n{traceback.format_exc()}")
                time.sleep(1.0)
        self.device.close()

    def _look(self, after=0.0):
        """The latest frame at full size, if it was grabbed after `after` and is fresh; else (None, its time)."""
        got = self.capture.frame(self.index)
        if got is None:
            return None, None
        frame, t = got
        if t < after or time.monotonic() - t > 0.5:
            return None, t
        return full_size(frame), t

    def _step(self):
        frame, t = self._look()
        seconds, overtime = self.reader.numbers.read_timer(frame) if frame is not None else (None, False)
        if seconds is None:
            self.prev = [None] * 4
            time.sleep(self.TICK)
            return
        if t - self.last_timer > POSTROLL:
            self._new_battle()
        overtime = bool(overtime)
        self.last_timer, self.timer = t, (seconds, overtime)
        layout = Layout.for_frame(frame)
        self.elixir = self.reader._elixir(frame, layout)
        slots = [self.slot_state(layout.hand_slot(frame, k), self.prev[k]) for k in range(4)]
        self.prev = [s["feat"] for s in slots]
        choices = [k for k, s in enumerate(slots) if s["ready"] and s["lit"] and k != self.avoid]
        if not choices or (self.elixir < self.target and self.elixir < 9.5):
            time.sleep(self.TICK)
            return
        self._play(frame, layout, self.rng.choice(choices), slots, seconds, overtime)

    def slot_state(self, crop, prev):
        """A hand slot: its edge features, whether a card sits there still (not empty, not sliding in) and is lit."""
        art = hand_art(crop)
        feat = edge_features(art)
        g = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY).astype(np.float32)
        edge = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean() + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())
        stable = prev is not None and float(prev @ feat) / len(feat) > self.STABLE
        return {"feat": feat, "ready": edge >= self.EMPTY_EDGE and stable,
                "lit": saturation(art) > self.reader.color_threshold}

    def _card(self, feat):
        """This battle's record of the card with these edges (the same picture lit or greyed)."""
        for card in self.cards:
            if float(card[0] @ feat) / len(feat) > self.SAME_CARD:
                return card
        card = [feat, None]
        self.cards.append(card)
        return card

    def _play(self, frame, layout, slot, slots, seconds, overtime):
        crop = layout.hand_slot(frame, slot).copy()
        feat = slots[slot]["feat"]
        card = self._card(feat)
        enemy = card[1] == "spell" and self.rng.random() < self.ENEMY_HALF
        x, y = (self.rng.uniform(0.5, W - 0.5), self.rng.uniform(17.5, H - 0.5)) if enemy else own_tile(self.rng)
        self.device.play_card(slot, (x, y), method="tap")
        tapped = time.monotonic()
        left, clock = self.outcome(self._frames_after(tapped), slot, feat, (x, y), enemy)
        if left:
            if clock is not None:
                card[1] = "troop"  # (a troop's clock missed once does not make it a spell)
            elif card[1] is None:
                card[1] = "spell"
        self.avoid = slot if left is False else None
        self.played += 1
        self.left += bool(left)
        landed = None if not left else clock_tile(clock) if clock is not None else (x, y)
        self.plays.put({
            "wall": tapped, "side": self.side, "slot": slot, "tile": [round(x, 2), round(y, 2)],
            "tile_blue": [round(x, 2), round(y, 2)] if self.index == 0 else [round(W - x, 2), round(H - y, 2)],
            "half": "enemy" if enemy else "own", "elixir": round(self.elixir, 2), "target": round(self.target, 2),
            "timer": seconds, "overtime": overtime, "left_hand": left, "kind": card[1] if left else None,
            "clock": None if clock is None else [round(clock[0], 2), round(clock[1], 2)],
            "landed_blue": None if landed is None else
            [round(landed[0], 2), round(landed[1], 2)] if self.index == 0 else
            [round(W - landed[0], 2), round(H - landed[1], 2)],
            "crop": crop})
        self.target = self._draw_target()

    def _frames_after(self, tapped):
        """Fresh frames from LOOK_FROM after the tap on, one per grab, until CLOCK_TO: (frame, seconds after the tap)."""
        seen = 0.0
        while True:
            frame, t = self._look(after=max(tapped + self.LOOK_FROM, seen + 1e-3))
            if frame is None:
                if time.monotonic() > tapped + self.CLOCK_TO + 0.5:
                    return  # (no fresh frames: the window is minimised)
                time.sleep(0.02)
                continue
            seen = t
            yield frame, t - tapped
            if t - tapped > self.CLOCK_TO:
                return

    def outcome(self, frames, slot, feat, tile, enemy):
        """(did the card leave the hand, the tile of our deploy clock or None) from the frames after a play.
        A card still in its slot (raised = selected, or not) shows the same edges; a played one leaves the slot empty
        for 1 s, then the next card comes. A troop or a building shows our (blue) deploy clock where it comes down,
        ~1.3 s after the tap: at the spot, or for a troop aimed at the enemy's half on the nearest tile it may go
        (the river's edge, or a pocket's). A spell shows none. left None: no frame to tell."""
        x, y = tile
        spots = [tile] + ([(min(max(x, 1.5), 16.5), 14.5), (x, 20.0)] if enemy else [])
        left = None
        for frame, _dt in frames:
            layout = Layout.for_frame(frame)
            if left is None:
                best = max(float(edge_features(hand_art(layout.hand_slot(frame, slot, r))) @ feat) / len(feat)
                           for r in range(0, Layout.HAND_MAX_RAISE + 1, 5))
                left = best < self.SAME_CARD
                if not left:
                    return False, None
            for c in find_clocks(frame, near=spots, layout=layout):
                if c.side == "blue" or (c.side == "?" and not enemy):  # (a red one, or a new one over there: theirs)
                    return left, c.tile
        return left, None


class Battle:
    """One recorded battle: the videos, the plays file and the card pictures."""

    def __init__(self, name, capture, instances):
        self.name, self.capture = name, capture
        self.paths = {i: os.path.join(OUT, f"{name}{SUFFIX.get(i, f'_{i}')}.mp4") for i in instances}
        self.cards_dir = os.path.join(OUT, f"{name}_cards")
        os.makedirs(self.cards_dir, exist_ok=True)
        self.plays = open(os.path.join(OUT, f"{name}_plays.jsonl"), "w", encoding="utf-8")
        self.count = collections.Counter()
        self.left = collections.Counter()
        self.pending = []
        self.started = time.monotonic()
        self.date = time.strftime("%Y-%m-%d %H:%M:%S")
        capture.begin(self.paths)

    def add(self, play):
        self.pending.append(play)
        self.flush()

    def flush(self):
        """Writes the plays waiting for the video's start time (the capture thread opens the files at its next tick)."""
        t0 = self.capture.start_time()
        if t0 is None or not self.pending:
            return
        for p in self.pending:
            side = p["side"]
            self.count[side] += 1
            self.left[side] += bool(p["left_hand"])
            crop_name = f"{side}_{self.count[side]:03d}.jpg"
            cv2.imwrite(os.path.join(self.cards_dir, crop_name), p["crop"], [cv2.IMWRITE_JPEG_QUALITY, 92])
            row = {k: v for k, v in p.items() if k not in ("wall", "crop")}
            row = {"t": round(p["wall"] - t0, 3), **row, "card": f"{self.name}_cards/{crop_name}"}
            self.plays.write(json.dumps(row, default=plain) + "\n")
        self.plays.flush()
        self.pending.clear()

    def finish(self, fps, timers):
        self.flush()
        self.capture.end()
        self.plays.close()
        summary = {"name": self.name, "date": self.date, "seconds": round(time.monotonic() - self.started, 1),
                   "fps": fps, "videos": {SIDES.get(i, str(i)): os.path.basename(p) for i, p in self.paths.items()},
                   "plays": dict(self.count), "left_hand": dict(self.left), "last_timer": timers}
        with open(os.path.join(OUT, f"{self.name}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, default=plain)
        return summary


def check(instances):
    reader = ScreenReader()
    for i in instances:
        screen = WindowScreen(index=i)
        frame = full_size(screen.screenshot())
        screen.close()
        layout = Layout.for_frame(frame)
        seconds, overtime = reader.numbers.read_timer(frame)
        sats = [saturation(hand_art(layout.hand_slot(frame, k))) for k in range(4)]
        print(f"LDPlayer {i} ({SIDES.get(i, i)}): " + (
            f"timer {seconds // 60}:{seconds % 60:02d}{' overtime' if overtime else ''}, elixir "
            f"{reader._elixir(frame, layout):.1f}, card colour " +
            " ".join(f"{s:.0f}{'(lit)' if s > reader.color_threshold else ''}" for s in sats)
            if seconds is not None else "no battle on screen"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--battles", type=int, default=10, help="stop after this many recorded battles")
    p.add_argument("--instances", type=int, nargs="+", default=[0, 1], help="LDPlayer instances (0 = blue)")
    p.add_argument("--minutes", type=float, default=None, help="also stop after this long (a battle on is cut short)")
    p.add_argument("--fps", type=float, default=15.0, help="video frame rate")
    p.add_argument("--seed", type=int, default=None, help="random seed (default: different every run)")
    p.add_argument("--no-record", action="store_true", help="bots only, no videos")
    p.add_argument("--check", action="store_true", help="read each screen once (timer, elixir, lit cards) and exit")
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.check:
        check(args.instances)
        return
    if not args.no_record and free_gb() < MIN_FREE_GB:
        sys.exit(f"only {free_gb():.1f} GB free on the disk: free some space first (videos take ~0.2 GB a battle)")
    log = Log(os.path.join(OUT, time.strftime("log_%Y%m%d_%H%M%S.txt")))
    stop, closing, closed = threading.Event(), threading.Event(), threading.Event()
    handler = on_window_close(closing, closed)  # noqa: F841 - (kept alive)
    capture = Capture(args.instances, args.fps)
    plays = queue.Queue()
    bots = [RandomBot(i, capture, plays, stop, log, args.seed) for i in args.instances]
    for b in bots:
        b.start()
    log(f"bots on LDPlayer {', '.join(f'{b.index} ({b.side})' for b in bots)}; "
        f"{'no recording' if args.no_record else f'battles go to raw/classic ({free_gb():.1f} GB free)'}. "
        f"Start a friendly battle; Ctrl+C to stop.")
    battle, done, next_status, next_disk = None, 0, 0.0, 0.0
    deadline = time.monotonic() + args.minutes * 60 if args.minutes else float("inf")
    try:
        while done < args.battles and not closing.is_set():
            now = time.monotonic()
            if now > deadline:
                log(f"{args.minutes:g} minutes are up")
                break
            seen = max(b.last_timer for b in bots)
            if battle is None and now - seen < 1.0:
                if args.no_record:
                    battle = "on"
                    log("battle started")
                else:
                    battle = Battle(next_name(), capture, args.instances)
                    log(f"battle {done + 1}: recording {battle.name}")
            elif battle is not None and now - seen > POSTROLL:
                timers = {b.side: b.timer for b in bots}
                if isinstance(battle, Battle):
                    s = battle.finish(args.fps, timers)
                    log(f"{s['name']} saved: {s['seconds']:.0f} s, plays " +
                        ", ".join(f"{side} {n} ({s['left_hand'].get(side, 0)} left the hand)"
                                  for side, n in s["plays"].items()))
                else:
                    log("battle over: " + ", ".join(f"{b.side} {b.played} plays ({b.left} left the hand)" for b in bots))
                battle, done = None, done + 1
            while True:
                try:
                    play = plays.get_nowait()
                except queue.Empty:
                    break
                if isinstance(battle, Battle):
                    battle.add(play)
            if isinstance(battle, Battle):
                battle.flush()
            if battle is not None and now >= next_status:
                next_status = now + 15.0
                parts = []
                for b in bots:
                    if b.timer is None or now - b.last_timer > 2.0:
                        parts.append(f"{b.side}: no timer")
                        continue
                    sec, ot = b.timer
                    parts.append(f"{b.side} {'+' if ot else ''}{sec // 60}:{sec % 60:02d} elixir {b.elixir:.1f}, "
                                 f"{b.played} plays ({b.left} left the hand)")
                log("  " + " | ".join(parts))
            if isinstance(battle, Battle) and now >= next_disk:
                next_disk = now + 30.0
                if free_gb() < MIN_FREE_GB:
                    log(f"only {free_gb():.1f} GB left on the disk: stopping")
                    break
            time.sleep(0.2)
        if done >= args.battles:
            log(f"{done} battles done")
    except KeyboardInterrupt:
        log("stopped (Ctrl+C)")
    finally:
        stop.set()
        if isinstance(battle, Battle):
            s = battle.finish(args.fps, {b.side: b.timer for b in bots})
            log(f"{s['name']} saved (cut short): {s['seconds']:.0f} s")
        capture.close()
        closed.set()  # (the files are finished: a closing console window may go now)
        for b in bots:
            b.join(timeout=3.0)


if __name__ == "__main__":
    main()
