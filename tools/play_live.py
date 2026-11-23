"""Plays Clash Royale in LDPlayer: screen -> perception -> live simulator copy -> brain -> touches.

    .venv\\Scripts\\python.exe tools\\play_live.py --check                 connection, screenshot speed, one reading
    .venv\\Scripts\\python.exe tools\\play_live.py --brain heuristic         watch only: shows what it would play
    .venv\\Scripts\\python.exe tools\\play_live.py --brain heuristic --act   really plays
    .venv\\Scripts\\python.exe tools\\play_live.py --brain runs\\starter_v1\\latest.pt --act

Only on the test (guest) account, in friendly battles. The preview window: P = pause/resume playing,
Q = quit. Without --act nothing is touched.
"""
import argparse
import os
import sys
import threading
import time
import traceback

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.device import Device, measure  # noqa: E402
from live.finisher import finishing_spell  # noqa: E402
from live.window import WindowScreen  # noqa: E402
from live.session import BLUE, LiveMatch  # noqa: E402
import twin  # noqa: E402
from perception.clock import find_clocks  # noqa: E402
from perception.layout import Layout  # noqa: E402
from perception.reader import ScreenReader  # noqa: E402

COLORS = {"blue": (255, 140, 40), "red": (40, 40, 255), "any": (0, 220, 255)}  # any: a spell hit (no side)


# --- brains: simulated battle -> (hand slot, position) or None -------------------------------------
class HeuristicBrain:
    def __init__(self, seed=0):
        self.bot = crsim.make_bot("heuristic", seed)

    def __call__(self, battle):
        move = self.bot.decide(battle, BLUE)
        return None if move is None else (move.hand_slot, (move.pos.x, move.pos.y))


class PolicyBrain:
    def __init__(self, path):
        import torch
        from model import PolicyPlayer, deck_pool_of, load_policy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, ckpt = load_policy(path, device)
        self.player = PolicyPlayer(net, device, greedy=True)
        self.key_spots = int(ckpt.get("key_spots", 0))  # the action space it was trained with
        self.deck_pool = deck_pool_of(ckpt)  # the decks its opponents had: the enemy deck block picks the fitting one
        # ... and the rhythm: it decided every decision_ticks x 50 ms in training
        self.decide_every = ckpt.get("args", {}).get("decision_ticks", 6) * 0.05
        self.finisher = False  # --finisher: a hard rule; the policies are to learn it (endgame starts in train_v7)
        self.data = crsim.GameData.load(os.path.join(ROOT, "data"))
        self.net, self.ckpt, self.device = net, ckpt, device
        self.search = None  # --search: lookahead (python/search.py) over the twin, the opponent's hand as it guesses

    def use_search(self, temperature):
        from search import SearchPlayer
        self.search = SearchPlayer(self.net, self.ckpt, self.device, temperature=temperature)

    def __call__(self, battle, enemy_elixir=None):
        """enemy_elixir: the opponent's elixir as the twin best knows it now (LiveMatch.enemy_elixir_seen: their
        troops and spells paid for as soon as seen), for the net and the lookahead instead of the simulator's."""
        if enemy_elixir is not None and abs(enemy_elixir - battle.player(crsim.Team.RED).elixir) > 1e-6:
            battle = battle.clone()
            battle.set_elixir(crsim.Team.RED, enemy_elixir)
        finish = finishing_spell(battle, BLUE, self.data) if self.finisher else None
        if finish is not None:  # a tower a spell in hand surely destroys: the policy never learned to take it
            return finish
        if self.search is not None:
            action = self.search.decide(battle, BLUE)
        else:
            action = int(self.player(crsim.rl.encode(battle, BLUE, key_spots=self.key_spots,
                                                     deck_pool=self.deck_pool)))
        if action == 0:
            return None
        slot, cell = divmod(action - 1, crsim.rl.CELLS)
        pos = crsim.rl.cell_position(battle, BLUE, cell)
        return slot, (pos.x, pos.y)


def make_brain(spec):
    if spec in (None, "none"):
        return None
    if spec == "heuristic":
        return HeuristicBrain()
    if os.path.exists(spec):
        return PolicyBrain(spec)
    sys.exit(f"--brain: none, heuristic or a checkpoint path, not {spec!r}")


# --- preview --------------------------------------------------------------------------------------
NAMES = {"knight": "knight", "archer": "archer", "minion": "minion", "giant": "giant", "mini_pekka": "pekka",
         "musketeer": "musket", "fireball": "fireball", "arrows": "arrows", "fireball_hit": "fireball!",
         "arrows_hit": "arrows!", "archers": "archer", "minions": "minion"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 900  # side panel (full-size pixels) with the status and the latest events


def label(img, text, x, y, color, size=0.9):
    cv2.putText(img, text, (int(x), int(y)), FONT, size, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), FONT, size, color, 2, cv2.LINE_AA)


PREVIEW_H = 900  # px: the preview window's height


def draw(frame, match, detections, status, last_move, events, clocks=()):
    """Left: the screen with what the detector sees (thin boxes, name on top), the deploy clocks found (circles)
    and where the simulator has its units (rings with name and HP bar). Right: status and the latest events.
    Drawn at the preview's own size (drawing on the full frame and shrinking it cost ~20 ms a time)."""
    k = PREVIEW_H / frame.shape[0]
    img = cv2.resize(frame, (int(round(frame.shape[1] * k)), PREVIEW_H), interpolation=cv2.INTER_LINEAR)
    L = Layout.for_frame(frame)

    def px(x, y):
        return int(x * k), int(y * k)

    def lab(text, x, y, color, size):
        cv2.putText(img, text, (int(x), int(y)), FONT, size * k, (0, 0, 0), max(1, round(6 * k)), cv2.LINE_AA)
        cv2.putText(img, text, (int(x), int(y)), FONT, size * k, color, max(1, round(2 * k)), cv2.LINE_AA)

    for d in detections:
        x0, y0 = px(d.box[0], d.box[1])
        x1, y1 = px(d.box[2], d.box[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS.get(d.team, COLORS["any"]), 1)
        lab(NAMES.get(d.cls, d.cls), x0, y0 - 3, COLORS.get(d.team, COLORS["any"]), 0.75)
    for c in clocks:
        cv2.circle(img, px(*c.px), max(2, int(31 * k)), (0, 215, 255), 2)
    if match.battle is not None:
        for e in match.battle.entities:
            if e.is_building:
                continue
            x, y = px(*L.tile_to_px(e.pos.x, e.pos.y))
            color = COLORS["blue" if e.team == BLUE else "red"]
            r = max(3, int(L.s(26) * k))
            cv2.circle(img, (x, y), r, (255, 255, 255), 3)
            cv2.circle(img, (x, y), r, color, 2 if e.deploy_ms_left <= 0 else 1)
            frac = max(0.0, e.hp / max(1, e.max_hp))
            cv2.rectangle(img, (x - r, y + r + 2), (x + r, y + r + 6), (40, 40, 40), -1)
            cv2.rectangle(img, (x - r, y + r + 2), (x - r + int(2 * r * frac), y + r + 6), color, -1)
            lab("sim " + NAMES.get(e.def_id, e.def_id), x - r, y + r + 19, color, 0.75)
        for p in match.battle.projectiles:
            if p.is_spell and p.delay_ms <= 0:
                cv2.drawMarker(img, px(*L.tile_to_px(p.pos.x, p.pos.y)), COLORS["blue" if p.team == BLUE else "red"],
                               cv2.MARKER_STAR, int(L.s(50) * k), 2)
    if last_move is not None:
        x, y = px(*L.tile_to_px(*last_move[1]))
        cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, int(L.s(60) * k), 3)
        lab("bot: " + last_move[0], x + 10, y - 10, (0, 255, 255), 0.9)

    panel = np.full((PREVIEW_H, int(PANEL_W * k), 3), 32, np.uint8)
    lines = status + ["", "box  = seen by the detector", "ring = unit in the simulator", "+    = bot's last move",
                      "gold circle = a deploy clock", "", "latest:"] + events[-24:]
    for n, line in enumerate(lines):
        color = (255, 255, 255)
        if " red " in f" {line} ":
            color = COLORS["red"]
        elif " blue " in f" {line} " or "bot" in line:
            color = (255, 190, 120)
        cv2.putText(panel, line, (int(16 * k), int((50 + 44 * n) * k)), FONT, 0.8 * k, color, 1, cv2.LINE_AA)
    return np.hstack([img, panel])


class FramePrefetch:
    """Takes the next screenshot in a background thread while the bot works on this one (capture ~35 ms of a
    ~90 ms loop before): one frame ahead at most, so it never captures more often than the loop takes frames.
    screenshot() hands over the frame (frame_t: when it was taken); a capture error is raised there, as the
    screen itself would."""

    def __init__(self, screen):
        self.screen = screen
        self.cond = threading.Condition()
        self.ready, self.stop = False, False
        self.frame = self.error = None
        self.frame_t = self._t = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="capture")
        self.thread.start()

    def _run(self):
        while True:
            with self.cond:
                while self.ready and not self.stop:
                    self.cond.wait(0.5)
                if self.stop:
                    return
            t0 = time.monotonic()
            try:
                frame, error = self.screen.screenshot(), None
            except BaseException as err:  # noqa: BLE001 - handed to the loop, which knows what to do
                frame, error = None, err
            t = (t0 + time.monotonic()) / 2  # (the moment the picture shows, about)
            with self.cond:
                self.frame, self.error, self._t, self.ready = frame, error, t, True
                self.cond.notify_all()
            if error is not None:
                time.sleep(0.05)

    STALE = 0.12  # s: a frame that waited longer (the loop was busy: a lookahead search, a tap) is dropped

    def screenshot(self):
        fresh_only = False
        with self.cond:
            while True:
                while not self.ready:
                    if not self.thread.is_alive():
                        raise RuntimeError("the capture thread stopped")
                    self.cond.wait(0.5)
                if self.error is None and not fresh_only and time.monotonic() - self._t > self.STALE:
                    self.ready, fresh_only = False, True  # taken before the loop's last long step: take a new one
                    self.cond.notify_all()
                    continue
                break
            frame, error, self.frame_t = self.frame, self.error, self._t
            self.ready = False
            self.cond.notify_all()
        if error is not None:
            raise error
        return frame

    def close(self):
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        self.thread.join(timeout=2.0)
        self.screen.close()


class VideoScreen:
    """A recording played back in real time instead of the emulator, to try the loop without LDPlayer.
    Touches are only printed."""

    def __init__(self, path, start=0.0):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.start, self.t0 = start, time.monotonic()
        self.serial = os.path.basename(path)

    def screenshot(self):
        index = int((self.start + time.monotonic() - self.t0) * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise KeyboardInterrupt  # end of the recording
        return frame

    def play_card(self, slot, tile, method="drag"):
        print(f"  (video) touch: slot {slot} -> tile ({tile[0]:.1f}, {tile[1]:.1f})")

    def close(self):
        self.cap.release()


def full_size(frame):
    """Perception is calibrated on 1080x1920: bring a window capture of another size to it."""
    if frame.shape[1] != 1080:
        frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)
    return frame


def sim_view(match, battle, seen):
    """The simulator copy in one line: our elixir, each tower's HP on the screen / in the simulator, and every unit
    with where it is (b = ours, r = theirs)."""
    def sim_hp(name):
        e = battle.find(match._tower_id(name))
        return e.hp if e is not None else 0
    shown = seen.get("towers", {})
    towers = " ".join(f"{name} {(shown.get(name) or (None,))[0] or '?'}/{sim_hp(name)}" for name in match.TOWERS)
    units = " ".join(f"{'b' if e.team == BLUE else 'r'}:{twin.UNIT_OF_CARD.get(e.def_id, e.def_id)}"
                     f"({e.pos.x:.1f},{e.pos.y:.1f})" for e in battle.entities if not e.is_building)
    return f"elixir {battle.player(BLUE).elixir:.1f}  towers {towers}  units {units or '-'}"


def check(device, screen, reader):
    w, h = device.size()
    print(f"device {device.serial}: game screen {w}x{h}" + ("" if abs(w / h - 9 / 16) < 0.01 else
                                                           "  <-- not 9:16; the layout expects 1080x1920"))
    frame = screen.screenshot()
    print(f"capture {type(screen).__name__}: {frame.shape[1]}x{frame.shape[0]}, {measure(screen, 10) * 1000:.0f} ms each")
    frame = full_size(frame)
    path = os.path.join(ROOT, "raw", "perception", "live_check.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, frame)
    try:
        seen = reader.read(frame)
        print("hand:", [h["card"] if not h["unknown"] else "?" for h in seen["hand"]], " next:", seen["next"]["card"],
              f" elixir: {seen['elixir']:.1f}  timer: {seen['timer']}")
    except ValueError as err:
        print("cannot read this screen:", err)
    print(f"saved {path}")


class Tee:
    """Writes to the console and to a log file (what happened stays when the window is closed)."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="test the connection and exit")
    p.add_argument("--brain", default="none", help="none | heuristic | path to a policy checkpoint (.pt)")
    p.add_argument("--act", action="store_true", help="really touch the screen (default: only show)")
    p.add_argument("--method", default="tap", choices=("tap", "drag"),
                   help="how to put a card down: tap the card then the spot (exact), or drag it (lands short)")
    p.add_argument("--preview-every", type=int, default=3, help="draw the preview window every N frames")
    p.add_argument("--decide-every", type=float, default=None,
                   help="seconds between brain decisions (default: as the policy was trained, else 0.3)")
    p.add_argument("--temperature", type=float, default=0.5,
                   help="a policy samples its moves at this temperature (0 = always its most likely move). 0.5 beat "
                        "greedy in the simulator (v7e top vs v7e best 60%% vs 52%%, vs v7b 84%% vs 79%%) and is less "
                        "predictable for a human")
    p.add_argument("--search-ab", action="store_true",
                   help="lookahead in every other battle (the 1st, 3rd, ... with it, the 2nd, 4th, ... without), each "
                        "battle's choice in the log (\"battle N: with lookahead\"): does it help against a human?")
    p.add_argument("--search", action="store_true",
                   help="lookahead before a move (python/search.py): the policy's best few moves played forward in "
                        "copies of the twin; ~0.12 s a search (up to 0.2), about every third decision. In the simulator, "
                        "seeing only what the twin can (search_eval.py --fair), cycle 9 with it beat cycle 9 68%%, "
                        "cycle 8 85%% (47%% without), cycle 6 70%% (60%% without)")
    p.add_argument("--finisher", action="store_true",
                   help="hard rule: cast a spell in hand that surely destroys an enemy tower (python/live/finisher.py)")
    p.add_argument("--no-detector", action="store_true", help="no unit detection (the opponent stays invisible)")
    p.add_argument("--weights", default=None, help="unit detector weights (default: the latest)")
    p.add_argument("--serial", default=None, help="adb device (default: the first emulator)")
    p.add_argument("--instance", type=int, default=None,
                   help="LDPlayer instance number with several running (0 = the first; see LDMultiPlayer)")
    p.add_argument("--device", default="0", help="detector on: 0 = GPU, cpu = processor")
    p.add_argument("--capture", default="window", choices=("window", "adb"),
                   help="window = copy the LDPlayer window (fast), adb = emulator screenshots (slow)")
    p.add_argument("--video", default=None, help="use a recording instead of the emulator (for testing)")
    p.add_argument("--video-start", type=float, default=0.0)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--sync-capture", action="store_true",
                   help="take each screenshot when the frame before is done (default: the next one is taken in the "
                        "background meanwhile)")
    p.add_argument("--no-clocks", action="store_true",
                   help="don't look for deploy clocks (a new red troop then counts as played only after ~0.5 s)")
    p.add_argument("--no-quick-answer", action="store_true",
                   help="decide only on the policy's rhythm (default: also at once when the twin takes a new red play)")
    p.add_argument("--profile", action="store_true", help="time each step of the loop (printed with the status)")
    args = p.parse_args()
    if not args.check:
        os.makedirs(os.path.join(ROOT, "raw", "live_logs"), exist_ok=True)
        path = os.path.join(ROOT, "raw", "live_logs", time.strftime("live_%Y%m%d_%H%M%S.txt"))
        sys.stdout, sys.stderr = Tee(sys.stdout, path), Tee(sys.stderr, path)
        print(f"log: {path}")

    if args.video:
        screen = device = VideoScreen(args.video, args.video_start)
    else:
        serial = args.serial or (f"emulator-{5554 + 2 * args.instance}" if args.instance is not None else None)
        device = Device(serial)  # touches always go through ADB
        screen = WindowScreen(index=args.instance) if args.capture == "window" else device
    reader = ScreenReader()
    if args.check:
        check(device, screen, reader)
        return
    if not args.sync_capture and (screen is not device or args.video):  # (not for slow ADB screenshots)
        screen = FramePrefetch(screen)

    detector = None
    if not args.no_detector:
        from perception.units import DEFAULT_WEIGHTS, UnitDetector
        detector = UnitDetector(args.weights or DEFAULT_WEIGHTS, device=0 if args.device == "0" else args.device)
    data = load_data()
    from perception.unit_hp import UnitHpReader
    hp_reader = UnitHpReader()
    match = LiveMatch(data)
    brain = make_brain(args.brain)
    if args.finisher and isinstance(brain, PolicyBrain):
        brain.finisher = True
    if isinstance(brain, PolicyBrain) and args.temperature > 0:
        brain.player.temperature = args.temperature
    if isinstance(brain, PolicyBrain) and (args.search or args.search_ab):
        brain.use_search(args.temperature if args.temperature > 0 else 0.5)
    searcher = brain.search if isinstance(brain, PolicyBrain) else None
    battle_start, battle_no = None, 0  # --search-ab: the battle's start (wall s) and number
    if args.decide_every is None:
        args.decide_every = getattr(brain, "decide_every", 0.3)
    playing = args.act
    next_decision = next_status = 0.0
    warned = False
    last_move = None
    events = []  # latest events for the preview
    frames, fps_t0, fps = 0, time.monotonic(), 0.0
    errors = 0
    last_refusal = None
    loop_no = 0
    print(f"brain: {args.brain}{' + lookahead' if args.search else ' + lookahead every other battle' if args.search_ab else ''}, "
          f"{'PLAYING' if playing else 'watching only (no --act)'}; P = pause, Q = quit")
    red_taken = 0  # the twin's count of red plays taken: a new one = decide at once (--no-quick-answer: not)
    # No decision on a frame taken sooner than decide_every after our last tap: until then the screen (and the
    # twin) may still show the card in hand and the elixir unspent - the same card was tapped twice (2026-11-16).
    own_ready = 0.0
    clocks = []
    spent = {}  # --profile: step -> [seconds, times]

    def lap(step, since):
        now = time.monotonic()
        if args.profile:
            s = spent.setdefault(step, [0.0, 0])
            s[0] += now - since
            s[1] += 1
        return now

    try:
        while True:
            t_cap = time.monotonic()
            try:
                frame = screen.screenshot()
                t = getattr(screen, "frame_t", None) or time.monotonic()  # when the picture was taken
                warned = False
            except RuntimeError as err:  # LDPlayer minimised: slow ADB screenshots until it is back
                if not warned:
                    print(f"{err}; using ADB screenshots (slow) until the window is back")
                    warned = True
                frame = device.screenshot()
                t = time.monotonic()
            try:
                t_step = lap("capture (waiting)", t_cap)
                frame = full_size(frame)
                try:
                    seen = reader.read(frame)
                except ValueError:
                    continue
                t_step = lap("screen read", t_step)
                detections = detector.detect(frame) if detector else []
                t_step = lap("detector", t_step)
                bars = hp_reader.bars(frame, detections) if detector else None
                clocks = (find_clocks(frame, near=match.clock_spots(detections))
                          if detector and not args.no_clocks else None)
                t_step = lap("bars + clocks", t_step)
                match.update(t, seen, detections, bars, clocks)
                clocks = clocks or []
                t_step = lap("twin", t_step)
                if match.red_taken != red_taken:  # a new red play: answer it now, not at the next beat
                    red_taken = match.red_taken
                    if not args.no_quick_answer:
                        next_decision = min(next_decision, t)
                frames += 1
                if t - fps_t0 > 2.0:
                    fps, frames, fps_t0 = frames / (t - fps_t0), 0, t

                g = match.game_time(t)
                if args.search_ab and searcher is not None and match.start is not None and (
                        battle_start is None or abs(match.start - battle_start) > 30.0):
                    battle_start, battle_no = match.start, battle_no + 1  # a new battle
                    brain.search = searcher if battle_no % 2 == 1 else None
                    print(f"battle {battle_no}: {'with' if brain.search is not None else 'without'} lookahead")
                battle = match.battle
                if (battle is not None and brain is not None and t >= next_decision and t >= own_ready
                        and g is not None and g >= 0):
                    next_decision = t + args.decide_every
                    t_step = time.monotonic()
                    move = (brain(battle, enemy_elixir=match.enemy_elixir_seen(t)) if isinstance(brain, PolicyBrain)
                            else brain(battle))
                    t_step = lap("brain" + (" (lookahead)" if getattr(brain, "search", None) is not None else ""),
                                 t_step)
                    if move is not None:
                        slot, pos = move
                        card = battle.player(BLUE).hand[slot]
                        real = seen["hand"][slot]
                        cost = data.card(card).elixir if card else 99
                        elixir = seen.get("elixir")  # None: not readable this frame - wait for the next one
                        why_not = ("the slot is empty in the simulator" if not card else
                                   f"the screen shows {real['card'] or 'nothing'} there"
                                   if real["unknown"] or real["card"] != card
                                   else "elixir not readable" if elixir is None
                                   else f"elixir {elixir:.1f} < {cost}" if elixir + 0.05 < cost else None)
                        if why_not and (card, why_not) != last_refusal:  # (once per reason, not every 0.3 s)
                            print(f"{g:6.1f}s  bot wants {card} at ({pos[0]:.1f}, {pos[1]:.1f}), not played: {why_not}")
                        last_refusal = (card, why_not) if why_not else None
                        if not why_not:
                            last_move = (card, pos)
                            if playing:
                                device.play_card(slot, pos, args.method)
                                tapped = time.monotonic()
                                match.own_play(tapped, card, pos)  # (now, not when the frame was taken)
                                own_ready = tapped + args.decide_every
                            text = f"{'bot plays' if playing else 'bot would play'} {card} at ({pos[0]:.1f}, {pos[1]:.1f})"
                            print(f"{g:6.1f}s  {text}")
                            if playing or not events or not events[-1].endswith(text):
                                events.append(f"{g:5.1f}s {text}")
                for gt, text in match.log:
                    print(f"{gt:6.1f}s  {text}")
                    events.append(f"{gt:5.1f}s {text}")
                match.log.clear()
                del events[:-40]
                if t >= next_status:
                    next_status = t + 5.0
                    hand = [h["card"] if not h["unknown"] else "?" for h in seen["hand"]]
                    when = f"game {g:5.1f}s" if g is not None else "no battle"
                    units = sum(1 for e in battle.entities if not e.is_building) if battle is not None else 0
                    print(f"[{fps:4.1f} fps] {when}  elixir {seen['elixir']:.1f}  hand {hand}  units in sim {units}")
                    if battle is not None:  # to check the simulator copy against the game afterwards
                        print(f"          sim: {sim_view(match, battle, seen)}")
                    if args.profile and spent:
                        print("          ms: " + "  ".join(f"{k} {1000 * s / max(1, n):.1f}"
                                                         for k, (s, n) in spent.items()))
                        spent.clear()

            except Exception:  # one bad frame must not stop the bot in the middle of a battle
                errors += 1
                print(f"error in this frame ({errors} in a row), carrying on:\n{traceback.format_exc()}")
                if errors > 100:
                    raise
                continue
            errors = 0

            loop_no += 1
            if not args.no_window and loop_no % max(1, args.preview_every) == 0:  # (drawing costs frame rate)
                status = [f"{fps:.1f} fps  game {g:.1f}s" if g is not None else f"{fps:.1f} fps  waiting for a battle",
                          f"elixir {seen['elixir']:.1f}  " + ("PLAYING" if playing else "watching") +
                          f"  brain {args.brain}"]
                t_step = time.monotonic()
                try:  # the preview is only for us to watch: a drawing error must not stop the bot
                    cv2.imshow("cr-neiro live", draw(frame, match, detections, status, last_move, events, clocks))
                except Exception:
                    print(f"preview error, skipped:\n{traceback.format_exc()}")
                key = cv2.waitKey(1) & 0xFF
                lap("preview", t_step)
                if key == ord("q"):
                    break
                # The window's close button: it would only come back with the next frame - take it as quit.
                if cv2.getWindowProperty("cr-neiro live", cv2.WND_PROP_VISIBLE) < 1:
                    print("preview window closed: quitting")
                    break
                if key == ord("p") and args.act:
                    playing = not playing
                    print("playing" if playing else "paused")
    except KeyboardInterrupt:
        pass
    finally:
        device.close()
        if screen is not device:
            screen.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
