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
        from model import PolicyPlayer, load_policy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, ckpt = load_policy(path, device)
        self.player = PolicyPlayer(net, device, greedy=True)
        self.key_spots = int(ckpt.get("key_spots", 0))  # the action space it was trained with
        # ... and the rhythm: it decided every decision_ticks x 50 ms in training
        self.decide_every = ckpt.get("args", {}).get("decision_ticks", 6) * 0.05
        self.finisher = False  # --finisher: a hard rule; the policies are to learn it (endgame starts in train_v7)
        self.data = crsim.GameData.load(os.path.join(ROOT, "data"))
        self.net, self.ckpt, self.device = net, ckpt, device
        self.search = None  # --search: lookahead (python/search.py) over the twin, the opponent's hand as it guesses

    def use_search(self, temperature):
        from search import SearchPlayer
        self.search = SearchPlayer(self.net, self.ckpt, self.device, temperature=temperature)

    def __call__(self, battle):
        finish = finishing_spell(battle, BLUE, self.data) if self.finisher else None
        if finish is not None:  # a tower a spell in hand surely destroys: the policy never learned to take it
            return finish
        if self.search is not None:
            action = self.search.decide(battle, BLUE)
        else:
            action = int(self.player(crsim.rl.encode(battle, BLUE, key_spots=self.key_spots)))
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


def draw(frame, match, detections, status, last_move, events):
    """Left: the screen with what the detector sees (thin boxes, name on top) and where the simulator
    has its units (rings with name and HP bar). Right: status and the latest events."""
    img = frame.copy()
    L = Layout.for_frame(img)
    for d in detections:
        x0, y0, x1, y1 = map(int, d.box)
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS.get(d.team, COLORS["any"]), 2)
        label(img, NAMES.get(d.cls, d.cls), x0, y0 - 6, COLORS.get(d.team, COLORS["any"]), 0.75)
    if match.battle is not None:
        for e in match.battle.entities:
            if e.is_building:
                continue
            x, y = (int(v) for v in L.tile_to_px(e.pos.x, e.pos.y))
            color = COLORS["blue" if e.team == BLUE else "red"]
            r = L.s(26)
            cv2.circle(img, (x, y), r, (255, 255, 255), 6)
            cv2.circle(img, (x, y), r, color, 3 if e.deploy_ms_left <= 0 else 1)
            frac = max(0.0, e.hp / max(1, e.max_hp))
            cv2.rectangle(img, (x - r, y + r + 4), (x + r, y + r + 12), (40, 40, 40), -1)
            cv2.rectangle(img, (x - r, y + r + 4), (x - r + int(2 * r * frac), y + r + 12), color, -1)
            label(img, "sim " + NAMES.get(e.def_id, e.def_id), x - r, y + r + 40, color, 0.75)
        for p in match.battle.projectiles:
            if p.is_spell and p.delay_ms <= 0:
                x, y = (int(v) for v in L.tile_to_px(p.pos.x, p.pos.y))
                cv2.drawMarker(img, (x, y), COLORS["blue" if p.team == BLUE else "red"], cv2.MARKER_STAR, L.s(50), 4)
    if last_move is not None:
        x, y = (int(v) for v in L.tile_to_px(*last_move[1]))
        cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, L.s(60), 5)
        label(img, "bot: " + last_move[0], x + 20, y - 20, (0, 255, 255))

    panel = np.full((img.shape[0], PANEL_W, 3), 32, np.uint8)
    lines = status + ["", "box  = seen by the detector", "ring = unit in the simulator", "+    = bot's last move", "",
                      "latest:"] + events[-24:]
    for k, line in enumerate(lines):
        color = (255, 255, 255)
        if " red " in f" {line} ":
            color = COLORS["red"]
        elif " blue " in f" {line} " or "bot" in line:
            color = (255, 190, 120)
        cv2.putText(panel, line, (16, 50 + 44 * k), FONT, 0.8, color, 2, cv2.LINE_AA)
    out = np.hstack([img, panel])
    h = 900
    return cv2.resize(out, (int(out.shape[1] * h / out.shape[0]), h), interpolation=cv2.INTER_AREA)


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
    p.add_argument("--search", action="store_true",
                   help="lookahead before a move (python/search.py): the policy's best few moves played forward in "
                        "copies of the twin; ~0.12 s a search. In the simulator it beat the plain policy 87%%")
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
    if isinstance(brain, PolicyBrain) and args.search:
        brain.use_search(args.temperature if args.temperature > 0 else 0.5)
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
    print(f"brain: {args.brain}{' + lookahead' if args.search else ''}, {'PLAYING' if playing else 'watching only (no --act)'}; P = pause, Q = quit")
    try:
        while True:
            try:
                frame = screen.screenshot()
                warned = False
            except RuntimeError as err:  # LDPlayer minimised: slow ADB screenshots until it is back
                if not warned:
                    print(f"{err}; using ADB screenshots (slow) until the window is back")
                    warned = True
                frame = device.screenshot()
            try:
                frame = full_size(frame)
                t = time.monotonic()
                try:
                    seen = reader.read(frame)
                except ValueError:
                    continue
                detections = detector.detect(frame) if detector else []
                match.update(t, seen, detections, hp_reader.bars(frame, detections) if detector else None)
                frames += 1
                if t - fps_t0 > 2.0:
                    fps, frames, fps_t0 = frames / (t - fps_t0), 0, t

                g = match.game_time(t)
                battle = match.battle
                if battle is not None and brain is not None and t >= next_decision and g is not None and g >= 0:
                    next_decision = t + args.decide_every
                    move = brain(battle)
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
                                match.own_play(t, card, pos)
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
                try:  # the preview is only for us to watch: a drawing error must not stop the bot
                    cv2.imshow("cr-neiro live", draw(frame, match, detections, status, last_move, events))
                except Exception:
                    print(f"preview error, skipped:\n{traceback.format_exc()}")
                key = cv2.waitKey(1) & 0xFF
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
