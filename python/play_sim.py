"""Play the bot yourself in the simulator: a window laid out like the game, you at the bottom (blue), the bot at
the top (red), both with the starter deck.

    scripts\\play_sim.bat [--bot runs\\h1\\final_cycle11.pt] [--seed 7]

Pick a card (click it, or keys 1-4), then click a tile; or drag the card onto the arena. Right click or Esc:
cancel. Space: start / pause. N: the next game once this one is over.
Only what the game would show you is drawn: the bot's hand, elixir and blueprints are hidden, and its cards
appear in the list on the right when its troops (or spells) show up. The bot plays as live: the net's moves
sampled at temperature 0.5, at the rhythm it was trained at, but it sees the battle exactly (no screen reading,
no delays): what it loses here is the net's own play.

Every game is saved to raw/sim_games/<date_time>_seed<seed>.json: the seed and every play with its tick, so
replay() rebuilds the battle exactly under the same simulator ("sim": its fingerprint; --check replays them all).
Those games are exact data of the user's moves (for a clone of him) and the bot's score against him.
Card pictures come from raw/perception/card_art (tools/card_art.py; outside git: cut from game art). Without them
the cards are drawn as coloured tiles with names.
"""
import argparse
import ctypes
import ctypes.wintypes
import datetime
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import crsim  # noqa: E402

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
DECK = "knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer".split(",")
DEFAULT_BOT = "runs/h1/final_cycle11.pt"
ART_DIR = os.path.join(ROOT, "raw", "perception", "card_art")
GAMES_DIR = os.path.join(ROOT, "raw", "sim_games")

NAMES = {"knight": "Рыцарь", "archers": "Лучницы", "minions": "Миньоны", "arrows": "Стрелы",
         "fireball": "Огненный шар", "giant": "Гигант", "mini_pekka": "Мини-пекка", "musketeer": "Мушкетёр"}
TINTS = {"knight": (190, 160, 90), "archers": (215, 105, 165), "minions": (85, 105, 200), "arrows": (150, 185, 215),
         "fireball": (235, 125, 40), "giant": (205, 135, 75), "mini_pekka": (85, 115, 150),
         "musketeer": (165, 115, 195)}
REASONS = {"not enough elixir": "мало эликсира", "cannot deploy there": "сюда нельзя",
           "slot is empty": "карта ещё не пришла", "match is finished": "матч окончен"}

TEAM = {BLUE: (60, 130, 240), RED: (230, 60, 60)}
TEAM_DARK = {BLUE: (32, 66, 132), RED: (124, 32, 32)}
GRASS = ((104, 168, 78), (96, 158, 72))
RIVER = (70, 140, 215)
BRIDGE = (156, 116, 74)
BG = (24, 26, 32)
PANEL = (36, 39, 48)
TEXT = (236, 236, 242)
DIM = (150, 152, 166)
ELIXIR = (214, 64, 230)
GOLD = (255, 212, 84)


def sim_version():
    """A fingerprint of the simulator a game was played with (its data files and the built module): a saved game
    replays exactly only under the same one."""
    import hashlib
    h = hashlib.sha1()
    for name in ("cards.json", "overrides.json", "rules.json", "towers.json", "arena.json"):
        with open(os.path.join(ROOT, "data", name), "rb") as f:
            h.update(f.read())
    with open(crsim.__file__, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def side_name(team):
    return "blue" if team == BLUE else "red"


def clock_text(ms):
    s = (max(0, int(ms)) + 999) // 1000
    return f"{s // 60}:{s % 60:02d}"


def bot_label(path):
    m = re.search(r"cycle(\d+)", os.path.basename(path))
    return f"цикл {m.group(1)}" if m else os.path.splitext(os.path.basename(path))[0]


def towers(battle):
    """Every tower still standing: [side, king/princess, x, hp], for the saved result and the replay check."""
    return sorted([side_name(e.team), "king" if e.kind == crsim.EntityKind.KING_TOWER else "princess",
                   round(e.pos.x, 2), int(e.hp)] for e in battle.entities if e.is_building)


class Bot:
    """A trained net as one side, as the live bot plays it: moves sampled at temperature 0.5, every
    decision_ticks ticks (its training rhythm)."""

    def __init__(self, path, temperature=0.5):
        import torch
        from model import PolicyPlayer, load_policy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, ckpt = load_policy(path, device)
        self.player = PolicyPlayer(net, device, temperature=temperature)
        self.key_spots = int(ckpt.get("key_spots", 0))
        self.decision_ticks = int(ckpt.get("args", {}).get("decision_ticks") or 4)

    def decide(self, battle, team):
        """(hand slot, position) or None to wait. The same move as crsim.rl.apply_action, made here so it can be
        recorded."""
        action = self.player(crsim.rl.encode(battle, team, key_spots=self.key_spots))
        if action <= 0:
            return None
        slot, cell = divmod(action - 1, crsim.rl.CELLS)
        return slot, crsim.rl.cell_position(battle, team, cell)


def replay(record, data):
    """The battle of a saved game rebuilt: the same seed and decks, every play at its tick in the order made."""
    tick_ms = data.rules.tick_ms
    b = crsim.Battle(data, record["decks"]["blue"], record["decks"]["red"], record["seed"])
    plays, k, end = record["plays"], 0, record["result"]["time_ms"]
    while not b.finished and b.time_ms < end:
        tick = b.time_ms // tick_ms
        while k < len(plays) and plays[k][0] <= tick:
            _, side, card, x, y = plays[k]
            team = BLUE if side == "blue" else RED
            hand = list(b.player(team).hand)
            if card not in hand or not b.play(team, hand.index(card), crsim.Vec2(x, y)):
                raise ValueError(f"play {k} ({side} {card} at tick {tick}) does not replay")
            k += 1
        b.step()
    return b


def saved_games(bot=None):
    games = []
    for path in sorted(glob.glob(os.path.join(GAMES_DIR, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except (OSError, ValueError):
            continue
        if rec.get("result") and (bot is None or rec.get("bot") == bot):
            games.append((path, rec))
    return games


def check_saved(data):
    bad, other = 0, 0
    games, now = saved_games(), sim_version()
    for path, rec in games:
        if rec.get("sim") != now:
            other += 1
            print(f"{os.path.basename(path)}: played with another simulator version ({rec.get('sim', 'not noted')})")
            continue
        try:
            b = replay(rec, data)
            same = towers(b) == rec["result"]["towers"] and b.time_ms == rec["result"]["time_ms"]
        except ValueError as err:
            same, b = False, None
            print(f"{os.path.basename(path)}: {err}")
        bad += not same
        print(f"{os.path.basename(path)}: {'replays exactly' if same else 'DIFFERS'}")
    print(f"{len(games) - other - bad}/{len(games) - other} saved games of this simulator replay exactly"
          + (f"; {other} from other versions" if other else ""))
    return bad == 0


def font(size, bold=False):
    for name in ("segoeui", "arial", "consolas"):
        path = pygame.font.match_font(name, bold=bold)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.Font(None, size)


def draw_crown(surf, cx, cy, w, color):
    h = w * 0.72
    pts = [(cx - w / 2, cy + h / 2), (cx - w / 2, cy - h / 4), (cx - w / 4, cy + h / 10), (cx, cy - h / 2),
           (cx + w / 4, cy + h / 10), (cx + w / 2, cy - h / 4), (cx + w / 2, cy + h / 2)]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, (40, 30, 10), pts, 1)


class Game:
    def __init__(self, args):
        self.data = crsim.GameData.load(os.path.join(ROOT, "data"))
        self.rules, self.arena = self.data.rules, self.data.arena
        self.tick_ms = self.rules.tick_ms
        self.bot_path = args.bot.replace("\\", "/")
        bot_file = args.bot if os.path.isabs(args.bot) else os.path.join(ROOT, args.bot)
        if not os.path.exists(bot_file):
            sys.exit(f"нет сетки {bot_file}")
        self.bot = Bot(bot_file)
        self.bot_name = bot_label(self.bot_path)
        self.seed = args.seed if args.seed is not None else int(time.time()) % 1000000
        self.sim = sim_version()
        self.init_screen(args.tile)
        self.load_art()
        self.score = self.tally()
        warm = crsim.Battle(self.data, DECK, DECK, 0)  # the net's first move takes a while on the GPU: not in play
        self.bot.decide(warm, RED)
        self.new_match()

    # --- window -------------------------------------------------------------------------------------------
    def init_screen(self, tile):
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # real pixels: the size below is worked out from them
        except (AttributeError, OSError):
            pass
        pygame.init()
        if not tile:
            avail = 0
            try:  # the screen's height without the taskbar, less the window's title bar
                area = ctypes.wintypes.RECT()
                if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(area), 0):  # SPI_GETWORKAREA
                    avail = area.bottom - area.top - 40
            except (AttributeError, OSError):
                pass
            if avail <= 0:
                avail = (pygame.display.Info().current_h or 1000) - 90
            tile = int(max(14, min(36, (avail - 92) / 36.7)))
        self.T = T = tile
        self.M, self.TOP = 10, 46
        self.AW, self.AH = int(self.arena.width * T), int(self.arena.height * T)
        self.ax, self.ay = self.M, self.M + self.TOP
        self.card_w = int((self.AW - 40) / 4.7)
        self.card_h = int(self.card_w * 215 / 175)
        self.next_w, self.next_h = int(self.card_w * 0.66), int(self.card_h * 0.66)
        self.hand_y = self.ay + self.AH + 12
        self.bar_y = self.hand_y + self.card_h + 8
        self.px, self.PW = self.ax + self.AW + 16, max(250, int(T * 9.5))
        size = (self.px + self.PW + self.M, self.bar_y + 24 + self.M)
        pygame.display.set_caption("Против бота (симулятор)")
        self.screen = pygame.display.set_mode(size)
        self.f_small, self.f_text = font(max(11, T // 2)), font(max(13, int(T * 0.62)))
        self.f_bold = font(max(14, int(T * 0.7)), bold=True)
        self.f_big = font(max(20, int(T * 1.25)), bold=True)
        self.f_huge = font(max(28, int(T * 1.9)), bold=True)
        self.bg = self.arena_surface()

    def arena_surface(self):
        a, T = self.arena, self.T
        s = pygame.Surface((self.AW, self.AH))
        for ty in range(int(a.height)):
            for tx in range(int(a.width)):
                s.fill(GRASS[(tx + ty) % 2], (tx * T, ty * T, T, T))
        top = (a.height - a.river_max_y) * T
        s.fill(RIVER, (0, top, self.AW, (a.river_max_y - a.river_min_y) * T))
        for bx in a.bridge_x:
            s.fill(BRIDGE, ((bx - a.bridge_half_width) * T, top - 3, a.bridge_half_width * 2 * T,
                            (a.river_max_y - a.river_min_y) * T + 6))
        return s

    def load_art(self):
        self.art, self.cards, self.greys, self.minis, self.nexts, self.tokens = {}, {}, {}, {}, {}, {}
        missing = []
        for cid in DECK + ["_empty"]:
            path = os.path.join(ART_DIR, f"{cid}.png")
            if os.path.exists(path):
                img = pygame.image.load(path).convert()
            else:
                missing.append(cid)
                img = self.drawn_card(cid)
            self.art[cid] = img
            self.cards[cid] = pygame.transform.smoothscale(img, (self.card_w, self.card_h))
            grey = pygame.transform.grayscale(self.cards[cid])
            grey.fill((120, 120, 120), special_flags=pygame.BLEND_RGB_MULT)
            self.greys[cid] = grey
            self.nexts[cid] = pygame.transform.smoothscale(img, (self.next_w, self.next_h))
            self.minis[cid] = pygame.transform.smoothscale(img, (int(self.T * 1.4), int(self.T * 1.72)))
        if missing:
            print(f"нет картинок {missing} (tools/card_art.py): рисую карты цветом")

    def drawn_card(self, cid):
        s = pygame.Surface((175, 215))
        s.fill(TINTS.get(cid, (60, 60, 70)) if cid != "_empty" else (40, 52, 90))
        pygame.draw.rect(s, (25, 25, 30), s.get_rect(), 6, border_radius=10)
        if cid != "_empty":
            f = font(26, bold=True)
            for i, word in enumerate(NAMES[cid].split()):
                t = f.render(word, True, (255, 255, 255))
                s.blit(t, t.get_rect(center=(88, 70 + i * 30)))
            pygame.draw.circle(s, ELIXIR, (88, 180), 22)
            t = font(30, bold=True).render(str(self.data.card(cid).elixir), True, (255, 255, 255))
            s.blit(t, t.get_rect(center=(88, 180)))
        return s

    def token(self, cid, d):
        """The card's picture in a circle, d pixels across: a unit on the field."""
        key = (cid, d)
        if key not in self.tokens:
            img = self.art.get(cid)
            out = pygame.Surface((d, d), pygame.SRCALPHA)
            if img is None:
                pygame.draw.circle(out, (200, 200, 200), (d // 2, d // 2), d // 2)
                t = self.f_small.render(cid[:3], True, (20, 20, 20))
                out.blit(t, t.get_rect(center=(d // 2, d // 2)))
            else:
                w, h = img.get_size()
                face = img.subsurface(pygame.Rect(int(w * 0.12), int(h * 0.07), int(w * 0.76), int(h * 0.52)))
                pygame.draw.circle(out, (255, 255, 255, 255), (d // 2, d // 2), d // 2)
                out.blit(pygame.transform.smoothscale(face, (d, d)), (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.tokens[key] = out
        return self.tokens[key]

    # --- games --------------------------------------------------------------------------------------------
    def tally(self):
        won = lost = drawn = 0
        for _, rec in saved_games(self.bot_path):
            r = rec["result"]
            if r.get("finished"):
                won += r["winner"] == "blue"
                lost += r["winner"] == "red"
                drawn += r["winner"] is None
        return won, lost, drawn

    def new_match(self):
        self.battle = crsim.Battle(self.data, DECK, DECK, self.seed)
        self.started = self.paused = self.saved = False
        self.acc = 0.0
        self.selected = self.drag = None
        self.effects, self.spells, self.snapshot = [], {}, {}
        self.status, self.status_until = "", 0
        self.zone_key, self.zone = None, None
        self.saved_as = ""
        self.tower_spots = [(e.team, e.kind, e.pos.x, e.pos.y, e.footprint) for e in self.battle.entities
                            if e.is_building]
        now = datetime.datetime.now()
        self.file = os.path.join(GAMES_DIR, f"{now:%Y%m%d_%H%M%S}_seed{self.seed}.json")
        self.record = {"version": 1, "sim": self.sim, "seed": self.seed, "decks": {"blue": DECK, "red": DECK},
                       "bot": self.bot_path, "bot_temperature": 0.5, "decision_ticks": self.bot.decision_ticks, "tick_ms": self.tick_ms,
                       "started": now.isoformat(timespec="seconds"), "plays": [], "result": None}

    def play(self, team, slot, pos):
        card = self.battle.player(team).hand[slot]
        if not self.battle.play(team, slot, pos):
            return False
        self.record["plays"].append([self.battle.time_ms // self.tick_ms, side_name(team), card,
                                     float(pos.x), float(pos.y)])
        return True

    def tick(self):
        b = self.battle
        if (b.time_ms // self.tick_ms) % self.bot.decision_ticks == 0:
            move = self.bot.decide(b, RED)
            if move is not None:
                self.play(RED, *move)
        self.snapshot = {e.id: (e.pos.x, e.pos.y, e.team, e.is_building, e.footprint) for e in b.entities}
        b.step()
        now = b.time_ms
        for ev in b.events:
            spot = self.snapshot.get(ev.entity)
            if ev.type == crsim.EventType.DAMAGE and spot:  # a hit: a quick flash on whoever took it
                self.effects.append([now, 180, spot[0], spot[1], 0.9 if spot[3] else 0.42, (255, 250, 210)])
            elif ev.type == crsim.EventType.DEATH and spot and not spot[3]:
                self.effects.append([now, 320, spot[0], spot[1], 0.55, (225, 225, 225)])
            elif ev.type == crsim.EventType.TOWER_DESTROYED and spot:
                self.effects.append([now, 900, spot[0], spot[1], 3.0, (255, 150, 40)])
                self.say("Башня бота разрушена!" if spot[2] == RED else "Бот разрушил твою башню", 3000)
        b.clear_events()
        spells = {}
        for p in b.projectiles:
            if p.is_spell:
                spells[(p.card_index, int(p.team), p.aim.x, p.aim.y)] = (p.aim.x, p.aim.y, p.splash_radius)
        for key, (x, y, r) in self.spells.items():
            if key not in spells:  # it landed
                fire = self.data.card_ids[key[0]] == "fireball"
                self.effects.append([now, 380, x, y, r, (255, 140, 30) if fire else (235, 240, 255)])
        self.spells = spells
        if b.finished:
            self.finish()

    def update(self, dt_ms):
        if not self.started or self.paused or self.battle.finished:
            self.acc = 0.0
            return
        self.acc = min(self.acc + dt_ms, 250.0)  # after a stall the clock waits rather than racing
        while self.acc >= self.tick_ms and not self.battle.finished:
            self.tick()
            self.acc -= self.tick_ms

    def finish(self):
        if self.saved:
            return
        b = self.battle
        if not b.finished and not self.record["plays"]:
            return  # nothing played: nothing worth keeping
        self.record["result"] = {
            "finished": bool(b.finished),
            "winner": None if b.winner is None else side_name(b.winner),
            "crowns": [b.player(BLUE).crowns, b.player(RED).crowns],
            "time_ms": b.time_ms,
            "towers": towers(b),
        }
        os.makedirs(GAMES_DIR, exist_ok=True)
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(self.record, f, ensure_ascii=False)
        self.saved = True
        self.saved_as = os.path.relpath(self.file, ROOT)
        self.score = self.tally()

    def next_match(self):
        if self.started and not self.battle.finished:
            self.finish()  # left early: kept as unfinished
        self.seed += 1
        self.new_match()

    # --- input --------------------------------------------------------------------------------------------
    def card_rect(self, slot):
        x0 = self.M + self.next_w + 14
        return pygame.Rect(x0 + slot * (self.card_w + 6), self.hand_y, self.card_w, self.card_h)

    def card_at(self, pos):
        for slot in range(4):
            if self.card_rect(slot).inflate(0, 24).move(0, -12).collidepoint(pos):
                return slot
        return None

    def over_arena(self, pos):
        return self.ax <= pos[0] < self.ax + self.AW and self.ay <= pos[1] < self.ay + self.AH

    def tile_at(self, pos):
        x = (pos[0] - self.ax) / self.T
        y = self.arena.height - (pos[1] - self.ay) / self.T
        return crsim.Vec2(min(int(x), int(self.arena.width) - 1) + 0.5, min(int(y), int(self.arena.height) - 1) + 0.5)

    def to_screen(self, x, y):
        return self.ax + x * self.T, self.ay + (self.arena.height - y) * self.T

    def say(self, text, ms=1800):
        self.status, self.status_until = text, pygame.time.get_ticks() + ms

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_SPACE and not self.battle.finished:
                if not self.started:
                    self.started = True
                else:
                    self.paused = not self.paused
            elif pygame.K_1 <= ev.key <= pygame.K_4:
                slot = ev.key - pygame.K_1
                self.selected = None if self.selected == slot else slot
            elif ev.key == pygame.K_ESCAPE:
                self.selected = self.drag = None
            elif ev.key == pygame.K_n and (self.battle.finished or not self.started):
                self.next_match()
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 3:
                self.selected = self.drag = None
            elif ev.button == 1:
                slot = self.card_at(ev.pos)
                if slot is not None:
                    self.drag = [slot, ev.pos, False, self.selected == slot]  # slot, from, moved, was selected
                    self.selected = slot
                elif self.selected is not None and self.over_arena(ev.pos):
                    self.place(ev.pos)
        elif ev.type == pygame.MOUSEMOTION and self.drag and not self.drag[2]:
            fx, fy = self.drag[1]
            if abs(ev.pos[0] - fx) + abs(ev.pos[1] - fy) > 10:
                self.drag[2] = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.drag:
            slot, _, moved, was = self.drag
            self.drag = None
            if moved and self.over_arena(ev.pos):
                self.place(ev.pos)
            elif not moved and was:
                self.selected = None  # a second click on the chosen card puts it back

    def place(self, spos):
        if self.battle.finished:
            return
        if not self.started or self.paused:
            self.say("пробел - " + ("начать" if not self.started else "продолжить"))
            return
        pos = self.tile_at(spos)
        reason = self.battle.check_play(BLUE, self.selected, pos)
        if reason:
            self.say(REASONS.get(reason, reason))
            return
        self.play(BLUE, self.selected, pos)
        self.selected = None

    # --- drawing ------------------------------------------------------------------------------------------
    def text(self, f, s, color, pos, anchor="topleft"):
        t = f.render(s, True, color)
        r = t.get_rect(**{anchor: (int(pos[0]), int(pos[1]))})
        self.screen.blit(t, r)
        return r

    def draw(self):
        s = self.screen
        s.fill(BG)
        self.draw_top()
        s.set_clip(pygame.Rect(self.ax, self.ay, self.AW, self.AH))
        s.blit(self.bg, (self.ax, self.ay))
        hand = self.battle.player(BLUE).hand
        card = None
        if self.selected is not None and hand[self.selected]:
            card = self.data.card(hand[self.selected])
            if not card.is_spell:
                s.blit(self.deploy_zone(), (self.ax, self.ay))
        self.draw_rubble()
        self.draw_field()
        if not self.battle.finished:  # (they would stay frozen on the last frame)
            self.draw_effects()
        if card is not None:
            self.draw_ghost(card)
        self.draw_overlay()
        s.set_clip(None)
        self.draw_hand()
        self.draw_panel()
        pygame.display.flip()

    def draw_top(self):
        b, y = self.battle, self.M + self.TOP // 2
        x = self.ax + 4
        for team, who in ((BLUE, "Ты"), (RED, "Бот")):
            draw_crown(self.screen, x + 11, y, 22, TEAM[team])
            r = self.text(self.f_bold, f"{who} {b.player(team).crowns}", TEXT, (x + 28, y), "midleft")
            x = r.right + 18
        if b.phase == crsim.Phase.OVERTIME:
            left, extra = self.rules.regulation_ms + self.rules.overtime_ms - b.time_ms, "овертайм"
        else:
            left, extra = self.rules.regulation_ms - b.time_ms, ""
        if b.elixir_multiplier > 1 and not b.finished:
            extra = (extra + " " if extra else "") + f"x{b.elixir_multiplier:g} эликсир"
        r = self.text(self.f_big, clock_text(left), TEXT, (self.ax + self.AW - 4, y), "midright")
        if extra:
            self.text(self.f_text, extra, ELIXIR, (r.left - 10, y), "midright")
        if self.status and pygame.time.get_ticks() < self.status_until:
            self.text(self.f_bold, self.status, GOLD, (self.ax + self.AW / 2, self.ay + 14), "midtop")

    def deploy_zone(self):
        """Tiles where troops may not go, shaded red as in the game (it grows when a tower falls)."""
        key = tuple(sorted(e.id for e in self.battle.entities if e.is_building))
        if key != self.zone_key:
            T = self.T
            self.zone = pygame.Surface((self.AW, self.AH), pygame.SRCALPHA)
            for ty in range(int(self.arena.height)):
                for tx in range(int(self.arena.width)):
                    if not self.battle.can_deploy_at(BLUE, crsim.Vec2(tx + 0.5, ty + 0.5)):
                        sy = (self.arena.height - ty - 1) * T
                        self.zone.fill((200, 30, 30, 70), (tx * T, sy, T, T))
            self.zone_key = key
        return self.zone

    def draw_rubble(self):
        alive = {(e.team, e.kind, round(e.pos.x, 2)) for e in self.battle.entities if e.is_building}
        for team, kind, x, y, fp in self.tower_spots:
            if (team, kind, round(x, 2)) not in alive:
                cx, cy = self.to_screen(x, y)
                size = int(fp * 2 * self.T)
                r = pygame.Rect(0, 0, size, size)
                r.center = (int(cx), int(cy))
                pygame.draw.rect(self.screen, (92, 88, 82), r, border_radius=6)
                pygame.draw.line(self.screen, (60, 56, 52), r.topleft, r.bottomright, 3)
                pygame.draw.line(self.screen, (60, 56, 52), r.topright, r.bottomleft, 3)

    def draw_field(self):
        b, s, T = self.battle, self.screen, self.T
        for e in sorted(b.entities, key=lambda e: (e.air, -e.pos.y)):
            if e.is_building:
                self.draw_tower(e)
            else:
                self.draw_unit(e)
        for p in b.pending:  # your own blueprints only: the bot's are hidden from you, as in the game
            if p.team != BLUE or p.unit > 0:
                continue
            cid = self.data.card_ids[p.card_index]
            cx, cy = self.to_screen(p.pos.x, p.pos.y)
            card = self.data.card(cid)
            rad = int((card.spell.radius if card.is_spell else max(card.unit.radius, 0.5)) * T)
            pygame.draw.circle(s, (240, 240, 240), (int(cx), int(cy)), max(rad, 8), 1)
            self.text(self.f_small, f"{p.ms_left / 1000:.1f}", TEXT, (cx, cy), "center")
        for p in b.projectiles:
            if p.is_spell:
                if p.team == BLUE:  # where your spell lands (the bot's aim is not shown to you)
                    pygame.draw.circle(s, TEAM[BLUE], self.to_screen(p.aim.x, p.aim.y), int(p.splash_radius * T), 2)
                if p.delay_ms == 0:
                    fire = self.data.card_ids[p.card_index] == "fireball"
                    pygame.draw.circle(s, (255, 130, 30) if fire else (245, 245, 255),
                                       self.to_screen(p.pos.x, p.pos.y), 7 if fire else 4)
            else:
                pygame.draw.circle(s, (255, 235, 100), self.to_screen(p.pos.x, p.pos.y), 3)

    def hp_bar(self, cx, top, width, hp, max_hp, color):
        width = int(width)
        pygame.draw.rect(self.screen, (15, 15, 18), (int(cx - width / 2) - 1, int(top) - 1, width + 2, 6))
        pygame.draw.rect(self.screen, color, (int(cx - width / 2), int(top), int(width * max(0.0, hp / max_hp)), 4))

    def draw_tower(self, e):
        s = self.screen
        cx, cy = self.to_screen(e.pos.x, e.pos.y)
        size = int(e.footprint * 2 * self.T)
        r = pygame.Rect(0, 0, size, size)
        r.center = (int(cx), int(cy))
        king = e.kind == crsim.EntityKind.KING_TOWER
        pygame.draw.rect(s, (70, 72, 80), r.inflate(6, 6), border_radius=8)
        pygame.draw.rect(s, TEAM[e.team] if e.active else TEAM_DARK[e.team], r, border_radius=7)
        if king:
            draw_crown(s, cx, cy, size * 0.45, GOLD if e.active else (150, 135, 90))
            if not e.active:
                self.text(self.f_small, "zz", TEXT, (r.right - 4, r.top + 2), "topright")
        else:
            pygame.draw.circle(s, (235, 235, 240), r.center, max(3, size // 6))
        self.hp_bar(cx, r.top - 9, size, e.hp, e.max_hp, TEAM[e.team])
        self.text(self.f_small, str(e.hp), TEXT, (cx, r.top - 11), "midbottom")

    def draw_unit(self, e):
        s, T = self.screen, self.T
        cid = self.data.card_ids[e.card_index] if e.card_index >= 0 else e.def_id
        cx, cy = self.to_screen(e.pos.x, e.pos.y)
        d = int(max(20, min(46, 2.3 * e.radius * T)))
        if e.air:
            pygame.draw.ellipse(s, (45, 80, 45), (cx - d / 2, cy - d / 6 + 3, d, d / 3))
            cy -= T * 0.4
        tok = self.token(cid, d)
        rect = tok.get_rect(center=(int(cx), int(cy)))
        if e.deploy_ms_left > 0:
            faded = tok.copy()
            faded.set_alpha(120)
            s.blit(faded, rect)
            pygame.draw.circle(s, TEAM[e.team], rect.center, d // 2 + 1, 1)
            self.text(self.f_small, f"{e.deploy_ms_left / 1000:.1f}", TEXT, rect.center, "center")
            return
        s.blit(tok, rect)
        pygame.draw.circle(s, TEAM[e.team], rect.center, d // 2 + 1, 3)
        if e.hp < e.max_hp:
            self.hp_bar(cx, rect.top - 6, max(d, 18), e.hp, e.max_hp, TEAM[e.team])

    def draw_effects(self):
        now, keep = self.battle.time_ms, []
        for fx in self.effects:
            t0, ms, x, y, r, color = fx
            k = (now - t0) / ms
            if k >= 1.0:
                continue
            keep.append(fx)
            rad = max(4, int(r * self.T * (0.7 + 0.5 * k)))
            surf = pygame.Surface((2 * rad + 2, 2 * rad + 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, int(160 * (1 - k))), (rad + 1, rad + 1), rad)
            cx, cy = self.to_screen(x, y)
            self.screen.blit(surf, (int(cx) - rad - 1, int(cy) - rad - 1))
        self.effects = keep

    def draw_ghost(self, card):
        mouse = pygame.mouse.get_pos()
        if not self.over_arena(mouse):
            return
        pos = self.tile_at(mouse)
        ok = not self.battle.check_play(BLUE, self.selected, pos)
        cx, cy = self.to_screen(pos.x, pos.y)
        color = (255, 255, 255) if ok else (255, 70, 70)
        if card.is_spell:
            pygame.draw.circle(self.screen, color, (int(cx), int(cy)), int(card.spell.radius * self.T), 2)
        else:
            d = int(max(22, min(46, 2.3 * card.unit.radius * self.T)))
            tok = self.token(card.id, d).copy()
            tok.set_alpha(165 if ok else 90)
            self.screen.blit(tok, tok.get_rect(center=(int(cx), int(cy))))
            pygame.draw.circle(self.screen, color, (int(cx), int(cy)), d // 2 + 1, 2)

    def shade(self, alpha=150):
        veil = pygame.Surface((self.AW, self.AH), pygame.SRCALPHA)
        veil.fill((10, 12, 18, alpha))
        self.screen.blit(veil, (self.ax, self.ay))

    def draw_overlay(self):
        b, mid = self.battle, (self.ax + self.AW / 2, self.ay + self.AH / 2)
        if b.finished:
            self.shade(130)
            w = b.winner
            title, color = ("Победа!", GOLD) if w == BLUE else ("Поражение", (255, 90, 90)) if w == RED else ("Ничья", TEXT)
            self.text(self.f_huge, title, color, (mid[0], mid[1] - 40), "center")
            self.text(self.f_bold, f"{b.player(BLUE).crowns} : {b.player(RED).crowns}", TEXT, (mid[0], mid[1] + 10),
                      "center")
            self.text(self.f_text, "N - следующая игра", TEXT, (mid[0], mid[1] + 48), "center")
            if self.saved_as:
                self.text(self.f_small, "сохранено: " + self.saved_as, DIM, (mid[0], mid[1] + 80), "center")
        elif not self.started:
            self.shade()
            self.text(self.f_big, "Пробел - начать", TEXT, (mid[0], mid[1] - 60), "center")
            lines = ["Ты внизу (синие), бот сверху (красные).",
                     "Карта: клик по ней или 1-4, потом клик по полю.",
                     "Или перетащи карту на поле.",
                     "ПКМ или Esc - отменить выбор.",
                     "Пробел во время игры - пауза."]
            for i, line in enumerate(lines):
                self.text(self.f_text, line, TEXT, (mid[0], mid[1] - 10 + i * (self.T * 0.9)), "center")
        elif self.paused:
            self.shade(110)
            self.text(self.f_big, "Пауза", TEXT, (mid[0], mid[1] - 20), "center")
            self.text(self.f_text, "пробел - продолжить", TEXT, (mid[0], mid[1] + 24), "center")

    def draw_hand(self):
        s, p = self.screen, self.battle.player(BLUE)
        nx, ny = self.M, self.hand_y + self.card_h - self.next_h
        self.text(self.f_small, "Далее", DIM, (nx + self.next_w / 2, ny - 4), "midbottom")
        nxt = p.queue[0] if len(p.queue) else ""
        s.blit(self.nexts.get(nxt, self.nexts["_empty"]), (nx, ny))
        for slot in range(4):
            r = self.card_rect(slot)
            cid = p.hand[slot]
            if not cid:
                s.blit(self.cards["_empty"], r)
                continue
            card = self.data.card(cid)
            chosen = self.selected == slot
            if chosen:
                r = r.move(0, -12)
            s.blit(self.cards[cid] if p.elixir + 1e-9 >= card.elixir else self.greys[cid], r)
            if chosen:
                pygame.draw.rect(s, GOLD, r.inflate(4, 4), 3, border_radius=8)
            self.text(self.f_small, str(slot + 1), (255, 255, 255), (r.left + 5, r.top + 3))
        x0, w = self.card_rect(0).left, self.card_rect(3).right - self.card_rect(0).left
        y, h, full = self.bar_y, 20, self.rules.max_elixir
        pygame.draw.rect(s, (52, 30, 62), (x0, y, w, h), border_radius=6)
        pygame.draw.rect(s, ELIXIR, (x0, y, int(w * min(1.0, p.elixir / full)), h), border_radius=6)
        for i in range(1, int(full)):
            gx = x0 + w * i / full
            pygame.draw.line(s, (30, 18, 38), (gx, y + 2), (gx, y + h - 3), 2)
        drop = (x0 - 16, y + h // 2)
        pygame.draw.circle(s, ELIXIR, drop, 15)
        pygame.draw.circle(s, (255, 255, 255), drop, 15, 2)
        self.text(self.f_bold, str(int(p.elixir + 1e-9)), (255, 255, 255), drop, "center")

    def draw_panel(self):
        s, b = self.screen, self.battle
        x, y = self.px, self.M
        pygame.draw.rect(s, PANEL, (x - 8, self.M, self.PW, self.screen.get_height() - 2 * self.M), border_radius=10)
        y = self.text(self.f_bold, "Против бота", TEXT, (x, y + 6)).bottom + 2
        y = self.text(self.f_small, f"{self.bot_name} ({os.path.basename(self.bot_path)})", DIM, (x, y)).bottom + 6
        won, lost, drawn = self.score
        line = f"Счёт: ты {won} - {lost} бот" + (f", ничьих {drawn}" if drawn else "")
        y = self.text(self.f_text, line, GOLD, (x, y)).bottom + 18
        y = self.text(self.f_text, "Карты бота:", TEXT, (x, y)).bottom + 8
        mh = self.minis[DECK[0]].get_height()
        plays = [p for p in b.player(RED).plays if p.reveal_ms >= 0]
        ls = self.f_small.get_linesize()
        hy = self.screen.get_height() - self.M - 10 - 5 * ls  # the help lines below
        room = (hy - 10 - y) // (mh + 6)
        for p in reversed(plays[-max(1, room):]):
            cid = self.data.card_ids[p.card_index]
            s.blit(self.minis[cid], (x, y))
            self.text(self.f_text, NAMES.get(cid, cid), TEXT, (x + self.minis[cid].get_width() + 10, y + 4))
            if p.reveal_ms < self.rules.regulation_ms:
                when = clock_text(self.rules.regulation_ms - p.reveal_ms)
            else:
                when = "ОТ " + clock_text(self.rules.regulation_ms + self.rules.overtime_ms - p.reveal_ms)
            self.text(self.f_small, when, DIM, (x + self.minis[cid].get_width() + 10, y + mh - 6), "bottomleft")
            y += mh + 6
        for i, line in enumerate(["1-4 или клик: выбрать карту", "клик по полю / перетащить: поставить",
                                  "ПКМ / Esc: отмена", "пробел: старт / пауза", "N: следующая игра"]):
            self.text(self.f_small, line, DIM, (x, hy + i * ls))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bot", default=DEFAULT_BOT, help="the net the bot plays with")
    p.add_argument("--seed", type=int, default=None, help="the first game's seed (then +1 a game)")
    p.add_argument("--tile", type=int, default=0, help="pixels per tile (0: as large as the screen allows)")
    p.add_argument("--check", action="store_true", help="replay every saved game and check it ends the same")
    args = p.parse_args()
    if args.check:
        sys.exit(0 if check_saved(crsim.GameData.load(os.path.join(ROOT, "data"))) else 1)
    game = Game(args)
    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                if game.started and not game.battle.finished:
                    game.finish()  # left mid-game: kept as unfinished
                pygame.quit()
                return
            game.handle(ev)
        game.update(clock.tick(60))
        game.draw()


if __name__ == "__main__":
    main()
