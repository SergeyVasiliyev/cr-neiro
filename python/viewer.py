"""Debug viewer for the crsim battle simulator.

Run from the project root:
    .venv\\Scripts\\python.exe python\\viewer.py                          you (blue) vs heuristic bot
    .venv\\Scripts\\python.exe python\\viewer.py --blue runs\\x\\latest.pt   trained model vs heuristic bot
    .venv\\Scripts\\python.exe python\\viewer.py --blue heuristic --red random --blue-deck giant,knight,...

Each side is controlled by: human | noop | random | heuristic | path to a checkpoint (.pt).
"""
import argparse
import os
import sys

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import crsim  # noqa: E402  (built into python/ by scripts\build.bat)

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
DEFAULT_DECK = "knight,archers,minions,arrows,fireball,giant,mini_pekka,musketeer"  # starter deck

DECISION_TICKS = 6  # bots and models act every 0.3 s (--decision-ticks: use what the model was trained with)
TILE = 22  # pixels per arena tile
MARGIN = 10
PANEL_W = 320
LOG_LINES = 14

TEAM_COLOR = {BLUE: (66, 125, 230), RED: (222, 66, 66)}
TEAM_DARK = {BLUE: (36, 70, 140), RED: (130, 36, 36)}
TEAM_NAME = {BLUE: "синие", RED: "красные"}
GRASS = [(98, 162, 74), (90, 152, 68)]
RIVER = (64, 128, 205)
BRIDGE = (150, 110, 70)
BG = (28, 30, 36)
TEXT = (230, 230, 235)
DIM = (150, 150, 160)
ELIXIR = (200, 70, 220)


def fmt_clock(ms):
    ms = max(0, ms)
    return f"{ms // 60000}:{ms // 1000 % 60:02d}"


class ModelAgent:
    """Plays a side with a trained checkpoint."""

    def __init__(self, path, search=False):
        import torch  # only needed when a model is used
        from model import PolicyPlayer, load_policy

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net, ckpt = load_policy(path, device)
        # as in live play: moves sampled at temperature 0.5, on the key spots it was trained with
        self.player = PolicyPlayer(net, device, temperature=0.5)
        self.search = None
        if search:  # "search:<path>": lookahead before a move (python/search.py)
            from search import SearchPlayer
            self.search = SearchPlayer(net, ckpt, device, temperature=0.5)
        self.key_spots = int(ckpt.get("key_spots", 0))
        from model import deck_pool_of
        self.deck_pool = deck_pool_of(ckpt)
        self.decision_ticks = ckpt.get("args", {}).get("decision_ticks")
        self.name = ("модель+поиск " if search else "модель ") + os.path.basename(os.path.dirname(os.path.abspath(path)))

    # Agents decide first and play afterwards, so both sides decide on the same state.
    def decide(self, battle, team):
        if self.search is not None:
            action = self.search.decide(battle, team)
        else:
            action = self.player(crsim.rl.encode(battle, team, key_spots=self.key_spots, deck_pool=self.deck_pool))
        return lambda: crsim.rl.apply_action(battle, team, action)


class BotAgent:
    def __init__(self, kind, seed):
        self.bot = crsim.make_bot(kind, seed)
        self.name = "бот " + kind

    def decide(self, battle, team):
        move = self.bot.decide(battle, team)
        if move is None:
            return lambda: None
        return lambda: battle.play(team, move.hand_slot, move.pos)


def make_agent(spec, seed):
    if spec == "human":
        return None
    if spec in ("noop", "random", "heuristic"):
        return BotAgent(spec, seed)
    if os.path.exists(spec):
        return ModelAgent(spec)
    if spec.startswith("search:") and os.path.exists(spec[7:]):
        return ModelAgent(spec[7:], search=True)
    sys.exit(f"не понимаю {spec!r}: нужно human, noop, random, heuristic, путь к .pt или search:путь к .pt")


class Viewer:
    def __init__(self, args):
        self.data = crsim.GameData.load(args.data)
        self.decision_ticks = args.decision_ticks
        self.decks = {BLUE: args.blue_deck.split(","), RED: args.red_deck.split(",")}
        for team, deck in self.decks.items():
            if len(deck) != 8:
                sys.exit(f"{TEAM_NAME[team]}: в колоде должно быть 8 карт, а не {len(deck)}")
            unknown = [c for c in deck if c not in self.data.cards]
            if unknown:
                sys.exit(f"неизвестные карты: {unknown}. Есть: {sorted(self.data.cards)}")
        self.seed = args.seed
        self.agents = {BLUE: make_agent(args.blue, args.seed + 1), RED: make_agent(args.red, args.seed + 2)}
        for agent in self.agents.values():  # a model decides at the rhythm it was trained at
            if getattr(agent, "decision_ticks", None):
                self.decision_ticks = agent.decision_ticks
        self.agent_on = {team: agent is not None for team, agent in self.agents.items()}
        self.control = BLUE if self.agents[BLUE] is None or self.agents[RED] is not None else RED
        self.selected = None
        self.speed = 1.0
        self.paused = False
        self.show_targets = True
        self.status = ""

        arena = self.data.arena
        self.arena_w = int(arena.width * TILE)
        self.arena_h = int(arena.height * TILE)
        pygame.init()
        pygame.display.set_caption("crsim viewer")
        self.screen = pygame.display.set_mode((MARGIN * 3 + self.arena_w + PANEL_W, MARGIN * 2 + self.arena_h))
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 11)
        self.big = pygame.font.SysFont("consolas", 18, bold=True)
        self.hand_rects = {}
        self.new_match()

    # --- simulation -------------------------------------------------------

    def new_match(self):
        self.battle = crsim.Battle(self.data, self.decks[BLUE], self.decks[RED], self.seed)
        self.log = [f"матч, seed {self.seed}"]
        self.acc_ms = 0.0
        self.ticks = 0
        self.selected = None

    def tick(self):
        if self.ticks % self.decision_ticks == 0:
            moves = [self.agents[t].decide(self.battle, t) for t in (BLUE, RED) if self.agent_on[t]]
            for move in moves:
                move()
        self.ticks += 1
        self.battle.step()
        for ev in self.battle.events:
            self.log.append(f"{fmt_clock(ev.time_ms)} {TEAM_NAME[ev.team]:<7} {ev.type.name.lower()} {ev.info}")
        self.battle.clear_events()
        del self.log[:-LOG_LINES]

    def update(self, dt_ms):
        if self.paused or self.battle.finished:
            return
        tick_ms = self.data.rules.tick_ms
        self.acc_ms = min(self.acc_ms + dt_ms * self.speed, 2000 * self.speed)
        while self.acc_ms >= tick_ms and not self.battle.finished:
            self.tick()
            self.acc_ms -= tick_ms

    # --- coordinates ------------------------------------------------------

    def to_screen(self, pos):
        return (MARGIN + pos.x * TILE, MARGIN + (self.data.arena.height - pos.y) * TILE)

    def to_world(self, px, py):
        return ((px - MARGIN) / TILE, self.data.arena.height - (py - MARGIN) / TILE)

    # --- input ------------------------------------------------------------

    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if pygame.K_1 <= k <= pygame.K_4:
                slot = k - pygame.K_1
                self.selected = None if self.selected == slot else slot
            elif k == pygame.K_TAB:
                self.control = RED if self.control == BLUE else BLUE
                self.selected = None
            elif k == pygame.K_ESCAPE:
                self.selected = None
            elif k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k == pygame.K_RIGHT and self.paused:
                self.tick()
            elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.speed = min(64.0, self.speed * 2)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(0.125, self.speed / 2)
            elif k == pygame.K_b and self.agents[BLUE]:
                self.agent_on[BLUE] = not self.agent_on[BLUE]
            elif k == pygame.K_r and self.agents[RED]:
                self.agent_on[RED] = not self.agent_on[RED]
            elif k == pygame.K_t:
                self.show_targets = not self.show_targets
            elif k == pygame.K_n:
                self.seed += 1
                self.new_match()
            elif k == pygame.K_BACKSPACE:
                self.new_match()
        elif ev.type == pygame.MOUSEBUTTONDOWN:
            if ev.button == 3:
                self.selected = None
            elif ev.button == 1:
                self.click(ev.pos)

    def click(self, mouse):
        for (team, slot), rect in self.hand_rects.items():
            if rect.collidepoint(mouse):
                self.control = team
                self.selected = None if self.selected == slot else slot
                return
        if self.selected is None:
            return
        x, y = self.to_world(*mouse)
        x, y = int(x) + 0.5, int(y) + 0.5  # cards snap to the centre of a tile, like in the game
        pos = crsim.Vec2(x, y)
        reason = self.battle.check_play(self.control, self.selected, pos)
        if reason:
            self.status = f"нельзя: {reason}"
            return
        self.battle.play(self.control, self.selected, pos)
        self.status = ""
        self.selected = None

    # --- drawing ----------------------------------------------------------

    def draw(self):
        self.screen.fill(BG)
        self.draw_arena()
        if self.selected is not None:
            self.draw_deploy_zone()
        self.draw_entities()
        self.draw_panel()
        pygame.display.flip()

    def draw_arena(self):
        a = self.data.arena
        for ty in range(int(a.height)):
            for tx in range(int(a.width)):
                rect = (MARGIN + tx * TILE, MARGIN + ty * TILE, TILE, TILE)
                pygame.draw.rect(self.screen, GRASS[(tx + ty) % 2], rect)
        river_top = MARGIN + (a.height - a.river_max_y) * TILE
        river_h = (a.river_max_y - a.river_min_y) * TILE
        pygame.draw.rect(self.screen, RIVER, (MARGIN, river_top, self.arena_w, river_h))
        for bx in a.bridge_x:
            left = MARGIN + (bx - a.bridge_half_width) * TILE
            pygame.draw.rect(self.screen, BRIDGE, (left, river_top - 3, a.bridge_half_width * 2 * TILE, river_h + 6))

    def draw_deploy_zone(self):
        card_id = self.battle.player(self.control).hand[self.selected]
        if not card_id:
            return
        card = self.data.card(card_id)
        spell = card.is_spell
        a = self.data.arena
        overlay = pygame.Surface((self.arena_w, self.arena_h), pygame.SRCALPHA)
        step = 0.5
        y = step / 2
        while y < a.height:
            x = step / 2
            while x < a.width:
                pos = crsim.Vec2(x, y)
                ok = self.battle.can_cast_at(pos) if spell else self.battle.can_deploy_at(self.control, pos)
                if ok:
                    px, py = self.to_screen(crsim.Vec2(x - step / 2, y + step / 2))
                    overlay.fill((255, 255, 255, 45), (px - MARGIN, py - MARGIN, step * TILE, step * TILE))
                x += step
            y += step
        self.screen.blit(overlay, (MARGIN, MARGIN))
        if spell:
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(self.screen, (255, 255, 255), (mx, my), int(card.spell.radius * TILE), 1)

    def hp_bar(self, cx, top, width, hp, max_hp, color):
        frac = max(0.0, hp / max_hp)
        pygame.draw.rect(self.screen, (20, 20, 20), (cx - width / 2, top, width, 4))
        pygame.draw.rect(self.screen, color, (cx - width / 2, top, width * frac, 4))

    def draw_entities(self):
        entities = self.battle.entities
        by_id = {e.id: e for e in entities}

        if self.show_targets:
            for e in entities:
                t = by_id.get(e.target)
                if t is not None and not e.is_building and e.deploy_ms_left == 0:
                    width = 2 if e.locked else 1
                    pygame.draw.line(self.screen, TEAM_DARK[e.team], self.to_screen(e.pos), self.to_screen(t.pos), width)

        for e in sorted(entities, key=lambda e: (e.air, -e.pos.y)):
            cx, cy = self.to_screen(e.pos)
            if e.is_building:
                self.draw_tower(e, cx, cy)
            else:
                self.draw_troop(e, cx, cy)

        # Blueprints (played, not on the field yet) are visible only to the side that played them.
        for p in self.battle.pending:
            if p.team != self.control or p.unit > 0:
                continue
            card = self.data.card(self.data.card_ids[p.card_index])
            cx, cy = self.to_screen(p.pos)
            radius = card.spell.radius if card.is_spell else max(card.unit.radius, 0.4)
            pygame.draw.circle(self.screen, (235, 235, 235), (cx, cy), int(radius * TILE), 1)
            self.blit_center(self.small, f"{card.id[:3]} {p.ms_left / 1000:.1f}", (cx, cy), (235, 235, 235))

        for p in self.battle.projectiles:
            if p.is_spell:  # a spell in flight: the ball and the circle where it lands
                pygame.draw.circle(self.screen, TEAM_COLOR[p.team], self.to_screen(p.aim), int(p.splash_radius * TILE), 1)
                if p.delay_ms == 0:
                    pygame.draw.circle(self.screen, (255, 120, 30), self.to_screen(p.pos), 6)
                continue
            color = (255, 150, 40) if p.splash_radius > 0 else (255, 235, 90)
            pygame.draw.circle(self.screen, color, self.to_screen(p.pos), 3)

    def draw_tower(self, e, cx, cy):
        size = e.footprint * 2 * TILE
        rect = pygame.Rect(0, 0, size, size)
        rect.center = (cx, cy)
        color = TEAM_COLOR[e.team] if e.active else TEAM_DARK[e.team]
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
        pygame.draw.rect(self.screen, (20, 20, 20), rect, 2, border_radius=4)
        label = "K" if e.kind == crsim.EntityKind.KING_TOWER else "P"
        if not e.active:
            label += " zz"
        self.blit_center(self.font, label, (cx, cy - 6), TEXT)
        self.blit_center(self.small, str(e.hp), (cx, cy + 9), TEXT)
        self.hp_bar(cx, rect.top - 7, size, e.hp, e.max_hp, (90, 230, 90))

    def draw_troop(self, e, cx, cy):
        r = max(4, int(e.radius * TILE))
        if e.air:
            pygame.draw.ellipse(self.screen, (40, 70, 40), (cx - r, cy - r / 3 + 4, 2 * r, 2 * r / 3))
            cy -= 8
        color = TEAM_COLOR[e.team]
        if e.deploy_ms_left > 0:
            pygame.draw.circle(self.screen, color, (cx, cy), r, 2)
            self.blit_center(self.small, f"{e.deploy_ms_left / 1000:.1f}", (cx, cy), TEXT)
            return
        pygame.draw.circle(self.screen, color, (cx, cy), r)
        pygame.draw.circle(self.screen, (255, 255, 255) if e.air else (20, 20, 20), (cx, cy), r, 1)
        self.blit_center(self.small, e.def_id[:3], (cx, cy), TEXT)
        if e.hp < e.max_hp:
            self.hp_bar(cx, cy - r - 6, max(2 * r, 16), e.hp, e.max_hp, (90, 230, 90))

    def blit_center(self, font, text, center, color):
        surf = font.render(text, True, color)
        self.screen.blit(surf, surf.get_rect(center=(int(center[0]), int(center[1]))))

    def text(self, x, y, s, color=TEXT, font=None):
        self.screen.blit((font or self.font).render(s, True, color), (x, y))

    def draw_panel(self):
        b = self.battle
        rules = self.data.rules
        x = MARGIN * 2 + self.arena_w
        y = MARGIN

        if b.finished:
            w = b.winner
            title = "КОНЕЦ: " + ("ничья" if w is None else f"победа, {TEAM_NAME[w]}")
        elif b.phase == crsim.Phase.OVERTIME:
            title = f"ОВЕРТАЙМ {fmt_clock(rules.regulation_ms + rules.overtime_ms - b.time_ms)}"
        else:
            title = f"Время {fmt_clock(rules.regulation_ms - b.time_ms)}"
        if not b.finished and b.elixir_multiplier > 1:
            title += f"  эликсир x{b.elixir_multiplier:g}"
        self.text(x, y, title, font=self.big)
        y += 26
        crowns = f"Короны: синие {b.player(BLUE).crowns} : {b.player(RED).crowns} красные"
        self.text(x, y, crowns)
        y += 24

        self.hand_rects = {}
        for team in (RED, BLUE):
            y = self.draw_player(x, y, team)

        speed = f"скорость x{self.speed:g}" + ("  ПАУЗА" if self.paused else "")
        self.text(x, y, speed, (255, 220, 120) if self.paused else TEXT)
        y += 18
        if self.status:
            self.text(x, y, self.status, (255, 120, 120))
        y += 22

        help_lines = [
            "1-4 / клик по карте: выбрать карту",
            "ЛКМ по арене: поставить, ПКМ: отмена",
            "Tab: за кого играешь (синие/красные)",
            "B / R: бот/модель за синих/красных вкл/выкл",
            "Пробел: пауза, -> : шаг (на паузе)",
            "+ / -: скорость, T: линии целей",
            "N: новый матч, Backspace: заново",
        ]
        for line in help_lines:
            self.text(x, y, line, DIM, self.small)
            y += 14
        y += 8
        for line in self.log:
            self.text(x, y, line[:46], DIM, self.small)
            y += 14

    def draw_player(self, x, y, team):
        p = self.battle.player(team)
        agent = self.agents[team]
        who = agent.name if agent and self.agent_on[team] else "ручн."
        mark = "  <- ты" if self.control == team else ""
        self.text(x, y, f"{TEAM_NAME[team]} ({who}){mark}", TEAM_COLOR[team])
        y += 18
        bar_w = PANEL_W - 60
        pygame.draw.rect(self.screen, (50, 40, 60), (x, y, bar_w, 12))
        pygame.draw.rect(self.screen, ELIXIR, (x, y, bar_w * p.elixir / self.data.rules.max_elixir, 12))
        for i in range(1, 10):
            pygame.draw.line(self.screen, BG, (x + bar_w * i / 10, y), (x + bar_w * i / 10, y + 11))
        self.text(x + bar_w + 6, y - 2, f"{p.elixir:.1f}")
        y += 18

        card_w = (PANEL_W - 18) // 4
        for slot, card_id in enumerate(p.hand):
            if not card_id:  # empty slot: the next card is on its way
                rect = pygame.Rect(x + slot * (card_w + 4), y, card_w, 40)
                self.hand_rects[(team, slot)] = rect
                pygame.draw.rect(self.screen, (35, 35, 42), rect, border_radius=4)
                self.text(rect.x + 4, rect.y + 22, f"{p.refill_ms_left[slot] / 1000:.1f}", DIM, self.small)
                continue
            card = self.data.card(card_id)
            rect = pygame.Rect(x + slot * (card_w + 4), y, card_w, 40)
            self.hand_rects[(team, slot)] = rect
            selected = self.control == team and self.selected == slot
            affordable = p.elixir + 1e-9 >= card.elixir
            pygame.draw.rect(self.screen, (70, 70, 85) if affordable else (45, 45, 52), rect, border_radius=4)
            pygame.draw.rect(self.screen, (255, 230, 120) if selected else (20, 20, 20), rect, 2, border_radius=4)
            self.text(rect.x + 4, rect.y + 3, f"{slot + 1}", DIM, self.small)
            self.text(rect.right - 14, rect.y + 3, str(card.elixir), ELIXIR, self.small)
            self.text(rect.x + 4, rect.y + 22, card.name[:10], TEXT, self.small)
        y += 44
        self.text(x, y, f"следующая: {self.data.card(p.queue[0]).name}", DIM, self.small)
        return y + 24


def main():
    parser = argparse.ArgumentParser(description="crsim debug viewer")
    parser.add_argument("--data", default=os.path.join(ROOT, "data"))
    parser.add_argument("--blue", default="human", help="human | noop | random | heuristic | путь к .pt | search:путь к .pt")
    parser.add_argument("--red", default="heuristic",
                        help="human | noop | random | heuristic | путь к .pt | search:путь к .pt (модель с поиском)")
    parser.add_argument("--blue-deck", default=DEFAULT_DECK, help="8 id карт через запятую")
    parser.add_argument("--red-deck", default=DEFAULT_DECK, help="8 id карт через запятую")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--decision-ticks", type=int, default=DECISION_TICKS,
                        help="тиков (50 мс) между решениями ботов и модели: 6 = 0.3 с, 10 = 0.5 с")
    viewer = Viewer(parser.parse_args())

    clock = pygame.time.Clock()
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            viewer.handle(ev)
        viewer.update(clock.tick(60))
        viewer.draw()


if __name__ == "__main__":
    main()
