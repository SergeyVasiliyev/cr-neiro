"""The live copy of a real match in the simulator, kept in step with the screen.

Every frame: the countdown timer fixes the battle clock; our hand, next card and elixir are copied from
the screen; the opponent's new units (and spell hits) are added as their plays, unless the simulator
already has that unit on the field or the card can't be back in their hand yet (twin.judge_enemy).
Our own plays are added by whoever makes them (the bot calls own_play). Units then move and fight in
the simulator, which keeps knowing about the ones hidden from view.

The policy sees this simulated battle exactly as in training (crsim.rl.encode).
"""
import math

import crsim
import twin
from crsim_env import DEFAULT_DECK
from perception.tracker import Tracker

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED


class LiveMatch:
    def __init__(self, data, deck=DEFAULT_DECK, enemy_deck=DEFAULT_DECK, deploy_delay=1.0, correct=True):
        self.data = data
        self.correct = correct   # pull the simulator towards what is seen (see _correct)
        self.decks = (list(deck), list(enemy_deck))
        self.deploy_delay = deploy_delay
        self.battle = None
        self.tracker = Tracker()
        self.lo, self.hi = -math.inf, math.inf  # battle start lies in (lo, hi], video/wall seconds
        self.conflicts = 0
        self.accepted_red = []
        self.handled = set()     # red track ids already turned into a play (or rejected)
        self.own_lock_until = 0.0  # just played: the screen lags behind our own play
        self.last_hand = [None] * 4  # last readable card in each slot
        self.manual = []         # our plays made by hand (not by the bot): [card, time], waiting for a position
        self.own_units = set()   # blue track ids already given to one of our plays
        self.sim_seen = {}       # simulator unit id -> last time a real unit matched it
        self.spawned_for = set()  # track ids the simulator got a unit for
        self.tower_reads = {}    # tower name -> (hp, how many frames in a row it was read)
        self.log = []            # (game seconds, text)

    # --- clock -----------------------------------------------------------------------------------
    @property
    def start(self):
        return None if math.isinf(self.lo) or math.isinf(self.hi) else (self.lo + self.hi) / 2

    def _clock(self, t, timer):
        sec, overtime = timer
        if sec is None or overtime:
            return
        lo, hi = t - (181 - sec), t - (180 - sec)  # the timer shows the seconds left, rounded up
        if max(self.lo, lo) <= min(self.hi, hi):
            self.lo, self.hi = max(self.lo, lo), min(self.hi, hi)
            self.conflicts = 0
        else:
            self.conflicts += 1
            if self.conflicts > 15:  # consistently different: a new match started
                self.reset()
                self.lo, self.hi = lo, hi

    def reset(self):
        self.__init__(self.data, *self.decks, deploy_delay=self.deploy_delay, correct=self.correct)

    def game_time(self, t):
        return None if self.start is None else t - self.start

    # --- per frame -------------------------------------------------------------------------------
    def update(self, t, seen, detections):
        """seen: ScreenReader.read(frame); detections: UnitDetector.detect(frame) (or [])."""
        if "timer" in seen:
            self._clock(t, (seen["timer"]["seconds"], seen["timer"]["overtime"]))
        g = self.game_time(t)
        if g is None or g < 0:
            return
        if self.battle is None:
            self.battle = crsim.Battle(self.data, *self.decks, 0)
            self.battle.set_follow_real(True)  # the real game decides when it is over
        tick = self.data.rules.tick_ms / 1000.0
        while self.battle.time_ms / 1000.0 + tick <= g and not self.battle.finished:
            self.battle.step()
        if t >= self.own_lock_until:
            self._watch_own_hand(t, seen)
            self._sync_own(seen)
        self.tracker.update(t, detections)
        self._place_manual_plays(t, g)
        self._enemy_plays(t, g)
        if self.correct:
            self._correct(t, g, seen)

    # --- keeping the simulator honest ------------------------------------------------------------
    TOWERS = {"blue_king": (BLUE, None), "blue_left": (BLUE, 0), "blue_right": (BLUE, 1),
              "red_king": (RED, None), "red_left": (RED, 0), "red_right": (RED, 1)}
    MOVE_BLEND = 0.5       # per frame, how far a simulated unit is pulled to where it is seen
    MATCH_DIST = 2.5       # tiles
    GONE_AFTER = {"minion": 2.5}  # seconds unseen in the open before the simulator drops a unit
    GONE_DEFAULT = 1.5
    GONE_HIDDEN = 4.0      # ... next to a giant or a tower, which can cover it
    NEW_AFTER = 1.0        # seconds a real unit has to be tracked before the simulator gets one for it

    def _tower_id(self, name):
        team, side = self.TOWERS[name]
        return self.battle.king_tower(team) if side is None else self.battle.princess_tower(team, side)

    def _maybe_hidden(self, x, y, real):
        for r in real:
            if r["cls"] == "giant" and abs(r["x"] - x) < 1.5 and -0.5 < y - r["y"] < 2.5:
                return True
        for name in self.TOWERS:
            e = self.battle.find(self._tower_id(name))
            if e is not None and math.dist((e.pos.x, e.pos.y), (x, y)) < 2.5:
                return True
        return y > 30.5  # the top edge of the arena is partly behind the HUD

    def _correct(self, t, g, seen):
        """Tower HP from the screen; simulated units pulled to where they are seen, dropped when the game
        clearly no longer has them, added when the game has one the simulator does not."""
        b = self.battle
        if b.finished:
            return
        for name, (hp, exact) in seen.get("towers", {}).items():
            # A number that agrees with the HP bar, read the same 3 frames in a row; never 0 (screens
            # without the HUD read as 0) - a destroyed tower is the simulator's own business.
            if not exact or not hp:
                self.tower_reads.pop(name, None)
                continue
            last, n = self.tower_reads.get(name, (None, 0))
            n = n + 1 if hp == last else 1
            self.tower_reads[name] = (hp, n)
            e = b.find(self._tower_id(name))
            if n >= 3 and e is not None and abs(e.hp - hp) > 30:
                b.twin_set_hp(e.id, int(hp))
        real = [st for st in self.tracker.state(t) if st["seen"] and st["cls"] in twin.CARD_OF_UNIT]
        # Units still in their deploy timer are on the field (and seen) too: match them as well.
        sim = [{"id": e.id, "cls": twin.UNIT_OF_CARD.get(e.def_id, e.def_id), "hp": e.hp,
                "deploying": e.deploy_ms_left > 0, "team": "blue" if e.team == BLUE else "red",
                "x": e.pos.x, "y": e.pos.y}
               for e in b.entities if not e.is_building]
        pairs = twin.match_units(sim, real, self.MATCH_DIST)
        matched_real = set()
        for i, j, d in pairs:
            s_, r = sim[i], real[j]
            self.sim_seen[s_["id"]] = t
            matched_real.add(j)
            if d > 0.3:
                a = self.MOVE_BLEND
                b.twin_move(s_["id"], crsim.Vec2(s_["x"] + a * (r["x"] - s_["x"]), s_["y"] + a * (r["y"] - s_["y"])))
        matched_sim = {sim[i]["id"] for i, _, _ in pairs}
        for s_ in sim:
            if s_["id"] in matched_sim or s_["hp"] <= 0 or s_["deploying"]:
                continue
            last = self.sim_seen.setdefault(s_["id"], t)  # first noticed now: give it time
            limit = self.GONE_HIDDEN if self._maybe_hidden(s_["x"], s_["y"], real) else \
                self.GONE_AFTER.get(s_["cls"], self.GONE_DEFAULT)
            if t - last > limit:
                b.twin_remove(s_["id"])
                self.log.append((g, f"sim {s_['team']} {s_['cls']} dropped: not seen for {t - last:.1f}s"))
        tracks = {tr.id: tr for tr in self.tracker.tracks}
        for j, r in enumerate(real):
            tr = tracks.get(r["id"])
            if j in matched_real or tr is None or r["id"] in self.spawned_for:
                continue
            if t - tr.first_seen < self.NEW_AFTER or tr.hits < 5:
                continue
            card = twin.CARD_OF_UNIT[r["cls"]]
            b.twin_spawn(BLUE if r["team"] == "blue" else RED, card, crsim.Vec2(r["x"], r["y"]))
            self.spawned_for.add(r["id"])
            self.log.append((g, f"sim {r['team']} {r['cls']} added: seen, simulator did not have it"))

    def _sync_own(self, seen):
        hand = [h["card"] for h in seen["hand"]]
        if not any(h["unknown"] for h in seen["hand"]):
            try:
                self.battle.set_hand(BLUE, [c or "" for c in hand], seen["next"]["card"] or "")
            except ValueError:
                pass  # a misread (a card twice, one not in the deck): keep the simulator's idea
        real = seen["elixir"]
        sim = self.battle.player(BLUE).elixir
        if int(real) != int(sim + 1e-6):  # whole units are read reliably, the filling segment less so
            self.battle.set_elixir(BLUE, int(real) + (0.05 if real > sim else 0.95))

    def _watch_own_hand(self, t, seen):
        """A card that leaves its slot without the bot playing it was played by hand."""
        for slot, h in enumerate(seen["hand"]):
            if h["unknown"]:
                continue  # being dragged or sliding in
            if h["card"] is None and self.last_hand[slot] is not None:
                self.manual.append([self.last_hand[slot], t])
            self.last_hand[slot] = h["card"]

    def _place_manual_plays(self, t, g):
        """Our hand-made plays go into the simulator once their units (or spell hit) are seen."""
        for play in list(self.manual):
            card, t0 = play
            if t - t0 > 6.0:
                self.manual.remove(play)  # never seen where: the hand sync already cycled it
                self.log.append((g, f"blue {card}: played by hand, position not found"))
                continue
            if card in twin.SPELL_CARDS:
                new = [tr for tr in self.tracker.tracks if tr.team == "blue" and tr.cls == card + "_hit"
                       and tr.first_seen > t0 + 0.5 and tr.id not in self.own_units]
                need = 1
            else:
                unit = twin.UNIT_OF_CARD.get(card)
                need = {"archers": 2, "minions": 3}.get(card, 1)
                new = [tr for tr in self.tracker.tracks if tr.team == "blue" and tr.cls == unit and tr.confirmed
                       and t0 + 0.3 < tr.first_seen < t0 + 4.0 and tr.id not in self.own_units]  # the slot empties when the drag starts
                if len(new) < need and t - t0 < 3.5:
                    continue  # the rest of the group may still show up
            if not new:
                continue
            new = sorted(new, key=lambda tr: tr.first_seen)[:need]
            self.own_units.update(tr.id for tr in new)
            self.spawned_for.update(tr.id for tr in new)  # these tracks are this play's units
            pos = (sum(tr.history[0][1] for tr in new) / len(new), sum(tr.history[0][2] for tr in new) / len(new))
            self.manual.remove(play)
            if card in twin.SPELL_CARDS:
                ago_ms = int(self.deploy_delay * 1000)  # it already landed; in the simulator it still flies
            else:
                ago_ms = int((t - t0) * 1000)
            self.battle.force_play(BLUE, card, crsim.Vec2(*pos), ago_ms)
            self.log.append((g, f"blue {card} (by hand) at ({pos[0]:.1f}, {pos[1]:.1f})"))

    def _enemy_plays(self, t, g):
        fresh = [tr for tr in self.tracker.tracks if tr.team == "red" and tr.confirmed and tr.id not in self.handled
                 and t - tr.first_seen < 1.0]
        for tr in fresh:
            if tr.id in self.handled:
                continue
            self.handled.add(tr.id)
            x, y = tr.history[0][1], tr.history[0][2]
            if tr.cls.endswith("_hit"):
                spell = tr.cls[:-4]
                cand = twin.Play("red", spell, tr.first_seen, (x, y), "hit", [tr.id])
            else:
                card = twin.CARD_OF_UNIT.get(tr.cls)
                if card is None or not 15.5 <= y <= 31.6:
                    continue
                count = {"archers": 2, "minions": 3}.get(card, 1)
                group = [tr] + [o for o in fresh if o is not tr and o.id not in self.handled and o.cls == tr.cls
                                and abs(o.first_seen - tr.first_seen) < 0.6
                                and math.dist((o.history[0][1], o.history[0][2]), (x, y)) < 3.0][:count - 1]
                self.handled.update(o.id for o in group)
                pos = (sum(o.history[0][1] for o in group) / len(group), sum(o.history[0][2] for o in group) / len(group))
                cand = twin.Play("red", card, tr.first_seen - self.deploy_delay, pos, "units", [o.id for o in group])
            cand.rejected = twin.judge_enemy(self.battle, self.data, cand, self.accepted_red)
            if cand.rejected != "card not back in hand yet":
                self.spawned_for.update(cand.units)  # a new play's units, or units the simulator has
            if cand.rejected:
                self.log.append((g, f"red {cand.card}? no: {cand.rejected}"))
                continue
            self.accepted_red.append(cand)
            # The unit is on the field already: its blueprint second is over. (A spell is only seen when
            # it lands; in the simulator it still has to fly - it lands late. To do: start it in flight.)
            ago_ms = int((t - cand.t) * 1000) if cand.source == "units" else int(self.deploy_delay * 1000)
            self.battle.force_play(RED, cand.card, crsim.Vec2(*cand.pos), ago_ms)
            self.log.append((g, f"red {cand.card} at ({cand.pos[0]:.1f}, {cand.pos[1]:.1f})"))

    # --- our side ----------------------------------------------------------------------------------
    def own_play(self, t, card, pos, lock=1.0):
        """We just played card at pos (simulator tiles)."""
        self.battle.force_play(BLUE, card, crsim.Vec2(*pos))
        self.own_lock_until = t + lock
        # Our next screen readings will see this card leave the hand: it is not a hand-made play.
        self.last_hand = [None if c == card else c for c in self.last_hand]
        self.log.append((self.game_time(t), f"blue {card} at ({pos[0]:.1f}, {pos[1]:.1f})"))
