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
        self.red_plays = []      # (game seconds, card, position, elixir the simulator thinks they lacked)
        self.red_rejected = []   # (game seconds, card, position, why): candidates judged not to be a new play
        self.handled = set()     # red track ids already turned into a play (or rejected)
        self.own_lock_until = 0.0  # just played: the screen lags behind our own play
        self.last_hand = [None] * 4  # last readable card in each slot
        self.empty_since = [None] * 4  # (time a slot first read empty, the card that was there)
        self.manual = []         # our plays made by hand (not by the bot): [card, time], waiting for a position
        self.own_units = set()   # blue track ids already given to one of our plays
        self.own_spells = []     # (card, pos, time) of spells the bot cast: their hits are ours
        self.sim_seen = {}       # simulator unit id -> last time a real unit matched it
        self.track_hp = {}       # track id -> (HP fraction from its bar, time); no bar = full HP
        self.track_no_bar = {}   # track id -> since when it has been seen without a bar
        self.spawned_for = {}    # track id -> when the simulator got a unit for it (no second one for a while)
        self.tower_reads = {}    # tower name -> (hp, how many frames in a row it was read)
        self.tower_ref = {}      # tower name -> (real hp, simulated hp right after syncing to it)
        self.tower_damage = {}   # tower name -> [real damage, simulated damage] between syncs (a mechanics check)
        self.tower_gone_since = {}  # princess tower name -> since when its HP bar has not been seen
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
    def update(self, t, seen, detections, bars=None):
        """seen: ScreenReader.read(frame); detections: UnitDetector.detect(frame) (or []);
        bars: UnitHpReader.bars(frame) - HP bars over damaged troops (None: not read)."""
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
        self._read_unit_hp(t, detections, bars)
        self._place_manual_plays(t, g)
        self._enemy_plays(t, g)
        if self.correct:
            self._correct(t, g, seen)

    TOWER_GONE_AFTER = 3.0  # s without a princess tower's HP bar (timer on screen) = it fell ...
    TOWER_GONE_BELOW = 1300  # ... if it was last read this low (bars are missing at the start of a match,
                             # and a unit can hide one for a moment)
    PRINCESS_HALF = 1.4    # tiles: a troop can't stand on a princess tower ...
    ON_TOWER_CONF = 0.6    # ... but the princess figure on it (other tower skins than the detector learned
                           # on) is sometimes taken for an archer / musketeer, at low confidence

    def _red_may_deploy(self, x, y, slack=0.7):
        """Could red have put a troop here (pockets after one of our princess towers fell), allowing for the
        detector's position error?"""
        for dx, dy in ((0, 0), (slack, 0), (-slack, 0), (0, slack), (0, -slack)):
            if self.battle.can_deploy_at(RED, crsim.Vec2(x + dx, y + dy)):
                return True
        return False

    def _on_princess_tower(self, x, y):
        for team in (BLUE, RED):
            for side in (0, 1):
                e = self.battle.find(self.battle.princess_tower(team, side))
                if e is not None and abs(x - e.pos.x) < self.PRINCESS_HALF and abs(y - e.pos.y) < self.PRINCESS_HALF:
                    return True
        return False

    def _read_unit_hp(self, t, detections, bars):
        if bars is None:
            return
        from perception.unit_hp import assign, fraction
        by_det = assign(bars, detections)
        # This frame's detections are where their tracks now are (the tracker took their positions).
        seen_tracks = [tr for tr in self.tracker.tracks if tr.last_seen == t]
        for di, d in enumerate(detections):
            tr = min(seen_tracks, key=lambda tr: abs(tr.x - d.tile[0]) + abs(tr.y - d.tile[1]), default=None)
            if tr is None or abs(tr.x - d.tile[0]) + abs(tr.y - d.tile[1]) > 1e-6 or tr.cls != d.cls:
                continue
            if di in by_det:
                self.track_hp[tr.id] = (fraction(by_det[di], d.cls), t)
                self.track_no_bar.pop(tr.id, None)
            else:
                self.track_no_bar.setdefault(tr.id, t)

    # --- keeping the simulator honest ------------------------------------------------------------
    TOWERS = {"blue_king": (BLUE, None), "blue_left": (BLUE, 0), "blue_right": (BLUE, 1),
              "red_king": (RED, None), "red_left": (RED, 0), "red_right": (RED, 1)}
    MOVE_BLEND = 0.5       # per frame, how far a simulated unit is pulled to where it is seen
    MOVE_DEADBAND = 0.6    # tiles: closer than this it is left alone (feet positions are ~0.2-0.3 tiles off)
    MATCH_DIST = 2.5       # tiles
    GONE_AFTER = {"minion": 2.5}  # seconds unseen in the open before the simulator drops a unit
    GONE_DEFAULT = 1.5
    GONE_HIDDEN = 4.0      # ... next to a giant or a tower, which can cover it
    NEW_AFTER = 1.0        # seconds a real unit has to be tracked before the simulator gets one for it
    RESPAWN_AFTER = 2.5    # a unit just put in (by a play or a correction) is not added again this soon

    def _tower_id(self, name):
        team, side = self.TOWERS[name]
        return self.battle.king_tower(team) if side is None else self.battle.princess_tower(team, side)

    HP_TOLERANCE = 0.08    # of max HP: smaller differences are left to the simulator
    FULL_AFTER = 1.0       # s seen without a bar = undamaged

    def _sync_unit_hp(self, unit_id, track_id, t):
        """The simulated unit gets the HP its bar shows (a unit without a bar for a while is undamaged)."""
        e = self.battle.find(unit_id)
        if e is None:
            return
        frac = None
        if track_id in self.track_hp and self.track_hp[track_id][1] == t:
            frac = self.track_hp[track_id][0]
        elif track_id in self.track_no_bar and t - self.track_no_bar[track_id] >= self.FULL_AFTER:
            frac = 1.0
        if frac is None:
            return
        target = max(1, round(frac * e.max_hp))
        if abs(e.hp - target) > self.HP_TOLERANCE * e.max_hp:
            self.battle.twin_set_hp(e.id, target)

    def _out_of_reach(self, unit_id, pos):
        """Would moving this attacking unit to pos put its target out of its attack range?"""
        e = self.battle.find(unit_id)
        if e is None or not e.locked:
            return False
        target = self.battle.find(e.target)
        if target is None:
            return False
        reach = self.data.card(e.def_id).unit.attack.range
        return math.dist(pos, (target.pos.x, target.pos.y)) - e.radius - target.radius > reach

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
        timer_shown = seen.get("timer", {}).get("seconds") is not None
        for name, (hp, exact) in seen.get("towers", {}).items():
            if not name.endswith("king") and timer_shown:
                # A princess tower always shows its HP bar while it stands: gone for a while = it fell.
                if hp is None:
                    since = self.tower_gone_since.setdefault(name, t)
                    e = b.find(self._tower_id(name))
                    last_hp = self.tower_ref.get(name, (None,))[0]  # last HP read for sure
                    if (e is not None and t - since >= self.TOWER_GONE_AFTER
                            and last_hp is not None and last_hp <= self.TOWER_GONE_BELOW):
                        b.twin_set_hp(e.id, 0)
                        self.log.append((g, f"{name} has fallen in the game: destroyed in the simulator too"))
                else:
                    self.tower_gone_since.pop(name, None)
            # A number that agrees with the HP bar, read the same 3 frames in a row; never 0 (screens
            # without the HUD read as 0) - a destroyed tower is the simulator's own business.
            if not exact or not hp:
                self.tower_reads.pop(name, None)
                continue
            last, n = self.tower_reads.get(name, (None, 0))
            n = n + 1 if hp == last else 1
            self.tower_reads[name] = (hp, n)
            e = b.find(self._tower_id(name))
            if n < 3 or e is None:
                continue
            prev = self.tower_ref.get(name)
            match_over = g > (self.data.rules.regulation_ms + self.data.rules.overtime_ms) / 1000.0
            if prev is not None and hp <= prev[0] and not match_over:  # (towers drain in the end animation)
                # Damage since the last reading: what the game did vs what the simulator did with the
                # same units on the field. A big gap means the simulated fighting is off.
                dmg = self.tower_damage.setdefault(name, [0, 0])
                dmg[0] += prev[0] - hp
                dmg[1] += max(0, prev[1] - e.hp)
            if abs(e.hp - hp) > 30:
                b.twin_set_hp(e.id, int(hp))
            self.tower_ref[name] = (hp, b.find(e.id).hp)
        tracks = {tr.id: tr for tr in self.tracker.tracks}
        real = [st for st in self.tracker.state(t) if st["seen"] and st["cls"] in twin.CARD_OF_UNIT
                and tracks[st["id"]].landed(t)]  # still dropping in: drawn above where it stands
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
            self._sync_unit_hp(s_["id"], r["id"], t)
            if d <= self.MOVE_DEADBAND:
                continue  # within what the detector's boxes jitter: let the simulator place it
            a = self.MOVE_BLEND
            new = (s_["x"] + a * (r["x"] - s_["x"]), s_["y"] + a * (r["y"] - s_["y"]))
            if self._out_of_reach(s_["id"], new):
                continue  # it is hitting something: don't pull it out of its attack range over a measuring error
            b.twin_move(s_["id"], crsim.Vec2(*new))
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
        for j, r in enumerate(real):
            tr = tracks.get(r["id"])
            if j in matched_real or tr is None or t - self.spawned_for.get(r["id"], -1e9) < self.RESPAWN_AFTER:
                continue
            if t - tr.first_seen < self.NEW_AFTER or tr.hits < 5:
                continue
            card = twin.CARD_OF_UNIT[r["cls"]]
            b.twin_spawn(BLUE if r["team"] == "blue" else RED, card, crsim.Vec2(r["x"], r["y"]))
            self.spawned_for[r["id"]] = t
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
            card = h["card"]
            if card is None:
                if self.last_hand[slot] is not None and self.empty_since[slot] is None:
                    self.empty_since[slot] = (t, self.last_hand[slot])
                since = self.empty_since[slot]
                if since is not None and t - since[0] >= twin.EMPTY_FOR:  # not a one-frame misread
                    self.manual.append([since[1], since[0]])
                    self.empty_since[slot] = None
                    self.last_hand[slot] = None
            else:
                if self.empty_since[slot] is not None and card == self.empty_since[slot][1]:
                    self.empty_since[slot] = None
                self.last_hand[slot] = card

    def _place_manual_plays(self, t, g):
        """Our hand-made plays go into the simulator once their units (or spell hit) are seen."""
        for play in list(self.manual):
            card, t0 = play
            if t - t0 > 6.0:
                self.manual.remove(play)  # never seen where: the hand sync already cycled it
                self.log.append((g, f"blue {card}: played by hand, position not found"))
                continue
            if card in twin.SPELL_CARDS:
                new = [tr for tr in self.tracker.tracks if tr.team in ("blue", "any") and tr.cls == card + "_hit"
                       and tr.first_seen > t0 + 0.5 and tr.id not in self.own_units]
                need = 1
            else:
                unit = twin.UNIT_OF_CARD.get(card)
                need = {"archers": 2, "minions": 3}.get(card, 1)
                new = [tr for tr in self.tracker.tracks if tr.team == "blue" and tr.cls == unit and tr.confirmed
                       and tr.landed(t)
                       and t0 + 0.3 < tr.first_seen < t0 + 4.0 and tr.id not in self.own_units]  # the slot empties when the drag starts
                if len(new) < need and t - t0 < 3.5:
                    continue  # the rest of the group may still show up
            if not new:
                continue
            new = sorted(new, key=lambda tr: tr.first_seen)[:need]
            self.own_units.update(tr.id for tr in new)
            self.spawned_for.update((tr.id, t) for tr in new)  # these tracks are this play's units
            pos = (sum(tr.landing_pos()[0] for tr in new) / len(new), sum(tr.landing_pos()[1] for tr in new) / len(new))
            self.manual.remove(play)
            if card in twin.SPELL_CARDS:
                ago_ms = int(self.deploy_delay * 1000)  # it already landed; in the simulator it still flies
            else:
                ago_ms = int((t - t0) * 1000)
            self.battle.force_play(BLUE, card, crsim.Vec2(*pos), ago_ms)
            if card in twin.SPELL_CARDS:  # the next waves' hits are ours too
                self.own_spells.append((card, pos, t0))
            self.log.append((g, f"blue {card} (by hand) at ({pos[0]:.1f}, {pos[1]:.1f})"))

    def _enemy_plays(self, t, g):
        fresh = [tr for tr in self.tracker.tracks if tr.confirmed and tr.id not in self.handled and tr.landed(t)
                 and t - tr.first_seen < 1.8 and tr.id not in self.own_units
                 and (tr.team == "red" or (tr.team == "any" and tr.cls.endswith("_hit")))]
        # A troop is judged once it has been seen a few times (a kind the detector is unsure about flickers);
        # spell hits right away.
        fresh = [tr for tr in fresh if tr.cls.endswith("_hit") or
                 (tr.hits >= self.NEW_PLAY_HITS and t - tr.first_seen >= self.NEW_PLAY_AGE)]
        for tr in fresh:
            if tr.id in self.handled:
                continue
            self.handled.add(tr.id)
            x, y = tr.landing_pos()
            if tr.cls.endswith("_hit"):
                spell = tr.cls[:-4]
                if self._own_spell_hit(spell, (x, y), tr.first_seen):
                    continue  # the hit of a spell we cast
                if any(c.card == spell and c.source == "hit" and abs(getattr(c, "hit_t", c.t) - tr.first_seen) < twin.SPELL_SAME_PLAY[0]
                       and math.dist(c.pos, (x, y)) < twin.SPELL_SAME_PLAY[1] for c in self.accepted_red[-3:]):
                    continue  # another wave of the same arrows
                # Played: the blueprint second plus the flight from their king tower before the hit.
                speed = self.data.card(spell).spell.projectile_speed
                flight = math.dist((x, y), twin.KING["red"]) / speed if speed > 0 else 0.0
                cand = twin.Play("red", spell, tr.first_seen - flight - self.deploy_delay, (x, y), "hit", [tr.id])
                cand.hit_t = tr.first_seen
            else:
                card = twin.CARD_OF_UNIT.get(tr.cls)
                if card is None or not (15.5 <= y <= 31.6 or self._red_may_deploy(x, y)):
                    continue  # not where they can put a troop (our half: only in a pocket of a fallen tower)
                if self._on_princess_tower(x, y) and tr.conf < self.ON_TOWER_CONF:
                    self.log.append((g, f"red {card}? no: on a tower, unsure"))
                    continue
                rival = self._rival_kind(tr)
                if rival is not None:
                    self.log.append((g, f"red {card}? no: more often seen as {rival}"))
                    continue
                count = {"archers": 2, "minions": 3}.get(card, 1)
                group = [tr] + [o for o in fresh if o is not tr and o.id not in self.handled and o.cls == tr.cls
                                and abs(o.first_seen - tr.first_seen) < 0.6
                                and math.dist(o.landing_pos(), (x, y)) < 3.0][:count - 1]
                self.handled.update(o.id for o in group)
                pos = (sum(o.landing_pos()[0] for o in group) / len(group), sum(o.landing_pos()[1] for o in group) / len(group))
                cand = twin.Play("red", card, tr.first_seen - self.deploy_delay, pos, "units", [o.id for o in group])
            unit = twin.UNIT_OF_CARD.get(cand.card)
            elsewhere = [(o.x, o.y) for o in self.tracker.tracks if o.cls == unit and o.team == "red"
                         and o.last_seen == t and o.id not in cand.units]
            cand.rejected = twin.judge_enemy(self.battle, self.data, cand, self.accepted_red, seen_elsewhere=elsewhere)
            if cand.rejected == "card not back in hand yet" and self._undo_phantom(cand, t, g):
                cand.rejected = None
            if cand.rejected != "card not back in hand yet":
                self.spawned_for.update((u, t) for u in cand.units)  # a new play's units, or units it has
            if cand.rejected:
                self.red_rejected.append((self.game_time(cand.t), cand.card, cand.pos, cand.rejected))
                self.log.append((g, f"red {cand.card}? no: {cand.rejected}"))
                continue
            cand.accepted_at = t
            self.accepted_red.append(cand)
            # The unit is on the field already: its blueprint second is over. (A spell is only seen when
            # it lands; in the simulator it still has to fly - it lands late. To do: start it in flight.)
            ago_ms = int((t - cand.t) * 1000) if cand.source == "units" else int(self.deploy_delay * 1000)
            missing = self.battle.force_play(RED, cand.card, crsim.Vec2(*cand.pos), ago_ms)
            self.red_plays.append((self.game_time(cand.t), cand.card, cand.pos, missing))  # when it was played
            self.log.append((g, f"red {cand.card} at ({cand.pos[0]:.1f}, {cand.pos[1]:.1f})"))

    NEW_PLAY_HITS = 4      # sightings before a new red troop counts as a play
    NEW_PLAY_AGE = 0.5     # s
    PHANTOM_HITS = 8       # a play whose units were seen fewer times than this, and are gone, was a mistake

    def _rival_kind(self, tr):
        """Another kind of red troop the detector sees on the same spot more often (one unit, two names)."""
        x, y = tr.history[-1][1], tr.history[-1][2]
        for o in self.tracker.tracks:
            if o is tr or o.team != tr.team or o.cls == tr.cls or o.cls not in twin.CARD_OF_UNIT:
                continue
            if math.dist((o.x, o.y), (x, y)) < 1.0 and o.hits > tr.hits:
                return o.cls
        return None

    def _undo_phantom(self, cand, t, g):
        """The same card can't be back already - unless the earlier play was a phantom: its units were seen
        only a few times and are gone. Then that play is taken back (its simulated units removed)."""
        same = [p for p in self.accepted_red[-3:] if p.card == cand.card and p.source == "units"]
        if not same or cand.source != "units":
            return False
        old = same[-1]
        tracks = {tr.id: tr for tr in self.tracker.tracks}
        live = [tracks[u] for u in old.units if u in tracks]
        seen = sum(tr.hits for tr in live)
        if any(t - tr.last_seen < 0.5 for tr in live) or seen >= self.PHANTOM_HITS:
            return False  # still there, or was well seen: the new one is the doubtful one
        unit = twin.UNIT_OF_CARD[old.card]
        for e in self.battle.entities:  # its simulated unit still matches something seen: not a phantom
            if (not e.is_building and e.team == RED and twin.UNIT_OF_CARD.get(e.def_id) == unit
                    and t - self.sim_seen.get(e.id, -1e9) < 1.0
                    and math.dist((e.pos.x, e.pos.y), old.pos) < 4.0 + 1.5 * (t - getattr(old, "accepted_at", t))):
                return False
        for e in list(self.battle.entities):
            if (not e.is_building and e.team == RED and twin.UNIT_OF_CARD.get(e.def_id) == unit
                    and math.dist((e.pos.x, e.pos.y), old.pos) < 4.0 + 1.5 * (t - getattr(old, "accepted_at", t))):
                self.battle.twin_remove(e.id)
        self.accepted_red.remove(old)
        self.red_plays = [p for p in self.red_plays if not (p[1] == old.card and p[2] == old.pos)]
        self.red_rejected.append((self.game_time(old.t), old.card, old.pos, "taken back: a phantom"))
        self.log.append((g, f"red {old.card} at ({old.pos[0]:.1f}, {old.pos[1]:.1f}) taken back: a phantom"))
        return True

    def _own_spell_hit(self, spell, pos, t):
        for card, p, t0 in self.own_spells:
            if card == spell and t0 + 0.8 < t < t0 + 4.5 and math.dist(p, pos) < 3.0:
                return True
        return False

    # --- our side ----------------------------------------------------------------------------------
    def own_play(self, t, card, pos, lock=1.0):
        """We just played card at pos (simulator tiles)."""
        self.battle.force_play(BLUE, card, crsim.Vec2(*pos))
        self.own_lock_until = t + lock
        if card in twin.SPELL_CARDS:
            self.own_spells.append((card, tuple(pos), t))
            del self.own_spells[:-8]
        # Our next screen readings will see this card leave the hand: it is not a hand-made play.
        self.last_hand = [None if c == card else c for c in self.last_hand]
        self.empty_since = [None if s and s[1] == card else s for s in self.empty_since]
        self.log.append((self.game_time(t), f"blue {card} at ({pos[0]:.1f}, {pos[1]:.1f})"))
