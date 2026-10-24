"""The live copy of a real match in the simulator, kept in step with the screen.

Everything the twin knows about the match is in its journal, in battle time:
  - plays: every card either side played - whose, which, when, where, how sure, and how it is known (the
    bot's own move, our hand, the opponent's units or spell seen by the detector, a recording's truth);
  - fixes: what the screen made the simulator change - a tower's HP, a unit's HP or position, a unit gone
    from the game, a unit of a play still alive in the game that the simulator had lost.
Every troop in the simulator belongs to a play (its play/unit tag), so no unit comes from nowhere: one the
game shows and no play explains is either an opponent's play not noticed yet (it becomes one) or a
detector mistake - our own side never gets a unit we did not play.

When the journal changes in the past - a play noticed late (units show a second after the card is put
down; one put behind the king's tower only when it walks out), a play found to be a phantom and taken
back - the simulator goes back to a copy of itself from before that moment and plays the journal forward
again: the late play's units walk from where and when they really started, a phantom's units and all the
damage they did are gone.

Every frame: the countdown timer fixes the battle clock; our hand and elixir are copied from the screen;
the opponent's new units (and spell hits) become their plays, unless the simulator already has that unit
or the card can't be back in their hand yet (twin.judge_enemy). The policy sees this simulated battle
exactly as in training (crsim.rl.encode).
"""
import bisect
import math
from dataclasses import dataclass, field

import crsim
import twin
from crsim_env import DEFAULT_DECK
from perception.tracker import DROP_TIME, Tracker

BLUE, RED = crsim.Team.BLUE, crsim.Team.RED
TEAM = {"blue": BLUE, "red": RED}
GROUP = {"archers": 2, "minions": 3}  # units per play (others: 1)


@dataclass
class Play:
    id: int
    side: str                # "blue" / "red"
    card: str
    g: float                 # battle seconds when the card was put down
    pos: tuple = None        # where (None: played, place unknown - only the cycle and elixir count)
    source: str = "seen"     # "bot", "hand", "seen", "late" (a unit no play explained), "truth"
    conf: float = 1.0
    units: list = field(default_factory=list)  # tracker ids of its units / spell hits
    on: bool = True          # False: taken back (a phantom)
    origin: tuple = None     # where it was put, when not where it was first seen (behind the king's tower)
    hit_t: float = None      # spells seen landing: when (video s)
    accepted_at: float = 0.0  # video s
    missing: float = 0.0     # elixir the simulator thought was missing for it


@dataclass
class Fix:
    g: float
    kind: str                # "tower", "hp", "move", "drop", "revive"
    key: object              # tower name, or (play id, unit index)
    value: object = None


@dataclass
class Snapshot:
    g: float
    battle: object
    sim_index: dict
    next_ev: int


class LiveMatch:
    def __init__(self, data, deck=DEFAULT_DECK, enemy_deck=DEFAULT_DECK, deploy_delay=1.0, correct=True,
                 red_truth=None, blue_truth=None, record_damage=False):
        self.data = data
        # Known plays (offline: from each player's own recording), [(game s, card, (x, y) or None)], instead
        # of working them out from the screen: a clean test of the simulated mechanics.
        self.red_truth = sorted(red_truth, key=lambda p: p[0]) if red_truth is not None else None
        self.blue_truth = sorted(blue_truth, key=lambda p: p[0]) if blue_truth is not None else None
        # Every correction the screen forced on the simulator - each one is a place the two disagree:
        # (game s, kind, unit, side, details)
        self.corrections = []
        self.record_damage = record_damage  # keep who hit whom in the simulator (to explain corrections)
        self.keep_before_correction = False  # set to keep .before_correction, the state before each frame's fixes
        self.before_correction = None
        self.damage_log = {}     # simulator entity id -> [(game s, source, amount)]
        self.correct = correct   # pull the simulator towards what is seen (see _correct)
        self.decks = (list(deck), list(enemy_deck))
        self.deploy_delay = deploy_delay
        self.tick = data.rules.tick_ms / 1000.0
        self.battle = None
        self.tracker = Tracker()
        self.lo, self.hi = -math.inf, math.inf  # battle start lies in (lo, hi], video/wall seconds
        self.conflicts = 0
        # the journal (plays and fixes, by battle time) and how far the simulator has applied it
        self.journal = []
        self.journal_g = []      # their times, for bisect
        self.next_ev = 0
        self.plays = []          # every Play recorded (also those taken back)
        self.sim_index = {}      # play id -> (side, index in the simulated player's plays)
        self.snaps = []
        self.rewinds = 0
        self.red_rejected = []   # (game seconds, card, position, why): candidates judged not to be a new play
        self.handled = set()     # red track ids already turned into a play (or rejected)
        self.own_lock_until = 0.0  # just played: the screen lags behind our own play
        self.last_hand = [None] * 4  # last readable card in each slot
        self.empty_since = [None] * 4  # (time a slot first read empty, the card that was there)
        self.manual = []         # our plays waiting to be placed: [card, time, the bot's Play or None (by hand)]
        self.own_units = set()   # blue track ids already given to one of our plays
        self.own_spells = []     # (card, pos, game s) of our spells: explosions there and then are ours
        self.sim_seen = {}       # unit key -> last time a real unit matched it
        self.last_pair = {}      # unit key -> track id it was matched with last frame (identities stay put)
        self.gone = {}           # unit key -> (game s, x, y, hp, "drop" / "death") of units the simulator lost
        self.track_hp = {}       # track id -> (HP fraction from its bar, time, steady, readings in a row agreeing)
        self.track_no_bar = {}   # track id -> since when it has been seen without a bar
        self.spawned_for = {}    # track id -> when it was last explained (a play, a revived unit, ignored)
        self.tower_reads = {}    # tower name -> (hp, how many frames in a row it was read)
        self.tower_ref = {}      # tower name -> (real hp, simulated hp right after syncing to it)
        self.tower_damage = {}   # tower name -> [real damage, simulated damage] between syncs (a mechanics check)
        self.tower_gone_since = {}  # princess tower name -> since when its HP bar has not been seen
        self.sim_deaths = []     # (game s, kind, side, x, y) of units the simulator killed
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
            if self.conflicts > 15:  # consistently different: the clock drifted (frames lost), or a new match
                if abs(lo - self.lo) < self.CLOCK_DRIFT:
                    self.log.append((self.game_time(t), f"clock off by {lo - self.lo:+.1f}s: following the timer"))
                    self.lo, self.hi = lo, hi
                    self.conflicts = 0
                    return
                self.reset()
                self.lo, self.hi = lo, hi

    CLOCK_DRIFT = 5.0  # s: the timer disagreeing by less than this is the same match, the clock re-set to it

    def reset(self):
        self.__init__(self.data, *self.decks, deploy_delay=self.deploy_delay, correct=self.correct,
                      red_truth=self.red_truth, blue_truth=self.blue_truth, record_damage=self.record_damage)

    def game_time(self, t):
        return None if self.start is None else t - self.start

    FRESH_SEEN = 2.5  # s: a red troop / spell seen this recently and not yet counted as a play is paid for already

    def enemy_elixir_seen(self, t):
        """The opponent's elixir for the net. The simulator's count takes a red play off only once the twin has
        judged it one (~0.6 s after its troop shows up, a spell when it lands: ~1.6 / 2.0 s after it was put down,
        when its elixir is long gone): here a troop already seen and a spell already flying are paid for at once -
        a troop unless one of the same kind is close by in the simulator (that one, found again after a while
        hidden), each troop its share of the card (an archer half of 3), each spell once."""
        if self.battle is None:
            return None
        e = self.battle.player(RED).elixir
        sim_red = [(twin.UNIT_OF_CARD.get(u.def_id, u.def_id), u.pos.x, u.pos.y) for u in self.battle.entities
                   if not u.is_building and u.team == RED]
        paid, spells = 0.0, []
        paired = set(self.last_pair.values())  # tracks the twin has matched with a simulated unit
        for tr in self.tracker.tracks:
            if tr.id in self.handled or tr.id in self.own_units or tr.id in paired or t - tr.first_seen > self.FRESH_SEEN:
                continue
            x, y = tr.x, tr.y
            if tr.cls in twin.CARD_OF_UNIT and tr.team == "red":
                if tr.hits < 2 or not 15.5 <= tr.history[0][2] <= 31.6:
                    continue  # a flicker, or first seen where they can't put it down: not fresh
                if any(c == tr.cls and math.dist((cx, cy), (x, y)) < 3.0 for c, cx, cy in sim_red):
                    continue
                card = self.data.card(twin.CARD_OF_UNIT[tr.cls])
                paid += card.elixir / max(1, card.count)
            elif tr.cls in ("fireball", "arrows") and (tr.team == "red" or (tr.team == "any" and tr.vy < -1.0)):
                if any(c == tr.cls and abs(ts - tr.first_seen) < 1.5 for c, ts in spells):
                    continue  # (one cast)
                if any(p.side == "red" and p.on and p.card == tr.cls and p.hit_t is not None
                       and 0.0 <= p.hit_t - tr.first_seen < 3.0 for p in self.plays[-4:]):
                    continue  # its hit already counted it
                spells.append((tr.cls, tr.first_seen))
                paid += self.data.card(tr.cls).elixir
        return max(0.0, e - paid)

    # --- the journal -----------------------------------------------------------------------------
    @property
    def now(self):
        """Battle seconds the simulator is at."""
        return self.battle.time_ms / 1000.0

    @property
    def red_plays(self):
        """The opponent's plays: [(game s, card, position, elixir the simulator thought they lacked)]."""
        return [(p.g, p.card, p.origin or p.pos, p.missing) for p in sorted(self.plays, key=lambda p: p.g)
                if p.side == "red" and p.on and p.source != "truth"]

    @property
    def accepted_red(self):
        return sorted((p for p in self.plays if p.side == "red" and p.on), key=lambda p: p.g)

    def _new_battle(self):
        b = crsim.Battle(self.data, *self.decks, 0)
        b.set_follow_real(True)  # the real game decides when it is over
        b.set_record_damage(self.record_damage)
        return b

    def add_play(self, side, card, g, pos, source, conf=1.0, units=(), **extra):
        """Puts a play into the journal (going back to replay from it, if it lies in the past)."""
        p = Play(len(self.plays), side, card, g, None if pos is None else tuple(pos), source, conf, list(units),
                 **extra)
        self.plays.append(p)
        self._record(p)
        return p

    def revise_play(self, play, g, pos, units=None):
        """A play happened at another time or place than first recorded: moved in the journal, replayed."""
        old = play.g
        i = self.journal.index(play)
        del self.journal[i], self.journal_g[i]
        if i < self.next_ev:
            self.next_ev -= 1
        play.g, play.pos = g, None if pos is None else tuple(pos)
        if units is not None:
            play.units = list(units)
        j = bisect.bisect_right(self.journal_g, g)
        self.journal.insert(j, play)
        self.journal_g.insert(j, g)
        if j < self.next_ev:
            self.next_ev += 1
        if min(old, g) < self.now - 1e-9:
            self._rewind(min(old, g))
        elif j == self.next_ev and g <= self.now + 1e-9:
            self._apply(play)
            self.next_ev += 1

    def take_back(self, play, why):
        """A play that wasn't one: out of the journal, the simulator replayed without it."""
        play.on = False
        self.red_rejected.append((play.g, play.card, play.origin or play.pos, why))
        self._rewind(play.g)

    def _fix(self, kind, key, value=None):
        self._record(Fix(self.now, kind, key, value))

    def _record(self, ev):
        i = bisect.bisect_right(self.journal_g, ev.g)
        self.journal.insert(i, ev)
        self.journal_g.insert(i, ev.g)
        if ev.g < self.now - 1e-9:  # in the past of what the simulator has played: go back and play it again
            self._rewind(ev.g)
        elif ev.g <= self.now + 1e-9:
            self._apply(ev)
            self.next_ev += 1

    REWIND_KEEP = 60.0  # s of snapshots kept; going back further replays the whole match
    SNAP_EVERY = 0.5    # s

    def _snapshot(self):
        if self.snaps and self.now - self.snaps[-1].g < self.SNAP_EVERY:
            return
        self.snaps.append(Snapshot(self.now, self.battle.clone(), dict(self.sim_index), self.next_ev))
        while self.snaps and self.now - self.snaps[0].g > self.REWIND_KEEP:
            self.snaps.pop(0)

    def _rewind(self, g):
        """The journal changed at battle time g: back to the last copy of the simulator from before g, then
        forward again to where it was, playing the journal on the way."""
        target = self.now
        while self.snaps and self.snaps[-1].g >= g - 1e-9:
            self.snaps.pop()
        if self.snaps:
            s = self.snaps[-1]
            self.battle, self.sim_index, self.next_ev = s.battle.clone(), dict(s.sim_index), s.next_ev
        else:
            self.battle, self.sim_index, self.next_ev = self._new_battle(), {}, 0
        back_to = self.now
        for log in self.damage_log.values():
            log[:] = [x for x in log if x[0] <= back_to]
        self.sim_deaths = [d for d in self.sim_deaths if d[0] <= back_to]
        self.gone = {k: v for k, v in self.gone.items() if v[0] <= back_to}
        self.rewinds += 1
        self._advance(target)

    def _advance(self, g):
        """Steps the simulator to battle time g, applying the journal on the way."""
        while True:
            now = self.now
            while self.next_ev < len(self.journal) and self.journal_g[self.next_ev] <= now + 1e-9:
                self._apply(self.journal[self.next_ev])
                self.next_ev += 1
            if now + self.tick > g + 1e-9 or self.battle.finished:
                return
            self.battle.step()
            self._take_events()

    def _apply(self, ev):
        b = self.battle
        if isinstance(ev, Play):
            if not ev.on:
                return
            team = TEAM[ev.side]
            before = len(b.player(team).plays)
            ev.missing = b.force_play(team, ev.card, None if ev.pos is None else crsim.Vec2(*ev.pos),
                                      max(0, int(round((self.now - ev.g) * 1000))))
            if len(b.player(team).plays) > before:
                self.sim_index[ev.id] = (ev.side, before)
            return
        if ev.kind == "tower":
            e = b.find(self._tower_id(ev.key))
            if e is not None:
                b.twin_set_hp(e.id, int(ev.value))
            return
        e = self._entity(ev.key)
        if ev.kind == "hp" and e is not None:
            b.twin_set_hp(e.id, int(ev.value))
        elif ev.kind == "move" and e is not None:
            b.twin_move(e.id, crsim.Vec2(*ev.value))
        elif ev.kind == "drop" and e is not None:
            self.gone[ev.key] = (self.now, e.pos.x, e.pos.y, e.hp, "drop")
            b.twin_remove(e.id)
        elif ev.kind == "revive" and e is None and ev.key[0] in self.sim_index:
            card, pos, hp = ev.value
            side, index = self.sim_index[ev.key[0]]
            uid = b.twin_spawn(TEAM[side], card, crsim.Vec2(*pos), index, ev.key[1])
            b.twin_set_hp(uid, max(1, int(hp)))
            self.gone.pop(ev.key, None)

    def _entity(self, key):
        """The simulated unit with this key (play id, unit index), None if the simulator doesn't have it."""
        if key is None or key[0] not in self.sim_index:
            return None
        side, index = self.sim_index[key[0]]
        team = TEAM[side]
        for e in self.battle.entities:
            if not e.is_building and e.team == team and e.play == index and e.unit == key[1]:
                return e
        return None

    def _plays_by_index(self):
        """(side, index in the simulated player's plays) -> play id."""
        return {v: pid for pid, v in self.sim_index.items()}

    def _key(self, e, by_index=None):
        """The unit key (play id, unit index) of a simulated troop."""
        pid = (by_index or self._plays_by_index()).get(("blue" if e.team == BLUE else "red", e.play))
        return None if pid is None else (pid, e.unit)

    def _take_events(self):
        by_index = None
        for ev in self.battle.events:
            if self.record_damage and ev.type == crsim.EventType.DAMAGE:
                source, amount = ev.info.rsplit(":", 1)
                self.damage_log.setdefault(ev.entity, []).append((ev.time_ms / 1000.0, source, int(amount)))
            elif ev.type == crsim.EventType.DEATH and ev.play >= 0:
                g = ev.time_ms / 1000.0
                side = "blue" if ev.team == BLUE else "red"
                by_index = by_index or self._plays_by_index()
                pid = by_index.get((side, ev.play))
                key = None if pid is None else (pid, ev.unit)
                if key in self.gone and self.gone[key][4] == "drop":
                    continue  # taken out by the twin (not seen for a while), not killed
                self.sim_deaths.append((g, twin.UNIT_OF_CARD.get(ev.info, ev.info), side, ev.pos.x, ev.pos.y))
                if key is not None:
                    self.gone[key] = (g, ev.pos.x, ev.pos.y, 0, "death")
        self.battle.clear_events()  # (the twin reads them every step; nothing else needs them kept)

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
            self.battle = self._new_battle()
            for side, truth in (("blue", self.blue_truth), ("red", self.red_truth)):
                for gp, card, pos in truth or ():
                    self.add_play(side, card, gp, pos, "truth")
                    if side == "blue" and card in twin.SPELL_CARDS and pos:
                        self.own_spells.append((card, tuple(pos), gp))
        self._advance(g)
        self.sim_deaths = [d for d in self.sim_deaths if self.now - d[0] < self.DEATH_GRACE]
        if t >= self.own_lock_until:
            if self.blue_truth is None:
                self._watch_own_hand(t, seen)
            self._sync_own(seen)
        self.tracker.update(t, detections)
        self._read_unit_hp(t, detections, bars)
        self._place_manual_plays(t, g)
        if self.red_truth is None:
            self._enemy_plays(t, g)
        if self.keep_before_correction:
            self.before_correction = twin.snapshot(self.battle)  # what the simulator made of it on its own
        if self.correct:
            self._correct(t, g, seen)
        self._snapshot()

    TOWER_GONE_AFTER = 3.0  # s without a princess tower's HP bar (timer on screen) = it fell ...
    TOWER_GONE_BELOW = 1300  # ... if it was last read this low (bars are missing at the start of a match,
                             # and a unit can hide one for a moment)
    PRINCESS_HALF = 1.4    # tiles: a troop can't stand on a princess tower ...
    KING_HALF = 2.2        # ... or on the king's (the king figure is taken for archers now and then)
    ON_TOWER_CONF = 0.6    # ... but the princess figure on it (other tower skins than the detector learned
                           # on) is sometimes taken for an archer / musketeer, at low confidence

    def _red_may_deploy(self, x, y, slack=0.7):
        """Could red have put a troop here (pockets after one of our princess towers fell), allowing for the
        detector's position error?"""
        for dx, dy in ((0, 0), (slack, 0), (-slack, 0), (0, slack), (0, -slack)):
            if self.battle.can_deploy_at(RED, crsim.Vec2(x + dx, y + dy)):
                return True
        return False

    def _on_princess_tower(self, x, y, cls=None):
        """On a crown tower, where no troop can be put: a princess tower, or for archers (the king figure is
        taken for them) the king's - other troops there were put right behind it."""
        for team in (BLUE, RED):
            for side in (0, 1):
                e = self.battle.find(self.battle.princess_tower(team, side))
                if e is not None and abs(x - e.pos.x) < self.PRINCESS_HALF and abs(y - e.pos.y) < self.PRINCESS_HALF:
                    return True
            e = self.battle.find(self.battle.king_tower(team))
            if (cls == "archer" and e is not None and abs(x - e.pos.x) < self.KING_HALF
                    and abs(y - e.pos.y) < self.KING_HALF):
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
            if di in by_det and by_det[di].filled == 0 and self.SKIP_EMPTY_BARS:
                continue  # a living unit's bar is never empty: a neighbour's dying bar, or one tinted by a spell
            if di in by_det:
                prev = self.track_hp.get(tr.id)
                frac = fraction(by_det[di], d.cls)
                # Trust a bar reading only when the ones before agree (a bar read wrong, or given to the
                # wrong minion of a bunch, would otherwise wreck the simulated unit): one before for most,
                # two for minions (tiny bars, three of them close together).
                run = prev[3] + 1 if prev is not None and t - prev[1] < 0.35 and abs(prev[0] - frac) < 0.12 else 0
                steady = run >= (2 if d.cls == "minion" else 1)
                self.track_hp[tr.id] = (frac, t, steady, run)
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
    NEW_AFTER = 1.0        # seconds a real unit has to be tracked before it is explained by a play
    RESPAWN_AFTER = 2.5    # a unit just explained (a play, a revived unit) is not looked at again this soon
    FAR_MATCH = 5.0        # tiles: an unmatched simulated unit this close is the one seen, walked off
    DEATH_GRACE = 1.0      # s: a unit the simulator has just killed is still seen dying in the game for a
                           # moment - not a unit the simulator lost (bringing it back put ghosts into fights)
    REVIVE_WINDOW = 60.0   # s: a lost unit of a play this old can still be the one seen (dropped while hidden) ...
    REVIVE_DEAD = 6.0      # ... but one the simulator killed only this recently: older, the one seen is another
                           # unit (live: own units "back" 40 s after dying were the detector naming a knight a
                           # mini pekka; theirs were new plays, refused by the cycle check and replaced by these)
    REVIVE_HP = 0.3        # of max HP, for a unit the simulator killed and the game didn't (until its bar is read)

    def _tower_id(self, name):
        team, side = self.TOWERS[name]
        return self.battle.king_tower(team) if side is None else self.battle.princess_tower(team, side)

    HP_TOLERANCE = 0.08    # of max HP: smaller differences are left to the simulator
    SKIP_EMPTY_BARS = True  # an empty bar is never a living unit's (see _read_unit_hp)
    # "Seen without a bar for a while = undamaged" is off: in a bunch (minions) bars are often not found, and
    # a wounded unit would be healed (checked against both-side recordings). Seconds, or None.
    FULL_AFTER = None

    def _sync_unit_hp(self, unit_id, key, track_id, t):
        """The simulated unit gets the HP its bar shows."""
        e = self.battle.find(unit_id)
        if e is None or key is None:
            return
        frac, source = None, None
        if track_id in self.track_hp and self.track_hp[track_id][1] == t and self.track_hp[track_id][2]:
            frac, source = self.track_hp[track_id][0], "bar"
        elif self.FULL_AFTER is not None and track_id in self.track_no_bar \
                and t - self.track_no_bar[track_id] >= self.FULL_AFTER:
            frac, source = 1.0, "no bar"
        if frac is None:
            return
        target = max(1, round(frac * e.max_hp))
        if abs(e.hp - target) > self.HP_TOLERANCE * e.max_hp:
            g = self.now
            recent = [(s, a) for gt, s, a in self.damage_log.get(e.id, []) if g - gt <= 2.0]
            self.corrections.append((g, "hp", twin.UNIT_OF_CARD.get(e.def_id, e.def_id),
                                     "blue" if e.team == BLUE else "red",
                                     {"sim": e.hp / e.max_hp, "real": frac, "id": e.id, "hit_by": recent,
                                      "from": source}))
            self._fix("hp", key, target)

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

    def _move_context(self, unit_id):
        """What the simulator's unit was doing when it had to be moved (for tools/sim_check.py): attacking or
        walking, at a troop or a building, other troops close by (own side / enemy, centres within 1.5 tiles),
        how far from the nearest building's edge, and where (own half, river / bridges, enemy half)."""
        b = self.battle
        e = b.find(unit_id)
        if e is None:
            return None
        target = b.find(e.target)
        own = enemy = 0
        building_gap = enemy_gap = 99.0
        for o in b.entities:
            if o.id == e.id or o.hp <= 0:
                continue
            d = math.dist((o.pos.x, o.pos.y), (e.pos.x, e.pos.y))
            if o.is_building:
                building_gap = min(building_gap, d - o.radius - e.radius)
                continue
            if o.team != e.team:
                enemy_gap = min(enemy_gap, d - o.radius - e.radius)
            if d < 1.5 and o.air == e.air:
                own += o.team == e.team
                enemy += o.team != e.team
        a = self.data.arena
        y = e.pos.y if e.team == BLUE else a.height - e.pos.y  # in its own frame: 0 = its back line
        zone = "own half" if y < a.river_min_y - 0.5 else "river" if y < a.river_max_y + 0.5 else "enemy half"
        key = self._key(e)
        age = self.now - self.plays[key[0]].g - self.deploy_delay if key is not None else None  # s since it appeared
        return {"locked": bool(e.locked), "at_troop": bool(target is not None and not target.is_building), "age": age,
                "own_close": own, "enemy_close": enemy, "building_gap": building_gap, "enemy_gap": enemy_gap,
                "zone": zone,
                "air": bool(e.air)}

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
        clearly no longer has them; a unit the game has and the simulator doesn't is a play's unit it lost
        (brought back), an opponent's play not noticed yet, or a detector mistake."""
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
                        self._fix("tower", name, 0)
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
                self.corrections.append((g, "tower", name, name.split("_")[0], {"sim": e.hp, "real": hp}))
                self._fix("tower", name, int(hp))
            e = b.find(self._tower_id(name))
            self.tower_ref[name] = (hp, e.hp if e is not None else 0)
        b = self.battle
        tracks = {tr.id: tr for tr in self.tracker.tracks}
        real = [st for st in self.tracker.state(t) if st["seen"] and st["cls"] in twin.CARD_OF_UNIT
                and tracks[st["id"]].landed(t)]  # still dropping in: drawn above where it stands
        # Units still in their deploy timer are on the field (and seen) too: match them as well.
        by_index = self._plays_by_index()
        sim = [{"id": e.id, "key": self._key(e, by_index), "cls": twin.UNIT_OF_CARD.get(e.def_id, e.def_id), "hp": e.hp,
                "max_hp": e.max_hp, "deploying": e.deploy_ms_left > 0, "team": "blue" if e.team == BLUE else "red",
                "x": e.pos.x, "y": e.pos.y}
               for e in b.entities if not e.is_building]
        for r in real:  # a steady HP bar reading of this frame helps tell look-alikes apart
            hp = self.track_hp.get(r["id"])
            r["hp_frac"] = hp[0] if hp is not None and hp[1] == t and hp[2] else None
        pairs = twin.match_units(sim, real, self.MATCH_DIST, prev=self.last_pair)
        self.last_pair = {sim[i]["key"]: real[j]["id"] for i, j, _ in pairs if sim[i]["key"] is not None}
        matched_real = set()
        for i, j, d in pairs:
            s_, r = sim[i], real[j]
            self.sim_seen[s_["key"]] = t
            matched_real.add(j)
            self._sync_unit_hp(s_["id"], s_["key"], r["id"], t)
            if d <= self.MOVE_DEADBAND or s_["key"] is None:
                continue  # within what the detector's boxes jitter: let the simulator place it
            a = self.MOVE_BLEND
            new = (s_["x"] + a * (r["x"] - s_["x"]), s_["y"] + a * (r["y"] - s_["y"]))
            if self._out_of_reach(s_["id"], new):
                continue  # it is hitting something: don't pull it out of its attack range over a measuring error
            self.corrections.append((g, "moved", s_["cls"], s_["team"],
                                     {"sim": (s_["x"], s_["y"]), "real": (r["x"], r["y"]), "id": s_["id"],
                                      "ctx": self._move_context(s_["id"])}))
            self._fix("move", s_["key"], new)
        matched_sim = {sim[i]["key"] for i, _, _ in pairs}
        for s_ in sim:
            if s_["key"] in matched_sim or s_["hp"] <= 0 or s_["deploying"] or s_["key"] is None:
                continue
            last = self.sim_seen.setdefault(s_["key"], t)  # first noticed now: give it time
            limit = self.GONE_HIDDEN if self._maybe_hidden(s_["x"], s_["y"], real) else \
                self.GONE_AFTER.get(s_["cls"], self.GONE_DEFAULT)
            if t - last > limit:
                self.corrections.append((g, "dropped", s_["cls"], s_["team"],
                                         {"pos": (s_["x"], s_["y"]), "hp": s_["hp"], "id": s_["id"]}))
                self._fix("drop", s_["key"])
                self.sim_seen.pop(s_["key"], None)
                self.log.append((g, f"sim {s_['team']} {s_['cls']} dropped: not seen for {t - last:.1f}s"))
        for j, r in enumerate(real):
            tr = tracks.get(r["id"])
            if j in matched_real or tr is None or t - self.spawned_for.get(r["id"], -1e9) < self.RESPAWN_AFTER:
                continue
            if t - tr.first_seen < self.NEW_AFTER or tr.hits < 5:
                continue
            if any(c == r["cls"] and side == r["team"] and math.dist((x, y), (r["x"], r["y"])) < 2.5
                   for _, c, side, x, y in self.sim_deaths):
                continue  # the simulator just killed one here: this is it, dying
            # The simulator may have the unit, just walked too far off to be matched: move that one here
            # (keeping its HP) rather than adding a twin of it.
            far = [s_ for s_ in sim if s_["cls"] == r["cls"] and s_["team"] == r["team"] and s_["hp"] > 0
                   and s_["key"] is not None and s_["key"] not in matched_sim
                   and math.dist((s_["x"], s_["y"]), (r["x"], r["y"])) < self.FAR_MATCH]
            if far:
                s_ = min(far, key=lambda s_: math.dist((s_["x"], s_["y"]), (r["x"], r["y"])))
                matched_sim.add(s_["key"])
                self.sim_seen[s_["key"]] = t
                self.corrections.append((g, "moved", s_["cls"], s_["team"],
                                         {"sim": (s_["x"], s_["y"]), "real": (r["x"], r["y"]), "id": s_["id"]}))
                self._fix("move", s_["key"], (r["x"], r["y"]))
                continue
            self._explain(r, tr, t, g)

    def _explain(self, r, tr, t, g):
        """A unit the game has and the simulator doesn't: which play is it from?"""
        self.spawned_for[r["id"]] = t
        card = twin.CARD_OF_UNIT[r["cls"]]
        lost = self._lost_unit(r["cls"], r["team"], r["x"], r["y"], reach=(4.0, 2.0))
        if lost is not None:
            key, (gg, x, y, hp, how) = lost
            max_hp = self.data.card(card).unit.hitpoints
            bar = self.track_hp.get(r["id"])
            if bar is not None and t - bar[1] < 1.0:
                hp = bar[0] * max_hp
            elif how == "death":
                hp = self.REVIVE_HP * max_hp
            self._fix("revive", key, (card, (r["x"], r["y"]), max(1, int(hp))))
            self.sim_seen[key] = t
            kind = "back" if how == "drop" else "added"  # killed too early: the simulated fighting is off
            self.corrections.append((g, kind, r["cls"], r["team"], {"pos": (r["x"], r["y"]), "key": key}))
            self.log.append((g, f"sim {r['team']} {r['cls']} back ({'dropped' if how == 'drop' else 'killed'} "
                                f"{self.now - gg:.1f}s ago): seen in the game"))
            return
        if r["team"] == "blue":
            # We know every card we played: a blue unit no play of ours explains is the detector's mistake.
            self.log.append((g, f"blue {r['cls']} seen, but none of our plays has it: not added"))
            return
        if r["id"] in self.handled or self.red_truth is not None:  # (with the truth every play is known)
            # Already judged not to be a play (another kind more often seen there, on a tower, the card not
            # back in their hand): the detector's confusion, not a unit to add.
            return
        # Theirs, and not from any play we know: a play we missed.
        x, y = tr.landing_pos()
        play = self.add_play("red", card, g - (t - tr.first_seen) - self.deploy_delay, (x, y), "late", 0.5,
                             [tr.id], accepted_at=t)
        self.handled.add(tr.id)
        self.corrections.append((g, "late play", r["cls"], r["team"], {"pos": (x, y)}))
        self.log.append((g, f"red {card} at ({x:.1f}, {y:.1f}): a play not noticed when made (seen {t - tr.first_seen:.1f}s)"))
        return play

    def _lost_unit(self, cls, side, x, y, window=None, reach=(2.0, 1.5)):
        """(key, gone record) of a unit of one of side's plays that the simulator has lost (killed, or dropped
        while hidden) and that could be the one seen at (x, y) - within reach[0] + reach[1] tiles per second
        since it was lost; None if there is none."""
        best = None
        for (pid, k), rec in self.gone.items():
            p = self.plays[pid]
            if not p.on or p.side != side or twin.UNIT_OF_CARD.get(p.card) != cls:
                continue
            age = self.now - rec[0]
            if age > (window if window is not None else self.REVIVE_WINDOW) or self._entity((pid, k)) is not None:
                continue
            if rec[4] == "death" and age > self.REVIVE_DEAD:
                continue  # killed long ago: what is seen now is another unit (a new play, or the detector's mistake)
            d = math.dist((rec[1], rec[2]), (x, y))
            if d <= reach[0] + reach[1] * age and (best is None or d < best[0]):
                best = (d, (pid, k), rec)
        return None if best is None else (best[1], best[2])

    BEHIND_KING_Y = 31.2   # where a troop put behind their king's tower stands (tiles)
    WALK_OUT_AGE = 1.0     # s: a troop first seen by their king's tower is judged once it is clear whether it
                           # walks (came from behind the tower) or stands (fresh, in its deploy timer)

    def _behind_king(self, x, y):
        return y >= 28.5 and abs(x - 9.0) <= 4.0

    def _walking_out(self, tr):
        """A red troop first seen by their king's tower already walking: it was put behind the tower."""
        x, y = tr.landing_pos()
        if not self._behind_king(x, y):
            return False
        after = [(h[1], h[2]) for h in tr.history if h[0] - tr.first_seen >= DROP_TIME]
        return len(after) >= 2 and math.dist(after[0], after[-1]) >= 0.4

    RETURN_WINDOW = 4.0    # s: a unit seen again this soon after the simulator dropped it (hidden behind a giant,
                           # under an explosion, under the top bar) near there is that unit, not a new play

    def _dropped_in(self, tr):
        """Did this red troop arrive like a fresh play - falling in from above (drawn higher at first), or
        standing still in its deploy timer - rather than walk back into view?"""
        lx, ly = tr.landing_pos()
        if tr.history[0][2] - ly > 0.4:  # the red half is the top of the screen: higher = larger y
            return True
        after = [(h[1], h[2]) for h in tr.history if h[0] - tr.first_seen >= DROP_TIME]
        return len(after) < 2 or math.dist(after[0], after[-1]) < 0.25

    def _sync_own(self, seen):
        hand = [h["card"] for h in seen["hand"]]
        if not any(h["unknown"] for h in seen["hand"]):
            try:
                self.battle.set_hand(BLUE, [c or "" for c in hand], seen["next"]["card"] or "")
            except ValueError:
                pass  # a misread (a card twice, one not in the deck): keep the simulator's idea
        real = seen.get("elixir")
        if real is None:
            return  # not readable this frame (the bar covered): keep the simulator's count
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
                    self.manual.append([since[1], since[0], None])
                    self.empty_since[slot] = None
                    self.last_hand[slot] = None
            else:
                if self.empty_since[slot] is not None and card == self.empty_since[slot][1]:
                    self.empty_since[slot] = None
                self.last_hand[slot] = card

    def _place_manual_plays(self, t, g):
        """Our hand-made plays go into the journal once their units (or spell hit) are seen - at the moment
        they were really made: the simulator goes back and plays them from there. The bot's own plays are
        in the journal from the moment it decided, where it aimed: once their units are seen, the play is
        corrected to when and where the card really came down (touches land late and a little off)."""
        for play in list(self.manual):
            card, t0, bot_play = play
            if t - t0 > 6.0:
                self.manual.remove(play)  # never seen where: the hand sync already cycled it
                if bot_play is None:
                    self.log.append((g, f"blue {card}: played by hand, position not found"))
                continue
            if card in twin.SPELL_CARDS:
                # Its explosion, whatever the detector calls it (a fireball's is taken for arrows' now and then):
                # the right kind first.
                new = [tr for tr in self.tracker.tracks if tr.team in ("blue", "any") and tr.cls.endswith("_hit")
                       and tr.first_seen > t0 + 0.5 and tr.id not in self.own_units]
                new.sort(key=lambda tr: (tr.cls != card + "_hit", tr.first_seen))
                need = 1
            else:
                unit = twin.UNIT_OF_CARD.get(card)
                need = GROUP.get(card, 1)
                new = [tr for tr in self.tracker.tracks if tr.team == "blue" and tr.cls == unit and tr.confirmed
                       and tr.landed(t)
                       and t0 + 0.3 < tr.first_seen < t0 + 4.0 and tr.id not in self.own_units]  # the slot empties when the drag starts
                if len(new) < need and t - t0 < 3.5:
                    continue  # the rest of the group may still show up
            if not new:
                continue
            new = new[:need] if card in twin.SPELL_CARDS else sorted(new, key=lambda tr: tr.first_seen)[:need]
            self.own_units.update(tr.id for tr in new)
            self.spawned_for.update((tr.id, t) for tr in new)  # these tracks are this play's units
            pos = (sum(tr.landing_pos()[0] for tr in new) / len(new), sum(tr.landing_pos()[1] for tr in new) / len(new))
            self.manual.remove(play)
            # When it really was played: its units appear after the blueprint second, its spell lands after
            # that and the flight from our king tower (the hand shows the card leave at the start of the drag,
            # a moment earlier).
            played = min(tr.first_seen for tr in new) - self.deploy_delay
            if card in twin.SPELL_CARDS:
                speed = self.data.card(card).spell.projectile_speed
                played -= math.dist(pos, twin.KING["blue"]) / speed if speed > 0 else 0.0
            gp = g - (t - played)
            if bot_play is not None:
                aimed, decided = bot_play.pos, bot_play.g
                self.revise_play(bot_play, gp, pos, [tr.id for tr in new])
                self.log.append((g, f"blue {card} came down at ({pos[0]:.1f}, {pos[1]:.1f}), "
                                    f"{math.dist(pos, aimed):.1f} tiles from where aimed, {gp - decided:+.1f}s"))
                continue
            if card in twin.SPELL_CARDS:  # the next waves' hits are ours too
                self.own_spells.append((card, pos, gp))
            self.add_play("blue", card, gp, pos, "hand", 1.0, [tr.id for tr in new], accepted_at=t)
            self.log.append((g, f"blue {card} (by hand) at ({pos[0]:.1f}, {pos[1]:.1f})"))

    def _enemy_plays(self, t, g):
        fresh = [tr for tr in self.tracker.tracks if tr.confirmed and tr.id not in self.handled and tr.landed(t)
                 and t - tr.first_seen < 1.8 and tr.id not in self.own_units
                 and (tr.team == "red" or (tr.team == "any" and tr.cls.endswith("_hit")))]
        # A troop is judged once it has been seen a few times (a kind the detector is unsure about flickers);
        # spell hits right away.
        fresh = [tr for tr in fresh if tr.cls.endswith("_hit") or
                 (tr.hits >= self.NEW_PLAY_HITS and t - tr.first_seen >= self.NEW_PLAY_AGE
                  and (not self._behind_king(*tr.landing_pos()) or t - tr.first_seen >= self.WALK_OUT_AGE))]
        for tr in fresh:
            if tr.id in self.handled:
                continue
            self.handled.add(tr.id)
            x, y = tr.landing_pos()
            seen_g = g - (t - tr.first_seen)  # battle time it was first seen
            if tr.cls.endswith("_hit"):
                spell = tr.cls[:-4]
                if self._own_spell_hit((x, y), seen_g):
                    continue  # the hit of a spell we cast
                if any(c.card == spell and c.hit_t is not None and abs(c.hit_t - tr.first_seen) < twin.SPELL_SAME_PLAY[0]
                       and math.dist(c.pos, (x, y)) < twin.SPELL_SAME_PLAY[1] for c in self.accepted_red[-3:]):
                    continue  # another wave of the same arrows
                # Played: the blueprint second plus the flight from their king tower before the hit.
                speed = self.data.card(spell).spell.projectile_speed
                flight = math.dist((x, y), twin.KING["red"]) / speed if speed > 0 else 0.0
                cand = twin.Play("red", spell, seen_g - flight - self.deploy_delay, (x, y), "hit", [tr.id])
                cand.hit_t = tr.first_seen
            else:
                card = twin.CARD_OF_UNIT.get(tr.cls)
                if card is None or not (15.5 <= y <= 32.5 or self._red_may_deploy(x, y)):
                    continue  # not where they can put a troop (our half: only in a pocket of a fallen tower)
                # A tower's own figure taken for a troop stays put; a troop placed behind the king's tower
                # walks out from behind it (first seen on it, moving).
                still = math.dist(tr.history[0][1:], tr.history[-1][1:]) < 0.4
                if self._on_princess_tower(x, y, tr.cls) and tr.conf < self.ON_TOWER_CONF and still:
                    self.log.append((g, f"red {card}? no: on a tower, unsure"))
                    continue
                rival = self._rival_kind(tr)
                if rival is not None:
                    self.log.append((g, f"red {card}? no: more often seen as {rival}"))
                    continue
                part = [p for p in self.accepted_red[-2:] if p.pos is not None and p.card != card
                        and p.hit_t is None and len(p.units) < GROUP.get(p.card, 1)
                        and t - p.accepted_at < 1.5 and math.dist(p.pos, (x, y)) < 2.5]
                if part:  # the rest of a group just played, taken for another kind (an archer seen as a knight)
                    part[0].units.append(tr.id)
                    self.log.append((g, f"red {card}? no: the rest of the {part[0].card} just played"))
                    continue
                if not self._dropped_in(tr) and self._lost_unit(tr.cls, "red", x, y, self.RETURN_WINDOW) is not None:
                    self.log.append((g, f"red {card}? no: a unit of theirs seen again (the simulator had lost it)"))
                    continue
                count = GROUP.get(card, 1)
                group = [tr] + [o for o in fresh if o is not tr and o.id not in self.handled and o.cls == tr.cls
                                and abs(o.first_seen - tr.first_seen) < 0.6
                                and math.dist(o.landing_pos(), (x, y)) < 3.0][:count - 1]
                self.handled.update(o.id for o in group)
                pos = (sum(o.landing_pos()[0] for o in group) / len(group), sum(o.landing_pos()[1] for o in group) / len(group))
                cand = twin.Play("red", card, seen_g - self.deploy_delay, pos, "units", [o.id for o in group])
                if self._walking_out(tr):
                    # Put behind their king's tower (where a push starts), it was hidden until it walked out:
                    # played its blueprint second, its deploy timer and the walk before it was first seen.
                    unit = self.data.card(card).unit
                    origin = (9.0 + max(-1.0, min(1.0, x - 9.0)), self.BEHIND_KING_Y)
                    cand.t -= unit.deploy_time_ms / 1000.0 + math.dist(origin, (x, y)) / max(0.3, unit.speed)
                    cand.origin = origin
            unit = twin.UNIT_OF_CARD.get(cand.card)
            elsewhere = [(o.x, o.y) for o in self.tracker.tracks if o.cls == unit and o.team == "red"
                         and o.last_seen == t and o.id not in cand.units]
            cand.rejected = twin.judge_enemy(self.battle, self.data, cand, self.accepted_red, seen_elsewhere=elsewhere)
            if cand.rejected == "card not back in hand yet" and self._undo_phantom(cand, t, g):
                cand.rejected = None
            if cand.rejected == "card not back in hand yet" and self._unmistakable(cand, tr):
                # The whole group came down (both archers, all three minions) or the spell hit: that is a play,
                # whatever the cycle says - an earlier play was misread, so the cycle is what is wrong.
                self.log.append((g, f"red {cand.card}: cycle out of step, the screen trusted"))
                cand.rejected = None
            if cand.rejected != "card not back in hand yet":
                self.spawned_for.update((u, t) for u in cand.units)  # a new play's units, or units it has
            if cand.rejected:
                self.red_rejected.append((cand.t, cand.card, cand.pos, cand.rejected))
                self.log.append((g, f"red {cand.card}? no: {cand.rejected}"))
                continue
            # In the journal at the moment it was played: the simulator goes back and plays it from there.
            self.add_play("red", cand.card, cand.t, cand.pos, "seen", min(1.0, tr.conf), cand.units,
                          origin=getattr(cand, "origin", None), hit_t=getattr(cand, "hit_t", None), accepted_at=t)
            self.log.append((g, f"red {cand.card} at ({cand.pos[0]:.1f}, {cand.pos[1]:.1f})"))

    NEW_PLAY_HITS = 4      # sightings before a new red troop counts as a play
    NEW_PLAY_AGE = 0.5     # s
    PHANTOM_HITS = 8       # a play whose units were seen fewer times than this, and are gone, was a mistake ...
    PHANTOM_WINDOW = 10.0  # ... if it was this recent (s): an older one was real and died - checked against
                           # both-side recordings (3 of 4 undos were of real plays 14-29 s old, the right one 8 s)

    UNMISTAKABLE_CONF = 0.6

    def _unmistakable(self, cand, tr):
        """A play the cycle check may not overrule: a spell's hit, or a whole group seen landing together.
        (A single troop may be a look-alike named wrong - a knight called a mini pekka - which the cycle check
        rightly catches.)"""
        if cand.source == "hit":
            return tr.conf >= self.UNMISTAKABLE_CONF
        return GROUP.get(cand.card, 1) > 1 and len(cand.units) >= GROUP[cand.card]

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
        only a few times and are gone. Then that play is taken back: the simulator replays the match from
        before it, without it (its units, and what they did, never happened)."""
        same = [p for p in self.accepted_red[-3:] if p.card == cand.card and p.hit_t is None and p.pos is not None
                and p.source in ("seen", "late")]
        if not same or cand.source != "units":
            return False
        old = same[-1]
        if t - old.accepted_at > self.PHANTOM_WINDOW:
            return False
        tracks = {tr.id: tr for tr in self.tracker.tracks}
        live = [tracks[u] for u in old.units if u in tracks]
        seen = sum(tr.hits for tr in live)
        if any(t - tr.last_seen < 0.5 for tr in live) or seen >= self.PHANTOM_HITS:
            return False  # still there, or was well seen: the new one is the doubtful one
        for k in range(GROUP.get(old.card, 1)):  # its simulated unit still matches something seen: not a phantom
            if t - self.sim_seen.get((old.id, k), -1e9) < 1.0:
                return False
        self.take_back(old, "taken back: a phantom")
        self.log.append((g, f"red {old.card} at ({old.pos[0]:.1f}, {old.pos[1]:.1f}) taken back: a phantom"))
        return True

    def _own_spell_hit(self, pos, g):
        """An explosion where and when one of our spells lands is ours, whatever the detector calls it."""
        for card, p, g0 in self.own_spells:
            if g0 + 0.8 < g < g0 + 4.5 + self.deploy_delay and math.dist(p, pos) < 3.0:
                return True
        return False

    # --- our side ----------------------------------------------------------------------------------
    def own_play(self, t, card, pos, lock=1.0):
        """We just played card at pos (simulator tiles)."""
        g = self.game_time(t)
        if g is None or self.battle is None:
            return
        play = self.add_play("blue", card, max(g, self.now), pos, "bot", 1.0, accepted_at=t)
        self.manual.append([card, t, play])  # to be corrected once its units are seen
        self.own_lock_until = t + lock
        if card in twin.SPELL_CARDS:
            self.own_spells.append((card, tuple(pos), g))
            del self.own_spells[:-8]
        # Our next screen readings will see this card leave the hand: it is not a hand-made play.
        self.last_hand = [None if c == card else c for c in self.last_hand]
        self.empty_since = [None if s and s[1] == card else s for s in self.empty_since]
        self.log.append((g, f"blue {card} at ({pos[0]:.1f}, {pos[1]:.1f})"))
