"""The live fight gap, on two-sided recorded battles: does the game deal the bot's side and the user's side the damage
the simulator says, given the same plays? Two measures.

1. Per troop: between two HP-bar readings of the same troop (<= 2.5 s apart), the HP the game shows it lost vs the
   damage the twin's simulator dealt it over the same span (the twin follows the screen; both players' plays come from
   their own recordings), summed per player and battle phase. Only totals mean something: the simulator's and the
   game's hits are out of phase, so splitting short spans by who hit (a tower, a troop) misleads.
2. Per fight: at each defence (a troop put down in its own half with an enemy troop within --near tiles), the twin's
   battle is copied and plays on alone for --horizon s with both players' real plays; each side's troop losses (shares
   of max HP) and tower losses vs the twin's. Fights in which a tower falls, the battle ends, or overtime comes are left
   out ("clean fights"); the end of a battle is no measure of the simulator.

    .venv\\Scripts\\python.exe tools\\fight_gap_check.py [video43_b01 ...]

Each name is the bot's recording of one battle (tools/split_battles.py parts: the bot on LDPlayer 0 is blue there); the
user's is the same name with _red. Default: every such pair in raw/perception/cache. First run, 2026-11-10, on the 22
battles of 2026-11-09: before 2:00 game/simulator 1.01 (bot's troops) and 0.98 (user's); 2:00-3:00 1.10 and 0.88;
overtime 1.23 and 0.70.
"""
import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crsim  # noqa: E402
import twin  # noqa: E402
from compare_red_truth import own_plays_timed, true_plays  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import BLUE, TEAM, LiveMatch, Play  # noqa: E402

CACHE = os.path.join(ROOT, "raw", "perception", "cache")
PHASES = (("before 2:00", 0.0, 120.0), ("2:00-3:00", 120.0, 180.0), ("overtime", 180.0, 1e9))


class ReadingMatch(LiveMatch):
    """The twin, also keeping every HP-bar reading of a troop: (game s, sim id, team, max HP, HP fraction read)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.readings = []

    def _sync_unit_hp(self, unit_id, key, track_id, t):
        e = self.battle.find(unit_id)
        th = self.track_hp.get(track_id)
        if e is not None and key is not None and th and th[1] == t and th[2]:
            self.readings.append((self.now, unit_id, "blue" if e.team == BLUE else "red", e.max_hp, th[0]))
        super()._sync_unit_hp(unit_id, key, track_id, t)


def battles(names):
    if names:
        return names
    out = []
    for path in sorted(glob.glob(os.path.join(CACHE, "video*_b[0-9][0-9].json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.exists(os.path.join(CACHE, f"{name}_red.json")):
            out.append(name)
    return out


def player_of(view, team):
    """Whose troop: in the bot's recording the bot is blue, in the user's the user is."""
    return ("bot" if team == "blue" else "user") if view == "bot" else ("user" if team == "blue" else "bot")


def per_troop(name, view, data, totals):
    bot, user = os.path.join(CACHE, f"{name}.json"), os.path.join(CACHE, f"{name}_red.json")
    blue_path, red_path = (bot, user) if view == "bot" else (user, bot)
    frames = twin.load_cache(blue_path)["frames"]
    red_frames = twin.load_cache(red_path)["frames"]
    m = ReadingMatch(data, red_truth=true_plays(red_frames, data), blue_truth=own_plays_timed(frames, data),
                     record_damage=True)
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr) if "bars" in fr else None)
        m.log.clear()
    by_unit = defaultdict(list)
    for rd in m.readings:
        by_unit[rd[1]].append(rd)
    for uid, rds in by_unit.items():
        hits = sorted(set(m.damage_log.get(uid, [])))
        for a, b in zip(rds, rds[1:]):
            if not 0 < b[0] - a[0] <= 2.5:
                continue
            sim = sum(amount for t, _, amount in hits if a[0] < t <= b[0] + 1e-6)
            phase = next(p for p, lo, hi in PHASES if lo <= a[0] < hi)
            row = totals[(player_of(view, a[2]), phase)][name]
            row[0] += (a[4] - b[4]) * a[3]
            row[1] += sim


def state(b, data):
    troops = {}
    for e in b.entities:
        if not e.is_building and e.play >= 0:
            troops[(e.team == BLUE, e.play, e.unit)] = (max(0, e.hp), e.max_hp)
    towers, standing = {True: 0, False: 0}, 0
    for team in (crsim.Team.BLUE, crsim.Team.RED):
        for i in (b.king_tower(team), b.princess_tower(team, 0), b.princess_tower(team, 1)):
            t = b.find(i)
            if t is not None and t.hp > 0:
                towers[team == crsim.Team.BLUE] += t.hp
                standing += 1
    return troops, towers, standing


def per_fight(name, view, data, horizon, near, fights):
    bot, user = os.path.join(CACHE, f"{name}.json"), os.path.join(CACHE, f"{name}_red.json")
    blue_path, red_path = (bot, user) if view == "bot" else (user, bot)
    frames = twin.load_cache(blue_path)["frames"]
    other = twin.load_cache(red_path)["frames"]
    own = sorted(p for p in own_plays_timed(frames, data) if p[2] is not None)
    m = LiveMatch(data, red_truth=true_plays(other, data), blue_truth=own)
    pending = [p for p in own if p[1] not in twin.SPELL_CARDS and p[2][1] < 16.0]
    starts, seen = [], []
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        g = m.game_time(fr["t"])
        if m.battle is None or g is None or g < 0:
            continue
        while pending and pending[0][0] <= m.now + 0.2:
            p = pending.pop(0)
            if p[0] >= m.now - 0.3:
                starts.append((p, m.now, m.battle.clone()))
        seen.append((m.now, m.battle.clone()))
    plays = sorted((ev for ev in m.journal if isinstance(ev, Play) and ev.on), key=lambda ev: ev.g)
    for (gp, card, pos), g0, b in starts:
        if g0 + horizon > 180.0:
            continue  # overtime (or about to be): sudden death ends fights
        troops0, towers0, standing0 = state(b, data)
        if not any(not e.is_building and e.team != BLUE and e.hp > 0 and e.play >= 0 and
                   math.hypot(e.pos.x - pos[0], e.pos.y - pos[1]) <= near for e in b.entities):
            continue  # nothing of the enemy's close: not a defence
        ahead = [s for s in seen if g0 < s[0] <= g0 + horizon]
        if not ahead or ahead[-1][0] < g0 + horizon - 0.3:
            continue
        todo = [ev for ev in plays if ev.g > g0 + 1e-9]
        k = 0
        g_end, real = ahead[-1]
        while b.time_ms / 1000.0 < g_end - 1e-9:
            now = b.time_ms / 1000.0
            while k < len(todo) and todo[k].g <= now + 1e-9:
                ev = todo[k]
                k += 1
                b.force_play(TEAM[ev.side], ev.card, None if ev.pos is None else crsim.Vec2(*ev.pos),
                             max(0, int(round((now - ev.g) * 1000))))
            b.step()
        sim_troops, sim_towers, sim_standing = state(b, data)
        real_troops, real_towers, real_standing = state(real, data)
        if b.finished or real.finished or min(sim_standing, real_standing) < standing0:
            continue  # a tower fell or the battle ended: no clean fight
        defender = "bot" if view == "bot" else "user"
        for side in (True, False):
            who = defender if side else ("user" if defender == "bot" else "bot")
            lost_sim = lost_real = 0.0
            for key, (hp, mx) in troops0.items():
                if key[0] == side and hp > 0:
                    lost_sim += (hp - sim_troops.get(key, (0, mx))[0]) / mx
                    lost_real += (hp - real_troops.get(key, (0, mx))[0]) / mx
            f = fights[(defender, who)]
            f[0] += lost_sim
            f[1] += lost_real
            f[2] += towers0[side] - sim_towers[side]
            f[3] += towers0[side] - real_towers[side]
        fights[(defender, "n")][0] += 1


def ratio_ci(per_battle):
    v = np.array(list(per_battle.values()), dtype=float)
    if v[:, 1].sum() <= 0:
        return None
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        s = v[rng.integers(0, len(v), len(v))].sum(axis=0)
        if s[1] > 0:
            boot.append(s[0] / s[1])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return v[:, 0].sum() / v[:, 1].sum(), lo, hi, len(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*")
    p.add_argument("--horizon", type=float, default=6.0, help="s the simulator plays a fight on by itself")
    p.add_argument("--near", type=float, default=6.0, help="tiles: an enemy troop this close makes a play a defence")
    args = p.parse_args()
    data = load_data()
    names = battles(args.names)
    totals = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    fights = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for name in names:
        for view in ("bot", "user"):
            try:
                per_troop(name, view, data, totals)
                per_fight(name, view, data, args.horizon, args.near, fights)
            except Exception as e:  # a battle the twin cannot follow
                print(f"{name} ({view}'s recording): skipped ({type(e).__name__}: {e})", flush=True)
    print(f"{len(names)} battles, each followed from both recordings")
    print("\n1. troop damage, game / simulator (95% by battle), between bar readings of the same troop:")
    for phase, _, _ in PHASES:
        parts = []
        for who in ("bot", "user"):
            r = ratio_ci(totals[(who, phase)]) if totals.get((who, phase)) else None
            parts.append(f"{who}'s troops " + (f"{r[0]:.2f} ({r[1]:.2f}..{r[2]:.2f}, {r[3]} battles)" if r else "-"))
        print(f"   {phase:11s}  " + "   ".join(parts))
    print(f"\n2. clean fights ({args.horizon:.0f} s, no tower falls, before overtime): losses in the simulator's own play "
          f"vs the game")
    for defender in ("bot", "user"):
        n = int(fights[(defender, "n")][0])
        if not n:
            continue
        print(f"   the {defender}'s defences: {n}")
        for who in (defender, "user" if defender == "bot" else "bot"):
            s, g, ts, tg = fights[(defender, who)]
            print(f"      {who}'s troops lost {s:6.1f} -> {g:6.1f} ({g / max(s, 1e-9):.2f})   "
                  f"{who}'s towers lost {ts:6.0f} -> {tg:6.0f} HP ({tg / max(ts, 1):.2f})")
    print("   (troops: shares of max HP summed; simulator -> game)")


if __name__ == "__main__":
    main()
