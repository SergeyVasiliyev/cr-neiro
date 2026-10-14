"""Replays a two-sided recording in the simulator with no help from the screen and compares, play by play,
how long the units live and what the towers lose. Made for "experiments": a few cards at a time
(a lone giant into a tower, minions under a tower, ...), where nothing else interferes.

    .venv\\Scripts\\python.exe tools\\scenario_check.py --match video12

Both players' plays come from their own hands, timed by when their units appear (or their spell lands).
Game: a unit lives from its first sighting to its last one (in its own player's recording, which keeps it
in view best). Simulator: from appearing to dying. Tower HP: read on the blue recording, against the
simulator's at the same moment.
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import twin  # noqa: E402
from crsim_env import DEFAULT_DECK, load_data  # noqa: E402

W, H = 18.0, 32.0
BLUE, RED = crsim.Team.BLUE, crsim.Team.RED


def plays_with_units(frames, data, deploy_delay=1.0):
    """A player's own plays: [{t, card, pos, lives: [(first, last) game s per unit]}] in its own view."""
    start, _ = twin.battle_start(frames)
    tracks = twin.run_tracker(frames)
    by_id = {tr.id: tr for tr in tracks}
    plays = twin.own_plays_from_hand(frames)
    twin.place_own_plays(plays, tracks)
    out = []
    for p in plays:
        seen = [by_id[u] for u in p.units if u in by_id]
        t = p.t
        if seen and p.card in twin.SPELL_CARDS and p.pos is not None:
            speed = data.card(p.card).spell.projectile_speed
            t = min(tr.first_seen for tr in seen) - math.dist(p.pos, twin.KING["blue"]) / speed - deploy_delay
        elif seen:
            t = min(tr.first_seen for tr in seen) - deploy_delay
        lives = [] if p.card in twin.SPELL_CARDS else [(tr.first_seen - start, tr.last_seen - start) for tr in seen]
        out.append({"t": t - start, "card": p.card, "pos": p.pos, "lives": lives})
    return out, start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--match", default="video12")
    args = p.parse_args()
    data = load_data()
    cache = os.path.join(ROOT, "raw", "perception", "cache")
    blue_frames = twin.load_cache(os.path.join(cache, f"{args.match}.json"))["frames"]
    red_frames = twin.load_cache(os.path.join(cache, f"{args.match}_red.json"))["frames"]
    blue, blue_start = plays_with_units(blue_frames, data)
    red, _ = plays_with_units(red_frames, data)
    for r in red:  # into blue's view
        if r["pos"] is not None:
            r["pos"] = (W - r["pos"][0], H - r["pos"][1])
    plays = sorted([dict(p, team=BLUE, side="blue") for p in blue] + [dict(p, team=RED, side="red") for p in red],
                   key=lambda p: p["t"])

    battle = crsim.Battle(data, DEFAULT_DECK, DEFAULT_DECK, 0)
    battle.set_follow_real(True)
    tick = data.rules.tick_ms / 1000.0
    born, died = {}, {}  # sim entity id -> game s
    owner = {}           # sim entity id -> play index
    k = 0
    towers_sim = []      # (game s, {tower: hp})
    end = max(fr["t"] for fr in blue_frames) - blue_start
    while battle.time_ms / 1000.0 < end:
        g = battle.time_ms / 1000.0
        while k < len(plays) and plays[k]["t"] <= g:
            pl = plays[k]
            battle.force_play(pl["team"], pl["card"], crsim.Vec2(*pl["pos"]) if pl["pos"] else None)
            k += 1
        before = {e.id for e in battle.entities}
        battle.step()
        g = battle.time_ms / 1000.0
        for e in battle.entities:
            if e.id not in before and not e.is_building:
                born[e.id] = g
                # the most recent play of this card by this side that still lacks units
                for i in range(k - 1, -1, -1):
                    pl = plays[i]
                    if pl["card"] == e.def_id and pl["team"] == e.team:
                        owner[e.id] = i
                        break
        for ev in battle.events:
            if ev.type == crsim.EventType.DEATH:
                died[ev.entity] = g
        battle.clear_events()
        if int(g * 10) % 5 == 0:
            towers_sim.append((g, twin.tower_hp(battle)))

    print(f"{args.match}: {len(plays)} plays; unit lifetimes (from appearing to dying), game vs simulator")
    print("   play at  side  card        game lives (s)                simulator lives (s)")
    for i, pl in enumerate(plays):
        if pl["card"] in twin.SPELL_CARDS:
            print(f"  {pl['t']:6.1f}s  {pl['side']:4s}  {pl['card']:10s}  (spell)")
            continue
        game = sorted(round(b - a, 1) for a, b in pl["lives"])
        sim = sorted(round(died.get(eid, end) - born[eid], 1) for eid, o in owner.items() if o == i)
        print(f"  {pl['t']:6.1f}s  {pl['side']:4s}  {pl['card']:10s}  {str(game):28s}  {str(sim)}")
    print("\n(a unit still alive at the end of the recording counts to the end; a game unit can also leave the view)")

    print("\ntower HP, game (blue recording) / simulator, every 10 s:")
    names = ("red_left", "red_king", "red_right", "blue_left", "blue_king", "blue_right")
    next_t = 0.0
    for fr in blue_frames:
        g = fr["t"] - blue_start
        if g < next_t:
            continue
        next_t += 10.0
        sim = min(towers_sim, key=lambda ts: abs(ts[0] - g))[1] if towers_sim else {}
        print(f"  {g:5.0f}s  " + "  ".join(f"{n.replace('_', ' ')} {fr['towers'][n][0] if fr['towers'][n][0] is not None else '-'}/{sim.get(n, '-')}"
                                       for n in names))


if __name__ == "__main__":
    main()
