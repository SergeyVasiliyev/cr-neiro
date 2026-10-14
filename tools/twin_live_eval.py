"""How closely the live simulator copy follows real matches: runs the live code (python/live/session.py)
over perception caches, with and without the corrections from what is seen.

    .venv\\Scripts\\python.exe tools\\twin_live_eval.py [--caches "raw/perception/cache/video?.json"]

Per match: the share of seen units that have a simulated unit nearby (< 1.5 tiles), the simulated units
nobody sees (and that are not plausibly hidden), tower HP error, and how many corrections were needed.
"""
import argparse
import glob
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import BLUE, LiveMatch  # noqa: E402


def evaluate(frames, data, correct):
    m = LiveMatch(data, correct=correct)
    covered = seen_total = 0
    ghosts = []
    tower_err = []
    events = {"dropped": 0, "added": 0, "red plays": 0}
    for fr in frames:
        t = fr["t"]
        m.update(t, twin.seen_from_cache(fr), twin.detections(fr))
        for _, text in m.log:
            if "dropped" in text:
                events["dropped"] += 1
            elif "added" in text:
                events["added"] += 1
            elif text.startswith("red ") and "?" not in text:
                events["red plays"] += 1
        m.log.clear()
        if m.battle is None:
            continue
        sim = [e for e in m.battle.entities if not e.is_building]
        real = [s for s in m.tracker.state(t) if s["seen"] and s["cls"] in twin.CARD_OF_UNIT]
        for r in real:
            seen_total += 1
            covered += any(twin.UNIT_OF_CARD.get(e.def_id) == r["cls"] and (e.team == BLUE) == (r["team"] == "blue")
                           and math.dist((e.pos.x, e.pos.y), (r["x"], r["y"])) < 1.5 for e in sim)
        ghosts.append(sum(1 for e in sim if not any(
            twin.UNIT_OF_CARD.get(e.def_id) == r["cls"] and math.dist((e.pos.x, e.pos.y), (r["x"], r["y"])) < 2.5
            for r in real)))
        for name, (hp, exact) in fr["towers"].items():
            if exact and hp is not None:
                e = m.battle.find(m._tower_id(name))
                if e is not None:
                    tower_err.append(abs(e.hp - hp))
    return {"covered": covered / max(1, seen_total), "ghosts": statistics.fmean(ghosts) if ghosts else 0,
            "tower_err": statistics.median(tower_err) if tower_err else 0, **events}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caches", default=os.path.join(ROOT, "raw", "perception", "cache", "video?.json"))
    args = p.parse_args()
    data = load_data()
    print("match    corrections  seen units the sim has  sim units unseen/frame  tower HP err  dropped  added")
    for path in sorted(glob.glob(args.caches)):
        frames = twin.load_cache(path)["frames"]
        for correct in (False, True):
            r = evaluate(frames, data, correct)
            print(f"{os.path.basename(path):8s} {'on ' if correct else 'off'}          {r['covered']:6.0%}"
                  f"                  {r['ghosts']:5.1f}                {r['tower_err']:6.0f}      "
                  f"{r['dropped']:5d}  {r['added']:5d}", flush=True)


if __name__ == "__main__":
    main()
