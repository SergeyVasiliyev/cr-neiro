"""Cuts a perception cache of several battles into one cache per battle, so the tools that follow a single battle
(the twin's clock, own_plays_timed, human_examples) can read them. A two-sided recording (name + name_red, both
windows recorded at once, so the same video times) is cut at the same moments on both sides.

    .venv\\Scripts\\python.exe tools\\split_battles.py video43 [video44 ...]

Writes raw/perception/cache/<name>_b01.json, _b02 ... (and <name>_b01_red.json ... for a pair). A battle starts
where the timer shows the start of regulation again after a later moment (or after a stretch without a timer);
a battle's part runs until 2 s before the next one's clock starts. Battles listed in raw/excluded_battles.json (by their
number in the recording, from 1) are left out, and so are clips under --min-battle s.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "raw", "perception", "cache")
REGULATION_S = 180.0


def game_s(fr):
    """The battle second a frame's timer shows (regulation only), or None."""
    sec, overtime = fr.get("timer") or (None, None)
    if sec is None or overtime:
        return None
    return REGULATION_S - sec


def battles(frames, min_battle=20.0):
    """[(first t, last t, start t)] of each battle: runs of frames whose timer goes forward; start t is when its
    clock started (from the first reading), last t the last timer reading (the timer is not always read to the end)."""
    runs, cur, last_g, last_t = [], None, None, None
    for fr in frames:
        g = game_s(fr)
        overtime = (fr.get("timer") or (None, None))[1]
        if g is None and not overtime:
            continue
        t = fr["t"]
        # between two battles there is always a stretch without a timer (the result, the menu, the loading), so a
        # clock going back without one is a misread, not a new battle
        gap = last_t is not None and t - last_t > 8.0
        new = cur is None or (gap and g is not None and last_g is not None and g < last_g - 5) or (t - last_t > 30)
        if new:
            if cur:
                runs.append(cur)
            cur = [t, t, t - (g if g is not None else 0.0)]  # first t, last t, the battle's start (video t)
        cur[1] = t
        if g is not None:
            last_g = g
        last_t = t
    if cur:
        runs.append(cur)
    # the timer is read from the battle's first frames on; a battle's last frames (the result) carry no timer
    return [r for r in runs if r[1] - r[0] >= min_battle]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+")
    p.add_argument("--min-battle", type=float, default=20.0)
    p.add_argument("--keep-excluded", action="store_true",
                   help="write the excluded battles too, as <name>_x03 ... (e.g. the user's tests, for checking the "
                        "simulator; nothing that learns reads those names)")
    args = p.parse_args()
    excluded_path = os.path.join(ROOT, "raw", "excluded_battles.json")
    excluded = json.load(open(excluded_path, encoding="utf-8")) if os.path.exists(excluded_path) else {}
    for name in args.names:
        cache = json.load(open(os.path.join(CACHE, f"{name}.json"), encoding="utf-8"))
        red_path = os.path.join(CACHE, f"{name}_red.json")
        red = json.load(open(red_path, encoding="utf-8")) if os.path.exists(red_path) else None
        runs = battles(cache["frames"], args.min_battle)
        if red is not None:
            # A side's timer can go unread between two battles close together (they then run into one): the side
            # that tells more battles apart cuts both.
            red_runs = battles(red["frames"], args.min_battle)
            if len(red_runs) != len(runs):
                more = "red" if len(red_runs) > len(runs) else "blue"
                print(f"{name}: {len(runs)} battles on the blue side but {len(red_runs)} on the red one - "
                      f"cut by the {more} side's", flush=True)
                if len(red_runs) > len(runs):
                    runs = red_runs
        # a battle runs until the next one's clock starts (its last seconds often have no timer reading)
        cuts = [-1e9] + [b[2] - 2.0 for b in runs[1:]] + [1e9]
        skip = {e["battle"] for e in excluded.get(name, [])}
        for i in range(len(runs)):
            lo, hi = cuts[i], cuts[i + 1]
            tag = f"{name}_b{i + 1:02d}"
            note = "excluded" if i + 1 in skip else ""
            if i + 1 in skip:
                if not args.keep_excluded:
                    print(f"{tag}: {runs[i][1] - runs[i][0]:.0f} s, left out (raw/excluded_battles.json)")
                    continue
                tag = f"{name}_x{i + 1:02d}"  # kept apart: not for learning, still a battle of the real game
            for src, out in ((cache, tag), (red, f"{tag}_red")):
                if src is None:
                    continue
                part = {k: v for k, v in src.items() if k != "frames"}
                part["part_of"] = src.get("video")
                part["frames"] = [fr for fr in src["frames"] if lo <= fr["t"] < hi]
                with open(os.path.join(CACHE, f"{out}.json"), "w", encoding="utf-8") as f:
                    json.dump(part, f)
            print(f"{tag}: {runs[i][0]:.0f}-{runs[i][1]:.0f} s of the video ({runs[i][1] - runs[i][0]:.0f} s) {note}",
                  flush=True)


if __name__ == "__main__":
    sys.exit(main())
