"""Replays a recorded match in the simulator and reports where the two disagree.

    .venv\\Scripts\\python.exe tools\\twin_replay.py --cache raw/perception/cache/video1.json [--video raw/videos/video1.mp4]

Needs a perception cache from tools/perceive_video.py. Prints the clock sync, the plays it found, then
tower HP / elixir / unit positions real vs simulated. With --video also writes raw/perception/twin.mp4:
the real units (solid dots, tracker ids) and the simulated ones (hollow rings, with HP bars) on top of
the recording, so a wrong path or a unit that should be dead is easy to spot.
"""
import argparse
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from crsim_env import load_data  # noqa: E402
from perception.tracker import Tracker  # noqa: E402

COLORS = {"blue": (255, 140, 40), "red": (40, 40, 255)}


def fmt_pos(p):
    return f"({p[0]:4.1f},{p[1]:4.1f})" if p else "   (?)    "


def real_tower_hp(frames, half_window=5):
    """Tower HP per frame as the median of the readings around it (single misread frames vanish);
    None where most frames show no HP bar."""
    out = []
    for k in range(len(frames)):
        window = frames[max(0, k - half_window):k + half_window + 1]
        row = {}
        for n in frames[k]["towers"]:
            values = [fr["towers"][n][0] for fr in window]
            known = sorted(v for v in values if v is not None)
            row[n] = known[len(known) // 2] if len(known) > len(values) // 2 else None
        out.append(row)
    return out


def real_states(frames):
    tracker = Tracker()
    out = []
    for fr in frames:
        tracker.update(fr["t"], twin.detections(fr))
        out.append([s for s in tracker.state(fr["t"]) if s["cls"] in twin.CARD_OF_UNIT])
    return out


def draw_video(path, frames, real, sim, start, out_path):
    import cv2
    from perception.layout import Layout

    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    writer = None
    for fr, rs, ss in zip(frames, real, sim):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(fr["t"] * vfps)))
        ok, img = cap.read()
        if not ok:
            break
        L = Layout.for_frame(img)
        for r in rs:
            x, y = (int(v) for v in L.tile_to_px(r["x"], r["y"]))
            cv2.circle(img, (x, y), 8 if r["seen"] else 5, COLORS[r["team"]], -1)
        if ss:
            for u in ss["units"]:
                x, y = (int(v) for v in L.tile_to_px(u["x"], u["y"]))
                color = COLORS[u["team"]]
                cv2.circle(img, (x, y), 22, (255, 255, 255), 5)
                cv2.circle(img, (x, y), 22, color, 3)
                frac = max(0.0, u["hp"] / u["max_hp"])
                cv2.rectangle(img, (x - 22, y - 34), (x + 22, y - 28), (40, 40, 40), -1)
                cv2.rectangle(img, (x - 22, y - 34), (x - 22 + int(44 * frac), y - 28), color, -1)
            for s in ss["spells"]:
                x, y = (int(v) for v in L.tile_to_px(s["x"], s["y"]))
                cv2.drawMarker(img, (x, y), COLORS[s["team"]], cv2.MARKER_STAR, 40, 4)
            text = (f"sim {ss['g']:5.1f}s  elixir {ss['elixir']:.1f} (real {fr['elixir']:.1f})  "
                    f"towers sim/real: " + " ".join(
                        f"{k.split('_')[1][0]}{k[0]}:{ss['towers'][k]}/{fr['towers'][k][0] or '-'}"
                        for k in ("red_left", "red_right", "blue_left", "blue_right")))
            cv2.putText(img, text, (10, 1580), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
            cv2.putText(img, text, (10, 1580), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        if writer is None:
            fps = 1.0 / max(1e-3, frames[1]["t"] - frames[0]["t"])
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer:
        writer.release()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--video", default=None, help="draw the comparison video from this recording")
    p.add_argument("--display", default="ceil", choices=("ceil", "floor"), help="how the timer rounds")
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "perception", "twin.mp4"))
    args = p.parse_args()

    cache = twin.load_cache(args.cache)
    frames = cache["frames"]
    data = load_data()
    speeds = {c: data.card(c).spell.projectile_speed for c in twin.SPELL_CARDS}

    start, agree = twin.battle_start(frames, args.display)
    print(f"battle starts at video {start:.2f}s (timer readings agreeing: {agree:.0%})")

    tracks = twin.run_tracker(frames)
    own = twin.own_plays_from_hand(frames)
    twin.place_own_plays(own, tracks)
    own_ids = {i for pl in own for i in pl.units}
    candidates = twin.enemy_candidates(tracks, own_ids, spell_speed=speeds)

    states, missing = twin.replay(own, candidates, frames, start, data)
    print("\nplays (game time, side, card, position, source, notes; short = simulator's missing elixir):")
    for pl in sorted(own + candidates, key=lambda pl: pl.t):
        miss = missing.get(id(pl))
        note = f"  short {miss:.1f}" if miss is not None and miss > 0.05 else ""
        if pl.pos is None:
            note += "  position unknown (cycle + elixir only)"
        if pl.rejected:
            note += f"  -- rejected: {pl.rejected}"
        print(f"  {pl.t - start:6.1f}s  {pl.team:4s}  {pl.card:10s}  {fmt_pos(pl.pos)}  {pl.source:5s}{note}")

    # Our elixir: the whole number on the bar is read reliably (the filling segment less so).
    pairs = [(int(fr["elixir"]), int(st["elixir"] + 1e-6)) for fr, st in zip(frames, states)
             if st is not None and fr["timer"][0] is not None]
    if pairs:
        same = sum(r == s for r, s in pairs) / len(pairs)
        near = sum(abs(r - s) <= 1 for r, s in pairs) / len(pairs)
        print(f"\nour elixir (whole units) real vs sim: equal {same:.0%}, within 1: {near:.0%}, "
              f"median real - sim {statistics.median(r - s for r, s in pairs):+.0f}")

    print("\ntower HP real / sim every 10 s:")
    names = ("red_left", "red_king", "red_right", "blue_left", "blue_king", "blue_right")
    print("   time  " + "  ".join(f"{n:>13s}" for n in names))
    next_print = 0.0
    first_gap = {}
    smooth = real_tower_hp(frames)
    for k, (fr, st) in enumerate(zip(frames, states)):
        if st is None:
            continue
        for n in names:
            real_hp = smooth[k][n]
            if (st["g"] > 3.0 and real_hp is not None and abs(real_hp - st["towers"][n]) > 150
                    and n not in first_gap):
                first_gap[n] = (st["g"], real_hp, st["towers"][n])
        if st["g"] >= next_print:
            row = []
            for n in names:
                real_hp = smooth[k][n]
                row.append(f"{(real_hp if real_hp is not None else '-')!s:>6}/{st['towers'][n]:<6d}")
            print(f"  {st['g']:5.1f}  " + "  ".join(row))
            next_print += 10.0
    for n, (g, r, s) in sorted(first_gap.items(), key=lambda kv: kv[1][0]):
        print(f"  {n} first differs by >150 HP at {g:.1f}s: real {r}, sim {s}")

    # Units: position error of matched pairs, and units only one side has.
    real = real_states(frames)
    errors, sim_only, real_only = {}, {}, {}
    for rs, st in zip(real, states):
        if st is None:
            continue
        sim_units = [u for u in st["units"] if not u["deploying"]]
        seen = [r for r in rs if r["seen"]]
        pairs = twin.match_units(sim_units, seen)
        for i, j, d in pairs:
            errors.setdefault(sim_units[i]["cls"], []).append(d)
        matched_s = {i for i, _, _ in pairs}
        matched_r = {j for _, j, _ in pairs}
        for i, u in enumerate(sim_units):
            if i not in matched_s:
                sim_only[u["cls"]] = sim_only.get(u["cls"], 0) + 1
        for j, r in enumerate(seen):
            if j not in matched_r:
                real_only[r["cls"]] = real_only.get(r["cls"], 0) + 1
    print("\nunits (counted over frames): median position error of matched pairs, unmatched sim / real")
    for cls in sorted(set(errors) | set(sim_only) | set(real_only)):
        e = errors.get(cls, [])
        med = f"{statistics.median(e):.2f} tiles" if e else "   -     "
        print(f"  {cls:11s}  {med}  (n={len(e):4d})   sim only {sim_only.get(cls, 0):4d}   real only {real_only.get(cls, 0):4d}")

    if args.video:
        draw_video(args.video, frames, real, states, start, args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
