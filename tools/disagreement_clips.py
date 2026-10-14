"""Short clips of the moments the simulator disagreed most with a recorded match, to look at by eye.

    .venv\\Scripts\\python.exe tools\\disagreement_clips.py --matches video8 video9 video10 video11 [--count 8]

For two-sided recordings (raw/videos/<name>.mp4 + <name>_red.mp4, read into raw/perception/cache). Both
players' plays are taken from their own hands, so what differs is the simulator itself. Picks the biggest
disagreements (a unit the simulator took much lower than its HP bar shows, a unit it walked far off, one it
lost that lives on in the game) and writes raw/perception/disagreements.mp4 at half speed: the game with
the simulator's own state on top (rings with HP bars, before the twin corrects it), the unit in question
in yellow, and on the right what disagreed and who hit that unit in the simulator.
"""
import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin  # noqa: E402
from compare_red_truth import own_plays_timed, true_plays  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import LiveMatch  # noqa: E402
from perception.layout import Layout  # noqa: E402

COLORS = {"blue": (255, 140, 40), "red": (40, 40, 255)}
BEFORE, AFTER = 2.5, 0.7  # seconds of each clip around the moment
PANEL_W = 900


def follow(name, data):
    blue = twin.load_cache(os.path.join(ROOT, "raw", "perception", "cache", f"{name}.json"))["frames"]
    red = twin.load_cache(os.path.join(ROOT, "raw", "perception", "cache", f"{name}_red.json"))["frames"]
    m = LiveMatch(data, red_truth=true_plays(red, data), blue_truth=own_plays_timed(blue, data), record_damage=True)
    m.keep_before_correction = True
    states = []
    for fr in blue:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr), twin.bars_from_cache(fr))
        m.log.clear()
        states.append((fr, m.game_time(fr["t"]), m.before_correction))
    return m, states


def pick(m, name, count_hint):
    """(score, game s, sim unit id or None, description) of the worst disagreements."""
    out = []
    for g, kind, cls, side, d in m.corrections:
        if kind == "hp" and d["real"] - d["sim"] > 0.35:
            out.append((d["real"] - d["sim"] + 0.5, g, d["id"],
                        f"{side} {cls}: HP in the simulator {d['sim']:.0%}, in the game {d['real']:.0%} - "
                        f"the simulator hit it too hard (or the game's attacker missed)"))
        elif kind == "hp" and d["sim"] - d["real"] > 0.35:
            out.append((d["sim"] - d["real"], g, d["id"],
                        f"{side} {cls}: HP in the simulator {d['sim']:.0%}, in the game {d['real']:.0%} - "
                        f"the game hit it harder"))
        elif kind == "moved":
            sx, sy = d["sim"]
            rx, ry = d["real"]
            dist = ((rx - sx) ** 2 + (ry - sy) ** 2) ** 0.5
            if dist > 1.5:
                out.append((dist / 3, g, d["id"],
                            f"{side} {cls}: {dist:.1f} tiles apart - simulator ({sx:.1f}, {sy:.1f}), "
                            f"game ({rx:.1f}, {ry:.1f})"))
        elif kind == "added":
            out.append((0.8, g, None, f"{side} {cls}: alive in the game, the simulator had lost it "
                                      f"(at {d['pos'][0]:.1f}, {d['pos'][1]:.1f})"))
    out.sort(reverse=True)
    chosen = []
    for item in out:
        if all(abs(item[1] - c[1]) > BEFORE + AFTER + 1 for c in chosen):
            chosen.append(item)
        if len(chosen) >= count_hint:
            break
    return [(name,) + c for c in chosen]


def draw(img, state, focus, lines):
    L = Layout.for_frame(img)
    if state:
        for u in state["units"]:
            x, y = (int(v) for v in L.tile_to_px(u["x"], u["y"]))
            color = COLORS[u["team"]]
            r = 26
            cv2.circle(img, (x, y), r, (0, 255, 255) if u["id"] == focus else (255, 255, 255), 7 if u["id"] == focus else 5)
            cv2.circle(img, (x, y), r, color, 3)
            frac = max(0.0, u["hp"] / max(1, u["max_hp"]))
            cv2.rectangle(img, (x - r, y + r + 4), (x + r, y + r + 12), (40, 40, 40), -1)
            cv2.rectangle(img, (x - r, y + r + 4), (x - r + int(2 * r * frac), y + r + 12), color, -1)
            cv2.putText(img, f"sim {u['cls'][:6]}", (x - r, y + r + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
            cv2.putText(img, f"sim {u['cls'][:6]}", (x - r, y + r + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for s in state["spells"]:
            x, y = (int(v) for v in L.tile_to_px(s["x"], s["y"]))
            cv2.drawMarker(img, (x, y), COLORS[s["team"]], cv2.MARKER_STAR, 50, 4)
    panel = np.full((img.shape[0], PANEL_W, 3), 32, np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(panel, line, (16, 60 + 46 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 255) if k == 0 else (235, 235, 235), 2, cv2.LINE_AA)
    out = np.hstack([img, panel])
    return cv2.resize(out, (int(out.shape[1] * 900 / out.shape[0]), 900), interpolation=cv2.INTER_AREA)


def wrap(text, width=58):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matches", nargs="+", default=["video8", "video9", "video10", "video11"])
    p.add_argument("--count", type=int, default=8, help="clips in all")
    p.add_argument("--out", default=os.path.join(ROOT, "raw", "perception", "disagreements.mp4"))
    args = p.parse_args()

    data = load_data()
    runs, events = {}, []
    for name in args.matches:
        m, states = follow(name, data)
        runs[name] = (m, states)
        events += pick(m, name, args.count)
    events.sort(key=lambda e: -e[1])
    events = sorted(events[:args.count], key=lambda e: (e[0], e[2]))

    writer = None
    listing = []
    for k, (name, score, g0, focus, text) in enumerate(events, 1):
        m, states = runs[name]
        cap = cv2.VideoCapture(os.path.join(ROOT, "raw", "videos", f"{name}.mp4"))
        log = m.damage_log.get(focus, []) if focus is not None else []
        listing.append(f"{k}. {name} {g0:.1f}s  {text}")
        for fr, g, state in states:
            if g is None or not g0 - BEFORE <= g <= g0 + AFTER:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, fr["t"] * 1000)
            ok, img = cap.read()
            if not ok:
                continue
            if img.shape[1] != 1080:
                img = cv2.resize(img, (1080, 1920))
            hits = [f"  {gt:6.1f}s  hit by {src} for {amt}" for gt, src, amt in log if g0 - BEFORE <= gt <= g]
            lines = [f"{k}/{len(events)}  {name}  game {g:6.1f}s  (moment {g0:.1f}s)"] + wrap(text) + \
                ["", "yellow ring = that unit in the simulator,", "rings = the simulator's own state", "",
                 "simulator damage to it so far:"] + (hits[-9:] or ["  none"])
            frame = draw(img, state, focus, lines)
            if writer is None:
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(frame)
    if writer:
        writer.release()
    print("\n".join(listing))
    print(f"\nwrote {args.out} (half speed)")


if __name__ == "__main__":
    main()
