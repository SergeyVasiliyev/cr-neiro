"""Checks a whole recorded match through the live simulator copy: did it describe the match right,
and did it break anywhere?

    .venv\\Scripts\\python.exe tools\\match_report.py [--caches "raw/perception/cache/video?.json"]

Runs the live code (python/live/session.py) over each perception cache and reports:
  clock      - battle start found from the timer, and whether it ever jumped;
  our plays  - cards seen leaving the hand, and how many got a position on the field;
  their plays- accepted / rejected (and why); elixir check: the simulator's opinion of how much elixir
               they lacked (phantom plays show up as big shortfalls);
  following  - share of the units on screen the simulator also has (within 1.5 tiles), simulated units
               nobody sees, and "broken" stretches (following < 60% for 5 s, or 4+ extra units for 3 s);
  towers     - damage between screen readings: the game's vs the simulator's own, with the same units
               on the field: a check of the simulated fighting itself.
Writes a chart per match to raw/perception/report_<video>.png.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from crsim_env import load_data  # noqa: E402
from live.session import BLUE, LiveMatch  # noqa: E402


def follow_stats(m, t):
    """(units seen, of them the simulator has nearby, simulated units nobody sees, simulated units)."""
    sim = [e for e in m.battle.entities if not e.is_building]
    real = [s for s in m.tracker.state(t) if s["seen"] and s["cls"] in twin.CARD_OF_UNIT]

    def same(e, r, d):
        return (twin.UNIT_OF_CARD.get(e.def_id) == r["cls"] and (e.team == BLUE) == (r["team"] == "blue")
                and math.dist((e.pos.x, e.pos.y), (r["x"], r["y"])) < d)
    covered = sum(1 for r in real if any(same(e, r, 1.5) for e in sim))
    ghosts = sum(1 for e in sim if not any(same(e, r, 2.5) for r in real))
    return len(real), covered, ghosts, len(sim)


def broken_stretches(series, bad, min_len):
    """[(start, end)] where bad(row) holds for at least min_len seconds."""
    out, start = [], None
    for g, *row in series:
        if bad(row):
            start = g if start is None else start
        else:
            if start is not None and g - start >= min_len:
                out.append((start, g))
            start = None
    if start is not None and series and series[-1][0] - start >= min_len:
        out.append((start, series[-1][0]))
    return out


def elixir_budget(g):
    """Elixir a player gets by game second g (start + regeneration at x1/x2/x3), ignoring the 10 cap."""
    with open(os.path.join(ROOT, "data", "rules.json"), encoding="utf-8") as f:
        r = json.load(f)
    regen = r["elixir_regen_ms"] / 1000.0
    x2, x3 = r["double_elixir_from_ms"] / 1000.0, r["triple_elixir_from_ms"] / 1000.0
    return (r["start_elixir"] + min(g, x2) / regen + max(0.0, min(g, x3) - x2) * 2 / regen
            + max(0.0, g - x3) * 3 / regen)


def report(path, data, plot=True):
    cache = twin.load_cache(path)
    frames = cache["frames"]
    name = os.path.splitext(cache["video"])[0]
    m = LiveMatch(data)
    series = []  # (game s, seen, covered, ghosts, sim units)
    events = Counter()
    rejected = Counter()
    starts = []
    for fr in frames:
        m.update(fr["t"], twin.seen_from_cache(fr), twin.detections(fr),
                 twin.bars_from_cache(fr) if "bars" in fr else None)
        for _, text in m.log:
            if "dropped" in text:
                events["dropped"] += 1
            elif "added" in text:
                events["added"] += 1
            elif "? no:" in text:
                rejected[text.split("no: ", 1)[1]] += 1
            elif "position not found" in text:
                events["own unplaced"] += 1
            elif text.startswith("blue "):
                events["own placed"] += 1
        m.log.clear()
        if m.start is not None:
            starts.append(m.start)
        g = m.game_time(fr["t"])
        if m.battle is not None and g is not None and g >= 0:
            series.append((g, *follow_stats(m, fr["t"])))

    lines = [f"=== {name} ==="]
    jumps = sum(1 for a, b in zip(starts, starts[1:]) if abs(a - b) > 1.0)
    lines.append(f"clock: battle starts at video {m.start:.2f}s, {'steady' if not jumps else f'{jumps} jumps!'}; "
                 f"match length {series[-1][0]:.0f}s" if series else "clock: no battle found")
    own_hand = len(twin.own_plays_from_hand(frames))
    lines.append(f"our plays: {own_hand} seen leaving the hand; {events['own placed']} placed by hand-tracking, "
                 f"{events['own unplaced']} without a position")
    short = [p for p in m.red_plays if p[3] > 0.05]
    budget = elixir_budget(series[-1][0] if series else 0.0)
    spent = sum(data.card(p[1]).elixir for p in m.red_plays)
    lines.append(f"their plays: {len(m.red_plays)} accepted ({', '.join(f'{c} {n}' for c, n in Counter(p[1] for p in m.red_plays).most_common())})")
    lines.append(f"  rejected: " + (", ".join(f"{n}x {r}" for r, n in rejected.most_common()) or "none"))
    lines.append(f"  elixir check: they spent {spent:.0f} of at most {budget:.0f} they could get (less if they sat at 10); "
                 f"{len(short)} plays with a shortfall" + (f", worst {max(p[3] for p in short):.1f}" if short else ""))
    if series:
        seen = sum(r[1] for r in series)
        cov = sum(r[2] for r in series) / max(1, seen)
        lines.append(f"following: {cov:.0%} of seen units are in the simulator; "
                     f"{statistics.fmean(r[3] for r in series):.1f} simulated units nobody sees per frame; "
                     f"corrections: {events['added']} added, {events['dropped']} dropped")
        bad_follow = broken_stretches([(r[0], r[1], r[2]) for r in series],
                                      lambda row: row[0] >= 3 and row[1] / row[0] < 0.6, 5.0)
        bad_ghost = broken_stretches([(r[0], r[3]) for r in series], lambda row: row[0] >= 4, 3.0)
        if bad_follow or bad_ghost:
            lines.append("  BROKEN stretches: " + ", ".join(
                [f"{a:.0f}-{b:.0f}s following <60%" for a, b in bad_follow] +
                [f"{a:.0f}-{b:.0f}s 4+ extra units" for a, b in bad_ghost]))
        else:
            lines.append("  no broken stretches")
    for side in ("blue", "red"):
        real = sum(v[0] for k, v in m.tower_damage.items() if k.startswith(side))
        sim = sum(v[1] for k, v in m.tower_damage.items() if k.startswith(side))
        if real:
            lines.append(f"towers ({'ours' if side == 'blue' else 'theirs'}): damage in the game {real}, "
                         f"by the simulator's own fighting {sim} ({sim / real:.0%})")
    if plot and series:
        plot_match(name, series, m)
    return "\n".join(lines)


def plot_match(name, series, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = [r[0] for r in series]
    win = 50  # 5 s at 10 frames/s

    def rolling(vals):
        return [statistics.fmean(vals[max(0, i - win):i + 1]) for i in range(len(vals))]
    cov = rolling([r[2] / r[1] if r[1] else 1.0 for r in series])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    ax.plot(g, [100 * c for c in cov], color="#2a7de1")
    ax.axhline(60, color="#999", lw=0.8, ls="--")
    ax.set_ylabel("units on screen\nthe sim also has, %")
    ax.set_ylim(0, 105)
    for gt, card, pos, miss in m.red_plays:
        ax.axvline(gt, color="#d33", alpha=0.25, lw=1)
    ax.set_title(f"{name}: how the simulator followed the match (red lines = their plays)")
    ax = axes[1]
    ax.plot(g, rolling([r[1] for r in series]), label="seen on screen", color="#444")
    ax.plot(g, rolling([r[4] for r in series]), label="in the simulator", color="#2a7de1")
    ax.plot(g, rolling([r[3] for r in series]), label="in the simulator, not seen", color="#e08a1e")
    ax.set_ylabel("units")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[2]
    names = [n for n in ("red_left", "red_king", "red_right", "blue_left", "blue_king", "blue_right") if n in m.tower_damage]
    real = [m.tower_damage[n][0] for n in names]
    sim = [m.tower_damage[n][1] for n in names]
    x = range(len(names))
    axes[2].remove()
    ax = fig.add_subplot(3, 1, 3)
    ax.bar([i - 0.2 for i in x], real, 0.4, label="damage in the game", color="#444")
    ax.bar([i + 0.2 for i in x], sim, 0.4, label="simulator's own fighting", color="#2a7de1")
    ax.set_xticks(list(x), names)
    ax.set_ylabel("tower damage")
    ax.legend(fontsize=8)
    axes[1].set_xlabel("game seconds")
    axes[0].tick_params(labelbottom=True)
    axes[1].tick_params(labelbottom=True)
    fig.tight_layout()
    out = os.path.join(ROOT, "raw", "perception", f"report_{name}.png")
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--caches", default=os.path.join(ROOT, "raw", "perception", "cache", "video?.json"))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    data = load_data()
    for path in sorted(glob.glob(args.caches)):
        print(report(path, data, plot=not args.no_plot), flush=True)
        print()


if __name__ == "__main__":
    main()
