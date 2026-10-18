"""How the HP bars of small troops read: the bar lengths found over minions and archers, and the HP fractions
they give, against the fractions a tower's shots leave (109 per shot: a minion 230 HP -> 53%, 5%; an archer
304 -> 64%, 28%), and the same for knights as a reference.

    .venv\\Scripts\\python.exe tools\\bar_check.py [caches...]
"""
import glob
import json
import os
import statistics
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import twin  # noqa: E402
from perception.unit_hp import BAR_LENGTH, assign, fraction  # noqa: E402
from perception.units import Detection  # noqa: E402

KINDS = {"minion": 230, "archer": 304, "knight": 1766, "musketeer": 721, "mini_pekka": 1390}


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "raw", "perception", "cache", "video*.json")))
    lengths = {k: Counter() for k in KINDS}
    fracs = {k: [] for k in KINDS}
    for path in files:
        for f in json.load(open(path, encoding="utf-8"))["frames"]:
            bars = twin.bars_from_cache(f)
            if not bars:
                continue
            ds = [Detection(d[0], d[1], d[2], tuple(d[3:7]), (d[7], d[8])) for d in f.get("det", [])]
            for di, bar in assign(bars, ds).items():
                cls = ds[di].cls
                if cls in KINDS:
                    lengths[cls][bar.length] += 1
                    fracs[cls].append(fraction(bar, cls))
    for cls, hp in KINDS.items():
        n = sum(lengths[cls].values())
        if not n:
            continue
        top = ", ".join(f"{L}px x{c}" for L, c in lengths[cls].most_common(6))
        print(f"{cls:11s} bars {n:6d}; full length assumed {BAR_LENGTH.get(cls)}px; lengths seen: {top}")
        after = [max(0, hp - 109 * k) / hp for k in (1, 2)]
        hist = Counter(round(x * 20) / 20 for x in fracs[cls])
        print(f"            fractions read (5% bins): " + ", ".join(f"{b:.2f}:{c}" for b, c in sorted(hist.items())))
        print(f"            one tower shot leaves {after[0]:.2f}, two {after[1]:.2f}")


if __name__ == "__main__":
    main()
