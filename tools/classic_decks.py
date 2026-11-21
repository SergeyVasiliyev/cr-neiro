"""The Classic Decks seen in the random-bot battles (tools/random_battles.py): each side's 8 cards per battle, the
unique cards over all battles and the decks.

    .venv\\Scripts\\python.exe tools\\classic_decks.py              all battles in raw/classic
    .venv\\Scripts\\python.exe tools\\classic_decks.py c003 c008    only these

Writes raw/classic/decks.json, raw/classic/cards/kNN.png (each unique card's picture, lit) and
raw/classic/cards/sheet.png (all cards, then every deck as a row). Cards are told apart by the edges of their
pictures in the hand (the same picture lit or greyed matches). Names come from raw/classic/card_names.json
({"k01": "goblin_barrel", ...}, sim card ids where the sim has the card), written by hand after looking at the sheet;
ids are kept from one run to the next by matching the saved pictures.
"""
import argparse
import json
import os
import re
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from perception.layout import Layout  # noqa: E402
from perception.reader import ScreenReader, edge_features, hand_art, saturation  # noqa: E402

OUT = os.path.join(ROOT, "raw", "classic")
CARDS = os.path.join(OUT, "cards")
STEP = 0.5        # s between looks at the hand
EMPTY_EDGE = 15.0  # as in random_battles.RandomBot: an empty slot 3-10, a card 20+
STABLE = 0.85     # the same slot's edges at the look before: not sliding in
SAME = 0.7        # edge correlation of two views of one card (a changed slot scored above it 1 time in 421)
MIN_SEEN = 4      # looks for a card to count in a side's deck (a stray view of a card in motion stays below)


def corr(a, b):
    return float(a @ b) / len(a)


def edge_energy(art):
    g = cv2.cvtColor(art, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean() + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())


def side_cards(path, reader):
    """The cards seen in the hand of one recording: [{feat, crop, sat, n}], most seen first."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    step = max(1, round(fps * STEP))
    layout = Layout()
    prev = [None] * 4
    found = []
    k = 0
    while cap.grab():
        if k % step == 0:
            ok, f = cap.retrieve()
            if ok:
                f = cv2.resize(f, (1080, 1920), interpolation=cv2.INTER_LINEAR) if f.shape[1] != 1080 else f
                if reader.numbers.read_timer(f)[0] is None:
                    prev = [None] * 4
                else:
                    for s in range(4):
                        crop = layout.hand_slot(f, s)
                        art = hand_art(crop)
                        feat = edge_features(art)
                        ready = edge_energy(art) >= EMPTY_EDGE and prev[s] is not None and corr(prev[s], feat) > STABLE
                        prev[s] = feat
                        sat = saturation(art)
                        if not ready or sat <= reader.color_threshold:
                            continue  # (a greyed card carries the elixir wedge, which moves its edges: lit only)
                        best = max(((corr(c["feat"], feat), c) for c in found), key=lambda x: x[0], default=(-1.0, None))
                        if best[0] > SAME:
                            c = best[1]
                            c["n"] += 1
                            if sat > c["sat"]:
                                c["crop"], c["sat"] = crop.copy(), sat
                        else:
                            found.append({"feat": feat, "crop": crop.copy(), "sat": sat, "n": 1})
        k += 1
    cap.release()
    return sorted(found, key=lambda c: -c["n"])


def load_known():
    """Unique cards saved by earlier runs: {id: features}, and the names written by hand."""
    known = {}
    if os.path.isdir(CARDS):
        for f in sorted(os.listdir(CARDS)):
            if re.match(r"k\d+\.png$", f):
                known[f[:-4]] = edge_features(hand_art(cv2.imread(os.path.join(CARDS, f))))
    names_path = os.path.join(OUT, "card_names.json")
    names = json.load(open(names_path, encoding="utf-8")) if os.path.exists(names_path) else {}
    return known, names


def label(img, text, y, color=(255, 255, 255), size=0.55):
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def tile(crop, top, bottom=""):
    img = cv2.resize(crop, (120, 148))
    img = cv2.copyMakeBorder(img, 0, 36, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    label(img, top, 162, (120, 230, 255))
    label(img, bottom, 179, (220, 220, 220), 0.42)
    return img


def sheet(rows):
    width = max(len(r) for r in rows)
    blank = np.zeros_like(rows[0][0])
    return cv2.vconcat([cv2.hconcat(r + [blank] * (width - len(r))) for r in rows])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("battles", nargs="*", help="battle names (default: all cNNN in raw/classic)")
    args = p.parse_args()
    names_all = sorted(f[:-5] for f in os.listdir(OUT) if re.match(r"c\d+\.json$", f))
    battles = args.battles or names_all
    reader = ScreenReader()
    known, names = load_known()
    pictures = {}  # unique id -> lit picture
    uniques = dict(known)
    dates = {}
    result = {"battles": {}, "cards": {}, "decks": []}
    for b in battles:
        summary = json.load(open(os.path.join(OUT, f"{b}.json"), encoding="utf-8"))
        if summary["date"] in dates:  # the same battle recorded twice (two script runs at once)
            print(f"{b}: the same battle as {dates[summary['date']]}, left out")
            continue
        dates[summary["date"]] = b
        result["battles"][b] = {}
        for side, video in summary["videos"].items():
            cards = side_cards(os.path.join(OUT, video), reader)
            deck = [c for c in cards if c["n"] >= MIN_SEEN][:8]
            ids = []
            for c in deck:
                best = max(((corr(f, c["feat"]), k) for k, f in uniques.items()), default=(-1.0, None))
                if best[0] > SAME:
                    uid = best[1]
                else:
                    uid = f"k{len(uniques) + 1:02d}"
                    uniques[uid] = c["feat"]
                if uid not in pictures or c["sat"] > pictures[uid][1]:
                    pictures[uid] = (c["crop"], c["sat"])
                ids.append(uid)
                result["cards"].setdefault(uid, {"name": names.get(uid), "seen": []})["seen"].append(f"{b} {side}")
            extra = len([c for c in cards if c["n"] >= MIN_SEEN]) - len(deck)
            result["battles"][b][side] = sorted(ids)
            print(f"{b} {side}: {len(ids)} cards " + " ".join(f"{i}({names.get(i, '?')})" for i in sorted(ids)) +
                  (f"  [{extra} more seen, left out]" if extra > 0 else "") +
                  ("  [fewer than 8: a card never sat in the hand]" if len(ids) < 8 else ""))
    def canon(uid):
        """One id per card: ids given the same name are one card split in two (a plain picture: the snowball)."""
        name = names.get(uid)
        return min((k for k, n in names.items() if n == name), key=lambda k: int(k[1:])) if name else uid

    def order(ids):
        return sorted(ids, key=lambda u: int(u[1:]))

    sides = [(f"{b} {side}", {canon(u) for u in ids}) for b, s in result["battles"].items() for side, ids in s.items()]
    decks = []  # a side with a card or two never seen lit (a short battle) joins the full deck it fits in
    for where, ids in sorted(sides, key=lambda s: -len(s[1])):
        home = next((d for d in decks if ids <= d["cards"]), None)
        if home is None:
            decks.append({"cards": ids, "seen": [where]})
        else:
            home["seen"].append(where)
    result["decks"] = [{"cards": order(d["cards"]), "names": [names.get(u) for u in order(d["cards"])],
                        "seen": d["seen"], "complete": len(d["cards"]) == 8}
                       for d in sorted(decks, key=lambda d: -len(d["seen"]))]
    for n, d in enumerate(result["decks"], 1):
        print(f"D{n} ({len(d['seen'])}x{'' if d['complete'] else ', not all 8 cards seen'}): " +
              ", ".join(nm or u for u, nm in zip(d["cards"], d["names"])))
    os.makedirs(CARDS, exist_ok=True)
    for uid, (crop, _sat) in pictures.items():
        if uid not in known:  # (a saved picture keeps its id; only new cards are written)
            cv2.imwrite(os.path.join(CARDS, f"{uid}.png"), crop)
    with open(os.path.join(OUT, "decks.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    # the sheet: every card, then every deck as a row
    def picture(uid):
        if uid in pictures:
            return pictures[uid][0]
        return cv2.imread(os.path.join(CARDS, f"{uid}.png"))
    ids = sorted(result["cards"], key=lambda u: int(u[1:]))
    rows = [[tile(picture(u), f"{u} {names.get(u, '')}"[:16], f"seen {len(result['cards'][u]['seen'])}x")
             for u in ids[i:i + 10]] for i in range(0, len(ids), 10)]
    for n, d in enumerate(result["decks"], 1):
        rows.append([tile(picture(u), f"D{n} {u}", (names.get(u) or "")[:18]) for u in d["cards"]] +
                    [tile(np.zeros((40, 40, 3), np.uint8), f"D{n}: {len(d['seen'])}x", ", ".join(d["seen"])[:22])])
    cv2.imwrite(os.path.join(CARDS, "sheet.png"), sheet(rows))
    print(f"{len(ids)} unique cards, {len(result['decks'])} decks; raw/classic/decks.json, raw/classic/cards/sheet.png")


if __name__ == "__main__":
    main()
