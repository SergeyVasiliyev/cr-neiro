"""Synthetic training pictures for the Classic Decks unit detector (round 0), from the seeds of tools/classic_labels.py.

Seeds are real units cut out of the random-bot battles with a mask. Pasted on the empty arena in crowds and alone,
at random spots, they make pictures whose every unit is labelled: the detector learns each kind's look without a
hand-labelled frame. The towers on these pictures come from the battles too (towers patch): hurt, destroyed, a
woken king, so the detector also sees the towers as they are in a battle.

    .venv\\Scripts\\python.exe tools\\classic_synth.py towers c008 c009 ...   tower pictures with no unit near
    .venv\\Scripts\\python.exe tools\\classic_synth.py build [--images 6000]  the dataset, raw/detector/classic0
    .venv\\Scripts\\python.exe tools\\classic_synth.py show [--n 6]           a few pictures with their boxes

The dataset is the arena crop of a 1080-wide frame (rows 120-1540), like the live detector's input
(python/perception/units.py), classes "<look>_<colour>" (blue = the bottom side of the picture's view).
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classic_labels import (ARENA_ROWS, COUNTDOWN, LOOK, SEEDS, TOWER_BOX, TOWERS, clean_arena,  # noqa: E402
                            video_path, view_tile)
from classic_replay import OUT, ROOT, Replay  # noqa: E402
from perception.layout import Layout  # noqa: E402

SYNTH = os.path.join(OUT, "synth")
DATASET = os.path.join(ROOT, "raw", "detector", "classic0")
REF_W = 1080
FREE_TILES = 4.5      # a tower picture is taken when no simulated unit is this close to the tower
TOWER_EVERY = 45      # frames between tower pictures (3 s)
HUD_FROM = 11.0       # battle seconds: the "VS" intro is gone


def pool_looks():
    """Every unit look of the pool (cards, the units they make, riders), sorted."""
    pool = json.load(open(os.path.join(ROOT, "data", "classic_decks.json"), encoding="utf-8"))["cards"]
    return looks_of(pool)


def looks_of(card_ids):
    """The unit looks these cards put on the arena (their units, the units those make, riders), sorted."""
    cards = json.load(open(os.path.join(ROOT, "data", "cards.json"), encoding="utf-8"))
    pool = card_ids
    units = set()

    def walk(cid, seen):
        if cid in seen or cid not in cards:
            return
        seen.add(cid)
        c = cards[cid]
        if c["type"] == "spell":
            if c["spell"].get("spawn"):
                walk(c["spell"]["spawn"]["unit"], seen)
            return
        units.add(LOOK.get(cid, cid))
        u = c["unit"]
        for key in ("spawn", "death_spawn", "riders"):
            if u.get(key):
                walk(u[key]["unit"], seen)
        if (u.get("hit_buff") or {}).get("unit"):
            walk(u["hit_buff"]["unit"], seen)
        if c.get("second"):
            walk(c["second"]["unit"], seen)

    for cid in pool:
        walk(cid, set())
    return sorted(units)


def class_names():
    return [f"{look}_{colour}" for look in pool_looks() for colour in ("blue", "red")]


# --- tower pictures -----------------------------------------------------------------------------------------
def tower_slots(layout):
    """Each tower's box in pixels of a view: slot name -> (x0, y0, x1, y1). The same in both views."""
    out = {}
    for cx, cy, kind in TOWERS:
        side = "far" if cy > 16 else "near"
        dx, below, above = TOWER_BOX[(kind, side)]
        xa, ya = layout.tile_to_px(cx - dx, cy + above)
        xb, yb = layout.tile_to_px(cx + dx, cy - below)
        name = f"{side}_{'king' if kind == 'king' else ('left' if cx < 9 else 'right')}"
        out[name] = (max(0, int(xa)), max(0, int(ya)), int(xb) + 1, int(yb) + 1)
    return out


def collect_towers(name):
    r = Replay(name)
    fps = r.tl["fps"]
    caps = {v: cv2.VideoCapture(video_path(name, v)) for v in ("blue", "red")}
    layout = None
    k, n = -1, 0
    centres = {}
    while True:
        frames = {v: c.read()[1] for v, c in caps.items()}
        if any(f is None for f in frames.values()):
            break
        k += 1
        if not r.advance(k / fps) or k % TOWER_EVERY:
            continue
        st = r.state()
        if st["g"] < HUD_FROM or any(a <= st["g"] < b for a, b in COUNTDOWN):
            continue  # (the "VS" intro covers the arena until 2:51; big numbers and banners at the end)
        if layout is None:
            layout = Layout.for_frame(frames["blue"])
            slots = tower_slots(layout)
            for cx, cy, kind in TOWERS:
                side = "far" if cy > 16 else "near"
                centres[f"{side}_{'king' if kind == 'king' else ('left' if cx < 9 else 'right')}"] = (cx, cy)
        for view, frame in frames.items():
            for slot, (x0, y0, x1, y1) in slots.items():
                cx, cy = centres[slot]
                near = [u for u in st["units"] if ((view_tile(u["x"], u["y"], view)[0] - cx) ** 2
                                                   + (view_tile(u["x"], u["y"], view)[1] - cy) ** 2) ** 0.5 < FREE_TILES]
                near += [s for s in st["spells"] if ((view_tile(s["x"], s["y"], view)[0] - cx) ** 2
                                                     + (view_tile(s["x"], s["y"], view)[1] - cy) ** 2) ** 0.5 < FREE_TILES + 2]
                if near:
                    continue
                d = os.path.join(SYNTH, "towers", slot)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{name}_{view}_{k:05d}.png"), frame[y0:y1, x0:x1])
                n += 1
    print(f"{name}: {n} tower pictures", flush=True)


# --- pictures -----------------------------------------------------------------------------------------------
# Looks that come in a group (a card's 2-6 units, a spawner's batch): their seeds turned out to be the group, not a unit
# (2026-11-22 sheets: skeletons and barbarians two to a box, spear goblins three, "goblins" mostly an inferno dragon),
# since a group walks bunched up and the simulator's spread of it is not the game's. They get no sprites here; real
# frames (tools/classic_round.py) label them one by one.
GROUP_LOOKS = {"skeletons", "barbarians", "spear_goblins", "goblins", "minions", "bats", "lava_pups", "golemite",
               "royal_hogs", "zappies"}


def load_sprites(exclude=()):
    recs = [json.loads(line) for line in open(os.path.join(SEEDS, "clean.jsonl"), encoding="utf-8")]
    by_class = collections.defaultdict(list)
    for r in recs:
        if r["battle"] not in exclude and r["unit"] not in GROUP_LOOKS:
            by_class[f"{r['unit']}_{r['seen']}"].append(r)
    return by_class


_SPRITES = {}


def sprite_image(path):
    if path not in _SPRITES:
        _SPRITES[path] = cv2.imread(os.path.join(SEEDS, "sprites", path), cv2.IMREAD_UNCHANGED)
    return _SPRITES[path]


_TOWER_PICS = {}


def tower_image(path):
    if path not in _TOWER_PICS:
        _TOWER_PICS[path] = cv2.imread(path)
    return _TOWER_PICS[path]


def paste(canvas, owner, sprite, x0, y0, idx):
    """Pastes an RGBA sprite with its top-left at (x0, y0); returns its box clipped to the canvas, or None."""
    h, w = sprite.shape[:2]
    H, W = canvas.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb - xa < 4 or yb - ya < 4:
        return None
    s = sprite[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    a = cv2.GaussianBlur(a, (3, 3), 0)[:, :, None] if min(a.shape[:2]) >= 3 else a
    region = canvas[ya:yb, xa:xb].astype(np.float32)
    canvas[ya:yb, xa:xb] = (s[:, :, :3] * a + region * (1 - a)).astype(np.uint8)
    solid = s[:, :, 3] > 127
    owner[ya:yb, xa:xb][solid] = idx
    ys, xs = np.nonzero(solid)
    if len(xs) == 0:
        return None
    return xa + int(xs.min()), ya + int(ys.min()), xa + int(xs.max()) + 1, ya + int(ys.max()) + 1, int(solid.sum())


def make_picture(rng, arena, towers, sprites, weights, layout, names_index):
    canvas = arena.copy()
    H, W = canvas.shape[:2]
    for slot, (x0, y0, x1, y1) in tower_slots(layout).items():
        pics = towers.get(slot)
        if pics and rng.random() < 0.8:
            patch = tower_image(rng.choice(pics))
            if patch is not None and patch.shape[:2] == (y1 - y0, x1 - x0):
                canvas[y0:y1, x0:x1] = patch
    classes = list(sprites)
    n = rng.randint(1, 26)
    centres = [(rng.uniform(1.0, 17.0), rng.uniform(2.0, 30.0)) for _ in range(rng.randint(1, 4))]
    items = []
    for _ in range(n):
        cls = rng.choices(classes, weights=weights)[0]
        rec = rng.choice(sprites[cls])
        if rng.random() < 0.65:
            cx, cy = rng.choice(centres)
            tx, ty = cx + rng.gauss(0, 1.4), cy + rng.gauss(0, 1.4)
        else:
            tx, ty = rng.uniform(0.3, 17.7), rng.uniform(0.5, 31.5)
        tx, ty = min(17.8, max(0.2, tx)), min(31.8, max(0.3, ty))
        items.append((cls, rec, tx, ty))
    # draw far to near; flying units over the ground ones
    items.sort(key=lambda it: (it[1]["air"], -it[3]))
    owner = np.full((H, W), -1, np.int32)
    placed = []
    for idx, (cls, rec, tx, ty) in enumerate(items):
        sp = sprite_image(rec["sprite"])
        if sp is None or sp.shape[2] != 4:
            continue
        sc = rng.uniform(0.93, 1.07)
        sp = cv2.resize(sp, (max(4, int(sp.shape[1] * sc)), max(4, int(sp.shape[0] * sc))))
        if rng.random() < 0.5:
            sp = sp[:, ::-1].copy()
        px, py = layout.tile_to_px(tx, ty)
        h, w = sp.shape[:2]
        if rec["air"]:  # the body over its shadow
            sx, sy = layout.tile_to_px(tx, ty - 1.15)
            rx, ry = max(3, int(w * 0.32)), max(2, int(w * 0.12))
            sh = np.zeros((H, W), np.uint8)
            cv2.ellipse(sh, (int(sx), int(sy)), (rx, ry), 0, 0, 360, 255, -1)
            canvas[sh > 0] = (canvas[sh > 0] * 0.62).astype(np.uint8)
            x0, y0 = int(px - w / 2), int(py - h / 2)
        else:  # feet (the box bottom) at the spot
            x0, y0 = int(px - w / 2), int(py - h)
        box = paste(canvas, owner, sp, x0, y0, idx)
        if box:
            placed.append((idx, cls, box))
    labels = []
    for idx, cls, (x0, y0, x1, y1, total) in placed:
        visible = int((owner[y0:y1, x0:x1] == idx).sum())
        if visible >= 0.3 * total and cls in names_index:
            labels.append((names_index[cls], x0, y0, x1, y1))
    return canvas, labels


def to_dataset_picture(canvas, labels):
    """The 1080-wide arena crop and its YOLO label lines."""
    s = REF_W / canvas.shape[1]
    big = cv2.resize(canvas, (REF_W, int(round(canvas.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    crop = big[ARENA_ROWS[0]:ARENA_ROWS[1]]
    h, w = crop.shape[:2]
    lines = []
    for c, x0, y0, x1, y1 in labels:
        bx0, by0, bx1, by1 = x0 * s, y0 * s - ARENA_ROWS[0], x1 * s, y1 * s - ARENA_ROWS[0]
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
        if bx1 - bx0 < 4 or by1 - by0 < 4:
            continue
        lines.append(f"{c} {(bx0 + bx1) / 2 / w:.6f} {(by0 + by1) / 2 / h:.6f} {(bx1 - bx0) / w:.6f} {(by1 - by0) / h:.6f}")
    return crop, lines


def good_towers(arena, layout, ring=6, max_off=0.15):
    """Tower pictures whose rim (arena floor around the tower) matches the empty arena: no darkened end screen,
    intro banner, name label or unit at the edge."""
    out = {}
    for slot, (x0, y0, x1, y1) in tower_slots(layout).items():
        ref = arena[y0:y1, x0:x1].astype(np.int16)
        rim = np.zeros(ref.shape[:2], bool)
        rim[:ring], rim[-ring:], rim[:, :ring], rim[:, -ring:] = True, True, True, True
        keep = []
        for f in glob.glob(os.path.join(SYNTH, "towers", slot, "*.png")):
            im = cv2.imread(f)
            if im is None or im.shape != ref.shape:
                continue
            off = (np.abs(im.astype(np.int16) - ref).max(axis=2) > 40)[rim].mean()
            if off <= max_off:
                keep.append(f)
        out[slot] = keep
    print("towers kept: " + ", ".join(f"{k} {len(v)}" for k, v in out.items()), flush=True)
    return out


def build(n_images, val_share=0.08, seed=0, exclude=(), name="classic0"):
    """name: the set under raw/detector (classic0 is round 0's; a new name leaves it as the trained detector had it)."""
    DATASET = os.path.join(ROOT, "raw", "detector", name)
    rng = random.Random(seed)
    arena = clean_arena()
    layout = Layout.for_frame(arena)
    names = class_names()
    names_index = {c: i for i, c in enumerate(names)}
    sprites = {c: v for c, v in load_sprites(exclude).items() if len(v) >= 3 and c in names_index}
    weights = [len(v) ** 0.3 for v in sprites.values()]
    towers = good_towers(arena, layout)
    for split in ("train", "val"):
        os.makedirs(os.path.join(DATASET, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET, "labels", split), exist_ok=True)
    counts = collections.Counter()
    for i in range(n_images):
        split = "val" if i < n_images * val_share else "train"
        canvas, labels = make_picture(rng, arena, towers, sprites, weights, layout, names_index)
        crop, lines = to_dataset_picture(canvas, labels)
        stem = f"s{i:05d}"
        cv2.imwrite(os.path.join(DATASET, "images", split, stem + ".jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, rng.randint(75, 93)])
        with open(os.path.join(DATASET, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines))
        for line in lines:
            counts[names[int(line.split()[0])]] += 1
        if i % 500 == 0:
            print(f"{i}/{n_images}", flush=True)
    with open(os.path.join(DATASET, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {DATASET}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(names):
            f.write(f"  {i}: {c}\n")
    json.dump({"classes": names, "sprites": {c: len(v) for c, v in sprites.items()}, "boxes": counts,
               "excluded_battles": sorted(exclude)},
              open(os.path.join(DATASET, "summary.json"), "w"), indent=1)
    print(f"{n_images} pictures, {sum(counts.values())} boxes, {len(sprites)} classes with sprites of {len(names)}")


def show(n, name="classic0"):
    DATASET = os.path.join(ROOT, "raw", "detector", name)
    names = class_names()
    pics = sorted(glob.glob(os.path.join(DATASET, "images", "train", "*.jpg")))[:n]
    rows = []
    for p in pics:
        im = cv2.imread(p)
        h, w = im.shape[:2]
        lab = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".txt"
        for line in open(lab).read().split("\n"):
            if not line.strip():
                continue
            c, x, y, bw, bh = line.split()
            x, y, bw, bh = float(x) * w, float(y) * h, float(bw) * w, float(bh) * h
            cv2.rectangle(im, (int(x - bw / 2), int(y - bh / 2)), (int(x + bw / 2), int(y + bh / 2)), (0, 0, 255), 2)
            cv2.putText(im, names[int(c)][:14], (int(x - bw / 2), int(y - bh / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
        rows.append(cv2.resize(im, (w // 2, h // 2)))
    out = os.path.join(DATASET, "show.jpg")
    cv2.imwrite(out, np.hstack(rows))
    print("show:", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("towers")
    s1.add_argument("battles", nargs="+")
    s2 = sub.add_parser("build")
    s2.add_argument("--images", type=int, default=6000)
    s2.add_argument("--exclude", nargs="*", default=[], help="battles whose seeds stay out (to test on them)")
    s2.add_argument("--name", default="classic0", help="the set under raw/detector")
    s3 = sub.add_parser("show")
    s3.add_argument("--n", type=int, default=6)
    s3.add_argument("--name", default="classic0")
    args = p.parse_args()
    if args.cmd == "towers":
        for b in args.battles:
            collect_towers(b)
    elif args.cmd == "build":
        build(args.images, exclude=set(args.exclude), name=args.name)
    else:
        show(args.n, args.name)


if __name__ == "__main__":
    main()
