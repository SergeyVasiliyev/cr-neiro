"""Labelling tool for units on battle frames (for training the unit detector).

    scripts\\label.bat            (or .venv\\Scripts\\python.exe python\\label_units.py)

Frames come from tools/extract_label_frames.py (raw/labels/frames/); labels are saved after every
change to raw/labels/labels.json.

Controls
  1..9, 0         choose the class (shown on the right)
  left click      a unit of OURS (blue, bottom) at the cursor
  right click     an ENEMY unit (red, top)
                  ground troops: click their feet; flying troops and spells: click their middle
  drag (L or R)   draw the box yourself instead of the default size
  D / Delete      remove the label under the cursor        Z  undo the last label
  T               swap blue/red of the label under the cursor
  R               set the label under the cursor to the chosen class
  C               remove all detector suggestions (dashed boxes) from this frame
  Space / Enter / ->   next frame (also marks this one done)    <-  previous frame
  X               skip this frame (bad / unclear), go to the next one
"""
import json
import os
import sys

import pygame

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FRAMES = os.path.join(ROOT, "raw", "labels", "frames")
LABELS = os.path.join(ROOT, "raw", "labels", "labels.json")

ARENA = (120, 1540)  # rows of the arena shown, at 1080x1920
# class id, name, default box (w, h) in 1080x1920 pixels, anchor: "feet" (click = bottom centre) or "centre"
CLASSES = [
    ("knight", "рыцарь", (62, 78), "feet"),
    ("archer", "лучница", (46, 66), "feet"),
    ("minion", "миньон", (56, 56), "centre"),
    ("giant", "гигант", (92, 112), "feet"),
    ("mini_pekka", "мини-пекка", (62, 72), "feet"),
    ("musketeer", "мушкетёр", (52, 72), "feet"),
    ("fireball", "огненный шар", (56, 56), "centre"),
    ("arrows", "стрелы (залп)", (150, 150), "centre"),
    # Where a spell lands (explosion / arrow rain): radius 2.5 / 3.5 tiles, a tile is 51 x 41.6 px.
    ("fireball_hit", "взрыв шара", (255, 208), "centre"),
    ("arrows_hit", "стрелы попали", (357, 291), "centre"),
]
CLASS_KEYS = "1234567890"  # key that selects each class, in CLASSES order
TEAM_COLOR = {"blue": (70, 150, 255), "red": (255, 70, 70)}
BG = (25, 27, 33)
TEXT = (230, 230, 235)
DIM = (150, 150, 160)
PANEL_W = 330


def load_labels():
    if os.path.exists(LABELS):
        with open(LABELS, encoding="utf-8") as f:
            return json.load(f)
    return {"classes": [c[0] for c in CLASSES], "frames": {}}


def save_labels(doc):
    tmp = LABELS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LABELS)


class Tool:
    def __init__(self):
        self.files = sorted(f for f in os.listdir(FRAMES) if f.lower().endswith(".jpg"))
        if not self.files:
            sys.exit(f"no frames in {FRAMES}: run tools/extract_label_frames.py first")
        self.doc = load_labels()
        pygame.init()
        info = pygame.display.Info()
        arena_h = ARENA[1] - ARENA[0]
        self.scale = min((info.current_h - 90) / arena_h, (info.current_w - PANEL_W - 40) / 1080)
        self.view_w, self.view_h = int(1080 * self.scale), int(arena_h * self.scale)
        self.screen = pygame.display.set_mode((self.view_w + PANEL_W, self.view_h))
        pygame.display.set_caption("Разметка юнитов")
        self.font = pygame.font.SysFont("consolas", 16)
        self.small = pygame.font.SysFont("consolas", 13)
        self.big = pygame.font.SysFont("consolas", 20, bold=True)
        self.cls = 0
        self.drag = None  # (button, start screen pos)
        done = {k for k, v in self.doc["frames"].items() if v.get("status") in ("labeled", "skip")}
        self.index = next((i for i, f in enumerate(self.files) if f not in done), 0)
        self.load_frame()

    # --- frames -------------------------------------------------------------------------
    def load_frame(self):
        name = self.files[self.index]
        img = pygame.image.load(os.path.join(FRAMES, name)).convert()
        img = img.subsurface(pygame.Rect(0, ARENA[0], img.get_width(), ARENA[1] - ARENA[0]))
        self.image = pygame.transform.smoothscale(img, (self.view_w, self.view_h))
        self.entry = self.doc["frames"].setdefault(name, {"status": None, "objects": []})

    def go(self, delta, mark_done=True):
        if mark_done and self.entry["status"] in (None, "prelabeled"):
            self.entry["status"] = "labeled"
        if mark_done:
            for o in self.entry["objects"]:  # moving on = the suggestions were reviewed
                o.pop("auto", None)
                o.pop("conf", None)
        save_labels(self.doc)
        index = max(0, min(len(self.files) - 1, self.index + delta))
        if mark_done and delta > 0:  # Space: on to the next frame still to do, skipping finished ones
            done = {k for k, v in self.doc["frames"].items() if v.get("status") in ("labeled", "skip")}
            index = next((i for i in range(self.index + 1, len(self.files)) if self.files[i] not in done), index)
        self.index = index
        self.load_frame()

    # --- coordinates --------------------------------------------------------------------
    def to_frame(self, sx, sy):
        return sx / self.scale, sy / self.scale + ARENA[0]

    def to_screen(self, fx, fy):
        return fx * self.scale, (fy - ARENA[0]) * self.scale

    # --- labels -------------------------------------------------------------------------
    def add_default(self, team, sx, sy):
        name, _, (w, h), anchor = CLASSES[self.cls]
        fx, fy = self.to_frame(sx, sy)
        if anchor == "feet":
            box = [fx - w / 2, fy - h, fx + w / 2, fy]
        else:
            box = [fx - w / 2, fy - h / 2, fx + w / 2, fy + h / 2]
        self.add(team, box)

    def add(self, team, box):
        x0, y0, x1, y1 = box
        box = [round(min(x0, x1), 1), round(min(y0, y1), 1), round(max(x0, x1), 1), round(max(y0, y1), 1)]
        self.entry["objects"].append({"cls": CLASSES[self.cls][0], "team": team, "box": box})
        self.entry["status"] = "labeled"
        save_labels(self.doc)

    def under_cursor(self, sx, sy):
        """Index of the smallest box under the cursor, or None."""
        fx, fy = self.to_frame(sx, sy)
        objs = self.entry["objects"]
        hits = [i for i, o in enumerate(objs) if o["box"][0] <= fx <= o["box"][2] and o["box"][1] <= fy <= o["box"][3]]
        if not hits:
            return None
        return min(hits, key=lambda i: (objs[i]["box"][2] - objs[i]["box"][0]) * (objs[i]["box"][3] - objs[i]["box"][1]))

    def remove_at(self, sx, sy):
        i = self.under_cursor(sx, sy)
        if i is not None:
            self.entry["objects"].pop(i)
            save_labels(self.doc)

    def edit_at(self, sx, sy, team=None, cls=None):
        i = self.under_cursor(sx, sy)
        if i is None:
            return
        o = self.entry["objects"][i]
        if team:
            o["team"] = "red" if o["team"] == "blue" else "blue"
        if cls:
            o["cls"] = cls
        o.pop("auto", None)
        o.pop("conf", None)
        save_labels(self.doc)

    # --- loop ---------------------------------------------------------------------------
    def handle(self, ev):
        if ev.type == pygame.KEYDOWN:
            k = ev.key
            if ev.unicode and ev.unicode in CLASS_KEYS[:len(CLASSES)]:
                self.cls = CLASS_KEYS.index(ev.unicode)
            elif k in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT):
                self.go(+1)
            elif k == pygame.K_LEFT:
                self.go(-1, mark_done=False)
            elif k == pygame.K_x:
                self.entry["status"] = "skip"
                save_labels(self.doc)
                self.go(+1, mark_done=False)
            elif k == pygame.K_z and self.entry["objects"]:
                self.entry["objects"].pop()
                save_labels(self.doc)
            elif k in (pygame.K_d, pygame.K_DELETE, pygame.K_BACKSPACE):
                self.remove_at(*pygame.mouse.get_pos())
            elif k == pygame.K_t:
                self.edit_at(*pygame.mouse.get_pos(), team=True)
            elif k == pygame.K_r:
                self.edit_at(*pygame.mouse.get_pos(), cls=CLASSES[self.cls][0])
            elif k == pygame.K_c:
                self.entry["objects"] = [o for o in self.entry["objects"] if not o.get("auto")]
                save_labels(self.doc)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button in (1, 3) and ev.pos[0] < self.view_w:
            self.drag = (ev.button, ev.pos)
        elif ev.type == pygame.MOUSEBUTTONUP and self.drag and ev.button == self.drag[0]:
            button, start = self.drag
            self.drag = None
            team = "blue" if button == 1 else "red"
            end = (min(ev.pos[0], self.view_w - 1), ev.pos[1])
            if abs(end[0] - start[0]) > 8 and abs(end[1] - start[1]) > 8:
                self.add(team, [*self.to_frame(*start), *self.to_frame(*end)])
            else:
                self.add_default(team, *start)

    def draw(self):
        self.screen.fill(BG)
        self.screen.blit(self.image, (0, 0))
        for o in self.entry["objects"]:
            x0, y0 = self.to_screen(o["box"][0], o["box"][1])
            x1, y1 = self.to_screen(o["box"][2], o["box"][3])
            color = TEAM_COLOR[o["team"]]
            ci = [c[0] for c in CLASSES].index(o["cls"]) if o["cls"] in [c[0] for c in CLASSES] else -1
            if o.get("auto"):  # detector suggestion not reviewed yet: dashed, with its confidence
                for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
                    n = max(1, int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) // 6))
                    for j in range(0, n, 2):
                        p0 = (a[0] + (b[0] - a[0]) * j / n, a[1] + (b[1] - a[1]) * j / n)
                        p1 = (a[0] + (b[0] - a[0]) * (j + 1) / n, a[1] + (b[1] - a[1]) * (j + 1) / n)
                        pygame.draw.line(self.screen, color, p0, p1, 2)
                self.screen.blit(self.small.render(f"{CLASS_KEYS[ci] if ci >= 0 else '?'} {o.get('conf', 0):.2f}", True, color), (x0 + 2, y0 + 1))
            else:
                pygame.draw.rect(self.screen, color, pygame.Rect(x0, y0, x1 - x0, y1 - y0), 2)
                self.screen.blit(self.small.render(CLASS_KEYS[ci] if ci >= 0 else "?", True, color), (x0 + 2, y0 + 1))
        mx, my = pygame.mouse.get_pos()
        if self.drag and self.drag[1][0] < self.view_w:
            sx, sy = self.drag[1]
            pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(min(sx, mx), min(sy, my), abs(mx - sx), abs(my - sy)), 1)
        elif mx < self.view_w:  # preview of the default box
            _, _, (w, h), anchor = CLASSES[self.cls]
            w, h = w * self.scale, h * self.scale
            rect = pygame.Rect(mx - w / 2, my - h if anchor == "feet" else my - h / 2, w, h)
            pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        x = self.view_w + 14
        y = 12
        done = sum(1 for f in self.files if self.doc["frames"].get(f, {}).get("status") in ("labeled", "skip"))
        self.screen.blit(self.big.render(f"осталось: {len(self.files) - done}", True, TEXT), (x, y))
        y += 26
        self.screen.blit(self.font.render(f"кадр {self.index + 1}/{len(self.files)}, готово {done}", True, DIM), (x, y))
        y += 22
        status = {"labeled": "готов", "skip": "пропущен", "prelabeled": "подсказки детектора"}.get(self.entry["status"], "новый")
        self.screen.blit(self.small.render(f"{self.files[self.index]}  [{status}]", True, DIM), (x, y))
        y += 30
        for i, (_, ru, _, anchor) in enumerate(CLASSES):
            color = (255, 230, 120) if i == self.cls else TEXT
            mark = "> " if i == self.cls else "  "
            where = "ноги" if anchor == "feet" else "центр"
            self.screen.blit(self.font.render(f"{mark}{CLASS_KEYS[i]} {ru} ({where})", True, color), (x, y))
            y += 22
        y += 14
        for line in ["ЛКМ — наш (синий)", "ПКМ — чужой (красный)", "тяни — своя рамка",
                     "D — удалить под курсором", "Z — отменить", "T — сменить сторону", "R — сменить класс",
                     "C — убрать все подсказки", "Пробел/→ — дальше",
                     "← — назад", "X — пропустить кадр"]:
            self.screen.blit(self.small.render(line, True, DIM), (x, y))
            y += 18
        self.screen.blit(self.small.render("пунктир = подсказка детектора", True, DIM), (x, y))
        y += 18
        y += 10
        counts = {}
        for o in self.entry["objects"]:
            counts[o["team"]] = counts.get(o["team"], 0) + 1
        self.screen.blit(self.font.render(f"на кадре: {counts.get('blue', 0)} наших, {counts.get('red', 0)} чужих",
                                          True, TEXT), (x, y))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    save_labels(self.doc)
                    pygame.quit()
                    return
                self.handle(ev)
            self.draw()
            clock.tick(60)


if __name__ == "__main__":
    Tool().run()
