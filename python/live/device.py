"""The Android emulator (LDPlayer) through ADB: screenshots in, touches out.

LDPlayer needs ADB switched on once: Settings -> Other settings -> ADB debugging -> Open local connection.
The adb.exe that ships with LDPlayer is used unless CR_ADB points to another one.

Use the bot only on the test (guest) account: automating play on a main account breaks Supercell's terms.
"""
import os
import queue
import re
import shutil
import struct
import subprocess
import threading
import time

import cv2
import numpy as np

from perception.layout import REF_W, Layout

LDPLAYER_ADB = r"C:\LDPlayer\LDPlayer9\adb.exe"


def find_adb():
    for path in (os.environ.get("CR_ADB"), LDPLAYER_ADB, shutil.which("adb")):
        if path and os.path.exists(path):
            return path
    raise RuntimeError("adb.exe not found: install LDPlayer 9 or set CR_ADB to adb.exe")


class Device:
    def __init__(self, serial=None, adb=None):
        self.adb = adb or find_adb()
        self.serial = serial or self._pick_device()
        self._shell = None
        self._touch_layout = None
        self._touch = None  # FastTouch, set up on the first touch (None if it can't be: input tap then)
        self._touch_tried = False

    # --- connection ------------------------------------------------------------------------------
    def _devices(self):
        out = subprocess.run([self.adb, "devices"], capture_output=True, text=True, timeout=15).stdout
        return [line.split("\t")[0] for line in out.splitlines()[1:] if line.endswith("\tdevice")]

    def _pick_device(self):
        found = self._devices()
        if not found:
            # LDPlayer's first instance also listens here when "local connection" is on.
            subprocess.run([self.adb, "connect", "127.0.0.1:5555"], capture_output=True, timeout=15)
            found = self._devices()
        if not found:
            raise RuntimeError("no emulator found over ADB: start LDPlayer and turn on "
                               "Settings -> Other settings -> ADB debugging -> Open local connection")
        emulators = [s for s in found if s.startswith("emulator-")]
        return (emulators or found)[0]

    def _run(self, *args, timeout=15):
        return subprocess.run([self.adb, "-s", self.serial, *args], capture_output=True, timeout=timeout,
                              check=True).stdout

    def size(self):
        """Screen size (width, height) of the portrait game screen in pixels. The emulator reports its
        landscape size; the game runs rotated."""
        out = self._run("shell", "wm", "size").decode(errors="replace")
        sizes = [line.split(":")[1].strip() for line in out.splitlines() if ":" in line]
        w, h = (int(v) for v in sizes[-1].split("x"))  # "Override size" (last) wins over "Physical size"
        return min(w, h), max(w, h)

    # --- eyes ------------------------------------------------------------------------------------
    def screenshot(self):
        """The current screen as a BGR image (raw framebuffer, no PNG encoding: much faster)."""
        raw = self._run("exec-out", "screencap")
        w, h, _fmt = struct.unpack_from("<III", raw)
        header = len(raw) - w * h * 4  # 12 bytes, or 16 on newer Android (colour space)
        if header < 12:
            raise RuntimeError(f"unexpected screencap size {len(raw)} for {w}x{h}")
        rgba = np.frombuffer(raw, np.uint8, count=w * h * 4, offset=header).reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    # --- hands -----------------------------------------------------------------------------------
    def _send(self, command):
        """Runs a command in one long-lived adb shell: no new process (and ~100 ms) per touch."""
        for attempt in range(2):
            if self._shell is None or self._shell.poll() is not None:
                self._shell = subprocess.Popen([self.adb, "-s", self.serial, "shell"], stdin=subprocess.PIPE,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            try:
                self._shell.stdin.write(command + "\n")
                self._shell.stdin.flush()
                return
            except (OSError, ValueError):  # the shell died (adb restarted, emulator hiccup): start a new one
                try:
                    self._shell.kill()
                except OSError:
                    pass
                self._shell = None
                if attempt:
                    raise

    def tap(self, x, y):
        self._send(f"input tap {int(x)} {int(y)}")

    def drag(self, x0, y0, x1, y1, ms=150):
        self._send(f"input swipe {int(x0)} {int(y0)} {int(x1)} {int(y1)} {int(ms)}")

    def play_card(self, slot, tile, method="drag"):
        """Puts the card in hand slot (0..3) on arena tile (x, y) in simulator coordinates."""
        if self._touch_layout is None:  # touches are in the device's own pixels, whatever we look at
            self._touch_layout = Layout(scale=self.size()[0] / REF_W)
        layout = self._touch_layout
        xa, xb = Layout.HAND_SLOTS_X[slot]
        ya, yb = Layout.HAND_Y
        cx, cy = layout.s((xa + xb) / 2), layout.s((ya + yb) / 2)
        tx, ty = layout.tile_to_px(*tile)
        if method == "drag":
            self.drag(cx, cy, tx, ty)
            return
        # Tap the card, then the spot. Fast: straight into the touchscreen (a few ms); otherwise through
        # Android's `input tap` (~0.2 s each: it starts a program every time).
        if not self._touch_tried:
            self._touch_tried = True
            try:
                self._touch = FastTouch(self.adb, self.serial)
            except Exception as err:  # noqa: BLE001 - any trouble: the slow way still works
                print(f"fast touches unavailable ({err}); using input tap")
        if self._touch is not None and self._touch.ok:
            self._touch.taps([(cx, cy), (tx, ty)])
        else:
            self._send(f"input tap {cx} {cy}; input tap {int(tx)} {int(ty)}")

    def close(self):
        if self._touch is not None:
            self._touch.close()
        if self._shell is not None and self._shell.poll() is None:
            self._shell.stdin.close()
            self._shell.wait(timeout=5)


class FastTouch:
    """Touches written straight into the emulator's touchscreen device (Linux input events through one
    long-lived adb pipe): no program started per touch, so a tap takes milliseconds instead of ~0.2 s.
    Taps run in a background thread, so the caller doesn't wait for them.

    The touchscreen reports in its own (landscape) frame; the game screen is rotated: positions are turned
    the way Android does it (TouchInputMapper), for the rotation it reports for that device."""

    HOLD = 0.012   # s a finger stays down (input tap holds ~0; a little for safety)
    BETWEEN = 0.035  # s between the taps of one play (the game selects the card, then takes the spot)

    def __init__(self, adb, serial):
        self.adb, self.serial = adb, serial
        self.dev, self.max_x, self.max_y = self._find_device()
        self.rotation = self._rotation()
        self.proc = subprocess.Popen([adb, "-s", serial, "exec-in", f"cat > {self.dev}"], stdin=subprocess.PIPE)
        self.tracking = 100
        self.jobs = queue.Queue()
        self.ok = True
        threading.Thread(target=self._run, daemon=True).start()

    def _shell(self, cmd):
        return subprocess.run([self.adb, "-s", self.serial, "shell", cmd], capture_output=True, text=True,
                              timeout=15).stdout

    def _find_device(self):
        out = self._shell("getevent -pl")
        for block in re.split(r"add device \d+: ", out)[1:]:
            mx = re.search(r"ABS_MT_POSITION_X\s*: value \d+, min \d+, max (\d+)", block)
            my = re.search(r"ABS_MT_POSITION_Y\s*: value \d+, min \d+, max (\d+)", block)
            name = re.search(r'name:\s*"([^"]*)"', block)
            if mx and my:
                self.name = name.group(1) if name else ""
                return block.split()[0], int(mx.group(1)), int(my.group(1))
        raise RuntimeError("no multi-touch screen device")

    def _rotation(self):
        """The rotation Android applies to this device's touches (its block in the input reader's state)."""
        out = self._shell("dumpsys input")
        reader = out[out.find("Input Reader State"):]
        blocks = re.split(r"\n\s*Device \d+: ", reader)
        for block in blocks[1:]:
            if block.split("\n", 1)[0].strip() == self.name:
                m = re.search(r"SurfaceOrientation: (\d)", block)
                if m:
                    return int(m.group(1))
        raise RuntimeError("touchscreen orientation unknown")

    def _raw(self, x, y):
        """Screen pixel (portrait game screen) -> touchscreen position (Android's rotation, inverted)."""
        x, y = int(round(x)), int(round(y))
        if self.rotation == 0:
            return x, y
        if self.rotation == 1:
            return self.max_x - y, x
        if self.rotation == 2:
            return self.max_x - x, self.max_y - y
        return y, self.max_y - x  # 3: checked on LDPlayer 9 (raw 1100,1039 -> screen 40,1100)

    @staticmethod
    def _ev(etype, code, value):
        return struct.pack("<qqHHi", 0, 0, etype, code, value)

    def _write(self, data):
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _tap(self, x, y):
        rx, ry = self._raw(x, y)
        self.tracking = self.tracking % 60000 + 1
        ev = self._ev
        self._write(ev(3, 0x2F, 0) + ev(3, 0x39, self.tracking) + ev(3, 0x35, rx) + ev(3, 0x36, ry)
                    + ev(3, 0x3A, 1) + ev(1, 0x14A, 1) + ev(0, 0, 0))
        time.sleep(self.HOLD)
        self._write(ev(3, 0x39, -1) + ev(1, 0x14A, 0) + ev(0, 0, 0))

    def _run(self):
        while True:
            points = self.jobs.get()
            if points is None:
                return
            try:
                for k, (x, y) in enumerate(points):
                    if k:
                        time.sleep(self.BETWEEN)
                    self._tap(x, y)
            except (OSError, ValueError) as err:
                print(f"fast touches broke ({err}); using input tap from now on")
                self.ok = False
                return

    def taps(self, points):
        self.jobs.put(list(points))

    def close(self):
        self.jobs.put(None)
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


def measure(device, n=10):
    """Average seconds per screenshot."""
    started = time.perf_counter()
    for _ in range(n):
        device.screenshot()
    return (time.perf_counter() - started) / n
