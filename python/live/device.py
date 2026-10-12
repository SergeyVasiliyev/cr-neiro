"""The Android emulator (LDPlayer) through ADB: screenshots in, touches out.

LDPlayer needs ADB switched on once: Settings -> Other settings -> ADB debugging -> Open local connection.
The adb.exe that ships with LDPlayer is used unless CR_ADB points to another one.

Use the bot only on the test (guest) account: automating play on a main account breaks Supercell's terms.
"""
import os
import shutil
import struct
import subprocess
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
        if self._shell is None or self._shell.poll() is not None:
            self._shell = subprocess.Popen([self.adb, "-s", self.serial, "shell"], stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        self._shell.stdin.write(command + "\n")
        self._shell.stdin.flush()

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
        else:  # tap the card, then the spot
            self._send(f"input tap {cx} {cy}; input tap {int(tx)} {int(ty)}")

    def close(self):
        if self._shell is not None and self._shell.poll() is None:
            self._shell.stdin.close()
            self._shell.wait(timeout=5)


def measure(device, n=10):
    """Average seconds per screenshot."""
    started = time.perf_counter()
    for _ in range(n):
        device.screenshot()
    return (time.perf_counter() - started) / n
