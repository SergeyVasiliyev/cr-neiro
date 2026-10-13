"""Fast screenshots of the LDPlayer window (~35 ms instead of ~900 ms through ADB).

The game area is LDPlayer's "TheRender" child window, asked to draw itself (PrintWindow with full
content), so other windows may cover it; it must not be minimised. Any window size works (it is 9:16);
bigger is better for the unit detector.
"""
import ctypes
import ctypes.wintypes as wt
import os
import subprocess

import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # real pixels, not scaled ones
except (AttributeError, OSError):
    user32.SetProcessDPIAware()

_EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
PW_RENDERFULLCONTENT = 2


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wt.DWORD), ("biWidth", wt.LONG), ("biHeight", wt.LONG), ("biPlanes", wt.WORD),
                ("biBitCount", wt.WORD), ("biCompression", wt.DWORD), ("biSizeImage", wt.DWORD),
                ("biXPelsPerMeter", wt.LONG), ("biYPelsPerMeter", wt.LONG), ("biClrUsed", wt.DWORD),
                ("biClrImportant", wt.DWORD)]


def _class_name(hwnd):
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


LDCONSOLE = r"C:\LDPlayer\LDPlayer9\ldconsole.exe"


def instances():
    """Running and stopped LDPlayer instances: [{index, title, render (game area window), running}].
    Several can run at once (LDMultiPlayer): instance N is adb device emulator-(5554 + 2N)."""
    if not os.path.exists(LDCONSOLE):
        return []
    out = subprocess.run([LDCONSOLE, "list2"], capture_output=True, text=True, timeout=15).stdout
    found = []
    for line in out.splitlines():
        f = line.strip().split(",")
        if len(f) >= 5 and f[0].isdigit():
            found.append({"index": int(f[0]), "title": f[1], "render": int(f[3]), "running": f[4] == "1"})
    return found


def find_render_window(index=None):
    """The LDPlayer game area window (of instance `index`, or the first one found), or None."""
    if index is not None:
        for inst in instances():
            if inst["index"] == index and inst["running"] and user32.IsWindow(inst["render"]):
                return inst["render"]
        return None
    found = []

    def top(hwnd, _):
        if user32.IsWindowVisible(hwnd) and _class_name(hwnd) == "LDPlayerMainFrame":
            def child(c, _):
                if _class_name(c) == "RenderWindow":
                    found.append(c)
                return True
            user32.EnumChildWindows(hwnd, _EnumProc(child), 0)
        return True

    user32.EnumWindows(_EnumProc(top), 0)
    return found[0] if found else None


class WindowScreen:
    def __init__(self, hwnd=None, index=None):
        self.hwnd = hwnd or find_render_window(index)
        if not self.hwnd:
            which = "LDPlayer" if index is None else f"LDPlayer instance {index}"
            raise RuntimeError(f"{which} window not found: start it (window visible, not minimised)")
        self._size = None
        self._screen_dc = user32.GetDC(0)
        self._mem_dc = gdi32.CreateCompatibleDC(self._screen_dc)
        self._bitmap = None

    def rect(self):
        """(x, y, width, height) of the game area on the desktop."""
        r = wt.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        p = wt.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(p))
        return p.x, p.y, r.right - r.left, r.bottom - r.top

    def _buffer(self, w, h):
        if self._size != (w, h):
            if self._bitmap:
                gdi32.DeleteObject(self._bitmap)
            info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
            bits = ctypes.c_void_p()
            self._bitmap = gdi32.CreateDIBSection(self._screen_dc, ctypes.byref(info), 0, ctypes.byref(bits), None, 0)
            gdi32.SelectObject(self._mem_dc, self._bitmap)
            self._pixels = np.ctypeslib.as_array((ctypes.c_ubyte * (w * h * 4)).from_address(bits.value)).reshape(h, w, 4)
            self._size = (w, h)

    def screenshot(self):
        """The game area as a BGR image (window size, 9:16)."""
        if user32.IsIconic(user32.GetParent(self.hwnd)):
            raise RuntimeError("LDPlayer window is minimised")
        x, y, w, h = self.rect()
        self._buffer(w, h)
        if not user32.PrintWindow(self.hwnd, self._mem_dc, PW_RENDERFULLCONTENT):
            raise RuntimeError("could not capture the LDPlayer window")
        return self._pixels[..., :3].copy()

    def close(self):
        if self._bitmap:
            gdi32.DeleteObject(self._bitmap)
        gdi32.DeleteDC(self._mem_dc)
        user32.ReleaseDC(0, self._screen_dc)
