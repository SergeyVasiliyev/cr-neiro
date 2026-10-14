"""Records several LDPlayer windows at once (LDPlayer's own recorder does one at a time).

    .venv\\Scripts\\python.exe tools\\record_windows.py              both instances -> raw/videos/videoN.mp4 + videoN_red.mp4
    .venv\\Scripts\\python.exe tools\\record_windows.py --instances 0 1 --fps 15 --name video8

Instance 0 is the blue player (its file is videoN.mp4), instance 1 the red one (videoN_red.mp4); N is
the next free number. Both windows are grabbed in one loop, so the two recordings share one clock.
Frames are the window size (9:16); the perception tools bring them to 1080x1920. Windows may be covered
by others but not minimised. Stop with Ctrl+C (or --minutes); closing the window also finishes the files.
"""
import argparse
import ctypes
import glob
import os
import re
import sys
import threading
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from live.window import WindowScreen  # noqa: E402

VIDEOS = os.path.join(ROOT, "raw", "videos")
SUFFIX = {0: "", 1: "_red"}


def next_name():
    numbers = [int(m.group(1)) for f in glob.glob(os.path.join(VIDEOS, "video*.mp4"))
               if (m := re.match(r"video(\d+)", os.path.basename(f)))]
    return f"video{max(numbers, default=0) + 1}"


def on_window_close(stop, closed):
    """Closing the console window (instead of Ctrl+C) would kill the recorder before the mp4 files are
    finished - unreadable (tools/repair_recording.py can save them). Windows gives a closing console a few
    seconds: stop the loop and wait for the files to be closed."""
    if os.name != "nt":
        return None

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
    def handler(event):
        if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            stop.set()
            closed.wait(4.5)
            return True
        return False  # Ctrl+C: KeyboardInterrupt as usual

    ctypes.windll.kernel32.SetConsoleCtrlHandler(handler, True)
    return handler  # keep a reference, or it is garbage collected


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances", type=int, nargs="+", default=[0, 1])
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--name", default=None, help="file name prefix (default: the next free videoN)")
    p.add_argument("--minutes", type=float, default=10.0, help="stop by itself after this long")
    args = p.parse_args()

    name = args.name or next_name()
    screens = {i: WindowScreen(index=i) for i in args.instances}
    writers = {}
    last = {}
    os.makedirs(VIDEOS, exist_ok=True)
    paths = {i: os.path.join(VIDEOS, f"{name}{SUFFIX.get(i, f'_{i}')}.mp4") for i in args.instances}
    print("recording " + ", ".join(f"LDPlayer {i} -> {os.path.basename(p)}" for i, p in paths.items()) +
          "   (Ctrl+C to stop)", flush=True)
    stop, closed = threading.Event(), threading.Event()
    handler = on_window_close(stop, closed)  # noqa: F841
    t0 = time.monotonic()
    written = 0
    grabbed = 0
    try:
        while time.monotonic() - t0 < args.minutes * 60 and not stop.is_set():
            for i, screen in screens.items():
                try:
                    frame = screen.screenshot()
                except RuntimeError:
                    continue  # minimised for a moment: repeat its last frame
                h, w = frame.shape[:2]
                frame = frame[:h - h % 2, :w - w % 2]  # even size for the encoder
                if i not in writers:
                    writers[i] = cv2.VideoWriter(paths[i], cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                                 (frame.shape[1], frame.shape[0]))
                if frame.shape[:2] != last.get(i, frame).shape[:2]:
                    frame = cv2.resize(frame, (last[i].shape[1], last[i].shape[0]))  # window resized
                last[i] = frame
            grabbed += 1
            # One frame per tick of the fixed frame rate, so video time is wall time (repeat if late).
            due = int((time.monotonic() - t0) * args.fps) + 1
            while written < due:
                for i, writer in writers.items():
                    writer.write(last[i])
                written += 1
            if grabbed % int(args.fps * 10) == 0:
                el = time.monotonic() - t0
                print(f"  {el:5.0f}s  {grabbed / el:.1f} grabs/s", flush=True)
            sleep = t0 + written / args.fps - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        for w in writers.values():
            w.release()
        for s in screens.values():
            s.close()
        closed.set()
    el = time.monotonic() - t0
    print(f"stopped after {el:.0f}s ({grabbed / max(el, 1e-6):.1f} grabs/s): " +
          ", ".join(os.path.basename(p) for p in paths.values()))


if __name__ == "__main__":
    main()
