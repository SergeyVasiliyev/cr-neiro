"""Accepts the user's friendly-battle invites on the guest account (LDPlayer instance 0) while the live bot
plays them: taps the yellow "Friendly battle" button of an invite at the bottom of the clan chat, and the "OK"
under a battle's result. Only on the guest account, only friendly battles.

    .venv\\Scripts\\python.exe tools\\auto_accept.py [--minutes 90]
Log: raw/auto_accept_log.txt
"""
import argparse
import os
import subprocess
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADB = r"C:\LDPlayer\LDPlayer9\adb.exe"
SERIAL = "emulator-5554"
INVITE_BUTTON = (846, 1404)   # the invite's "Friendly battle" button when it is the last message (1080x1920)
RESULT_OK = (540, 1734)       # "OK" under the winner banner
LOG = os.path.join(ROOT, "raw", "auto_accept_log.txt")


def log(text):
    line = time.strftime("%H:%M:%S ") + text
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def screen():
    png = subprocess.run([ADB, "-s", SERIAL, "exec-out", "screencap", "-p"], capture_output=True).stdout
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.resize(img, (1080, 1920))


def mean_bgr(img, xy, r=18):
    x, y = xy
    return img[y - r:y + r, x - 60:x + 60].reshape(-1, 3).mean(0)


def is_yellow(img, xy):
    """Most of the button yellow - its share of yellow pixels, not the mean colour: a two-line label ("Дружеский /
    бой") covers its middle."""
    x, y = xy
    reg = img[y - 46:y + 46, x - 120:x + 120].astype(int)
    b, g, r = reg[..., 0], reg[..., 1], reg[..., 2]
    return ((r > 200) & (g > 140) & (b < 100)).mean() > 0.4


def results_shown(img):
    b, g, r = mean_bgr(img, RESULT_OK)
    banner = img[180:260, 300:780].reshape(-1, 3).mean(0)  # "Winner!" over the winner's crowns: bright text
    return b > 190 and g > 140 and r < 120 and banner.max() > 90


def tap(xy):
    subprocess.run([ADB, "-s", SERIAL, "shell", "input", "tap", str(xy[0]), str(xy[1])])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=90)
    args = p.parse_args()
    end = time.time() + args.minutes * 60
    log("started")
    last_tap = 0.0
    while time.time() < end:
        img = screen()
        if img is not None and time.time() - last_tap > 6:
            if is_yellow(img, INVITE_BUTTON):
                tap(INVITE_BUTTON)
                last_tap = time.time()
                log("invite accepted")
            elif results_shown(img):
                time.sleep(2.0)  # let the result settle (and the bot see the end)
                tap(RESULT_OK)
                last_tap = time.time()
                log("result closed")
        time.sleep(2.0)
    log("stopped")


if __name__ == "__main__":
    main()
