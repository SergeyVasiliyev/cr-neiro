"""Makes sure recordings have perception caches: waits for any tools/perceive_video.py already running to
finish, then reads the ones still missing.

    .venv\\Scripts\\python.exe tools\\ensure_caches.py video22 video22_red ...
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def perceiving():
    out = subprocess.run(["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
                         capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "perceive_video.py" in line)


def main():
    names = sys.argv[1:]
    while perceiving():
        print(f"waiting for {perceiving()} reading(s) already running", flush=True)
        time.sleep(30)
    for name in names:
        if os.path.exists(os.path.join(ROOT, "raw", "perception", "cache", f"{name}.json")):
            continue
        print(f"reading {name}", flush=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "perceive_video.py"), "--video",
                        os.path.join(ROOT, "raw", "videos", f"{name}.mp4"), "--conf", "0.15"], check=True)
    print("caches ready", flush=True)


if __name__ == "__main__":
    main()
