"""Repairs a recording whose recorder was closed instead of stopped (Ctrl+C): the video data is all there,
but the index at the end of the mp4 (and with it the codec header) was never written - players and the
tools can't open it ("moov atom not found").

    .venv\\Scripts\\python.exe tools\\repair_recording.py --broken raw/videos/video13.mp4 --like raw/videos/video14.mp4

--like: a good recording made the same way (same window size, same recorder): its codec header is put in
front of the broken file's frames, and the result is written again as a normal mp4 at the recorder's
frame rate (15 fps: video time is wall time). The broken file is moved to raw/videos/broken/.
"""
import argparse
import os
import shutil
import struct
import sys
import tempfile

import cv2


def boxes(data):
    off = 0
    while off + 8 <= len(data):
        size, typ = struct.unpack(">I4s", data[off:off + 8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", data[off + 8:off + 16])[0], 16
        if size == 0:
            size = len(data) - off
        yield typ.decode("latin1"), off, off + head, off + size
        off += size


def codec_header(path):
    """MPEG-4 Part 2 decoder config (VOS/VOL headers) from the esds box of a good mp4."""
    data = open(path, "rb").read()
    moov = next(data[b:e] for t, _, b, e in boxes(data) if t == "moov")
    k = moov.find(b"esds")
    i = moov.find(b"\x05", k + 8)  # DecoderSpecificInfo tag (the ES / config descriptors come first)
    while i >= 0:
        n, j = 0, i + 1
        for _ in range(4):  # length: up to 4 bytes, 7 bits each
            n = (n << 7) | (moov[j] & 0x7F)
            j += 1
            if not moov[j - 1] & 0x80:
                break
        if moov[j:j + 3] == b"\x00\x00\x01":
            return moov[j:j + n]
        i = moov.find(b"\x05", i + 1)
    raise SystemExit(f"no MPEG-4 decoder config in {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--broken", required=True)
    p.add_argument("--like", required=True)
    p.add_argument("--fps", type=float, default=15.0)
    args = p.parse_args()

    data = open(args.broken, "rb").read()
    mdat = next(((b, e) for t, _, b, e in boxes(data) if t == "mdat"), None)
    if mdat is None:
        raise SystemExit(f"{args.broken}: no video data")
    raw = os.path.join(tempfile.mkdtemp(), "stream.m4v")
    with open(raw, "wb") as f:
        f.write(codec_header(args.like))
        f.write(data[mdat[0]:mdat[1]])
    cap = cv2.VideoCapture(raw)
    out = args.broken + ".fixed.mp4"
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        n += 1
    if writer is None:
        raise SystemExit("no frame could be decoded (different window size than --like?)")
    writer.release()
    os.makedirs(os.path.join(os.path.dirname(args.broken), "broken"), exist_ok=True)
    shutil.move(args.broken, os.path.join(os.path.dirname(args.broken), "broken", os.path.basename(args.broken)))
    os.replace(out, args.broken)
    print(f"{args.broken}: {n} frames ({n / args.fps:.0f} s) recovered")


if __name__ == "__main__":
    sys.exit(main())
