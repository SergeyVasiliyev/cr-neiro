"""After a training cycle's post-run checks: keep its final under its own name, pick the start of the next cycle (the
new final if it beat the old one at temp 0.5, else the old one), and make the better one the bot's default.

Writes "<start> <the other one>" to --out for the next cycle's script (scripts\\night_20261112.bat reads it).

    .venv\\Scripts\\python.exe tools\\pick_start.py --new runs/h1/final.pt --save-as runs/h1/final_cycle5.pt ^
        --old runs/h1/final_cycle4.pt --result raw/h1_cycle5_after.txt --out raw/h1_cycle6_start.txt
"""
import argparse
import datetime
import os
import re
import shutil


def scores(text):
    """{opponent as written in the checks: score} from a cycle's post-run checks (evaluate.py's output)."""
    return {m.group(1): float(m.group(2)) for m in
            re.finditer(r"=== cycle \d+ final vs (\S+), \d+ games.*?score ([0-9.]+) \+-", text, re.S)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", default="runs/h1/final.pt", help="the cycle's final (the next cycle overwrites it)")
    p.add_argument("--save-as", default="runs/h1/final_cycle5.pt")
    p.add_argument("--old", default="runs/h1/final_cycle4.pt", help="the cycle's start, the best net before it")
    p.add_argument("--result", default="raw/h1_cycle5_after.txt", help="the post-run checks (first: new vs old)")
    p.add_argument("--out", default="raw/h1_cycle6_start.txt")
    p.add_argument("--cycle", type=int, default=5, help="the new final's cycle number (for the bot scripts' note)")
    p.add_argument("--old-cycle", type=int, default=None, help="the old one's cycle number (default: --cycle - 1)")
    p.add_argument("--bot-scripts", nargs="*", default=["scripts/play_vs_bot.bat", "scripts/play_recorded.bat"])
    p.add_argument("--user-gain", type=float, default=None,
                   help="also weigh the user starts (evaluate.py --user-starts: the new and the old net as blue against "
                        "the old one, the user's real cards scripted; sections '=== user starts ...: new as blue' and "
                        "'...: old as blue'): the new one is taken if it won the head-to-head and is not this much worse "
                        "there, or if it is at least this much better there and scored --min-score head-to-head")
    p.add_argument("--min-score", type=float, default=0.45)
    args = p.parse_args()

    if os.path.exists(args.new) and not os.path.exists(args.save_as):
        shutil.copy2(args.new, args.save_as)
        print(f"kept {args.new} as {args.save_as}")
    text = open(args.result, encoding="utf-8", errors="replace").read() if os.path.exists(args.result) else ""
    found = scores(text)
    first = re.search(r"score ([0-9.]+) \+- ([0-9.]+)", text)  # the first match: the new final vs the old one
    score = float(first.group(1)) if first else None
    better = score is not None and score > 0.5 and os.path.exists(args.save_as)
    if args.user_gain is not None and score is not None and os.path.exists(args.save_as):
        user = {m.group(1): float(m.group(2)) for m in
                re.finditer(r"=== user starts[^\n]*?: (new|old) as blue.*?score ([0-9.]+) \+-", text, re.S)}
        if "new" in user and "old" in user:
            gain = user["new"] - user["old"]
            better = (score > 0.5 and gain > -args.user_gain) or (score >= args.min_score and gain >= args.user_gain)
            print(f"user starts: new {user['new']:.3f}, old {user['old']:.3f} (gain {gain:+.3f}); head-to-head {score:.3f}")
        else:
            print("user starts: not found, the head-to-head alone decides")
    if better:
        start, other = args.save_as, args.old
    else:
        start, other = args.old, args.save_as if os.path.exists(args.save_as) else args.old
    with open(args.out, "w", encoding="ascii", newline="") as f:
        f.write(f"{start} {other}\n")
    print(f"the new final vs the old: {score if score is not None else 'not found'}; next start: {start} "
          f"(the other: {other})")

    if better:  # the bot plays the better net from now on
        old_name, new_name = os.path.basename(args.old), os.path.basename(args.save_as)
        v7e = found.get("runs/v7e/top.pt")
        si2 = found.get("runs/si2/best.pt")
        note = (f"user's clone, {args.cycle} cycles; at temp 0.5 beats cycle {args.old_cycle or args.cycle - 1} {score:.0%}"
                + (f", v7e top {v7e:.0%}" if v7e is not None else "") + (f", si2 {si2:.0%}" if si2 is not None else "")
                + ")")
        for path in args.bot_scripts:
            if not os.path.exists(path):
                continue
            s = open(path, encoding="utf-8").read()
            s2 = s.replace(old_name, new_name)
            s2 = re.sub(r"user's clone, \d+ cycles; at temp 0\.5 beats [^\n]*\)", lambda _: note, s2)
            s2 = re.sub(r"\(the best one so far, \d{4}-\d\d-\d\d:", lambda _: "(the best one so far, "
                        f"{datetime.date.today().isoformat()}:", s2)
            if s2 != s:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(s2)
                print(f"{path}: now {new_name}")


if __name__ == "__main__":
    main()
