"""Does a policy finish the game? Overtime (0:0), an enemy princess tower low, a spell in hand that destroys it,
the elixir for it: greedy policy, nothing helping it (no finisher rule).

    .venv\\Scripts\\python.exe tools\\finisher_test.py runs\\v7b\\final.pt
Prints "finishes N/M": games it won within 6 s.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import crsim  # noqa: E402
import crsim_env as ce  # noqa: E402
from model import PolicyPlayer, load_policy  # noqa: E402

CASES = (("fireball", 120), ("arrows", 50))
SEEDS = 12


def main():
    path = sys.argv[1]
    data = ce.load_data()
    deck = ce.DEFAULT_DECK
    blue, red = crsim.Team.BLUE, crsim.Team.RED
    net, ck = load_policy(path, "cuda")
    player = PolicyPlayer(net, "cuda", greedy=True)
    ticks = ck.get("args", {}).get("decision_ticks", 6)
    key_spots = bool(ck.get("key_spots", False))
    won = total = 0
    for spell, hp in CASES:
        for seed in range(SEEDS):
            b = crsim.Battle(data, deck, deck, seed)
            while b.phase != crsim.Phase.OVERTIME:
                b.step()
            b.steps(200)
            towers = [e for e in b.entities if e.team == red and e.def_id == "princess_tower"]
            b.twin_set_hp(towers[seed % 2].id, hp)
            others = [c for c in deck if c != spell]
            hand = [spell] + others[seed % 4: seed % 4 + 3]
            b.set_hand(blue, hand, [c for c in deck if c not in hand][0])
            b.set_elixir(blue, 6.0)
            for _ in range(int(6.0 / (ticks * 0.05))):
                if b.finished:
                    break
                a = int(player(crsim.rl.encode(b, blue, key_spots=key_spots)))
                if a > 0:
                    crsim.rl.apply_action(b, blue, a)
                b.steps(ticks)
            b.steps(60)
            won += bool(b.finished and b.winner == blue)
            total += 1
    print(f"finishes {won}/{total}")


if __name__ == "__main__":
    main()
