#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "crsim/battle.h"

namespace cr {

enum class BotKind : uint8_t {
    Noop,       // never plays: a punching bag for curriculum training
    Random,     // random card at a random spot on its own half
    Heuristic,  // defends against troops on its half, pushes a lane when elixir is high
};

BotKind bot_kind_from_string(const std::string& name);  // "noop" | "random" | "heuristic"
const char* bot_kind_name(BotKind kind);

struct BotMove {
    int hand_slot = 0;
    Vec2 pos;  // absolute arena position
};

// Scripted opponent, asked once per decision step. decide() only looks at the battle, so both
// sides can decide on the same state before either move is applied (see act_simultaneously).
class Bot {
public:
    virtual ~Bot() = default;
    virtual std::optional<BotMove> decide(const Battle& battle, Team team) = 0;

    // Decide and play right away (fine when the other side is not deciding at the same moment).
    void act(Battle& battle, Team team) {
        if (auto m = decide(battle, team)) battle.play(team, m->hand_slot, m->pos);
    }
};

std::unique_ptr<Bot> make_bot(BotKind kind, uint64_t seed);

// Both bots decide on the same state, then both moves are applied: neither side gets to react to a
// card the other side played in the same instant.
void act_simultaneously(Battle& battle, Bot& blue, Bot& red);

}  // namespace cr
