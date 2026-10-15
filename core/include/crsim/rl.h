#pragma once

// Reinforcement-learning interface: observation encoding, action space and masks.
// Everything is expressed in the acting player's own frame (their side at the bottom),
// so one policy can play either side.
#include <cstdint>
#include <vector>

#include "crsim/battle.h"

namespace cr::rl {

// Deploy grid: one action per tile of the arena (own frame), per hand slot.
constexpr int kGridW = 18;
constexpr int kGridH = 32;
constexpr int kCells = kGridW * kGridH;
constexpr int kNoop = 0;
constexpr int kNumActions = 1 + kHandSize * kCells;  // action = 1 + slot * kCells + row * kGridW + col

constexpr int kMaxEntities = 64;     // towers first, then troops and flying spells closest to our side
constexpr int kEntityFeatures = 14;
constexpr int kGlobalFeatures = 25;
constexpr int kHandTokens = kDeckSize;  // own full cycle: 4 cards in hand, then the queue in order
constexpr int kHistory = kDeckSize;     // the opponent's last plays, newest first

// Token vocabulary for entities and cards: 0 = padding, 1/2 = towers, then CardDef::index + 3.
constexpr int kPadToken = 0;
constexpr int kPrincessToken = 1;
constexpr int kKingToken = 2;
constexpr int kFirstCardToken = 3;
inline int vocab_size(const GameData& data) { return kFirstCardToken + static_cast<int>(data.cards.size()); }
inline int card_token(const CardDef& card) { return kFirstCardToken + card.index; }

struct ObsConfig {
    // Not shown in the game, but computable from what is visible: the opponent's elixir as if their
    // cards still waiting as blueprints (hidden from us) had not been played yet.
    bool observe_enemy_elixir = true;
    // Only key spots are legal (key_spot_cells): ~30 per troop card, a few per spell, instead of every tile.
    bool key_spots = false;
};

// Caller-owned buffers for one player's observation.
struct ObsView {
    float* global;          // [kGlobalFeatures]
    float* entities;        // [kMaxEntities][kEntityFeatures]
    int32_t* entity_token;  // [kMaxEntities]
    uint8_t* entity_mask;   // [kMaxEntities], 1 = real entity
    int32_t* hand;          // [kHandTokens]
    int32_t* enemy_history;      // [kHistory] card tokens of revealed plays, 0 = none yet
    float* enemy_history_age;    // [kHistory] seconds since each was revealed / 10
    uint8_t* action_mask;   // [kNumActions], may be null
};

// Throws std::runtime_error if the arena does not match the deploy grid.
void check_arena(const GameData& data);

void encode(const Battle& battle, Team team, const ObsConfig& cfg, const ObsView& out);
void action_mask(const Battle& battle, Team team, uint8_t* out, bool key_spots = false);
// The spots a player would really use (cells, own frame): for troops - behind the king and the princess
// towers, the back corners, the bridges, the middle at a few depths, in front of the towers, the pockets
// of fallen enemy towers, and on / in front of the enemy troop deepest in our half; for spells - the enemy
// princess towers (the king's once awake), a tower with the troops next to it, the three densest groups of
// enemy troops and that deepest troop, troops aimed where they will be when the spell lands.
void key_spot_cells(const Battle& battle, Team team, std::vector<int>& troop_cells, std::vector<int>& spell_cells);

// Absolute arena position of the center of an own-frame grid cell.
Vec2 cell_position(const Battle& battle, Team team, int cell);
// Plays the card for `action`. kNoop and illegal actions do nothing and return false.
bool apply_action(Battle& battle, Team team, int action);
// The action for playing hand slot `slot` at `pos` (absolute tiles): the cell it falls in, own frame.
int encode_action(const Battle& battle, Team team, int slot, Vec2 pos);

}  // namespace cr::rl
