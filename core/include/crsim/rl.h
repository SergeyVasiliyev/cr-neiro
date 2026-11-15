#pragma once

// Reinforcement-learning interface: observation encoding, action space and masks.
// Everything is expressed in the acting player's own frame (their side at the bottom),
// so one policy can play either side.
#include <cstdint>
#include <random>
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
constexpr int kEntityFeatures = 18;
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

// The live bot sees the opponent through its digital twin (python/live/session.py), which follows the screen. On
// 32 two-sided battles (2026-11-15, tools/twin_plays_check.py) the twin missed 3.5% of the user's plays (their
// units never showed, their elixir was never taken off), added 3% phantom ones (elixir taken off for nothing), had a
// new troop ~0.45 s late on average, and the HP of ~5% of the units off by ~30% of their max. A LiveView makes an
// observation like that: its own count of the opponent's elixir (filling up, the plays it sees taken off, held
// between 0 and the max like the twin's), and per play and per unit choices drawn from a hash of its seed. One per
// match and observer: the caller keeps it between observations and resets it, with a new seed, every match.
struct LiveView {
    double miss_play = 0.0;     // chance an opponent play is missed
    double phantom_play = 0.0;  // chance per opponent play seen of a phantom one besides (a card of their deck)
    int late_ms = 0;            // an opponent play is made out (its troops, its place in the history) 0..late_ms late
    double hp_off = 0.0;        // chance a troop's HP is seen off ...
    double hp_off_sd = 0.0;     // ... by a normal share of its max, all its life
    uint64_t seed = 0;
    // the observer's state
    bool started = false;
    double count = 0.0;  // its count of the opponent's elixir
    int last_ms = 0;
    size_t judged = 0;   // opponent plays (in order) already taken off the count
    bool enabled() const { return miss_play > 0.0 || phantom_play > 0.0 || late_ms > 0 || hp_off > 0.0; }
    void reset(uint64_t new_seed) {
        seed = new_seed;
        started = false;
        count = 0.0;
        last_ms = 0;
        judged = 0;
    }
};

struct ObsConfig {
    // Not shown in the game, but computable from what is visible: the opponent's elixir as if their
    // cards still waiting as blueprints (hidden from us) had not been played yet.
    bool observe_enemy_elixir = true;
    // Which placements are legal: 0 = every tile; 1 = key spots only (key_spot_cells): ~30 per troop card, a
    // few per spell; 2 = key spots and the tactical ones next to the troops on the field (key_spot_cells).
    int key_spots = 0;
    // Training noise like the screen reader's (used only with rng): each enemy troop is missed with this
    // chance, and enemy troops are seen up to this many tiles off.
    double miss_enemy = 0.0;
    double jitter = 0.0;
    // Live, the opponent's elixir is the twin's estimate (off by ~1.6 on average, mostly too high: plays it
    // missed): the enemy elixir feature can be shifted by this much (an offset of the match's own) plus a
    // normal jitter of this spread per observation (with rng).
    double enemy_elixir_offset = 0.0;
    double enemy_elixir_jitter = 0.0;
    std::mt19937_64* rng = nullptr;
    // Seen like the live bot sees it (LiveView), if set and enabled; it is updated by every observation.
    LiveView* live = nullptr;
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
void action_mask(const Battle& battle, Team team, uint8_t* out, int key_spots = 0);
// The spots a player would really use (cells, own frame): for troops - behind the king and the princess
// towers, the back corners, the bridges, the middle at a few depths, in front of the towers, the pockets
// of fallen enemy towers, and on / in front of the enemy troop deepest in our half; for spells - the enemy
// princess towers (the king's once awake), a tower with the troops next to it, the three densest groups of
// enemy troops and that deepest troop, troops aimed where they will be when the spell lands.
// tactical: also troop spots next to the troops on the field - right behind our two foremost walking ground
// troops (to push behind a giant), and for the two enemy troops deepest in our half: on it and a little in front,
// at either side of it and at its back (to come at it from the side), and 3.5 tiles from it towards the middle in
// front of our king (to pull it between the towers); and once an enemy princess tower is down, the deepest corner
// of its pocket next to the middle.
void key_spot_cells(const Battle& battle, Team team, std::vector<int>& troop_cells, std::vector<int>& spell_cells,
                    bool tactical = false);

// Absolute arena position of the center of an own-frame grid cell.
Vec2 cell_position(const Battle& battle, Team team, int cell);
// Plays the card for `action`. kNoop and illegal actions do nothing and return false.
bool apply_action(Battle& battle, Team team, int action);
// The action for playing hand slot `slot` at `pos` (absolute tiles): the cell it falls in, own frame.
int encode_action(const Battle& battle, Team team, int slot, Vec2 pos);

}  // namespace cr::rl
