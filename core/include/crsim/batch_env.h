#pragma once

// N independent matches stepped in parallel (OpenMP), writing observations straight into flat
// buffers that Python exposes as numpy arrays without copying. Both players of every match are
// controllable; an action of kBotAction hands that side to a built-in scripted bot.
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

#include "crsim/battle.h"
#include "crsim/bots.h"
#include "crsim/rl.h"

namespace cr {

constexpr int32_t kBotAction = -1;

struct RewardConfig {
    double win = 1.0;           // +win for a victory, -win for a defeat, 0 for a draw
    double crown = 0.0;         // per crown taken (minus per crown lost)
    double tower_damage = 0.0;  // per one side's full tower HP worth of damage dealt (minus taken)
    double elixir_leak = 0.0;   // penalty per elixir wasted at max elixir
    // ... only for the leak the opponent does not match (both sitting at 10 and waiting is fine, as pros
    // open a match): the penalty is on my leak minus theirs, and their leak beyond mine is a gain
    bool leak_relative = false;
    // Elixir trades, the heart of the game: per elixir worth of the opponent's troops lost, minus our own.
    // A troop is worth its card's cost split over the troops it makes (an archer 1.5, a giant 5). Damage
    // takes it down only to trade_floor of that (a Musketeer left at 10% HP still shoots as hard as a
    // fresh one); the rest goes when it dies. A spell's cost counts as spent when cast: a Fireball that
    // kills a Musketeer is an even trade, one that also kills the Archers a good one.
    double elixir_trade = 0.0;
    double trade_floor = 0.25;
    // ... except damage from spells, with this on: it takes the worth down in full (a Fireball that leaves a
    // Musketeer at 10% did its job).
    bool spell_full_worth = false;
    // Patience: per elixir held, per second, counted up to bank_cap and only below a full bar (at full,
    // elixir leaks instead), so the best place to be is just under full - pushes built from a full bar.
    double elixir_bank = 0.0;
    double bank_cap = 9.0;
};

// Rewards come in parts, for critics of their own: the result, the shaping signals (towers, crowns, elixir
// trades, leak) and the style ones (bank). BatchEnv::reward is their plain sum.
constexpr int kRewardParts = 3;

struct Matchup {
    Deck blue;
    Deck red;
};

struct BatchEnvConfig {
    int num_envs = 1;
    int num_threads = 0;      // 0 = OpenMP default (all cores)
    int decision_ticks = 10;  // ticks simulated per step (10 x 50 ms = 0.5 s)
    uint64_t seed = 0;
    // true: finished matches restart inside step() (vectorized training).
    // false: a finished env keeps its terminal observation until reset_env() (Gymnasium-style).
    bool auto_reset = true;
    std::vector<Matchup> matchups;  // one is drawn uniformly at every reset
    BotKind bots[2] = {BotKind::Heuristic, BotKind::Heuristic};  // [blue, red], used on kBotAction
    rl::ObsConfig obs;
    int key_spots[2] = {0, 0};  // per side (blue, red): rl::ObsConfig::key_spots levels (obs.key_spots is ignored)
    RewardConfig reward;
    // A move lands this many ticks after the decision (uniform; may be past the next decision: meanwhile that
    // side can only wait), like the live bot's touches.
    int delay_ticks_min = 0;
    int delay_ticks_max = 0;
    // Observation noise like the screen reader's: an enemy troop missed with this chance per observation,
    // enemy troops seen up to this many tiles off.
    double obs_miss_enemy = 0.0;
    double obs_jitter = 0.0;
    // Enemy elixir as live (rl::ObsConfig::enemy_elixir_offset): per match and side an offset drawn from
    // N(elixir_noise_mean, elixir_noise_sd), plus elixir_noise_jitter per observation. All 0: exact.
    double elixir_noise_mean = 0.0;
    double elixir_noise_sd = 0.0;
    double elixir_noise_jitter = 0.0;
    // The opponent seen like the live bot's twin sees it (rl::LiveView): plays missed (their troops never seen,
    // their elixir never taken off), phantom plays (elixir taken off for nothing), plays made out up to
    // live_late_ms late, troops' HP seen off. All 0: exact.
    double live_miss_play = 0.0;
    double live_phantom_play = 0.0;
    int live_late_ms = 0;
    double live_hp_off = 0.0;
    double live_hp_off_sd = 0.0;
    // Endgame starts: this share of episodes begins late (a random time in [from, to) ms, nobody having played
    // so far: 0:0), the princess towers at random HP (a quarter within a spell's reach, 30-300 HP, a quarter
    // low), random elixir, a random point in the cycle - to learn what full games rarely show: finishing a low
    // tower (or not, when the spell would land too late), defending one, overtime.
    double endgame_share = 0.0;
    int endgame_from_ms = 140000;
    int endgame_to_ms = 280000;
};

class BatchEnv {
public:
    // Copies `data`; throws on invalid configuration (unknown cards, empty matchups, ...).
    BatchEnv(const GameData& data, BatchEnvConfig cfg);
    BatchEnv(const BatchEnv&) = delete;
    BatchEnv& operator=(const BatchEnv&) = delete;

    // Starts a fresh episode in every env and writes observations.
    void reset();
    void reset_env(int env);
    // actions: [num_envs][2] (blue, red). Finished episodes are reset automatically: after the
    // call, done[i] = 1 marks that rewards/results are final and the observation is the new episode's.
    void step(const int32_t* actions);
    // The scripted bot for one side of one env, from its next episode on (instead of config().bots).
    void set_bot(int env, int side, BotKind kind);
    // The placements one side of one env may use (rl::ObsConfig::key_spots level), instead of the config's: a
    // pool opponent trained on another set of spots keeps to its own.
    void set_key_spots(int env, int side, int level);
    // Positions to start from: this share of the episodes begins at a copy of one of them (picked uniformly)
    // instead of a fresh match - moments of recorded human battles, to learn what its own games never show
    // (a push built behind the king, a low tower, an elixir lead). Taken as they are but for the twin's
    // bookkeeping (it never finishes on its own, logs events): off. One that is over, or whose cards are not
    // a matchup's decks, is left out; returns how many were kept. The battles' GameData must outlive the env.
    int set_starts(const std::vector<const Battle*>& starts, double share);

    int num_envs() const { return cfg_.num_envs; }
    const BatchEnvConfig& config() const { return cfg_; }
    const GameData& data() const { return data_; }
    const Battle& battle(int env) const { return *slots_.at(env).battle; }

    // Observation buffers, layout [num_envs][2 players][...].
    std::vector<float> global;          // [.][.][rl::kGlobalFeatures]
    std::vector<float> entities;        // [.][.][rl::kMaxEntities][rl::kEntityFeatures]
    std::vector<int32_t> entity_token;  // [.][.][rl::kMaxEntities]
    std::vector<uint8_t> entity_mask;   // [.][.][rl::kMaxEntities]
    std::vector<int32_t> hand;          // [.][.][rl::kHandTokens]
    std::vector<int32_t> enemy_history;      // [.][.][rl::kHistory]
    std::vector<float> enemy_history_age;    // [.][.][rl::kHistory]
    std::vector<uint8_t> action_mask;   // [.][.][rl::kNumActions]

    // Step results.
    std::vector<float> reward;    // [num_envs][2]
    std::vector<float> reward_parts;  // [num_envs][2][kRewardParts]: result, shaping, style
    std::vector<uint8_t> done;    // [num_envs]
    // The move each side made this step, as an action (bots' moves encoded too; rl::kNoop if none):
    // a scripted bot's games become examples to imitate.
    std::vector<int32_t> taken;   // [num_envs][2]
    // Results of the episode that just ended (valid where done = 1).
    std::vector<int8_t> winner;   // 0 blue, 1 red, 2 draw
    std::vector<int32_t> crowns;  // [num_envs][2]
    std::vector<int32_t> episode_steps;
    std::vector<int32_t> episode_time_ms;
    std::vector<int32_t> episode_matchup;
    std::vector<int32_t> episode_start;  // set_starts: the position it started from, -1 = a fresh match
    std::vector<int32_t> matchup;  // matchup of the episode currently running in each env
    std::vector<int32_t> start;    // ... and the position it started from (-1 = fresh)

private:
    struct Slot {
        std::unique_ptr<Battle> battle;
        std::unique_ptr<Bot> bots[2];
        int bot_kind[2] = {-1, -1};  // set_bot: BotKind for the next episodes, -1 = config's
        int key_spots[2] = {-1, -1};  // set_key_spots: the level, -1 = config's
        std::mt19937_64 rng;
        std::mt19937_64 obs_rng;     // observation noise only, so it never changes how games go
        double elixir_offset[2] = {0.0, 0.0};  // this match's enemy elixir offset per side (elixir_noise_*)
        rl::LiveView live[2];  // each side's view of the other this match (live_*)
        // A move decided but not landed yet (delay_ticks): until it lands that side only waits, like the
        // live bot while its touches go through.
        struct Pending {
            int ticks = -1;  // ticks left before it lands, -1 = none
            int32_t action = 0;
            bool bot = false;
            BotMove move;
        };
        Pending pending[2];
        int matchup = 0;
        int start = -1;  // set_starts: the position this episode began at, -1 = a fresh match
        int steps = 0;
        int tower_hp[2] = {0, 0};
        int crowns[2] = {0, 0};
        double leaked[2] = {0.0, 0.0};
        // elixir trades: each troop's remaining worth (team, elixir), and plays already counted
        std::unordered_map<EntityId, std::pair<int, double>> troop_worth;
        size_t plays_counted[2] = {0, 0};
    };

    // Elixir lost by each side since the last step (troop worth gone, spells cast); updates the slot.
    void elixir_losses(Slot& s, double lost[2]) const;

    void reset_slot(int i);
    void start_endgame(Slot& s);
    void step_slot(int i, const int32_t* actions);
    void write_obs(int i);
    int tower_hp(const Battle& b, Team team) const;

    GameData data_;
    BatchEnvConfig cfg_;
    std::vector<Slot> slots_;
    std::vector<std::unique_ptr<Battle>> starts_;  // set_starts
    double start_share_ = 0.0;
    double full_tower_hp_ = 1.0;
};

}  // namespace cr
