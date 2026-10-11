#pragma once

// N independent matches stepped in parallel (OpenMP), writing observations straight into flat
// buffers that Python exposes as numpy arrays without copying. Both players of every match are
// controllable; an action of kBotAction hands that side to a built-in scripted bot.
#include <cstdint>
#include <memory>
#include <random>
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
};

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
    RewardConfig reward;
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
    std::vector<uint8_t> done;    // [num_envs]
    // Results of the episode that just ended (valid where done = 1).
    std::vector<int8_t> winner;   // 0 blue, 1 red, 2 draw
    std::vector<int32_t> crowns;  // [num_envs][2]
    std::vector<int32_t> episode_steps;
    std::vector<int32_t> episode_time_ms;
    std::vector<int32_t> episode_matchup;
    std::vector<int32_t> matchup;  // matchup of the episode currently running in each env

private:
    struct Slot {
        std::unique_ptr<Battle> battle;
        std::unique_ptr<Bot> bots[2];
        std::mt19937_64 rng;
        int matchup = 0;
        int steps = 0;
        int tower_hp[2] = {0, 0};
        int crowns[2] = {0, 0};
        double leaked[2] = {0.0, 0.0};
    };

    void reset_slot(int i);
    void step_slot(int i, const int32_t* actions);
    void write_obs(int i);
    int tower_hp(const Battle& b, Team team) const;

    GameData data_;
    BatchEnvConfig cfg_;
    std::vector<Slot> slots_;
    double full_tower_hp_ = 1.0;
};

}  // namespace cr
