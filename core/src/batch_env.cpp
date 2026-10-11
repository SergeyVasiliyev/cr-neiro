#include "crsim/batch_env.h"

#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cr {

BatchEnv::BatchEnv(const GameData& data, BatchEnvConfig cfg) : data_(data), cfg_(std::move(cfg)) {
    rl::check_arena(data_);
    if (cfg_.num_envs < 1) throw std::invalid_argument("num_envs must be >= 1");
    if (cfg_.decision_ticks < 1) throw std::invalid_argument("decision_ticks must be >= 1");
    if (cfg_.matchups.empty()) throw std::invalid_argument("at least one matchup is required");
    for (const auto& m : cfg_.matchups) {
        for (const auto& id : m.blue) data_.card(id);  // throws std::out_of_range on unknown ids
        for (const auto& id : m.red) data_.card(id);
    }
    full_tower_hp_ = data_.king_tower.hitpoints + 2.0 * data_.princess_tower.hitpoints;

    const size_t n = static_cast<size_t>(cfg_.num_envs);
    const size_t p = n * 2;
    global.resize(p * rl::kGlobalFeatures);
    entities.resize(p * rl::kMaxEntities * rl::kEntityFeatures);
    entity_token.resize(p * rl::kMaxEntities);
    entity_mask.resize(p * rl::kMaxEntities);
    hand.resize(p * rl::kHandTokens);
    enemy_history.resize(p * rl::kHistory);
    enemy_history_age.resize(p * rl::kHistory);
    action_mask.resize(p * rl::kNumActions);
    reward.resize(p);
    done.resize(n);
    winner.assign(n, -1);
    crowns.resize(p);
    episode_steps.resize(n);
    episode_time_ms.resize(n);
    episode_matchup.resize(n);
    matchup.resize(n);

    slots_.resize(n);
    for (size_t i = 0; i < n; ++i) slots_[i].rng.seed(cfg_.seed * 1000003ULL + i * 7919ULL + 1);
    reset();
}

void BatchEnv::reset() {
    for (int i = 0; i < cfg_.num_envs; ++i) reset_env(i);
}

void BatchEnv::reset_env(int i) {
    if (i < 0 || i >= cfg_.num_envs) throw std::out_of_range("env index out of range");
    reset_slot(i);
    write_obs(i);
    reward[2 * i] = reward[2 * i + 1] = 0.0f;
    done[i] = 0;
    winner[i] = -1;
}

int BatchEnv::tower_hp(const Battle& b, Team team) const {
    int total = 0;
    for (EntityId id : {b.king_tower(team), b.princess_tower(team, 0), b.princess_tower(team, 1)}) {
        if (const Entity* t = b.find(id)) total += std::max(0, t->hp);
    }
    return total;
}

void BatchEnv::reset_slot(int i) {
    Slot& s = slots_[i];
    s.matchup = static_cast<int>(s.rng() % cfg_.matchups.size());
    const Matchup& m = cfg_.matchups[s.matchup];
    s.battle = std::make_unique<Battle>(data_, m.blue, m.red, s.rng());
    s.battle->set_record_events(false);
    for (int side = 0; side < 2; ++side) s.bots[side] = make_bot(cfg_.bots[side], s.rng());
    s.steps = 0;
    for (int side = 0; side < 2; ++side) {
        s.tower_hp[side] = tower_hp(*s.battle, static_cast<Team>(side));
        s.crowns[side] = 0;
        s.leaked[side] = 0.0;
    }
    matchup[i] = s.matchup;
}

void BatchEnv::write_obs(int i) {
    const Battle& b = *slots_[i].battle;
    for (int side = 0; side < 2; ++side) {
        const size_t p = static_cast<size_t>(i) * 2 + side;
        rl::ObsView view{
            global.data() + p * rl::kGlobalFeatures,
            entities.data() + p * rl::kMaxEntities * rl::kEntityFeatures,
            entity_token.data() + p * rl::kMaxEntities,
            entity_mask.data() + p * rl::kMaxEntities,
            hand.data() + p * rl::kHandTokens,
            enemy_history.data() + p * rl::kHistory,
            enemy_history_age.data() + p * rl::kHistory,
            action_mask.data() + p * rl::kNumActions,
        };
        rl::encode(b, static_cast<Team>(side), cfg_.obs, view);
    }
}

void BatchEnv::step_slot(int i, const int32_t* actions) {
    Slot& s = slots_[i];
    Battle& b = *s.battle;
    if (b.finished()) {  // only without auto_reset: waiting for reset_env()
        reward[2 * i] = reward[2 * i + 1] = 0.0f;
        done[i] = 1;
        return;
    }
    // Both sides decide on the same state (agents already did, from the last observation), then
    // both moves are applied, so a bot never reacts to a card played in the same instant.
    std::optional<BotMove> bot_moves[2];
    for (int side = 0; side < 2; ++side) {
        if (actions[2 * i + side] == kBotAction) bot_moves[side] = s.bots[side]->decide(b, static_cast<Team>(side));
    }
    for (int side = 0; side < 2; ++side) {
        const Team team = static_cast<Team>(side);
        const int32_t a = actions[2 * i + side];
        if (a != kBotAction) rl::apply_action(b, team, a);
        else if (bot_moves[side]) b.play(team, bot_moves[side]->hand_slot, bot_moves[side]->pos);
    }
    for (int t = 0; t < cfg_.decision_ticks && !b.finished(); ++t) b.step();
    ++s.steps;

    const RewardConfig& rc = cfg_.reward;
    int hp[2];
    for (int side = 0; side < 2; ++side) hp[side] = tower_hp(b, static_cast<Team>(side));
    for (int side = 0; side < 2; ++side) {
        const Team team = static_cast<Team>(side);
        const int other = 1 - side;
        const PlayerState& me = b.player(team);
        double r = 0.0;
        const double dealt = s.tower_hp[other] - hp[other];
        const double taken = s.tower_hp[side] - hp[side];
        r += rc.tower_damage * (dealt - taken) / full_tower_hp_;
        r += rc.crown * ((me.crowns - s.crowns[side]) - (b.player(opponent(team)).crowns - s.crowns[other]));
        r -= rc.elixir_leak * (me.elixir_leaked - s.leaked[side]);
        if (b.finished()) {
            const auto w = b.winner();
            if (w) r += *w == team ? rc.win : -rc.win;
        }
        reward[2 * i + side] = static_cast<float>(r);
    }
    for (int side = 0; side < 2; ++side) {
        s.tower_hp[side] = hp[side];
        s.crowns[side] = b.player(static_cast<Team>(side)).crowns;
        s.leaked[side] = b.player(static_cast<Team>(side)).elixir_leaked;
    }

    if (b.finished()) {
        done[i] = 1;
        const auto w = b.winner();
        winner[i] = static_cast<int8_t>(w ? idx(*w) : 2);
        crowns[2 * i] = s.crowns[0];
        crowns[2 * i + 1] = s.crowns[1];
        episode_steps[i] = s.steps;
        episode_time_ms[i] = b.time_ms();
        episode_matchup[i] = s.matchup;
        if (cfg_.auto_reset) reset_slot(i);
    } else {
        done[i] = 0;
        winner[i] = -1;
    }
    write_obs(i);
}

void BatchEnv::step(const int32_t* actions) {
    const int n = cfg_.num_envs;
#ifdef _OPENMP
    // Per-loop thread count: never touch the process-wide OpenMP setting (other envs, torch).
    const int threads = cfg_.num_threads > 0 ? cfg_.num_threads : omp_get_num_procs();
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int i = 0; i < n; ++i) step_slot(i, actions);
#else
    for (int i = 0; i < n; ++i) step_slot(i, actions);
#endif
}

}  // namespace cr
