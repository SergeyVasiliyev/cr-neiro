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
    reward_parts.assign(p * kRewardParts, 0.0f);
    done.resize(n);
    taken.assign(p, rl::kNoop);
    winner.assign(n, -1);
    crowns.resize(p);
    episode_steps.resize(n);
    episode_time_ms.resize(n);
    episode_matchup.resize(n);
    matchup.resize(n);

    slots_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        slots_[i].rng.seed(cfg_.seed * 1000003ULL + i * 7919ULL + 1);
        slots_[i].obs_rng.seed(cfg_.seed * 2000029ULL + i * 6151ULL + 3);
    }
    reset();
}

void BatchEnv::set_bot(int env, int side, BotKind kind) {
    if (env < 0 || env >= cfg_.num_envs) throw std::out_of_range("env index out of range");
    if (side < 0 || side > 1) throw std::out_of_range("side must be 0 or 1");
    slots_[env].bot_kind[side] = static_cast<int>(kind);
}

void BatchEnv::reset() {
    for (int i = 0; i < cfg_.num_envs; ++i) reset_env(i);
}

void BatchEnv::reset_env(int i) {
    if (i < 0 || i >= cfg_.num_envs) throw std::out_of_range("env index out of range");
    reset_slot(i);
    write_obs(i);
    reward[2 * i] = reward[2 * i + 1] = 0.0f;
    std::fill_n(reward_parts.begin() + static_cast<size_t>(i) * 2 * kRewardParts, 2 * kRewardParts, 0.0f);
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

void BatchEnv::elixir_losses(Slot& s, double lost[2]) const {
    const Battle& b = *s.battle;
    lost[0] = lost[1] = 0.0;
    std::unordered_map<EntityId, std::pair<int, double>> now;
    for (const auto& e : b.entities()) {
        if (e.is_building() || e.card_index < 0 || e.max_hp <= 0) continue;
        const CardDef& card = data_.card(data_.card_ids[e.card_index]);
        const double full = static_cast<double>(card.elixir) / std::max(1, card.count);
        const double floor = cfg_.reward.trade_floor;
        // Lost HP takes the worth down to `floor`; HP lost to spells (spell_full_worth) all the way.
        const double left = static_cast<double>(std::max(0, e.hp)) / e.max_hp;
        const double spelled = cfg_.reward.spell_full_worth ? std::min(1.0, static_cast<double>(e.spell_damage) / e.max_hp) : 0.0;
        const double worth = full * (1.0 - spelled - (1.0 - floor) * std::max(0.0, 1.0 - left - spelled));
        const int side = idx(e.team);
        now[e.id] = {side, worth};
        auto it = s.troop_worth.find(e.id);
        if (it != s.troop_worth.end()) lost[side] += std::max(0.0, it->second.second - worth);
    }
    for (const auto& [id, tw] : s.troop_worth) {
        if (!now.count(id)) lost[tw.first] += tw.second;  // died
    }
    s.troop_worth = std::move(now);
    for (int side = 0; side < 2; ++side) {
        const auto& plays = b.player(static_cast<Team>(side)).plays;
        for (size_t k = s.plays_counted[side]; k < plays.size(); ++k) {
            const CardDef& card = data_.card(data_.card_ids[plays[k].card_index]);
            if (card.type == CardType::Spell) lost[side] += card.elixir;
        }
        s.plays_counted[side] = plays.size();
    }
}

void BatchEnv::reset_slot(int i) {
    Slot& s = slots_[i];
    s.matchup = static_cast<int>(s.rng() % cfg_.matchups.size());
    const Matchup& m = cfg_.matchups[s.matchup];
    s.battle = std::make_unique<Battle>(data_, m.blue, m.red, s.rng());
    s.battle->set_record_events(false);
    for (int side = 0; side < 2; ++side) {
        const BotKind kind = s.bot_kind[side] >= 0 ? static_cast<BotKind>(s.bot_kind[side]) : cfg_.bots[side];
        s.bots[side] = make_bot(kind, s.rng());
    }
    s.steps = 0;
    for (int side = 0; side < 2; ++side) {
        s.tower_hp[side] = tower_hp(*s.battle, static_cast<Team>(side));
        s.crowns[side] = 0;
        s.leaked[side] = 0.0;
        s.plays_counted[side] = 0;
    }
    s.troop_worth.clear();
    for (auto& pd : s.pending) pd = Slot::Pending{};
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
        rl::ObsConfig obs = cfg_.obs;
        obs.key_spots = cfg_.key_spots[side];
        if (cfg_.obs_miss_enemy > 0.0 || cfg_.obs_jitter > 0.0) {
            obs.miss_enemy = cfg_.obs_miss_enemy;
            obs.jitter = cfg_.obs_jitter;
            obs.rng = &slots_[i].obs_rng;
        }
        rl::encode(b, static_cast<Team>(side), obs, view);
        if (slots_[i].pending[side].ticks >= 0) {  // a move still on its way: nothing to do but wait
            std::fill_n(view.action_mask, rl::kNumActions, uint8_t{0});
            view.action_mask[rl::kNoop] = 1;
        }
    }
}

void BatchEnv::step_slot(int i, const int32_t* actions) {
    Slot& s = slots_[i];
    Battle& b = *s.battle;
    float* parts = reward_parts.data() + static_cast<size_t>(i) * 2 * kRewardParts;
    if (b.finished()) {  // only without auto_reset: waiting for reset_env()
        reward[2 * i] = reward[2 * i + 1] = 0.0f;
        std::fill_n(parts, 2 * kRewardParts, 0.0f);
        done[i] = 1;
        return;
    }
    // Both sides decide on the same state (agents already did, from the last observation), then both moves
    // go on their way, so a bot never reacts to a card played in the same instant. A side whose last move is
    // still on its way (delay_ticks) only waits: its mask allowed nothing else, and its bot is not asked.
    for (int side = 0; side < 2; ++side) {
        Slot::Pending& pd = s.pending[side];
        if (pd.ticks >= 0) continue;
        const int32_t a = actions[2 * i + side];
        if (a == kBotAction) {
            const auto m = s.bots[side]->decide(b, static_cast<Team>(side));
            if (!m) continue;
            pd.bot = true;
            pd.move = *m;
        } else {
            if (a == rl::kNoop) continue;
            pd.bot = false;
            pd.action = a;
        }
        pd.ticks = 0;
        if (cfg_.delay_ticks_max > 0) {
            const int lo = std::max(0, cfg_.delay_ticks_min);
            const int hi = std::max(lo, cfg_.delay_ticks_max);
            pd.ticks = lo + static_cast<int>(s.rng() % static_cast<uint64_t>(hi - lo + 1));
        }
    }
    auto land = [&](int side) {
        const Team team = static_cast<Team>(side);
        Slot::Pending& pd = s.pending[side];
        if (!pd.bot) {
            if (rl::apply_action(b, team, pd.action)) taken[2 * i + side] = pd.action;
        } else {
            const int move = rl::encode_action(b, team, pd.move.hand_slot, pd.move.pos);
            if (b.play(team, pd.move.hand_slot, pd.move.pos)) taken[2 * i + side] = move;
        }
        pd.ticks = -1;
    };
    taken[2 * i] = taken[2 * i + 1] = rl::kNoop;  // (the moves that landed during this step)
    for (int t = 0; t < cfg_.decision_ticks && !b.finished(); ++t) {
        for (int side = 0; side < 2; ++side) {
            if (s.pending[side].ticks == 0) land(side);
        }
        b.step();
        for (int side = 0; side < 2; ++side) {
            if (s.pending[side].ticks > 0) --s.pending[side].ticks;
        }
    }
    ++s.steps;

    const RewardConfig& rc = cfg_.reward;
    double lost[2] = {0.0, 0.0};
    if (rc.elixir_trade != 0.0) elixir_losses(s, lost);
    int hp[2];
    for (int side = 0; side < 2; ++side) hp[side] = tower_hp(b, static_cast<Team>(side));
    const double step_s = cfg_.decision_ticks * data_.rules.tick_ms / 1000.0;
    for (int side = 0; side < 2; ++side) {
        const Team team = static_cast<Team>(side);
        const int other = 1 - side;
        const PlayerState& me = b.player(team);
        double result = 0.0, shaping = 0.0, style = 0.0;
        const double dealt = s.tower_hp[other] - hp[other];
        const double received = s.tower_hp[side] - hp[side];
        shaping += rc.tower_damage * (dealt - received) / full_tower_hp_;
        shaping += rc.crown * ((me.crowns - s.crowns[side]) - (b.player(opponent(team)).crowns - s.crowns[other]));
        shaping -= rc.elixir_leak * (me.elixir_leaked - s.leaked[side]);
        shaping += rc.elixir_trade * (lost[other] - lost[side]);
        if (rc.elixir_bank != 0.0 && me.elixir < data_.rules.max_elixir - 1e-6) {
            style += rc.elixir_bank * std::min(me.elixir, rc.bank_cap) * step_s;
        }
        if (b.finished()) {
            const auto w = b.winner();
            if (w) result += *w == team ? rc.win : -rc.win;
        }
        reward[2 * i + side] = static_cast<float>(result + shaping + style);
        parts[side * kRewardParts + 0] = static_cast<float>(result);
        parts[side * kRewardParts + 1] = static_cast<float>(shaping);
        parts[side * kRewardParts + 2] = static_cast<float>(style);
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
