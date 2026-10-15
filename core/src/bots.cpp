#include "crsim/bots.h"

#include <algorithm>
#include <random>
#include <utility>
#include <stdexcept>
#include <vector>

#include "crsim/frame.h"

namespace cr {
namespace {

double uniform(std::mt19937_64& rng, double lo, double hi) {
    return lo + (hi - lo) * static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);
}

// pos (own frame) or a few jittered spots around it where the card can be played.
std::optional<BotMove> spot_near(const Battle& b, Team team, int slot, Vec2 own_pos, std::mt19937_64& rng) {
    const ArenaDef& a = b.data().arena;
    for (int attempt = 0; attempt < 6; ++attempt) {
        Vec2 p = own_pos;
        if (attempt > 0) p = p + Vec2{uniform(rng, -1.5, 1.5), uniform(rng, -1.5, 1.0)};
        p.x = std::clamp(p.x, 0.5, a.width - 0.5);
        p.y = std::clamp(p.y, 0.5, a.river_min_y - 0.5);
        const Vec2 abs = from_own_frame(a, team, p);
        if (b.check_play(team, slot, abs).empty()) return BotMove{slot, abs};
    }
    return std::nullopt;
}

std::optional<BotMove> if_playable(const Battle& b, Team team, int slot, Vec2 pos) {
    if (b.check_play(team, slot, pos).empty()) return BotMove{slot, pos};
    return std::nullopt;
}

bool is_spell(const CardDef& card) { return card.type == CardType::Spell; }

// Best spot for an affordable spell in hand: the enemy troop around which the spell would destroy
// at least its own elixir cost (each unit is worth its card's cost / unit count, scaled by the share
// of its HP the spell removes). Returns false if no spell is worth casting.
bool find_spell_target(const Battle& b, Team team, int* slot_out, Vec2* pos_out) {
    const GameData& data = b.data();
    const PlayerState& me = b.player(team);
    double best_value = 0.0;
    for (int slot = 0; slot < kHandSize; ++slot) {
        if (me.hand[slot].empty()) continue;
        const CardDef& card = data.card(me.hand[slot]);
        if (!is_spell(card) || me.elixir + 1e-9 < card.elixir) continue;
        const SpellDef& s = card.spell;
        for (const auto& center : b.entities()) {
            if (center.team == team || center.is_building()) continue;
            double value = 0.0;
            for (const auto& e : b.entities()) {
                if (e.team == team || e.is_building() || e.card_index < 0) continue;
                if (e.air ? !s.targets.air : !s.targets.ground) continue;
                if (distance(center.pos, e.pos) - e.radius > s.radius) continue;
                const CardDef& unit_card = data.card(data.card_ids[e.card_index]);
                const double removed = std::min<double>(e.hp, static_cast<double>(s.damage) * s.waves);
                value += removed / e.max_hp * unit_card.elixir / unit_card.count;
            }
            if (value >= card.elixir && value > best_value) {
                best_value = value;
                *slot_out = slot;
                *pos_out = center.pos;
            }
        }
    }
    return best_value > 0.0;
}

class NoopBot final : public Bot {
public:
    std::optional<BotMove> decide(const Battle&, Team) override { return std::nullopt; }
};

class RandomBot final : public Bot {
public:
    explicit RandomBot(uint64_t seed) : rng_(seed) {}

    std::optional<BotMove> decide(const Battle& b, Team team) override {
        if (rng_() % 2) return std::nullopt;  // plays on about half of the decision steps
        const ArenaDef& a = b.data().arena;
        const int slot = static_cast<int>(rng_() % kHandSize);
        if (b.player(team).hand[slot].empty()) return std::nullopt;
        if (is_spell(b.data().card(b.player(team).hand[slot]))) {  // somewhere on the enemy half
            const Vec2 pos{uniform(rng_, 1.0, a.width - 1.0), uniform(rng_, a.river_max_y + 1.0, a.height - 1.0)};
            return if_playable(b, team, slot, from_own_frame(a, team, pos));
        }
        const Vec2 pos{uniform(rng_, 1.0, a.width - 1.0), uniform(rng_, 1.0, a.river_min_y - 1.0)};
        return spot_near(b, team, slot, pos, rng_);
    }

private:
    std::mt19937_64 rng_;
};

class HeuristicBot final : public Bot {
public:
    explicit HeuristicBot(uint64_t seed) : rng_(seed) {}

    std::optional<BotMove> decide(const Battle& b, Team team) override {
        const ArenaDef& a = b.data().arena;
        const PlayerState& me = b.player(team);

        int spell_slot = -1;
        Vec2 spell_pos;
        if (find_spell_target(b, team, &spell_slot, &spell_pos)) return if_playable(b, team, spell_slot, spell_pos);

        // The enemy troop deepest into our half is the threat to answer.
        const Entity* threat = nullptr;
        double threat_y = a.river_max_y + 1.0;
        for (const auto& e : b.entities()) {
            if (e.team == team || e.is_building()) continue;
            const double y = to_own_frame(a, team, e.pos).y;
            if (y < threat_y) {
                threat = &e;
                threat_y = y;
            }
        }

        if (threat) {
            const int slot = best_answer(b, team, *threat);
            if (slot < 0) return std::nullopt;
            const CardDef& card = b.data().card(me.hand[slot]);
            const Vec2 t = to_own_frame(a, team, threat->pos);
            // Melee meets the threat a bit ahead of it; ranged units stay further back.
            const double back = card.unit.attack.range >= 3.0 ? 4.0 : 2.0;
            return spot_near(b, team, slot, {t.x, std::min(t.y, a.river_min_y) - back}, rng_);
        }

        if (me.elixir < 8.5) return std::nullopt;
        // Push the lane whose enemy princess tower is weaker (or already gone).
        const Team enemy = opponent(team);
        auto tower_hp = [&](int side) {
            const Entity* t = b.find(b.princess_tower(enemy, side));
            return t ? t->hp : 0;
        };
        const int left = own_left_side(team);
        const double lane_x = tower_hp(left) <= tower_hp(1 - left) ? a.bridge_x.front() : a.bridge_x.back();

        const int slot = static_cast<int>(rng_() % kHandSize);
        if (me.hand[slot].empty()) return std::nullopt;
        const CardDef& card = b.data().card(me.hand[slot]);
        if (is_spell(card)) {  // chip the tower of the chosen lane
            const int side = lane_x < a.width / 2.0 ? left : 1 - left;
            const Entity* tower = b.find(b.princess_tower(enemy, side));
            if (!tower) tower = b.find(b.king_tower(enemy));
            if (!tower) return std::nullopt;
            return if_playable(b, team, slot, tower->pos);
        }
        const bool tank = card.unit.attack.targets.buildings_only || card.unit.hitpoints >= 2000;
        const double y = tank ? a.river_min_y - 1.0 : a.river_min_y - 3.0;
        return spot_near(b, team, slot, {lane_x, y}, rng_);
    }

private:
    // Strongest affordable card in hand that can hit the threat, or -1.
    int best_answer(const Battle& b, Team team, const Entity& threat) {
        const PlayerState& me = b.player(team);
        int best = -1;
        int best_cost = -1;
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (me.hand[slot].empty()) continue;
            const CardDef& card = b.data().card(me.hand[slot]);
            if (is_spell(card)) continue;
            const TargetRules& t = card.unit.attack.targets;
            if (t.buildings_only || (threat.air ? !t.air : !t.ground)) continue;
            if (me.elixir + 1e-9 < card.elixir) continue;
            if (card.elixir > best_cost || (card.elixir == best_cost && rng_() % 2)) {
                best = slot;
                best_cost = card.elixir;
            }
        }
        return best;
    }

    std::mt19937_64 rng_;
};


// Plays like a beatdown player: never spends elixir it doesn't need to, starts a push with its tank
// (the giant) from behind the king tower when it has a full bar, and backs the tank up once it reaches
// the bridge - ranged troops behind it, melee next to it, spells on what defends. On defence it answers
// with the cheapest troop that does the job, placed in the middle to pull the attacker away from the
// towers (and the strongest troop that can hit it right in front of an enemy tank).
class PusherBot final : public Bot {
public:
    explicit PusherBot(uint64_t seed) : rng_(seed) { push_at_ = uniform(rng_, 8.8, 10.0); }

    std::optional<BotMove> decide(const Battle& b, Team team) override {
        const ArenaDef& a = b.data().arena;
        const PlayerState& me = b.player(team);

        int spell_slot = -1;
        Vec2 spell_pos;
        if (find_spell_target(b, team, &spell_slot, &spell_pos)) return if_playable(b, team, spell_slot, spell_pos);

        // Our tank on its way: back it up when it nears the bridge.
        const Entity* tank = nullptr;
        for (const auto& e : b.entities()) {
            if (e.team == team && !e.is_building() && is_tank(b, e)) {
                const double y = to_own_frame(a, team, e.pos).y;
                if (y > 9.0 && (!tank || y > to_own_frame(a, team, tank->pos).y)) tank = &e;
            }
        }
        if (tank) {
            const Vec2 t = to_own_frame(a, team, tank->pos);
            int helpers = 0;
            for (const auto& e : b.entities()) {
                if (e.team == team && !e.is_building() && &e != tank && distance(e.pos, tank->pos) < 5.0) ++helpers;
            }
            if (t.y < a.river_max_y + 2.0 && helpers < 2) {
                const int slot = best_support(b, team);
                if (slot >= 0) {
                    const CardDef& card = b.data().card(me.hand[slot]);
                    const bool ranged = card.unit.attack.range >= 3.0;
                    const double y = std::min(t.y - (ranged ? 2.5 : 1.0), a.river_min_y - 0.5);
                    return spot_near(b, team, slot, {t.x, y}, rng_);
                }
            }
        }

        // Defence: the enemy troop deepest into our half.
        const Entity* threat = nullptr;
        double threat_y = a.river_min_y;
        for (const auto& e : b.entities()) {
            if (e.team == team || e.is_building()) continue;
            const double y = to_own_frame(a, team, e.pos).y;
            if (y < threat_y) {
                threat = &e;
                threat_y = y;
            }
        }
        if (threat) {
            const Vec2 t = to_own_frame(a, team, threat->pos);
            if (is_tank(b, *threat)) {  // their tank: the strongest troop that can hit it, right in front of it
                const int slot = answer(b, team, *threat, true);
                if (slot < 0) return std::nullopt;
                return spot_near(b, team, slot, {t.x, std::max(1.0, t.y - 1.5)}, rng_);
            }
            const int slot = answer(b, team, *threat, false);  // anything else: cheap, pulled to the middle
            if (slot < 0) return std::nullopt;
            const double mid_x = t.x < a.width / 2.0 ? a.width / 2.0 - 0.5 : a.width / 2.0 + 0.5;
            return spot_near(b, team, slot, {mid_x, std::clamp(t.y - 3.0, 6.0, a.river_min_y - 1.5)}, rng_);
        }

        // A counter-push: troops of ours left over from defending, on their way: the tank goes straight to the
        // bridge in front of them (no time to walk it from the back).
        int tank_slot = -1;
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (!me.hand[slot].empty() && is_tank_card(b.data().card(me.hand[slot]))) tank_slot = slot;
        }
        if (tank_slot >= 0 && me.elixir + 1e-9 >= b.data().card(me.hand[tank_slot]).elixir) {
            int left_lane = 0, right_lane = 0;
            for (const auto& e : b.entities()) {
                if (e.team != team || e.is_building() || is_tank(b, e)) continue;
                const Vec2 p = to_own_frame(a, team, e.pos);
                if (p.y >= 7.0 && p.y < a.river_min_y) (p.x < a.width / 2.0 ? left_lane : right_lane) += 1;
            }
            if (std::max(left_lane, right_lane) >= 2) {
                const double x = left_lane >= right_lane ? a.bridge_x.front() : a.bridge_x.back();
                return spot_near(b, team, tank_slot, {x, a.river_min_y - 0.5}, rng_);
            }
        }

        // A push: wait for a full bar, then the tank behind the king, on the weaker enemy tower's lane.
        if (me.elixir < push_at_) return std::nullopt;
        push_at_ = uniform(rng_, 8.8, 10.0);
        const Team enemy = opponent(team);
        auto tower_hp = [&](int side) {
            const Entity* t = b.find(b.princess_tower(enemy, side));
            return t ? t->hp : 0;
        };
        const int left = own_left_side(team);
        const bool go_left =
            tower_hp(left) < tower_hp(1 - left) || (tower_hp(left) == tower_hp(1 - left) && rng_() % 2);
        const double behind_king_x = a.width / 2.0 + (go_left ? -0.5 : 0.5);
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (!me.hand[slot].empty() && is_tank_card(b.data().card(me.hand[slot]))) {
                return spot_near(b, team, slot, {behind_king_x, 0.5}, rng_);
            }
        }
        // No tank in hand: cycle towards it with the cheapest troop, from the back of the push lane.
        int cheapest = -1;
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (me.hand[slot].empty()) continue;
            const CardDef& card = b.data().card(me.hand[slot]);
            if (is_spell(card)) continue;
            if (cheapest < 0 || card.elixir < b.data().card(me.hand[cheapest]).elixir) cheapest = slot;
        }
        if (cheapest < 0) return std::nullopt;
        const double lane_x = go_left ? a.bridge_x.front() : a.bridge_x.back();
        return spot_near(b, team, cheapest, {lane_x, 3.5}, rng_);
    }

private:
    static bool is_tank_card(const CardDef& card) {
        return card.type != CardType::Spell && (card.unit.attack.targets.buildings_only || card.unit.hitpoints >= 2500);
    }
    static bool is_tank(const Battle& b, const Entity& e) {
        return e.card_index >= 0 && is_tank_card(b.data().card(b.data().card_ids[e.card_index]));
    }

    // A troop to back up the tank: ranged first (musketeer, archers), then melee or air.
    int best_support(const Battle& b, Team team) {
        const PlayerState& me = b.player(team);
        int best = -1;
        double best_score = -1.0;
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (me.hand[slot].empty()) continue;
            const CardDef& card = b.data().card(me.hand[slot]);
            if (is_spell(card) || is_tank_card(card) || me.elixir + 1e-9 < card.elixir) continue;
            const double score = (card.unit.attack.range >= 3.0 ? 10.0 : 5.0) + card.elixir;
            if (score > best_score) {
                best_score = score;
                best = slot;
            }
        }
        return best;
    }

    // Against a tank the strongest affordable troop that can hit it; otherwise the cheapest one.
    int answer(const Battle& b, Team team, const Entity& threat, bool strongest) {
        const PlayerState& me = b.player(team);
        int best = -1;
        int best_cost = strongest ? -1 : 99;
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (me.hand[slot].empty()) continue;
            const CardDef& card = b.data().card(me.hand[slot]);
            if (is_spell(card) || is_tank_card(card)) continue;
            const TargetRules& t = card.unit.attack.targets;
            if (t.buildings_only || (threat.air ? !t.air : !t.ground)) continue;
            if (me.elixir + 1e-9 < card.elixir) continue;
            if (strongest ? card.elixir > best_cost : card.elixir < best_cost) {
                best = slot;
                best_cost = card.elixir;
            }
        }
        return best;
    }

    std::mt19937_64 rng_;
    double push_at_ = 9.5;
};

}  // namespace

BotKind bot_kind_from_string(const std::string& name) {
    if (name == "noop") return BotKind::Noop;
    if (name == "random") return BotKind::Random;
    if (name == "heuristic") return BotKind::Heuristic;
    if (name == "pusher") return BotKind::Pusher;
    throw std::invalid_argument("unknown bot kind: " + name + " (expected noop, random, heuristic or pusher)");
}

const char* bot_kind_name(BotKind kind) {
    switch (kind) {
        case BotKind::Noop: return "noop";
        case BotKind::Random: return "random";
        case BotKind::Heuristic: return "heuristic";
        case BotKind::Pusher: return "pusher";
    }
    return "?";
}

void act_simultaneously(Battle& battle, Bot& blue, Bot& red) {
    const auto blue_move = blue.decide(battle, Team::Blue);
    const auto red_move = red.decide(battle, Team::Red);
    if (blue_move) battle.play(Team::Blue, blue_move->hand_slot, blue_move->pos);
    if (red_move) battle.play(Team::Red, red_move->hand_slot, red_move->pos);
}

std::unique_ptr<Bot> make_bot(BotKind kind, uint64_t seed) {
    switch (kind) {
        case BotKind::Noop: return std::make_unique<NoopBot>();
        case BotKind::Random: return std::make_unique<RandomBot>(seed);
        case BotKind::Heuristic: return std::make_unique<HeuristicBot>(seed);
        case BotKind::Pusher: return std::make_unique<PusherBot>(seed);
    }
    throw std::invalid_argument("bad bot kind");
}

}  // namespace cr
