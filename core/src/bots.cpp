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

}  // namespace

BotKind bot_kind_from_string(const std::string& name) {
    if (name == "noop") return BotKind::Noop;
    if (name == "random") return BotKind::Random;
    if (name == "heuristic") return BotKind::Heuristic;
    throw std::invalid_argument("unknown bot kind: " + name + " (expected noop, random or heuristic)");
}

const char* bot_kind_name(BotKind kind) {
    switch (kind) {
        case BotKind::Noop: return "noop";
        case BotKind::Random: return "random";
        case BotKind::Heuristic: return "heuristic";
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
    }
    throw std::invalid_argument("bad bot kind");
}

}  // namespace cr
