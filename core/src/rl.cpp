#include "crsim/rl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "crsim/frame.h"

namespace cr::rl {
namespace {

// Entity feature layout.
enum EntityFeature {
    kOwn,
    kX,
    kY,
    kHpFrac,
    kHpAbs,
    kAir,
    kBuilding,
    kKing,
    kDeploying,
    kAttacking,
    kRadius,
    kActive,
    kSpell,    // a spell in flight: x/y = where it lands, kDeploying = seconds until impact
    kPending,  // a played card not on the field yet: kDeploying = seconds until it appears
};
static_assert(kPending + 1 == kEntityFeatures);

constexpr double kHpScale = 5000.0;

double hp_frac(const Entity* e) { return e ? static_cast<double>(e->hp) / e->max_hp : 0.0; }

}  // namespace

void check_arena(const GameData& data) {
    if (data.arena.width != kGridW || data.arena.height != kGridH) {
        throw std::runtime_error("rl: arena must be 18x32 tiles to match the deploy grid");
    }
}

void encode(const Battle& b, Team team, const ObsConfig& cfg, const ObsView& out) {
    const GameData& data = b.data();
    const ArenaDef& a = data.arena;
    const RulesDef& r = data.rules;
    const Team enemy = opponent(team);
    const PlayerState& me = b.player(team);
    const PlayerState& them = b.player(enemy);

    // --- global features ---
    float* g = out.global;
    int k = 0;
    const bool overtime = b.phase() == Phase::Overtime;
    const int phase_end = overtime ? r.regulation_ms + r.overtime_ms : r.regulation_ms;
    g[k++] = static_cast<float>(b.time_ms()) / r.regulation_ms;
    g[k++] = overtime ? 1.0f : 0.0f;
    g[k++] = static_cast<float>(std::max(0, phase_end - b.time_ms())) / r.regulation_ms;
    g[k++] = static_cast<float>(b.elixir_multiplier() / 3.0);
    g[k++] = static_cast<float>(me.elixir / r.max_elixir);
    double visible_enemy_elixir = them.elixir;
    for (const auto& p : b.pending()) {  // their blueprints are hidden from us: not spent as far as we know
        if (p.team == enemy && p.unit <= 0) visible_enemy_elixir += data.card(data.card_ids[p.card_index]).elixir;
    }
    visible_enemy_elixir = std::min(visible_enemy_elixir, r.max_elixir);
    g[k++] = cfg.observe_enemy_elixir ? static_cast<float>(visible_enemy_elixir / r.max_elixir) : 0.0f;
    g[k++] = me.crowns / 3.0f;
    g[k++] = them.crowns / 3.0f;
    const int left = own_left_side(team);
    for (Team side : {team, enemy}) {
        const Entity* king = b.find(b.king_tower(side));
        g[k++] = static_cast<float>(hp_frac(king));
        g[k++] = static_cast<float>(hp_frac(b.find(b.princess_tower(side, left))));
        g[k++] = static_cast<float>(hp_frac(b.find(b.princess_tower(side, 1 - left))));
        g[k++] = king && king->active ? 1.0f : 0.0f;
    }
    for (int slot = 0; slot < kHandSize; ++slot) {
        if (me.hand[slot].empty()) {  // waiting for the next card
            g[k++] = 0.0f;
            g[k++] = 0.0f;
            continue;
        }
        const CardDef& card = data.card(me.hand[slot]);
        g[k++] = card.elixir / 10.0f;
        g[k++] = me.elixir + 1e-9 >= card.elixir ? 1.0f : 0.0f;
    }
    g[k++] = data.card(me.queue.front()).elixir / 10.0f;
    if (k != kGlobalFeatures) throw std::logic_error("rl: global feature count mismatch");

    // --- own cycle: hand, then the queue in the order the cards come back ---
    for (int slot = 0; slot < kHandSize; ++slot) {
        out.hand[slot] = me.hand[slot].empty() ? kPadToken : card_token(data.card(me.hand[slot]));
    }
    for (int i = 0; i < kHandTokens - kHandSize; ++i) {
        out.hand[kHandSize + i] = i < static_cast<int>(me.queue.size()) ? card_token(data.card(me.queue[i])) : kPadToken;
    }

    // --- opponent's recent plays we have seen (revealed), newest first ---
    int filled = 0;
    for (auto it = them.plays.rbegin(); it != them.plays.rend() && filled < kHistory; ++it) {
        if (it->reveal_ms < 0) continue;  // still a blueprint only they can see
        out.enemy_history[filled] = kFirstCardToken + it->card_index;
        out.enemy_history_age[filled] = (b.time_ms() - it->reveal_ms) / 10000.0f;
        ++filled;
    }
    for (; filled < kHistory; ++filled) {
        out.enemy_history[filled] = kPadToken;
        out.enemy_history_age[filled] = 0.0f;
    }

    // --- entities: own towers, enemy towers, then troops ordered by how deep they are into our half ---
    struct Item {
        int group;  // 0 own tower, 1 enemy tower, 2 troop or spell
        Vec2 own_pos;
        const Entity* e;
        const Projectile* spell;
        const PendingDeploy* pending;
    };
    std::vector<Item> order;
    order.reserve(b.entities().size() + b.projectiles().size());
    for (const auto& e : b.entities()) {
        const int group = e.is_building() ? (e.team == team ? 0 : 1) : 2;
        order.push_back({group, to_own_frame(a, team, e.pos), &e, nullptr, nullptr});
    }
    for (const auto& p : b.projectiles()) {
        if (p.is_spell()) order.push_back({2, to_own_frame(a, team, p.aim), nullptr, &p, nullptr});
    }
    for (const auto& p : b.pending()) {  // only our own blueprints: the opponent's are hidden from us
        if (p.team == team && p.unit <= 0) order.push_back({2, to_own_frame(a, team, p.pos), nullptr, nullptr, &p});
    }
    std::stable_sort(order.begin(), order.end(), [](const Item& x, const Item& y) {
        if (x.group != y.group) return x.group < y.group;
        if (x.own_pos.y != y.own_pos.y) return x.own_pos.y < y.own_pos.y;
        return x.own_pos.x < y.own_pos.x;
    });

    std::memset(out.entities, 0, sizeof(float) * kMaxEntities * kEntityFeatures);
    std::memset(out.entity_token, 0, sizeof(int32_t) * kMaxEntities);
    std::memset(out.entity_mask, 0, sizeof(uint8_t) * kMaxEntities);
    const int n = std::min<int>(kMaxEntities, static_cast<int>(order.size()));
    for (int i = 0; i < n; ++i) {
        float* f = out.entities + i * kEntityFeatures;
        if (const PendingDeploy* pd = order[i].pending) {
            const CardDef& card = data.card(data.card_ids[pd->card_index]);
            const Vec2 p = order[i].own_pos;
            f[kOwn] = 1.0f;
            f[kX] = static_cast<float>(p.x / a.width);
            f[kY] = static_cast<float>(p.y / a.height);
            f[kAir] = card.type == CardType::Troop && card.unit.air ? 1.0f : 0.0f;
            f[kDeploying] = pd->ms_left / 1000.0f;
            f[kRadius] = static_cast<float>(card.type == CardType::Spell ? card.spell.radius : card.unit.radius);
            f[kSpell] = card.type == CardType::Spell ? 1.0f : 0.0f;
            f[kPending] = 1.0f;
            out.entity_token[i] = card_token(card);
            out.entity_mask[i] = 1;
            continue;
        }
        if (const Projectile* s = order[i].spell) {
            const Vec2 p = order[i].own_pos;
            f[kOwn] = s->team == team ? 1.0f : 0.0f;
            f[kX] = static_cast<float>(p.x / a.width);
            f[kY] = static_cast<float>(p.y / a.height);
            f[kDeploying] = static_cast<float>((s->delay_ms / 1000.0) + distance(s->pos, s->aim) / s->speed);
            f[kRadius] = static_cast<float>(s->splash_radius);
            f[kSpell] = 1.0f;
            out.entity_token[i] = kFirstCardToken + s->card_index;
            out.entity_mask[i] = 1;
            continue;
        }
        const Entity& e = *order[i].e;
        const Vec2 p = to_own_frame(a, team, e.pos);
        f[kOwn] = e.team == team ? 1.0f : 0.0f;
        f[kX] = static_cast<float>(p.x / a.width);
        f[kY] = static_cast<float>(p.y / a.height);
        f[kHpFrac] = static_cast<float>(e.hp) / e.max_hp;
        f[kHpAbs] = static_cast<float>(e.hp / kHpScale);
        f[kAir] = e.air ? 1.0f : 0.0f;
        f[kBuilding] = e.is_building() ? 1.0f : 0.0f;
        f[kKing] = e.kind == EntityKind::KingTower ? 1.0f : 0.0f;
        f[kDeploying] = e.deploy_ms_left / 1000.0f;
        f[kAttacking] = e.locked ? 1.0f : 0.0f;
        f[kRadius] = static_cast<float>(e.radius);
        f[kActive] = e.active ? 1.0f : 0.0f;
        out.entity_token[i] = e.kind == EntityKind::KingTower       ? kKingToken
                              : e.kind == EntityKind::PrincessTower ? kPrincessToken
                                                                    : kFirstCardToken + e.card_index;
        out.entity_mask[i] = 1;
    }

    if (out.action_mask) action_mask(b, team, out.action_mask, cfg.key_spots);
}

Vec2 cell_position(const Battle& b, Team team, int cell) {
    const Vec2 own{cell % kGridW + 0.5, cell / kGridW + 0.5};
    return from_own_frame(b.data().arena, team, own);
}

namespace {

// Troop key spots, own frame, as cells (col, row) on the right side; each is also used mirrored on the left.
constexpr int kTroopSpots[][2] = {
    {14, 4},   // right behind the princess tower
    {17, 0},   // the back corner
    {9, 0},    // behind the king, right of the middle (walks the right lane)
    {11, 0},   // the back row beside the king
    {14, 14},  // the bridge
    {13, 14},  // the bridge, inner side
    {14, 12},  // just behind the bridge
    {14, 9},   // in front of the princess tower
    {16, 7},   // right beside the princess tower, outer side (a little forward of its middle)
    {12, 7},   // right beside the princess tower, inner side
    {11, 8},   // between the king and the princess tower (pulls attackers to the middle)
    {12, 11},  // mid-field
    {9, 13},   // the middle, near the river
    {9, 11},   // the middle
    {9, 9},    // the middle, lower
    {9, 7},    // the middle, in front of the king
    {9, 5},    // right in front of the king
    {12, 5},   // beside the princess tower, inner side, low (between it and the king)
    {10, 10},  // the middle, a little off centre
    {14, 18},  // the pocket of a fallen enemy princess tower
    {14, 20},  // deeper in the pocket
    {17, 20},  // the far corner of the pocket (by the side wall)
};

int cell_at(Vec2 own) {
    const int col = std::clamp(static_cast<int>(std::floor(own.x)), 0, kGridW - 1);
    const int row = std::clamp(static_cast<int>(std::floor(own.y)), 0, kGridH - 1);
    return row * kGridW + col;
}

}  // namespace

void key_spot_cells(const Battle& b, Team team, std::vector<int>& troops, std::vector<int>& spells) {
    const ArenaDef& a = b.data().arena;
    troops.clear();
    spells.clear();
    for (const auto& s : kTroopSpots) {
        troops.push_back(s[1] * kGridW + s[0]);
        troops.push_back(s[1] * kGridW + (kGridW - 1 - s[0]));
    }
    // The enemy troop deepest into our half (or about to cross): meet it there, or a little before it.
    const Entity* deepest = nullptr;
    double deepest_y = a.river_max_y + 3.0;
    // own-frame position where a spell cast now would find it (the blueprint second and the flight from
    // our king tower: a walking troop has moved on towards its target meanwhile), worth (elixir per unit)
    std::vector<std::pair<Vec2, double>> enemy;
    const Vec2 our_king = to_own_frame(a, team, b.find(b.king_tower(team)) ? b.find(b.king_tower(team))->pos
                                                                           : Vec2{a.width / 2.0, 3.0});
    for (const auto& e : b.entities()) {
        if (e.team == team || e.is_building()) continue;
        const Vec2 p = to_own_frame(a, team, e.pos);
        double worth = 1.0;
        if (e.card_index >= 0) {
            const CardDef& card = b.data().card(b.data().card_ids[e.card_index]);
            worth = static_cast<double>(card.elixir) / std::max(1, card.count);
        }
        Vec2 ahead = p;
        if (!e.locked && e.deploy_ms_left <= 0 && e.speed > 0.0) {
            const Entity* t = b.find(e.target);
            const Vec2 goal = t ? to_own_frame(a, team, t->pos) : Vec2{p.x, 0.0};
            const double lead = std::min(3.0, e.speed * (1.0 + distance(our_king, p) / 15.0));
            const double d = distance(p, goal);
            if (d > 1e-6) ahead = p + (goal + p * -1.0) * (std::min(lead, d) / d);
        }
        enemy.push_back({ahead, worth});
        if (p.y < deepest_y) {
            deepest = &e;
            deepest_y = p.y;
        }
    }
    if (deepest) {
        const Vec2 p = to_own_frame(a, team, deepest->pos);
        const double top = a.river_min_y - 0.5;
        troops.push_back(cell_at({p.x, std::min(p.y, top)}));
        troops.push_back(cell_at({p.x, std::min(p.y - 2.5, top)}));
        spells.push_back(cell_at(p));
    }
    // Spells: the enemy princess towers still standing, and the king's only once he is awake (a spell on a
    // sleeping king wakes him up: a gift to the opponent) ...
    const Team enemy_team = team == Team::Blue ? Team::Red : Team::Blue;
    std::vector<Vec2> towers;
    for (EntityId id : {b.king_tower(enemy_team), b.princess_tower(enemy_team, 0), b.princess_tower(enemy_team, 1)}) {
        const Entity* t = b.find(id);
        if (!t || (t->kind == EntityKind::KingTower && !t->active)) continue;
        towers.push_back(to_own_frame(a, team, t->pos));
        spells.push_back(cell_at(towers.back()));
    }
    // ... a tower together with the troops next to it (between them, as far into the tower as still takes
    // the troops' middle) ...
    for (const Vec2& tw : towers) {
        double total = 0.0;
        Vec2 mid{0.0, 0.0};
        for (const auto& [q, v] : enemy) {
            if (distance(q, tw) <= 4.5) {
                total += v;
                mid = mid + q * v;
            }
        }
        if (total > 0.0) spells.push_back(cell_at((mid * (1.0 / total) + tw) * 0.5));
    }
    // ... and the three densest groups of enemy troops (worth within 2.5 tiles, centred on their middle).
    std::vector<std::pair<double, Vec2>> groups;
    for (const auto& [p, w] : enemy) {
        double total = 0.0;
        Vec2 mid{0.0, 0.0};
        for (const auto& [q, v] : enemy) {
            if (distance(p, q) <= 2.5) {
                total += v;
                mid = mid + q * v;
            }
        }
        groups.push_back({total, mid * (1.0 / total)});
    }
    std::sort(groups.begin(), groups.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
    std::vector<Vec2> taken;
    for (const auto& [total, mid] : groups) {
        if (taken.size() >= 3) break;
        if (std::any_of(taken.begin(), taken.end(), [&](Vec2 q) { return distance(q, mid) < 3.0; })) continue;
        taken.push_back(mid);
        spells.push_back(cell_at(mid));
    }
}

void action_mask(const Battle& b, Team team, uint8_t* out, bool key_spots) {
    std::memset(out, 0, kNumActions);
    out[kNoop] = 1;
    if (b.finished()) return;
    const PlayerState& me = b.player(team);
    bool affordable[kHandSize];
    bool any = false;
    for (int slot = 0; slot < kHandSize; ++slot) {
        affordable[slot] = !me.hand[slot].empty() && me.elixir + 1e-9 >= b.data().card(me.hand[slot]).elixir;
        any = any || affordable[slot];
    }
    if (!any) return;
    bool spell[kHandSize];
    for (int slot = 0; slot < kHandSize; ++slot) {
        spell[slot] = !me.hand[slot].empty() && b.data().card(me.hand[slot]).type == CardType::Spell;
    }
    if (key_spots) {
        std::vector<int> troops, spells;
        key_spot_cells(b, team, troops, spells);
        for (int cell : troops) {
            if (!b.can_deploy_at(team, cell_position(b, team, cell))) continue;
            for (int slot = 0; slot < kHandSize; ++slot) {
                if (affordable[slot] && !spell[slot]) out[1 + slot * kCells + cell] = 1;
            }
        }
        for (int cell : spells) {
            if (!b.can_cast_at(cell_position(b, team, cell))) continue;
            for (int slot = 0; slot < kHandSize; ++slot) {
                if (affordable[slot] && spell[slot]) out[1 + slot * kCells + cell] = 1;
            }
        }
        return;
    }
    for (int cell = 0; cell < kCells; ++cell) {
        const Vec2 pos = cell_position(b, team, cell);
        const bool troop_ok = b.can_deploy_at(team, pos);
        const bool spell_ok = b.can_cast_at(pos);
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (affordable[slot] && (spell[slot] ? spell_ok : troop_ok)) out[1 + slot * kCells + cell] = 1;
        }
    }
}

int encode_action(const Battle& b, Team team, int slot, Vec2 pos) {
    const Vec2 own = to_own_frame(b.data().arena, team, pos);
    const int col = std::clamp(static_cast<int>(std::floor(own.x)), 0, kGridW - 1);
    const int row = std::clamp(static_cast<int>(std::floor(own.y)), 0, kGridH - 1);
    return 1 + slot * kCells + row * kGridW + col;
}

bool apply_action(Battle& b, Team team, int action) {
    if (action <= kNoop || action >= kNumActions) return false;
    const int slot = (action - 1) / kCells;
    const int cell = (action - 1) % kCells;
    return b.play(team, slot, cell_position(b, team, cell));
}

}  // namespace cr::rl
