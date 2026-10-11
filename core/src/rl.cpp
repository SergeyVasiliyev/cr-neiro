#include "crsim/rl.h"

#include <algorithm>
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
        const CardDef& card = data.card(me.hand[slot]);
        g[k++] = card.elixir / 10.0f;
        g[k++] = me.elixir + 1e-9 >= card.elixir ? 1.0f : 0.0f;
    }
    g[k++] = data.card(me.queue.front()).elixir / 10.0f;
    if (k != kGlobalFeatures) throw std::logic_error("rl: global feature count mismatch");

    // --- own cycle: hand, then the queue in the order the cards come back ---
    for (int slot = 0; slot < kHandSize; ++slot) out.hand[slot] = card_token(data.card(me.hand[slot]));
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

    if (out.action_mask) action_mask(b, team, out.action_mask);
}

Vec2 cell_position(const Battle& b, Team team, int cell) {
    const Vec2 own{cell % kGridW + 0.5, cell / kGridW + 0.5};
    return from_own_frame(b.data().arena, team, own);
}

void action_mask(const Battle& b, Team team, uint8_t* out) {
    std::memset(out, 0, kNumActions);
    out[kNoop] = 1;
    if (b.finished()) return;
    const PlayerState& me = b.player(team);
    bool affordable[kHandSize];
    bool any = false;
    for (int slot = 0; slot < kHandSize; ++slot) {
        affordable[slot] = me.elixir + 1e-9 >= b.data().card(me.hand[slot]).elixir;
        any = any || affordable[slot];
    }
    if (!any) return;
    bool spell[kHandSize];
    for (int slot = 0; slot < kHandSize; ++slot) spell[slot] = b.data().card(me.hand[slot]).type == CardType::Spell;
    for (int cell = 0; cell < kCells; ++cell) {
        const Vec2 pos = cell_position(b, team, cell);
        const bool troop_ok = b.can_deploy_at(team, pos);
        const bool spell_ok = b.can_cast_at(pos);
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (affordable[slot] && (spell[slot] ? spell_ok : troop_ok)) out[1 + slot * kCells + cell] = 1;
        }
    }
}

bool apply_action(Battle& b, Team team, int action) {
    if (action <= kNoop || action >= kNumActions) return false;
    const int slot = (action - 1) / kCells;
    const int cell = (action - 1) % kCells;
    return b.play(team, slot, cell_position(b, team, cell));
}

}  // namespace cr::rl
