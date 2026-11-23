#include "crsim/rl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
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
    // What it is busy with (appended last: nets trained before ignore them): the way to its current target in
    // tiles / 10 (troops chase or march on it, towers shoot it), whether that target is a troop (a fight) rather
    // than a building, and seconds until its next hit while it is attacking.
    kTargetDx,
    kTargetDy,
    kTargetTroop,
    kNextHit,
};
static_assert(kNextHit + 1 == kEntityFeatures);

constexpr double kHpScale = 5000.0;

double hp_frac(const Entity* e) { return e ? static_cast<double>(e->hp) / e->max_hp : 0.0; }

// --- LiveView: per play and per unit choices from a hash, so each is made once and kept ---
uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
double draw(const LiveView& lv, uint64_t what, uint64_t which) {  // uniform in [0, 1)
    return static_cast<double>(mix64(lv.seed ^ mix64((which << 4) | what)) >> 11) * (1.0 / 9007199254740992.0);
}
enum Draw : uint64_t { kMissed = 1, kLate = 2, kPhantom = 3, kPhantomCard = 4, kHpOff = 5, kHpU1 = 6, kHpU2 = 7 };
bool missed(const LiveView& lv, size_t play) { return draw(lv, kMissed, play) < lv.miss_play; }
int seen_from(const LiveView& lv, const CardPlay& p, size_t play) {  // battle ms the observer first sees the play
    return p.reveal_ms < 0 ? -1 : p.reveal_ms + static_cast<int>(draw(lv, kLate, play) * lv.late_ms);
}
// The opponent's play `play` as the observer knows it: nothing (missed, or not seen yet) or seen.
bool play_seen(const LiveView& lv, const PlayerState& them, size_t play, int now) {
    if (play >= them.plays.size() || missed(lv, play)) return false;
    const int from = seen_from(lv, them.plays[play], play);
    return from >= 0 && now >= from;
}
// A troop's HP fraction as seen: off by a normal share of its max for a few units (per unit, all its life).
double seen_hp_frac(const LiveView& lv, const Entity& e) {
    const double frac = static_cast<double>(e.hp) / e.max_hp;
    if (lv.hp_off <= 0.0 || draw(lv, kHpOff, e.id) >= lv.hp_off) return frac;
    const double u1 = std::max(1e-12, draw(lv, kHpU1, e.id)), u2 = draw(lv, kHpU2, e.id);
    const double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    return std::clamp(frac + lv.hp_off_sd * z, 0.02, 1.0);
}
// The observer's count of the opponent's elixir now: filling up as theirs does, each play it sees taken off when
// it is seen, a phantom play's elixir taken off now and then; held between 0 and the max like the twin's count.
double live_enemy_elixir(LiveView& lv, const Battle& b, const PlayerState& them, double exact) {
    const GameData& data = b.data();
    const RulesDef& r = data.rules;
    const int now = b.time_ms();
    if (!lv.started) {  // (a fresh match, or one that starts from a recorded position: its count is right)
        lv.started = true;
        lv.count = std::clamp(exact, 0.0, r.max_elixir);
        lv.last_ms = now;
        lv.judged = 0;
        while (lv.judged < them.plays.size() && them.plays[lv.judged].reveal_ms >= 0) ++lv.judged;
    }
    lv.count = std::min(r.max_elixir, lv.count + static_cast<double>(std::max(0, now - lv.last_ms)) *
                                                     static_cast<double>(b.elixir_multiplier()) / r.elixir_regen_ms);
    lv.last_ms = now;
    while (lv.judged < them.plays.size()) {
        const CardPlay& p = them.plays[lv.judged];
        // Paid for once it shows (the twin's count takes a troop off the moment it is first seen,
        // LiveMatch.enemy_elixir_seen, before its play is made out): not late like the troops themselves.
        if (p.reveal_ms < 0) break;  // (plays come to light in the order they were made)
        if (!missed(lv, lv.judged)) lv.count -= data.card(data.card_ids[p.card_index]).elixir;
        if (draw(lv, kPhantom, lv.judged) < lv.phantom_play) {  // one of their cards, played as far as it knows
            std::vector<const std::string*> deck;
            for (const auto& c : them.hand) {
                if (!c.empty()) deck.push_back(&c);
            }
            for (const auto& c : them.queue) deck.push_back(&c);
            if (!deck.empty()) {
                const size_t k = std::min(deck.size() - 1, static_cast<size_t>(draw(lv, kPhantomCard, lv.judged) * deck.size()));
                lv.count -= data.card(*deck[k]).elixir;
            }
        }
        lv.count = std::max(0.0, lv.count);
        ++lv.judged;
    }
    return lv.count;
}

}  // namespace

void encode_enemy_deck(const Battle& b, Team team, const ObsConfig& cfg, int32_t* tokens, float* info) {
    const PlayerState& them = b.player(opponent(team));
    const LiveView* live = cfg.live && cfg.live->enabled() ? cfg.live : nullptr;
    const int now = b.time_ms();
    // The plays we have seen, in the order they were made.
    std::vector<int> seen, seen_ms;
    for (size_t i = 0; i < them.plays.size(); ++i) {
        const CardPlay& p = them.plays[i];
        if (p.reveal_ms < 0) continue;  // still a blueprint only they can see
        if (live && !play_seen(*live, them, i, now)) continue;  // missed, or not made out yet
        seen.push_back(p.card_index);
        seen_ms.push_back(p.reveal_ms);
    }
    auto has = [](const std::vector<int>& v, int c) { return std::find(v.begin(), v.end(), c) != v.end(); };
    std::vector<int> played;  // distinct cards seen, in the order first seen
    for (int c : seen) {
        if (!has(played, c)) played.push_back(c);
    }
    // The pool decks they may hold: those with the most of the cards seen (all of them, unless one was misread).
    std::vector<const std::vector<int>*> fits;
    if (cfg.deck_pool) {
        int best = -1;
        for (const auto& deck : *cfg.deck_pool) {
            int n = 0;
            for (int c : played) n += has(deck, c);
            if (n > best) {
                best = n;
                fits.clear();
            }
            if (n == best) fits.push_back(&deck);
        }
    }
    auto in_deck = [&](int c) {
        if (has(played, c)) return 1.0;
        if (fits.empty()) return 0.0;
        int n = 0;
        for (const auto* d : fits) n += has(*d, c);
        return static_cast<double>(n) / static_cast<double>(fits.size());
    };
    // The cards to show: the ones seen, then the fitting decks' others, likeliest first.
    std::vector<int> cards = played;
    std::vector<std::pair<double, int>> others;
    for (const auto* d : fits) {
        for (int c : *d) {
            if (!has(cards, c) && std::none_of(others.begin(), others.end(), [c](const auto& o) { return o.second == c; })) {
                others.push_back({in_deck(c), c});
            }
        }
    }
    std::sort(others.begin(), others.end(), [](const auto& x, const auto& y) {
        return x.first != y.first ? x.first > y.first : x.second < y.second;
    });
    for (const auto& o : others) cards.push_back(o.second);
    // The cycle: a played card goes to the back of the queue and the front one comes into the hand, so their last
    // 4 plays are the queue, oldest at the front; before 4 plays its front holds cards never seen.
    const int recent_from = std::max(0, static_cast<int>(seen.size()) - kHandSize);
    std::vector<int> recent(seen.begin() + recent_from, seen.end());
    std::vector<int> recent_distinct;
    for (int c : recent) {
        if (!has(recent_distinct, c)) recent_distinct.push_back(c);
    }
    const double hand_share = static_cast<double>(kHandSize) / (kDeckSize - static_cast<int>(recent_distinct.size()));
    const float known = fits.empty() ? 0.0f : 1.0f / static_cast<float>(fits.size());
    for (int s = 0; s < kEnemyDeck; ++s) {
        float* f = info + s * kEnemyDeckFeatures;
        std::fill(f, f + kEnemyDeckFeatures, 0.0f);
        if (s >= static_cast<int>(cards.size())) {
            tokens[s] = kPadToken;
            continue;
        }
        const int c = cards[s];
        tokens[s] = kFirstCardToken + c;
        const double p = in_deck(c);
        f[kDeckPlayed] = has(played, c) ? 1.0f : 0.0f;
        f[kDeckIn] = static_cast<float>(p);
        int place = -1;  // its latest place among the recent plays
        for (int j = 0; j < static_cast<int>(recent.size()); ++j) {
            if (recent[j] == c) place = j;
        }
        if (place >= 0) {
            f[kDeckQueue] = static_cast<float>(kHandSize - static_cast<int>(recent.size()) + place + 1) / kHandSize;
        } else {
            f[kDeckInHand] = static_cast<float>(p * std::min(1.0, hand_share));
        }
        f[kDeckKnown] = known;
        for (int j = static_cast<int>(seen.size()) - 1; j >= 0; --j) {
            if (seen[j] == c) {
                f[kDeckSincePlay] = static_cast<float>(now - seen_ms[j]) / 10000.0f;
                break;
            }
        }
    }
}

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
    LiveView* live = cfg.live && cfg.live->enabled() ? cfg.live : nullptr;
    if (live) visible_enemy_elixir = live_enemy_elixir(*live, b, them, visible_enemy_elixir);
    if (cfg.rng && (cfg.enemy_elixir_offset != 0.0 || cfg.enemy_elixir_jitter > 0.0)) {
        visible_enemy_elixir += cfg.enemy_elixir_offset +
                                std::normal_distribution<double>(0.0, cfg.enemy_elixir_jitter)(*cfg.rng);
    }
    visible_enemy_elixir = std::clamp(visible_enemy_elixir, 0.0, r.max_elixir);
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
        if (live && !play_seen(*live, them, static_cast<size_t>(std::distance(them.plays.begin(), it.base()) - 1),
                               b.time_ms())) {
            continue;  // missed, or not made out yet
        }
        out.enemy_history[filled] = kFirstCardToken + it->card_index;
        out.enemy_history_age[filled] = (b.time_ms() - it->reveal_ms) / 10000.0f;
        ++filled;
    }
    for (; filled < kHistory; ++filled) {
        out.enemy_history[filled] = kPadToken;
        out.enemy_history_age[filled] = 0.0f;
    }

    // --- the opponent's deck as far as we know it: the cards seen, the pool decks that fit, their cycle ---
    if (out.enemy_deck && out.enemy_deck_info) encode_enemy_deck(b, team, cfg, out.enemy_deck, out.enemy_deck_info);

    // --- entities: own towers, enemy towers, then troops ordered by how deep they are into our half ---
    struct Item {
        int group;  // 0 own tower, 1 enemy tower, 2 troop or spell
        Vec2 own_pos;
        const Entity* e;
        const Projectile* spell;
        const PendingDeploy* pending;
        const AreaEffect* area = nullptr;   // a spell on the ground (Poison, Tornado, ...)
        const Roller* roller = nullptr;     // a rolling one (The Log)
    };
    std::vector<Item> order;
    order.reserve(b.entities().size() + b.projectiles().size());
    for (const auto& e : b.entities()) {
        const int group = e.is_tower() ? (e.team == team ? 0 : 1) : 2;
        order.push_back({group, to_own_frame(a, team, e.pos), &e, nullptr, nullptr});
    }
    for (const auto& p : b.projectiles()) {
        if (p.is_spell()) order.push_back({2, to_own_frame(a, team, p.aim), nullptr, &p, nullptr});
    }
    for (const auto& ar : b.areas()) {
        if (ar.card_index >= 0) order.push_back({2, to_own_frame(a, team, ar.pos), nullptr, nullptr, nullptr, &ar, nullptr});
    }
    for (const auto& roll : b.rollers()) {
        order.push_back({2, to_own_frame(a, team, roll.pos), nullptr, nullptr, nullptr, nullptr, &roll});
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
        if (order[i].area || order[i].roller) {  // a spell on the ground: where, how big, how long it has left
            const Vec2 p = order[i].own_pos;
            const AreaEffect* ar = order[i].area;
            const Roller* roll = order[i].roller;
            f[kOwn] = (ar ? ar->team : roll->team) == team ? 1.0f : 0.0f;
            f[kX] = static_cast<float>(p.x / a.width);
            f[kY] = static_cast<float>(p.y / a.height);
            f[kDeploying] = ar ? (ar->delay_ms + ar->ms_left) / 1000.0f : static_cast<float>(roll->left / std::max(0.1, roll->speed));
            f[kRadius] = static_cast<float>(ar ? ar->radius : roll->half_width);
            f[kSpell] = 1.0f;
            out.entity_token[i] = kFirstCardToken + (ar ? ar->card_index : roll->card_index);
            out.entity_mask[i] = 1;
            continue;
        }
        const Entity& e = *order[i].e;
        Vec2 p = to_own_frame(a, team, e.pos);
        if (live && e.team != team && !e.is_tower() && e.play >= 0 &&
            !play_seen(*live, them, static_cast<size_t>(e.play), b.time_ms())) {
            continue;  // a troop of a play it missed, or has not made out yet: the slot stays empty
        }
        if (cfg.rng && e.team != team && !e.is_tower()) {
            std::uniform_real_distribution<double> u(0.0, 1.0);
            if (cfg.miss_enemy > 0.0 && u(*cfg.rng) < cfg.miss_enemy) continue;  // not seen: the slot stays empty
            if (cfg.jitter > 0.0) {
                p.x = std::clamp(p.x + (2.0 * u(*cfg.rng) - 1.0) * cfg.jitter, 0.0, static_cast<double>(a.width));
                p.y = std::clamp(p.y + (2.0 * u(*cfg.rng) - 1.0) * cfg.jitter, 0.0, static_cast<double>(a.height));
            }
        }
        f[kOwn] = e.team == team ? 1.0f : 0.0f;
        f[kX] = static_cast<float>(p.x / a.width);
        f[kY] = static_cast<float>(p.y / a.height);
        f[kHpFrac] = static_cast<float>(e.hp) / e.max_hp;
        f[kHpAbs] = static_cast<float>(e.hp / kHpScale);
        if (live && !e.is_tower()) {  // (the towers' HP is read from their digits: exact)
            const double seen = seen_hp_frac(*live, e);
            f[kHpFrac] = static_cast<float>(seen);
            f[kHpAbs] = static_cast<float>(seen * e.max_hp / kHpScale);
        }
        f[kAir] = e.air ? 1.0f : 0.0f;
        f[kBuilding] = e.is_building() ? 1.0f : 0.0f;
        f[kKing] = e.kind == EntityKind::KingTower ? 1.0f : 0.0f;
        f[kDeploying] = e.deploy_ms_left / 1000.0f;
        f[kAttacking] = e.locked ? 1.0f : 0.0f;
        f[kRadius] = static_cast<float>(e.radius);
        f[kActive] = e.active ? 1.0f : 0.0f;
        if (const Entity* t = b.find(e.target); t && t->alive()) {
            const Vec2 d = to_own_frame(a, team, t->pos) - p;
            f[kTargetDx] = static_cast<float>(std::clamp(d.x / 10.0, -3.0, 3.0));
            f[kTargetDy] = static_cast<float>(std::clamp(d.y / 10.0, -3.0, 3.0));
            f[kTargetTroop] = t->is_building() ? 0.0f : 1.0f;
        }
        if (e.locked) f[kNextHit] = std::max(0, e.attack_cd_ms) / 1000.0f;
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

// Troop spots next to the troops on the field (key_spot_cells with tactical).
void tactical_troop_spots(const Battle& b, Team team, std::vector<int>& troops) {
    const ArenaDef& a = b.data().arena;
    const double top = a.river_min_y - 0.5;  // the last row on our side of the river (own frame)
    auto add = [&](Vec2 q) {
        q.x = std::clamp(q.x, 0.0, a.width - 1e-3);
        troops.push_back(cell_at({q.x, std::clamp(q.y, 0.0, top)}));
    };
    // Right behind our two foremost walking ground troops: a push follows its tank.
    std::vector<std::pair<double, const Entity*>> ours, theirs;
    for (const auto& e : b.entities()) {
        if (e.is_building() || !e.alive()) continue;
        const Vec2 p = to_own_frame(a, team, e.pos);
        if (e.team == team) {
            if (!e.air && !e.locked && e.deploy_ms_left <= 0 && p.y < a.river_max_y) ours.push_back({-p.y, &e});
        } else if (p.y < a.river_max_y + 3.0) {
            theirs.push_back({p.y, &e});
        }
    }
    std::sort(ours.begin(), ours.end());
    for (size_t i = 0; i < ours.size() && i < 2; ++i) {
        const Vec2 p = to_own_frame(a, team, ours[i].second->pos);
        add({p.x, p.y - ours[i].second->radius - 0.9});
    }
    // The two enemy troops deepest in our half (a second one only if apart from the first).
    std::sort(theirs.begin(), theirs.end());
    std::vector<Vec2> done;
    const Vec2 middle{a.width / 2.0, 7.5};  // in front of our king, between the princess towers
    for (const auto& [y, e] : theirs) {
        if (done.size() >= 2) break;
        const Vec2 p = to_own_frame(a, team, e->pos);
        if (std::any_of(done.begin(), done.end(), [&](Vec2 q) { return distance(p, q) < 1.5; })) continue;
        if (!done.empty()) {  // (the first one has these two already)
            add(p);
            add({p.x, p.y - 2.5});
        }
        done.push_back(p);
        const double side = e->radius + 1.0;
        add({p.x - side, p.y});
        add({p.x + side, p.y});
        add({p.x, p.y + side});  // at its back
        const double d = distance(p, middle);
        if (d > 1e-6) add(p + (middle + p * -1.0) * (std::min(3.5, d) / d));
    }
    // A fallen enemy princess tower opens its pocket: its deepest corner next to the middle (a fallen left tower:
    // the pocket's right far corner, and the other way about).
    const Team enemy_team = team == Team::Blue ? Team::Red : Team::Blue;
    for (int side = 0; side < 2; ++side) {  // (princess towers by side of the arena: 0 = left half)
        if (b.find(b.princess_tower(enemy_team, side)) != nullptr) continue;
        const double deep = a.river_max_y + a.pocket_depth - 0.5;  // own frame
        const Vec2 corner{side == 0 ? a.width / 2.0 - 0.5 : a.width / 2.0 + 0.5,
                          team == Team::Blue ? deep : a.height - deep};
        troops.push_back(cell_at(to_own_frame(a, team, corner)));
    }
}

}  // namespace

void key_spot_cells(const Battle& b, Team team, std::vector<int>& troops, std::vector<int>& spells, bool tactical) {
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
    if (tactical) tactical_troop_spots(b, team, troops);
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

void action_mask(const Battle& b, Team team, uint8_t* out, int key_spots) {
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
    // Where each card may go: 0 our half (troops, buildings), 1 anywhere (spells), 2 anywhere but on a building
    // (Goblin Drill), 3 our half, buildings too (The Log).
    int where[kHandSize];
    bool any_dig = false;
    for (int slot = 0; slot < kHandSize; ++slot) {
        where[slot] = 0;
        if (me.hand[slot].empty()) continue;
        const CardDef& card = b.data().card(me.hand[slot]);
        where[slot] = card.anywhere ? 2 : card.own_half ? 3 : card.type == CardType::Spell ? 1 : 0;
        any_dig = any_dig || where[slot] == 2;
    }
    auto allowed = [&](Vec2 pos, bool troop_ok, bool spell_ok, int slot) {
        switch (where[slot]) {
            case 0: return troop_ok;
            case 1: return spell_ok;
            case 2: return spell_ok && b.can_dig_at(pos);
            default: return b.can_roll_from(team, pos);
        }
    };
    if (key_spots > 0) {
        std::vector<int> troops, spells;
        key_spot_cells(b, team, troops, spells, key_spots >= 2);
        for (int cell : troops) {
            if (!b.can_deploy_at(team, cell_position(b, team, cell))) continue;
            for (int slot = 0; slot < kHandSize; ++slot) {
                if (affordable[slot] && (where[slot] == 0 || where[slot] == 3)) out[1 + slot * kCells + cell] = 1;
            }
        }
        for (int cell : spells) {
            const Vec2 pos = cell_position(b, team, cell);
            if (!b.can_cast_at(pos)) continue;
            for (int slot = 0; slot < kHandSize; ++slot) {
                if (affordable[slot] && where[slot] != 0 && allowed(pos, false, true, slot)) out[1 + slot * kCells + cell] = 1;
            }
        }
        return;
    }
    for (int cell = 0; cell < kCells; ++cell) {
        const Vec2 pos = cell_position(b, team, cell);
        const bool troop_ok = b.can_deploy_at(team, pos);
        const bool spell_ok = b.can_cast_at(pos);
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (affordable[slot] && allowed(pos, troop_ok, spell_ok, slot)) out[1 + slot * kCells + cell] = 1;
        }
    }
    (void)any_dig;
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
