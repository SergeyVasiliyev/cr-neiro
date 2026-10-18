#include "crsim/battle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;
// A locked attacker keeps hitting a target that drifts slightly out of range.
constexpr double kRangeSlack = 0.3;
// Elixir is accumulated in doubles; don't reject a play because of rounding.
constexpr double kElixirEps = 1e-9;
// Fraction of an overlap between two troops resolved per tick (1 = instant, jittery).
constexpr double kCollisionStiffness = 0.5;
constexpr double kGoldenAngle = 2.39996322972865332;
// How much closer (tiles) the other bridge must be before a troop on its own half switches lanes.
constexpr double kLaneHysteresis = 1.0;
// How far off a bridge's centre line a troop may cross (tiles; see next_waypoint).
constexpr double kBridgeLine = 0.45;
// Same-side troops nearly in line (|sideways part of the push| below this) get this much sideways push too,
// when the one in front has stopped to fight.
constexpr double kFollowSlipBelow = 0.15;
constexpr double kFollowSlip = 0.35;
// Following a slower troop of one's side (follow_step): it is in the way when it is ahead, at most this far
// past touching, and off the line by less than this share of the two radii.
constexpr double kFollowGap = 0.3;
constexpr double kFollowWidth = 0.95;

double edge_gap(const Entity& a, const Entity& b) {
    return distance(a.pos, b.pos) - a.radius - b.radius;
}

bool matches_targets(const TargetRules& rules, const Entity& t) {
    if (rules.buildings_only) return t.is_building();
    return t.air ? rules.air : rules.ground;
}

Vec2 move_towards(Vec2 from, Vec2 to, double max_step) {
    const Vec2 d = to - from;
    const double len = d.length();
    if (len <= max_step || len == 0.0) return to;
    return from + d * (max_step / len);
}

}  // namespace

const char* event_name(EventType type) {
    switch (type) {
        case EventType::CardPlayed: return "card_played";
        case EventType::Death: return "death";
        case EventType::Damage: return "damage";
        case EventType::TowerDestroyed: return "tower_destroyed";
        case EventType::KingActivated: return "king_activated";
        case EventType::OvertimeStarted: return "overtime_started";
        case EventType::MatchEnded: return "match_ended";
    }
    return "?";
}

Battle::Battle(const GameData& data, const Deck& blue, const Deck& red, uint64_t seed)
    : data_(data), rng_(seed) {
    const Deck* decks[2] = {&blue, &red};
    for (Team team : {Team::Blue, Team::Red}) {
        PlayerState& p = players_[idx(team)];
        Deck order = *decks[idx(team)];
        for (const auto& id : order) data_.card(id);  // throws on unknown ids
        // Hand-rolled Fisher-Yates so the deal does not depend on the standard library.
        for (size_t i = order.size() - 1; i > 0; --i) std::swap(order[i], order[rng_() % (i + 1)]);
        for (int i = 0; i < kHandSize; ++i) p.hand[i] = order[i];
        for (int i = kHandSize; i < kDeckSize; ++i) p.queue.push_back(order[i]);
        p.elixir = data_.rules.start_elixir;
        spawn_towers(team);
    }
}

Vec2 Battle::to_absolute(Team team, Vec2 blue_pos) const {
    if (team == Team::Blue) return blue_pos;
    return {blue_pos.x, data_.arena.height - blue_pos.y};
}

void Battle::spawn_towers(Team team) {
    const ArenaDef& a = data_.arena;
    auto make = [&](EntityKind kind, const TowerDef& def, Vec2 blue_pos, const char* name) {
        Entity e;
        e.id = next_id_++;
        e.kind = kind;
        e.team = team;
        e.def_id = name;
        e.pos = to_absolute(team, blue_pos);
        e.radius = def.radius;
        e.footprint = def.footprint;
        e.hp = e.max_hp = def.hitpoints;
        e.attack = def.attack;
        e.sight_range = def.attack.range;
        e.active = kind != EntityKind::KingTower;
        entities_.push_back(e);
        return e.id;
    };
    TowerIds& ids = towers_[idx(team)];
    ids.king = make(EntityKind::KingTower, data_.king_tower, a.king_pos, "king_tower");
    for (int i = 0; i < 2; ++i) {
        ids.princess[i] =
            make(EntityKind::PrincessTower, data_.princess_tower, a.princess_pos[i], "princess_tower");
    }
}

void Battle::spawn_unit(Team team, const CardDef& card, Vec2 pos, int i) {
    const ArenaDef& a = data_.arena;
    // Small groups stand on a circle, big ones (Skeleton Army) fill a disc.
    Vec2 offset;
    if (card.count > 1 && card.count <= 6) {
        const double angle = 2.0 * kPi * i / card.count;
        offset = {std::cos(angle) * card.spawn_spread, std::sin(angle) * card.spawn_spread};
    } else if (card.count > 6) {
        const double r = card.spawn_spread * std::sqrt((i + 0.5) / card.count);
        offset = {std::cos(i * kGoldenAngle) * r, std::sin(i * kGoldenAngle) * r};
    }
    Entity e;
    e.id = next_id_++;
    e.kind = EntityKind::Troop;
    e.team = team;
    e.def_id = card.id;
    e.card_index = card.index;
    e.pos = pos + offset;
    e.pos.x = std::clamp(e.pos.x, 0.0, a.width);
    e.pos.y = std::clamp(e.pos.y, 0.0, a.height);
    e.radius = card.unit.radius;
    e.mass = card.unit.mass;
    e.ignore_pushback = card.unit.ignore_pushback;
    e.air = card.unit.air;
    e.hp = e.max_hp = card.unit.hitpoints;
    e.speed = card.unit.speed;
    e.sight_range = card.unit.sight_range;
    e.attack = card.unit.attack;
    e.deploy_ms_left = card.unit.deploy_time_ms;
    e.first_attack_delay_ms = card.unit.first_attack_delay_ms;
    if (!e.air) keep_out_of_river(e);
    entities_.push_back(e);
}

void Battle::materialize_pending(int dt) {
    for (auto& p : pending_) {
        p.ms_left -= dt;
        if (p.ms_left > 0) continue;
        const CardDef& card = data_.card(data_.card_ids[p.card_index]);
        // ms_left below 0: the play was made longer ago than the blueprint lasts (force_play with ago_ms),
        // and the card has been on the field that long already.
        const int overdue = -p.ms_left;
        if (card.type == CardType::Spell) {
            cast_spell(p.team, card, p.pos, overdue);
        } else {
            spawn_unit(p.team, card, p.pos, p.unit);
            Entity& e = entities_.back();
            e.deploy_ms_left = std::max(0, e.deploy_ms_left - overdue);
            e.play = p.play_index;
            e.unit = p.unit;
            // Put down on top of a troop of ours from an earlier play, it lands in front of it (as in the
            // game: the newer one ends up ahead), not pushed back by the collision.
            const double forward = p.team == Team::Blue ? 1.0 : -1.0;
            for (const auto& o : entities_) {
                if (&o == &e || o.is_building() || o.team != e.team || o.air != e.air || o.play == e.play) continue;
                if (distance(o.pos, e.pos) < (o.radius + e.radius) * 0.9) {
                    e.pos.y = o.pos.y + forward * (o.radius + e.radius) * 0.9;
                    if (!e.air) keep_out_of_river(e);
                }
            }
        }
        CardPlay& play = players_[idx(p.team)].plays[p.play_index];
        if (play.reveal_ms < 0) play.reveal_ms = time_ms_;
    }
    std::erase_if(pending_, [](const PendingDeploy& p) { return p.ms_left <= 0; });
}

void Battle::refill_hands(int dt) {
    for (auto& p : players_) {
        for (int slot = 0; slot < kHandSize; ++slot) {
            if (!p.hand[slot].empty()) continue;
            p.refill_ms_left[slot] = std::max(0, p.refill_ms_left[slot] - dt);
            if (p.refill_ms_left[slot] == 0 && !p.queue.empty()) {
                p.hand[slot] = p.queue.front();
                p.queue.pop_front();
            }
        }
    }
}

int Battle::card_refill_ms() const {
    const double m = elixir_multiplier();
    return data_.rules.card_refill_ms[m >= 3.0 ? 2 : (m >= 2.0 ? 1 : 0)];
}

int Battle::deploy_delay_ms() const {
    const double m = elixir_multiplier();
    return data_.rules.deploy_delay_ms[m >= 3.0 ? 2 : (m >= 2.0 ? 1 : 0)];
}

Entity* Battle::find_mut(EntityId id) {
    if (id == kNoEntity) return nullptr;
    for (auto& e : entities_) {
        if (e.id == id) return &e;
    }
    return nullptr;
}

const Entity* Battle::find(EntityId id) const {
    return const_cast<Battle*>(this)->find_mut(id);
}

bool Battle::can_deploy_at(Team team, Vec2 pos) const {
    const ArenaDef& a = data_.arena;
    if (pos.x < 0.5 || pos.x > a.width - 0.5 || pos.y < 0.5 || pos.y > a.height - 0.5) return false;

    // Work in the player's own frame: own side is y < river_min_y.
    const double y = team == Team::Blue ? pos.y : a.height - pos.y;
    bool allowed = y < a.river_min_y;
    if (!allowed && y > a.river_max_y && y < a.river_max_y + a.pocket_depth) {
        // Pocket: the enemy princess tower on this side of the arena is down.
        const int side = pos.x < a.width / 2.0 ? 0 : 1;
        allowed = find(towers_[idx(opponent(team))].princess[side]) == nullptr;
    }
    if (!allowed) return false;

    for (const auto& e : entities_) {
        if (e.is_building() && std::abs(pos.x - e.pos.x) < e.footprint && std::abs(pos.y - e.pos.y) < e.footprint) {
            return false;
        }
    }
    return true;
}

bool Battle::can_cast_at(Vec2 pos) const {
    const ArenaDef& a = data_.arena;
    return pos.x >= 0.5 && pos.x <= a.width - 0.5 && pos.y >= 0.5 && pos.y <= a.height - 0.5;
}

std::string Battle::check_play(Team team, int hand_slot, Vec2 pos) const {
    if (finished()) return "match is finished";
    if (hand_slot < 0 || hand_slot >= kHandSize) return "bad hand slot";
    const PlayerState& p = players_[idx(team)];
    if (p.hand[hand_slot].empty()) return "slot is empty";
    const CardDef& card = data_.card(p.hand[hand_slot]);
    if (p.elixir + kElixirEps < card.elixir) return "not enough elixir";
    if (card.type == CardType::Spell ? !can_cast_at(pos) : !can_deploy_at(team, pos)) return "cannot deploy there";
    return {};
}

bool Battle::play(Team team, int hand_slot, Vec2 pos) {
    if (!check_play(team, hand_slot, pos).empty()) return false;
    start_play(team, hand_slot, pos);
    return true;
}

double Battle::force_play(Team team, const std::string& card_id, std::optional<Vec2> pos, int ago_ms) {
    const CardDef& card = data_.card(card_id);
    if (finished()) return 0.0;
    PlayerState& p = players_[idx(team)];
    int slot = -1;
    for (int i = 0; i < kHandSize; ++i) {
        if (p.hand[i] == card_id) slot = i;
    }
    if (slot < 0) {
        // Not in the simulated hand: take it out of the queue into a hand slot (swapping with the card
        // there, or filling an empty slot), so the deck still holds each card once.
        auto it = std::find(p.queue.begin(), p.queue.end(), card_id);
        if (it == p.queue.end()) throw std::invalid_argument("force_play: " + card_id + " is not in the deck");
        for (int i = 0; i < kHandSize && slot < 0; ++i) {
            if (!p.hand[i].empty()) slot = i;
        }
        if (slot >= 0) {
            std::swap(*it, p.hand[slot]);
        } else {
            slot = 0;
            p.hand[slot] = card_id;
            p.queue.erase(it);
        }
    }
    if (pos && card.type == CardType::Troop && !can_deploy_at(team, *pos)) {
        // Perception is off by a fraction of a tile now and then: use the closest legal spot nearby.
        double best = 1e9;
        Vec2 best_pos = *pos;
        for (double dy = -3.0; dy <= 3.0; dy += 0.25) {
            for (double dx = -3.0; dx <= 3.0; dx += 0.25) {
                const Vec2 q{pos->x + dx, pos->y + dy};
                const double d = dx * dx + dy * dy;
                if (d < best && can_deploy_at(team, q)) {
                    best = d;
                    best_pos = q;
                }
            }
        }
        pos = best_pos;
    }
    const double missing = std::max(0.0, card.elixir - p.elixir);
    const size_t first_new = pending_.size();
    start_play(team, slot, pos);
    for (size_t i = first_new; i < pending_.size(); ++i) pending_[i].ms_left -= std::max(0, ago_ms);  // may go below 0
    return missing;
}

void Battle::set_hand(Team team, const std::array<std::string, kHandSize>& hand, const std::string& next) {
    PlayerState& p = players_[idx(team)];
    // Every card of the deck, in the simulated cycle order: hand first, then the queue.
    std::vector<std::string> pool;
    for (const auto& c : p.hand) {
        if (!c.empty()) pool.push_back(c);
    }
    pool.insert(pool.end(), p.queue.begin(), p.queue.end());
    auto take = [&pool](const std::string& c) {
        auto it = std::find(pool.begin(), pool.end(), c);
        if (it == pool.end()) throw std::invalid_argument("set_hand: " + c + " is not in the deck");
        pool.erase(it);
    };
    for (const auto& c : hand) {
        if (!c.empty()) take(c);
    }
    std::deque<std::string> queue;
    if (!next.empty()) {
        take(next);
        queue.push_back(next);
    }
    // The simulated order of the hand's old cards is unknown to the screen; put the queue's own cards
    // first (they were already waiting), then cards the simulator still had in hand.
    std::vector<std::string> rest;
    for (const auto& c : p.queue) {
        auto it = std::find(pool.begin(), pool.end(), c);
        if (it != pool.end()) {
            rest.push_back(c);
            pool.erase(it);
        }
    }
    rest.insert(rest.end(), pool.begin(), pool.end());
    queue.insert(queue.end(), rest.begin(), rest.end());
    for (int i = 0; i < kHandSize; ++i) {
        if (hand[i].empty() && !p.hand[i].empty()) p.refill_ms_left[i] = card_refill_ms();
        if (!hand[i].empty()) p.refill_ms_left[i] = 0;
        p.hand[i] = hand[i];
    }
    p.queue = std::move(queue);
}

bool Battle::twin_move(EntityId id, Vec2 pos) {
    Entity* e = find_mut(id);
    if (!e || e->is_building()) return false;
    e->pos = {std::clamp(pos.x, 0.0, data_.arena.width), std::clamp(pos.y, 0.0, data_.arena.height)};
    if (!e->air) keep_out_of_river(*e);
    return true;
}

bool Battle::twin_set_hp(EntityId id, int hp) {
    Entity* e = find_mut(id);
    if (!e) return false;
    e->hp = std::clamp(hp, 0, e->max_hp);  // 0: removed (and a tower counted as destroyed) on the next step
    return true;
}

bool Battle::twin_remove(EntityId id) {
    Entity* e = find_mut(id);
    if (!e || e->is_building()) return false;
    e->hp = 0;
    return true;
}

EntityId Battle::twin_spawn(Team team, const std::string& card_id, Vec2 pos, int play, int unit) {
    const CardDef& card = data_.card(card_id);
    if (card.type != CardType::Troop) throw std::invalid_argument("twin_spawn: " + card_id + " is not a troop");
    spawn_unit(team, card, pos, 0);
    Entity& e = entities_.back();
    e.play = play;
    e.unit = unit;
    e.pos = {std::clamp(pos.x, 0.0, data_.arena.width), std::clamp(pos.y, 0.0, data_.arena.height)};
    if (!e.air) keep_out_of_river(e);
    e.deploy_ms_left = 0;  // it was seen walking around: long past its deploy timer
    e.first_attack_delay_ms = 0;
    return e.id;
}

void Battle::set_elixir(Team team, double elixir) {
    players_[idx(team)].elixir = std::clamp(elixir, 0.0, static_cast<double>(data_.rules.max_elixir));
}

void Battle::start_play(Team team, int hand_slot, std::optional<Vec2> pos) {
    PlayerState& p = players_[idx(team)];
    const std::string card_id = p.hand[hand_slot];
    const CardDef& card = data_.card(card_id);

    p.elixir = std::max(0.0, p.elixir - card.elixir);
    // The played card goes to the back of the queue; its slot stays empty for a moment, then the
    // front of the queue moves in (refill_hands).
    p.queue.push_back(card_id);
    p.hand[hand_slot].clear();
    p.refill_ms_left[hand_slot] = card_refill_ms();
    const int play_index = static_cast<int>(p.plays.size());
    p.plays.push_back({card.index, time_ms_, pos ? -1 : time_ms_});
    log(EventType::CardPlayed, team, kNoEntity, card_id);
    if (!pos) return;  // twin: played somewhere we don't know; only the cycle and elixir change

    // The card shows up as a blueprint first; troops/spells materialise after the deploy delay.
    const int delay = deploy_delay_ms();
    if (card.type == CardType::Spell) {
        pending_.push_back({team, card.index, *pos, -1, delay, play_index});
    } else {
        for (int i = 0; i < card.count; ++i) {
            pending_.push_back({team, card.index, *pos, i, delay + i * card.spawn_stagger_ms, play_index});
        }
    }
}

void Battle::cast_spell(Team team, const CardDef& card, Vec2 pos, int elapsed_ms) {
    const SpellDef& s = card.spell;
    const Vec2 origin = to_absolute(team, data_.arena.king_pos);
    for (int wave = 0; wave < s.waves; ++wave) {
        Projectile p;
        p.team = team;
        p.pos = origin;
        p.aim = pos;
        // Cast elapsed_ms ago: a wave already launched is that much further along (at most at its aim:
        // it lands in this tick's projectile update).
        const int flown_ms = elapsed_ms - wave * s.wave_interval_ms;
        if (flown_ms > 0) {
            const double d = std::min(distance(origin, pos), s.projectile_speed * flown_ms / 1000.0);
            p.pos = move_towards(origin, pos, d);
        }
        p.speed = s.projectile_speed;
        p.damage = s.damage;
        p.tower_damage_pct = s.tower_damage_pct;
        p.splash_radius = s.radius;
        p.pushback = s.pushback;
        p.delay_ms = std::max(0, wave * s.wave_interval_ms - elapsed_ms);
        p.card_index = card.index;
        p.targets = s.targets;
        p.source = card.id;
        projectiles_.push_back(p);
    }
}

double Battle::elixir_multiplier() const {
    const RulesDef& r = data_.rules;
    if (r.triple_elixir_from_ms > 0 && time_ms_ >= r.triple_elixir_from_ms) return 3.0;
    return time_ms_ >= r.double_elixir_from_ms ? 2.0 : 1.0;
}

void Battle::step() {
    if (finished()) return;
    const int dt = data_.rules.tick_ms;
    time_ms_ += dt;

    regen_elixir(dt);
    refill_hands(dt);
    for (auto& e : entities_) {
        if (e.deploy_ms_left > 0) e.deploy_ms_left = std::max(0, e.deploy_ms_left - dt);
    }
    materialize_pending(dt);  // new troops start their full deploy timer from here
    // entities_ is not resized during the updates below, so references stay valid.
    for (auto& e : entities_) {
        if (!e.alive() || e.deploy_ms_left > 0) continue;
        if (e.is_building()) update_tower(e, dt);
        else update_troop(e, dt);
    }
    update_projectiles(dt);
    resolve_collisions();
    remove_dead();
    update_phase();
}

void Battle::regen_elixir(int dt) {
    const RulesDef& r = data_.rules;
    const double gain = static_cast<double>(dt) / r.elixir_regen_ms * elixir_multiplier();
    for (auto& p : players_) {
        const double next = p.elixir + gain;
        if (next > r.max_elixir) p.elixir_leaked += next - std::max(p.elixir, r.max_elixir);
        p.elixir = std::min(r.max_elixir, next);
    }
}

bool Battle::can_attack(const Entity& attacker, const Entity& target) const {
    // Troops still on their deploy timer are on the field: they can be targeted and hit.
    return target.alive() && target.team != attacker.team && matches_targets(attacker.attack.targets, target);
}

EntityId Battle::acquire_troop_target(const Entity& e) const {
    // Closest valid enemy within sight range.
    EntityId best = kNoEntity;
    double best_gap = kInf;
    for (const auto& t : entities_) {
        if (!can_attack(e, t)) continue;
        const double gap = edge_gap(e, t);
        if (gap <= e.sight_range && gap < best_gap) {
            best = t.id;
            best_gap = gap;
        }
    }
    if (best != kNoEntity) return best;

    // Nothing in sight: keep marching on the building we were already heading for, if it still stands.
    // (A building that comes into sight range is picked up by the search above.)
    if (const Entity* current = find(e.target); current && current->is_building() && can_attack(e, *current)) {
        return current->id;
    }

    // Otherwise pick the closest enemy building. Troops still on their own half measure from the far
    // end of their lane's bridge, so with the right princess tower down a troop on the right goes for
    // the king instead of crossing over to the left tower. Flyers follow the same lanes.
    Vec2 from = e.pos;
    if (e.lane >= 0 && on_own_bank(e)) {
        const ArenaDef& a = data_.arena;
        from = {a.bridge_x[e.lane], e.team == Team::Blue ? a.river_max_y : a.river_min_y};
    }
    double best_dist = kInf;
    for (const auto& t : entities_) {
        if (!t.is_building() || !can_attack(e, t)) continue;
        const double d = distance(from, t.pos);
        if (d < best_dist) {
            best = t.id;
            best_dist = d;
        }
    }
    return best;
}

bool Battle::on_own_bank(const Entity& e) const {
    const ArenaDef& a = data_.arena;
    return e.team == Team::Blue ? e.pos.y < a.river_min_y : e.pos.y > a.river_max_y;
}

void Battle::update_lane(Entity& e) const {
    const std::vector<double>& bridges = data_.arena.bridge_x;
    int nearest = 0;
    for (int i = 1; i < static_cast<int>(bridges.size()); ++i) {
        const double d = std::abs(e.pos.x - bridges[i]) - std::abs(e.pos.x - bridges[nearest]);
        // Exactly in the middle: the lane on the player's own left (Red sees the arena rotated).
        if (d < 0.0 || (d == 0.0 && e.team == Team::Red)) nearest = i;
    }
    if (e.lane < 0) {
        e.lane = nearest;
        return;
    }
    // A unit nudged across the middle by collisions keeps its lane; one that walked over switches.
    const double gain = std::abs(e.pos.x - bridges[e.lane]) - std::abs(e.pos.x - bridges[nearest]);
    if (gain > kLaneHysteresis) e.lane = nearest;
}

Vec2 Battle::next_waypoint(const Entity& e, Vec2 dst, int bridge) const {
    if (e.air) return dst;
    const ArenaDef& a = data_.arena;
    // -1: Blue bank, 0: river band (only reachable via a bridge), 1: Red bank.
    auto bank = [&](double y) { return y < a.river_min_y ? -1 : (y > a.river_max_y ? 1 : 0); };
    const int from = bank(e.pos.y);
    const int to = bank(dst.y);
    if (from == to) return dst;

    // Troops don't all funnel through a bridge's centre: they step on it on the side they come from and keep
    // that line across. Over 549 crossings in 47 recordings (tools/bridge_check.py): coming 1 tile off-centre
    // they cross ~0.35 off, from further out ~0.5-0.6; 90% of crossings within 0.6 of the centre.
    auto lane_x = [&](double bx) { return std::clamp(e.pos.x, bx - kBridgeLine, bx + kBridgeLine); };
    if (from == 0) {
        // On a bridge: keep walking straight to the bank the destination is on.
        double bx = a.bridge_x.front();
        for (double x : a.bridge_x) {
            if (std::abs(x - e.pos.x) < std::abs(bx - e.pos.x)) bx = x;
        }
        const double exit_y = to > 0 ? a.river_max_y + 1e-6 : a.river_min_y - 1e-6;
        return {lane_x(bx), exit_y};
    }

    // On a bank with the destination across the river: head for the entrance of the given bridge,
    // or else the one giving the shortest walk.
    double bx = bridge >= 0 ? a.bridge_x[bridge] : a.bridge_x.front();
    if (bridge < 0) {
        double best_cost = kInf;
        for (double x : a.bridge_x) {
            const double cost = std::abs(x - e.pos.x) + std::abs(x - dst.x);
            if (cost < best_cost) {
                bx = x;
                best_cost = cost;
            }
        }
    }
    return {lane_x(bx), from < 0 ? a.river_min_y : a.river_max_y};
}

void Battle::update_troop(Entity& e, int dt) {
    if (on_own_bank(e)) update_lane(e);
    Entity* target = find_mut(e.target);
    if (e.locked &&
        !(target && can_attack(e, *target) && edge_gap(e, *target) <= e.attack.range + kRangeSlack)) {
        e.locked = false;
    }
    if (!e.locked) {
        e.target = acquire_troop_target(e);
        target = find_mut(e.target);
    }
    if (!target) return;

    const double reach = e.attack.range + (e.locked ? kRangeSlack : 0.0);
    if (edge_gap(e, *target) <= reach) {
        attack_tick(e, *target, dt);
    } else {
        // Marching on a building from our own half follows the lane; chasing troops takes the shortest way.
        const int bridge = target->is_building() && !e.air && on_own_bank(e) ? e.lane : -1;
        const Vec2 waypoint = next_waypoint(e, target->pos, bridge);
        e.pos = move_towards(e.pos, waypoint, follow_step(e, waypoint, e.speed * dt / 1000.0, dt));
    }
}

double Battle::follow_step(const Entity& e, Vec2 waypoint, double step, int dt) const {
    // A troop walking into the back of a slower one of its side, on its way (even a little off to the side),
    // follows it at its pace instead of getting past: troops put behind a giant stack up behind it and push
    // together - pushes in the game are built on this. Once the one in front stops to fight, the troop goes
    // round it (resolve_collisions).
    const Vec2 to = waypoint - e.pos;
    const double len = to.length();
    if (len < 1e-9) return step;
    const Vec2 dir = to * (1.0 / len);
    for (const auto& l : entities_) {
        if (&l == &e || l.team != e.team || l.is_building() || !l.alive() || l.air != e.air) continue;
        if (l.locked || l.speed >= e.speed) continue;
        const Vec2 rel = l.pos - e.pos;
        const double along = rel.x * dir.x + rel.y * dir.y;
        const double side = std::abs(rel.x * dir.y - rel.y * dir.x);
        const double contact = e.radius + l.radius;
        if (along <= 0.0 || along > contact + kFollowGap || side >= contact * kFollowWidth) continue;
        const double front_moves = l.deploy_ms_left > 0 ? 0.0 : l.speed * dt / 1000.0;
        step = std::min(step, std::max(0.0, along - contact) + front_moves);
    }
    return step;
}

void Battle::update_tower(Entity& e, int dt) {
    if (!e.active || e.deploy_ms_left > 0) return;
    Entity* target = find_mut(e.target);
    if (!(target && can_attack(e, *target) && edge_gap(e, *target) <= e.attack.range)) {
        e.locked = false;
        e.target = kNoEntity;
        target = nullptr;
        double best_gap = kInf;
        for (auto& t : entities_) {
            if (t.is_building() || !can_attack(e, t)) continue;
            const double gap = edge_gap(e, t);
            if (gap <= e.attack.range && gap < best_gap) {
                target = &t;
                best_gap = gap;
            }
        }
        if (target) e.target = target->id;
    }
    if (target) attack_tick(e, *target, dt);
}

void Battle::attack_tick(Entity& e, Entity& target, int dt) {
    if (!e.locked) {
        e.locked = true;
        e.attack_cd_ms = e.attack.first_hit_ms + e.first_attack_delay_ms;
        e.first_attack_delay_ms = 0;  // only the first attack after deploying waits longer
    }
    e.attack_cd_ms -= dt;
    if (e.attack_cd_ms <= 0) {
        fire(e, target);
        e.attack_cd_ms += e.attack.hit_speed_ms;
    }
}

void Battle::fire(Entity& e, Entity& target) {
    const AttackDef& a = e.attack;
    hit_source_ = e.def_id;
    if (a.projectile_speed <= 0.0) {
        if (a.splash_radius > 0.0) {
            apply_splash(e.team, a.splash_around_self ? e.pos : target.pos, a.splash_radius, a.damage,
                         a.tower_damage_pct, a.targets);
        } else {
            apply_damage(target, a.damage, a.tower_damage_pct);
        }
        return;
    }
    Projectile p;
    p.team = e.team;
    p.target = target.id;
    p.pos = e.pos;
    p.aim = target.pos;
    p.speed = a.projectile_speed;
    p.damage = a.damage;
    p.tower_damage_pct = a.tower_damage_pct;
    p.splash_radius = a.splash_radius;
    p.targets = a.targets;
    p.source = e.def_id;
    projectiles_.push_back(p);
}

void Battle::update_projectiles(int dt) {
    for (auto& p : projectiles_) {
        if (p.delay_ms > 0) {  // a later spell wave still waiting at the king tower
            p.delay_ms = std::max(0, p.delay_ms - dt);
            continue;
        }
        Entity* target = find_mut(p.target);
        if (target && target->alive()) p.aim = target->pos;
        const double step = p.speed * dt / 1000.0;
        if (distance(p.pos, p.aim) > step) {
            p.pos = move_towards(p.pos, p.aim, step);
            continue;
        }
        p.done = true;
        hit_source_ = p.source;
        spell_hit_ = p.is_spell();
        if (p.splash_radius > 0.0) {
            apply_splash(p.team, p.aim, p.splash_radius, p.damage, p.tower_damage_pct, p.targets, p.pushback);
        } else if (target && target->alive()) {
            apply_damage(*target, p.damage, p.tower_damage_pct);
        }
        spell_hit_ = false;
    }
    std::erase_if(projectiles_, [](const Projectile& p) { return p.done; });
}

void Battle::apply_damage(Entity& target, int damage, int tower_damage_pct) {
    // Crown towers take a share of some damage, rounded up: the game takes off the reduction rounded down
    // (Fireball 688 -> 159, Arrows 122 -> 25 per wave at level 11, as seen in friendly battles).
    if (target.is_building() && tower_damage_pct != 100) damage -= damage * (100 - tower_damage_pct) / 100;
    if (spell_hit_ && !target.is_building()) target.spell_damage += std::min(damage, std::max(0, target.hp));
    target.hp -= damage;
    if (record_damage_ && record_events_) {
        events_.push_back({time_ms_, EventType::Damage, target.team, target.id, hit_source_ + ":" + std::to_string(damage)});
    }
    if (target.kind == EntityKind::KingTower && !target.active && target.alive()) activate_king(target);
}

void Battle::apply_splash(Team source, Vec2 center, double radius, int damage, int tower_damage_pct,
                          const TargetRules& targets, double pushback) {
    TargetRules hit = targets;
    hit.buildings_only = false;  // splash damages everything of the right layer around the impact
    for (auto& t : entities_) {
        if (!t.alive() || t.team == source || !matches_targets(hit, t)) continue;
        if (distance(center, t.pos) - t.radius > radius) continue;
        apply_damage(t, damage, tower_damage_pct);
        if (pushback > 0.0 && t.alive() && !t.is_building() && !t.ignore_pushback) knock_back(t, center, pushback);
    }
}

void Battle::knock_back(Entity& e, Vec2 from, double dist) const {
    const Vec2 d = e.pos - from;
    const double len = d.length();
    // Hit dead center: pushed back towards its own side.
    const Vec2 dir = len > 1e-9 ? d * (1.0 / len) : Vec2{0.0, e.team == Team::Blue ? -1.0 : 1.0};
    e.pos = e.pos + dir * dist;
    e.locked = false;  // the attack restarts from scratch (first_hit_ms wind-up)
}

void Battle::resolve_collisions() {
    const ArenaDef& a = data_.arena;
    // Troops on the same layer (ground/air) push each other apart, heavier ones move less.
    for (size_t i = 0; i < entities_.size(); ++i) {
        Entity& p = entities_[i];
        if (p.is_building() || !p.alive()) continue;
        for (size_t j = i + 1; j < entities_.size(); ++j) {
            Entity& q = entities_[j];
            if (q.is_building() || !q.alive() || p.air != q.air) continue;
            const Vec2 d = q.pos - p.pos;
            const double dist = d.length();
            const double overlap = p.radius + q.radius - dist;
            if (overlap <= 0.0) continue;
            // Units spawned on the exact same spot separate sideways, deterministically.
            Vec2 n = dist > 1e-9 ? d * (1.0 / dist) : Vec2{(i + j) % 2 ? 1.0 : -1.0, 0.0};
            // Two of a side in line, the one in front stopped to fight (a giant at the tower): pushed straight
            // along the line the one behind would wait there forever; in the game it goes round. (While the one
            // in front still walks, the one behind follows it instead: update_troop.)
            if (p.team == q.team && (p.locked || q.locked) && std::abs(n.x) < kFollowSlipBelow) {
                const double side = std::abs(d.x) > 1e-3 ? (d.x > 0 ? 1.0 : -1.0) : ((i + j) % 2 ? 1.0 : -1.0);
                n = Vec2{n.x + side * kFollowSlip, n.y};
                n = n * (1.0 / n.length());
            }
            const double push = overlap * kCollisionStiffness;
            const double total = p.mass + q.mass;
            p.pos = p.pos - n * (push * q.mass / total);
            q.pos = q.pos + n * (push * p.mass / total);
        }
    }
    // Buildings don't move; ground troops slide around them. Air troops fly over.
    for (auto& e : entities_) {
        if (e.is_building() || e.air || !e.alive()) continue;
        for (const auto& b : entities_) {
            if (!b.is_building()) continue;
            const Vec2 d = e.pos - b.pos;
            const double dist = d.length();
            const double min_dist = e.radius + b.radius;
            if (dist >= min_dist) continue;
            const Vec2 n = dist > 1e-9 ? d * (1.0 / dist) : Vec2{0.0, e.team == Team::Blue ? -1.0 : 1.0};
            e.pos = b.pos + n * min_dist;
        }
        keep_out_of_river(e);
    }
    for (auto& e : entities_) {
        e.pos.x = std::clamp(e.pos.x, 0.0, a.width);
        e.pos.y = std::clamp(e.pos.y, 0.0, a.height);
    }
}

void Battle::keep_out_of_river(Entity& e) const {
    const ArenaDef& a = data_.arena;
    if (e.pos.y <= a.river_min_y || e.pos.y >= a.river_max_y) return;
    for (double bx : a.bridge_x) {
        if (std::abs(e.pos.x - bx) <= a.bridge_half_width) return;  // on a bridge
    }
    // In the water: move to the closest dry spot (a bank or the edge of a bridge).
    Vec2 best{e.pos.x, a.river_min_y};
    double best_d = e.pos.y - a.river_min_y;
    if (a.river_max_y - e.pos.y < best_d) {
        best = {e.pos.x, a.river_max_y};
        best_d = a.river_max_y - e.pos.y;
    }
    for (double bx : a.bridge_x) {
        const double edge = e.pos.x < bx ? bx - a.bridge_half_width : bx + a.bridge_half_width;
        if (std::abs(e.pos.x - edge) < best_d) {
            best = {edge, e.pos.y};
            best_d = std::abs(e.pos.x - edge);
        }
    }
    e.pos = best;
}

void Battle::activate_king(Entity& king) {
    if (king.active) return;
    king.active = true;
    king.deploy_ms_left = data_.king_tower.activation_ms;  // waking up: no shots yet
    log(EventType::KingActivated, king.team, king.id);
}

void Battle::remove_dead() {
    for (auto& e : entities_) {
        if (e.alive()) continue;
        log(EventType::Death, e.team, e.id, e.def_id);
        if (record_events_) {
            events_.back().pos = e.pos;
            events_.back().play = e.play;
            events_.back().unit = e.unit;
        }
        if (e.kind == EntityKind::PrincessTower) {
            players_[idx(opponent(e.team))].crowns += 1;
            log(EventType::TowerDestroyed, e.team, e.id, e.def_id);
            if (Entity* king = find_mut(towers_[idx(e.team)].king); king && king->alive()) activate_king(*king);
        } else if (e.kind == EntityKind::KingTower) {
            players_[idx(opponent(e.team))].crowns = 3;
            king_destroyed_ = true;
            log(EventType::TowerDestroyed, e.team, e.id, e.def_id);
        }
    }
    std::erase_if(entities_, [](const Entity& e) { return !e.alive(); });
}

void Battle::update_phase() {
    const RulesDef& r = data_.rules;
    const int blue = players_[idx(Team::Blue)].crowns;
    const int red = players_[idx(Team::Red)].crowns;
    auto leader = [&]() -> std::optional<Team> {
        if (blue == red) return std::nullopt;
        return blue > red ? Team::Blue : Team::Red;
    };

    if (king_destroyed_ && !follow_real_) {
        finish(leader());
        return;
    }
    if (phase_ == Phase::Regulation && time_ms_ >= r.regulation_ms) {
        if (blue != red && !follow_real_) {
            finish(leader());
            return;
        }
        phase_ = Phase::Overtime;
        log(EventType::OvertimeStarted, Team::Blue, kNoEntity);
    }
    if (phase_ != Phase::Overtime || follow_real_) return;
    if (blue != red) {  // sudden death: first crown in overtime wins
        finish(leader());
        return;
    }
    if (time_ms_ >= r.regulation_ms + r.overtime_ms) {
        // Tiebreaker: the player whose weakest tower has less HP loses.
        int weakest[2] = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
        for (const auto& e : entities_) {
            if (e.is_building()) weakest[idx(e.team)] = std::min(weakest[idx(e.team)], e.hp);
        }
        if (weakest[0] == weakest[1]) finish(std::nullopt);
        else finish(weakest[0] > weakest[1] ? Team::Blue : Team::Red);
    }
}

void Battle::finish(std::optional<Team> winner) {
    phase_ = Phase::Finished;
    winner_ = winner;
    log(EventType::MatchEnded, winner.value_or(Team::Blue), kNoEntity, winner ? team_name(*winner) : "draw");
}

void Battle::log(EventType type, Team team, EntityId entity, std::string info) {
    if (!record_events_) return;
    events_.push_back({time_ms_, type, team, entity, std::move(info), {}, -1, -1});
}

}  // namespace cr
