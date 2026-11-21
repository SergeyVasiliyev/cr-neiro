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
constexpr double kBridgeLine = 0.55;
// How far before the river a troop walking to a bridge gets onto its line (tiles), and how close to the line
// counts as on it.
constexpr double kBridgeApproach = 3.0;
constexpr double kLinedUp = 0.1;
// Bridge entrances and exits are aimed this far past the bank's edge (tiles): aimed right at the edge, a troop
// creeps up to it in ever smaller steps, and the least push from a troop beside it (a knight brushing a giant at the
// entrance, video37 128 s) keeps it short of the edge for good.
constexpr double kOntoBank = 0.3;
// Room left between a troop walking round a building and the building (tiles).
constexpr double kSteerMargin = 0.05;
// How long a knockback (fireball) takes.
constexpr int kKnockMs = 300;
// Same-side troops nearly in line (|sideways part of the push| below this) get this much sideways push too,
// when the one in front has stopped to fight.
constexpr double kFollowSlipBelow = 0.15;
constexpr double kFollowSlip = 0.35;
// Two of a side walking into each other off line: the sideways part of the push is rules.pass_slide times larger.
// Following a slower troop of one's side (follow_step): it is in the way when it is ahead, at most this far
// past touching, and off the line by less than this share of the two radii; only when both walk nearly the
// same way (cos of the angle between them), and the one in front is within this much sideways per tile ahead.
// In the recordings a mini pekka coming up right behind its giant (off its line by 0-0.5) gets past it 1 time
// in 58 within 4 s, one ~0.8 off (2/3 of the radii) slides along its side and past (video36, 61-64 s), as does
// one level with it.
// (The gap and the width: rules.follow_gap and rules.follow_width, data/rules.json "movement".)
constexpr double kFollowAlign = 0.94;
constexpr double kFollowCone = 1.0;

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

void Battle::spawn_unit(Team team, const CardDef& card, Vec2 pos, int i, const CardDef* as) {
    // Small groups stand on a circle, big ones (Skeleton Army) fill a disc. On the circle an odd group has one
    // in front (minions: one ahead, two behind it to the sides - 119 plays in the recordings), an even one
    // stands side by side (archers). A second kind of unit (Goblin Gang) stands on the same circle.
    const int n = card.count + (card.second.any() ? card.second.count : 0);
    Vec2 offset;
    if (card.formation == Formation::Line && card.count > 1) {
        offset = {-card.formation_width / 2 + card.formation_width * (i + 0.5) / card.count, 0.0};
    } else if (card.formation == Formation::FullLane && card.count > 1) {
        const double w = card.formation_width;
        offset = {data_.arena.width / 2 - w / 2 + w * (i + 0.5) / card.count - pos.x, 0.0};
    } else if (n > 1 && n <= 6) {
        const double forward = team == Team::Blue ? kPi / 2 : -kPi / 2;
        const double angle = forward + (n % 2 ? 0.0 : kPi / 2) + 2.0 * kPi * i / n;
        offset = {std::cos(angle) * card.spawn_spread, std::sin(angle) * card.spawn_spread};
    } else if (n > 6) {
        const double r = card.spawn_spread * std::sqrt((i + 0.5) / n);
        offset = {std::cos(i * kGoldenAngle) * r, std::sin(i * kGoldenAngle) * r};
    }
    add_unit(team, as ? *as : card, pos + offset);
}

Entity& Battle::add_unit(Team team, const CardDef& def, Vec2 pos) {
    const ArenaDef& a = data_.arena;
    const UnitDef& u = def.unit;
    Entity e;
    e.id = next_id_++;
    e.kind = u.building ? EntityKind::Building : EntityKind::Troop;
    e.team = team;
    e.def_id = def.id;
    e.card_index = def.index;
    e.def = &def;
    e.pos = {std::clamp(pos.x, 0.0, a.width), std::clamp(pos.y, 0.0, a.height)};
    e.radius = u.radius;
    e.mass = u.mass;
    e.ignore_pushback = u.ignore_pushback || u.building;
    e.air = u.air;
    e.hp = e.max_hp = u.hitpoints;
    e.speed = u.speed;
    e.sight_range = u.sight_range;
    e.attack = u.attack;
    e.deploy_ms_left = u.deploy_time_ms;
    e.first_attack_delay_ms = u.first_attack_delay_ms;
    e.shield = u.shield;
    if (u.building) {
        e.footprint = u.radius;
        if (u.lifetime_ms > 0) e.decay_per_ms = static_cast<double>(u.hitpoints) / u.lifetime_ms;
    }
    if (u.spawn.any()) e.spawn_ms = u.spawn.start_ms;
    if (!e.air && !u.building) keep_out_of_river(e);
    entities_.push_back(e);
    Entity& added = entities_.back();
    if (u.riders.any()) {
        const CardDef& rider = data_.card(u.riders.unit);
        for (int k = 0; k < u.riders.count; ++k) {
            const double side = u.riders.count == 1 ? 0.0 : (2.0 * k / (u.riders.count - 1) - 1.0) * 0.45;
            spawn_queue_.push_back({team, &rider, added.pos, added.deploy_ms_left, added.id, {side, 0.0}});
        }
    }
    return added;  // (until the next unit is added)
}

void Battle::queue_spawns(Team team, const SpawnDef& s, Vec2 at, Vec2 forward) {
    if (!s.any()) return;
    const CardDef& def = data_.card(s.unit);
    const double ahead = std::atan2(forward.y, forward.x);
    for (int i = 0; i < s.count; ++i) {
        Vec2 p = at;
        if (s.radius > 0.0 && s.count > 1) {
            const double angle = ahead + (s.count % 2 ? 0.0 : kPi / s.count) + 2.0 * kPi * i / s.count;
            p = at + Vec2{std::cos(angle) * s.radius, std::sin(angle) * s.radius};
        }
        spawn_queue_.push_back({team, &def, p, s.deploy_ms, kNoEntity, {0.0, 0.0}});
    }
}

void Battle::flush_spawns() {
    while (!spawn_queue_.empty()) {
        std::vector<QueuedSpawn> queue;
        queue.swap(spawn_queue_);
        for (const auto& s : queue) {
            Entity& e = add_unit(s.team, *s.def, s.pos);  // (its riders are queued for the next round)
            if (s.deploy_ms >= 0) e.deploy_ms_left = s.deploy_ms;
            if (s.host != kNoEntity) {
                e.host = s.host;
                e.host_offset = s.host_offset;
                e.pos = s.pos + s.host_offset;
            }
        }
    }
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
            const CardDef* as = p.unit_card >= 0 ? &data_.card(data_.card_ids[p.unit_card]) : nullptr;
            spawn_unit(p.team, card, p.pos, p.unit, as);
            Entity& e = entities_.back();
            e.deploy_ms_left = std::max(0, e.deploy_ms_left - overdue);
            e.play = p.play_index;
            e.unit = p.unit;
            // Put down on top of a troop of ours from an earlier play, it lands in front of it (as in the
            // game: the newer one ends up ahead), not pushed back by the collision.
            const double forward = p.team == Team::Blue ? 1.0 : -1.0;
            for (const auto& o : entities_) {
                if (e.is_building()) break;
                if (&o == &e || o.is_building() || o.team != e.team || o.air != e.air || o.play == e.play) continue;
                if (distance(o.pos, e.pos) < (o.radius + e.radius) * 0.9) {
                    e.pos.y = o.pos.y + forward * (o.radius + e.radius) * 0.9;
                    if (!e.air) keep_out_of_river(e);
                }
            }
            const AreaDamageDef& landing = (as ? *as : card).unit.deploy_damage;
            if (landing.any() && p.unit <= 0) {  // the Electro Wizard's zap, the Goblin Drill coming up
                hit_source_ = card.id;
                apply_area(p.team, e.pos, landing);
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
    if (!can_roll_from(team, pos)) return false;
    for (const auto& e : entities_) {
        if (e.is_building() && std::abs(pos.x - e.pos.x) < e.footprint && std::abs(pos.y - e.pos.y) < e.footprint) {
            return false;
        }
    }
    return true;
}

bool Battle::can_roll_from(Team team, Vec2 pos) const {
    const ArenaDef& a = data_.arena;
    if (pos.x < 0.5 || pos.x > a.width - 0.5 || pos.y < 0.5 || pos.y > a.height - 0.5) return false;
    // The outer tile at each end of the rows next to the river (x 0-1 and 17-18, on both sides of it) is not part
    // of the game's arena: a troop aimed there lands one tile in (live, 7 of 7 aimed at (17.5, 14.5) came down at
    // ~16.6; the user, 2026-11-21: that tile does not exist, left and right).
    if ((pos.x < 1.0 || pos.x > a.width - 1.0) &&
        ((pos.y >= a.river_min_y - 1.0 && pos.y < a.river_min_y) || (pos.y > a.river_max_y && pos.y <= a.river_max_y + 1.0))) {
        return false;
    }

    // Work in the player's own frame: own side is y < river_min_y.
    const double y = team == Team::Blue ? pos.y : a.height - pos.y;
    bool allowed = y < a.river_min_y;
    if (!allowed && y > a.river_max_y && y < a.river_max_y + a.pocket_depth) {
        // Pocket: the enemy princess tower on this side of the arena is down.
        const int side = pos.x < a.width / 2.0 ? 0 : 1;
        allowed = find(towers_[idx(opponent(team))].princess[side]) == nullptr;
    }
    return allowed;
}

bool Battle::can_cast_at(Vec2 pos) const {
    const ArenaDef& a = data_.arena;
    return pos.x >= 0.5 && pos.x <= a.width - 0.5 && pos.y >= 0.5 && pos.y <= a.height - 0.5;
}

bool Battle::can_dig_at(Vec2 pos) const {
    if (!can_cast_at(pos)) return false;
    for (const auto& e : entities_) {
        if (e.is_building() && std::abs(pos.x - e.pos.x) < e.footprint && std::abs(pos.y - e.pos.y) < e.footprint) {
            return false;
        }
    }
    return true;
}

std::string Battle::check_play(Team team, int hand_slot, Vec2 pos) const {
    if (finished()) return "match is finished";
    if (hand_slot < 0 || hand_slot >= kHandSize) return "bad hand slot";
    const PlayerState& p = players_[idx(team)];
    if (p.hand[hand_slot].empty()) return "slot is empty";
    const CardDef& card = data_.card(p.hand[hand_slot]);
    if (p.elixir + kElixirEps < card.elixir) return "not enough elixir";
    const bool ok = card.anywhere                    ? can_dig_at(pos)
                    : card.own_half                ? can_roll_from(team, pos)
                    : card.type == CardType::Spell ? can_cast_at(pos)
                                                   : can_deploy_at(team, pos);
    if (!ok) return "cannot deploy there";
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
    auto legal = [&](Vec2 q) { return card.own_half ? can_roll_from(team, q) : can_deploy_at(team, q); };
    if (pos && !card.placed_anywhere() && !legal(*pos)) {
        // Perception is off by a fraction of a tile now and then: use the closest legal spot nearby.
        double best = 1e9;
        Vec2 best_pos = *pos;
        for (double dy = -3.0; dy <= 3.0; dy += 0.25) {
            for (double dx = -3.0; dx <= 3.0; dx += 0.25) {
                const Vec2 q{pos->x + dx, pos->y + dy};
                const double d = dx * dx + dy * dy;
                if (d < best && legal(q)) {
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
    if (card.type == CardType::Spell) throw std::invalid_argument("twin_spawn: " + card_id + " is a spell");
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

double Battle::forget_blueprints(Team team) {
    std::vector<int> hidden;  // plays none of whose units is out yet (a group's first unit comes first)
    for (const auto& d : pending_) {
        if (d.team == team && d.unit <= 0 &&
            std::find(hidden.begin(), hidden.end(), d.play_index) == hidden.end()) {
            hidden.push_back(d.play_index);
        }
    }
    if (hidden.empty()) return 0.0;
    double refund = 0.0;
    for (const auto& d : pending_) {
        if (d.team == team && d.unit <= 0) refund += data_.card(data_.card_ids[d.card_index]).elixir;
    }
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [&](const PendingDeploy& d) {
                       return d.team == team &&
                              std::find(hidden.begin(), hidden.end(), d.play_index) != hidden.end();
                   }),
                   pending_.end());
    set_elixir(team, players_[idx(team)].elixir + refund);
    return refund;
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
        // A building dug there (Goblin Drill) comes up after its way underground from the king tower.
        const int dig = card.anywhere && card.dig_speed > 0.0
                            ? static_cast<int>(1000.0 * distance(to_absolute(team, data_.arena.king_pos), *pos) / card.dig_speed)
                            : 0;
        for (int i = 0; i < card.count; ++i) {
            pending_.push_back({team, card.index, *pos, i, delay + dig + i * card.spawn_stagger_ms, play_index});
        }
        if (card.second.any()) {
            const int second = data_.card(card.second.unit).index;
            for (int k = 0; k < card.second.count; ++k) {
                const int i = card.count + k;
                pending_.push_back({team, card.index, *pos, i, delay + i * card.spawn_stagger_ms, play_index, second});
            }
        }
    }
}

void Battle::cast_spell(Team team, const CardDef& card, Vec2 pos, int elapsed_ms) {
    const SpellDef& s = card.spell;
    if (s.kind == SpellKind::Area) {
        AreaEffect a;
        a.team = team;
        a.pos = pos;
        a.radius = s.radius;
        a.duration_ms = std::max(data_.rules.tick_ms, s.duration_ms);
        // It stays until its last tick or strike, which may come after its pull or aura ends (Poison's 8th tick).
        int last = a.duration_ms;
        if (s.tick_ms > 0 && s.tick_count > 0) last = std::max(last, s.first_ms + (s.tick_count - 1) * s.tick_ms);
        if (s.max_targets > 0) last = std::max(last, s.first_ms + (s.max_targets - 1) * s.hit_ms);
        a.ms_left = last + data_.rules.tick_ms;
        a.damage = s.damage;
        a.pushback = s.pushback;
        a.tower_damage_pct = s.tower_damage_pct;
        a.tower_damage = s.tower_damage;
        a.tower_tick_damage = s.tower_tick_damage;
        a.tick_ms = s.tick_ms;
        a.tick_left = s.first_ms;
        a.ticks_left = s.tick_count > 0 ? s.tick_count : -1;
        a.tick_damage = s.tick_damage;
        a.pull = s.pull;
        a.max_targets = s.max_targets;
        a.hit_ms = s.hit_ms;
        a.hit_left = s.first_ms > 0 ? s.first_ms : s.hit_ms;
        a.buff = s.buff;
        a.targets = s.targets;
        a.card_index = card.index;
        a.source = card.id;
        areas_.push_back(a);
        return;
    }
    if (s.kind == SpellKind::Roll) {
        Roller r;
        r.team = team;
        r.pos = pos;
        r.dir = team == Team::Blue ? 1.0 : -1.0;
        r.half_width = s.radius;
        r.half_depth = s.depth;
        r.left = s.range;
        r.speed = s.projectile_speed;
        r.damage = s.damage;
        r.tower_damage_pct = s.tower_damage_pct;
        r.pushback = s.pushback;
        r.spawn = s.spawn;
        r.card_index = card.index;
        r.source = card.id;
        rollers_.push_back(r);
        return;
    }
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
        p.buff = s.buff;
        p.spawn = s.spawn;
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
    flush_spawns();           // (the riders of what just landed)
    for (auto& e : entities_) {
        if (e.alive()) update_timers(e, dt);
    }
    for (auto& e : entities_) {  // riders sit on their unit; it gone, they fall off (remove_dead)
        if (e.host == kNoEntity || !e.alive()) continue;
        const Entity* h = find(e.host);
        if (h && h->alive()) e.pos = h->pos + e.host_offset;
        else e.hp = 0;
    }
    // entities_ is not resized during the updates below (units made meanwhile wait in spawn_queue_), so
    // references stay valid.
    for (auto& e : entities_) {
        if (!e.alive() || e.deploy_ms_left > 0) continue;
        if (e.is_building()) update_tower(e, dt);
        else if (e.knock_ms_left > 0) knocked(e, dt);
        else update_troop(e, dt);
    }
    update_projectiles(dt);
    update_areas(dt);
    update_rollers(dt);
    flush_spawns();
    resolve_collisions();
    remove_dead();
    update_phase();
}

void Battle::update_timers(Entity& e, int dt) {
    if (e.stun_ms > 0) e.stun_ms = std::max(0, e.stun_ms - dt);
    if (e.slow_ms > 0 && (e.slow_ms -= dt) <= 0) {
        e.slow_ms = 0;
        e.slow = 0.0;
    }
    if (e.rage_ms > 0 && (e.rage_ms -= dt) <= 0) {
        e.rage_ms = 0;
        e.rage = 0.0;
    }
    if (e.curse_ms > 0) e.curse_ms = std::max(0, e.curse_ms - dt);
    if (e.dash_cd_ms > 0) e.dash_cd_ms = std::max(0, e.dash_cd_ms - dt);
    if (e.hook_cd_ms > 0) e.hook_cd_ms = std::max(0, e.hook_cd_ms - dt);
    if (e.def == nullptr || e.deploy_ms_left > 0) return;
    if (e.decay_per_ms > 0.0) {  // a building's lifetime runs out
        e.decay_acc += e.decay_per_ms * dt;
        const int lost = static_cast<int>(e.decay_acc);
        if (lost > 0) {
            e.decay_acc -= lost;
            e.hp -= lost;
        }
    }
    if (e.spawn_ms >= 0 && e.stun_ms <= 0 && e.host == kNoEntity) {
        e.spawn_ms -= static_cast<int>(std::lround(dt * attack_factor(e)));
        if (e.spawn_ms <= 0) {
            const SpawnDef& s = e.def->unit.spawn;
            queue_spawns(e.team, s, e.pos, Vec2{0.0, e.team == Team::Blue ? 1.0 : -1.0});
            e.spawn_ms += std::max(s.interval_ms, data_.rules.tick_ms);
        }
    }
}

double Battle::speed_factor(const Entity& e) const { return (1.0 - e.slow) * (1.0 + e.rage); }
double Battle::attack_factor(const Entity& e) const { return (1.0 - e.slow) * (1.0 + e.rage); }

void Battle::apply_buff(Entity& e, const BuffDef& b, Team source) {
    if (!e.alive()) return;
    switch (b.kind) {
        case BuffKind::Stun:
            e.stun_ms = std::max(e.stun_ms, b.ms);
            e.locked = false;  // the attack's wind-up starts over, an inferno cools down, a charge is lost
            e.charging = false;
            e.walked = 0.0;
            e.ramp_ms = 0;
            e.hook_ms = -1;
            e.dashing = false;
            break;
        case BuffKind::Slow:
            e.slow_ms = std::max(e.slow_ms, b.ms);
            e.slow = std::max(e.slow, b.amount);
            break;
        case BuffKind::Curse:
            if (e.is_building()) break;
            e.curse_ms = std::max(e.curse_ms, b.ms);
            e.curse_team = source;
            e.curse_unit = b.unit.empty() ? nullptr : &data_.card(b.unit);
            break;
        case BuffKind::None:
            break;
    }
}

void Battle::heal_around(Team team, Vec2 center, double radius, int amount) {
    for (auto& t : entities_) {
        if (!t.alive() || t.team != team || t.is_building() || distance(t.pos, center) - t.radius > radius) continue;
        t.hp = std::min(t.max_hp, t.hp + amount);
    }
}

void Battle::apply_area(Team source, Vec2 center, const AreaDamageDef& a) {
    apply_splash(source, center, a.radius, a.damage, a.tower_damage_pct, TargetRules{true, true, false}, a.pushback,
                 a.buff.kind != BuffKind::None ? &a.buff : nullptr, kNoEntity, a.tower_damage);
}

void Battle::update_areas(int dt) {
    for (size_t k = 0; k < areas_.size(); ++k) {
        AreaEffect& a = areas_[k];  // (nothing here adds areas: deaths happen in remove_dead)
        if (a.done) continue;
        if (a.delay_ms > 0 && (a.delay_ms -= dt) > 0) continue;
        hit_source_ = a.source;
        spell_hit_ = a.card_index >= 0;
        const BuffDef* buff = a.buff.kind != BuffKind::None ? &a.buff : nullptr;
        if (!a.started) {
            a.started = true;
            if (a.rage <= 0.0 && a.max_targets == 0 && a.tick_ms == 0) {  // all at once (Zap, a death explosion)
                apply_splash(a.team, a.pos, a.radius, a.damage, a.tower_damage_pct, a.targets, a.pushback, buff,
                             kNoEntity, a.tower_damage);
            }
        }
        const bool lasting = a.elapsed_ms < a.duration_ms;  // (its pull and aura; ticks and bolts may come later)
        a.elapsed_ms += dt;
        if (a.rage > 0.0 && lasting) {  // friendly troops in it run and hit faster, for a second after leaving it too
            for (auto& t : entities_) {
                if (!t.alive() || t.team != a.team || t.is_building() || distance(t.pos, a.pos) - t.radius > a.radius) continue;
                t.rage = std::max(t.rage, a.rage);
                t.rage_ms = std::max(t.rage_ms, 1000);
            }
        }
        if (a.max_targets > 0 && a.hits < a.max_targets && (a.hit_left -= dt) <= 0) {  // the biggest one not hit yet
            a.hit_left += a.hit_ms;
            a.hits += 1;
            Entity* best = nullptr;
            for (auto& t : entities_) {
                if (!t.alive() || t.team == a.team || !t.targetable() || !matches_targets(a.targets, t)) continue;
                if (distance(t.pos, a.pos) - t.radius > a.radius) continue;
                if (std::find(a.hit.begin(), a.hit.end(), t.id) != a.hit.end()) continue;
                if (!best || t.hp + t.shield > best->hp + best->shield) best = &t;
            }
            if (best) {
                a.hit.push_back(best->id);
                if (buff) apply_buff(*best, *buff, a.team);
                apply_damage(*best, a.damage, a.tower_damage_pct, kNoEntity, a.tower_damage);
            }
        }
        if (a.tick_ms > 0 && a.ticks_left != 0 && (a.tick_left -= dt) <= 0) {  // damage over time (Poison, Tornado)
            a.tick_left += a.tick_ms;
            if (a.ticks_left > 0) a.ticks_left -= 1;
            apply_splash(a.team, a.pos, a.radius, a.tick_damage, a.tower_damage_pct, a.targets, 0.0, buff, kNoEntity,
                         a.tower_tick_damage);
        }
        if (a.pull > 0.0 && lasting) {
            const double step = a.pull * dt / 1000.0;
            for (auto& t : entities_) {
                if (!t.alive() || t.team == a.team || t.is_building() || !t.targetable() || !matches_targets(a.targets, t)) continue;
                if (distance(t.pos, a.pos) - t.radius > a.radius) continue;
                t.pos = move_towards(t.pos, a.pos, step);
            }
        }
        spell_hit_ = false;
        if ((a.ms_left -= dt) <= 0) a.done = true;
    }
    std::erase_if(areas_, [](const AreaEffect& a) { return a.done; });
}

void Battle::update_rollers(int dt) {
    for (auto& r : rollers_) {
        if (r.done) continue;
        hit_source_ = r.source;
        spell_hit_ = true;
        const double step = std::min(r.left, r.speed * dt / 1000.0);
        r.pos.y += r.dir * step;
        r.left -= step;
        for (auto& t : entities_) {
            if (!t.alive() || t.team == r.team || t.air || !t.targetable()) continue;
            if (std::abs(t.pos.x - r.pos.x) > r.half_width + t.radius || std::abs(t.pos.y - r.pos.y) > r.half_depth + t.radius) continue;
            if (std::find(r.hit.begin(), r.hit.end(), t.id) != r.hit.end()) continue;
            r.hit.push_back(t.id);
            apply_damage(t, r.damage, r.tower_damage_pct);
            // It pushes every troop it rolls over along, heavy ones too (PushbackAll).
            if (r.pushback > 0.0 && t.alive() && !t.is_building()) knock_back(t, t.pos - Vec2{0.0, r.dir}, r.pushback);
        }
        spell_hit_ = false;
        if (r.left <= 1e-9 || r.pos.y <= 0.0 || r.pos.y >= data_.arena.height) {
            r.done = true;
            queue_spawns(r.team, r.spawn, r.pos, Vec2{0.0, r.dir});
        }
    }
    std::erase_if(rollers_, [](const Roller& r) { return r.done; });
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
    // Troops still on their deploy timer are on the field: they can be targeted and hit. Riders on their unit and a
    // dashing Bandit are out of reach.
    return target.alive() && target.targetable() && target.team != attacker.team &&
           matches_targets(attacker.attack.targets, target);
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
    if (e.air || (e.def && e.def->unit.jump)) return dst;  // flyers, and jumpers over the river (Royal Hogs)
    const ArenaDef& a = data_.arena;
    // -1: Blue bank, 0: river band (only reachable via a bridge), 1: Red bank.
    auto bank = [&](double y) { return y < a.river_min_y ? -1 : (y > a.river_max_y ? 1 : 0); };
    const int from = bank(e.pos.y);
    const int to = bank(dst.y);
    if (from == to) return dst;

    // Troops don't all funnel through a bridge's centre: they step on it on the side they come from and keep
    // that line across. Over 549 crossings in 47 recordings (tools/bridge_check.py): coming 1 tile off-centre
    // they cross ~0.35 off, from further out ~0.5-0.6 (0.57 over 73 alone from 2+ tiles out); 90% within 0.6.
    auto lane_x = [&](double bx) { return std::clamp(e.pos.x, bx - kBridgeLine, bx + kBridgeLine); };
    if (from == 0) {
        // On a bridge: keep walking straight to the bank the destination is on.
        double bx = a.bridge_x.front();
        for (double x : a.bridge_x) {
            if (std::abs(x - e.pos.x) < std::abs(bx - e.pos.x)) bx = x;
        }
        const double exit_y = to > 0 ? a.river_max_y + kOntoBank : a.river_min_y - kOntoBank;
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
    // Far from the river a troop first gets onto its bridge's line, a few tiles before the river, and then walks
    // straight up it - not a straight line to the bridge: in the recordings a troop with 4-6 tiles to go sideways
    // heads off at ~45 degrees (a straight line would be ~30), and lines up ~3 tiles before the river
    // (tools/solo_path_check.py).
    const double lx = lane_x(bx);
    const double turn_y = from < 0 ? a.river_min_y - kBridgeApproach : a.river_max_y + kBridgeApproach;
    const bool before_turn = from < 0 ? e.pos.y < turn_y : e.pos.y > turn_y;
    if (before_turn && std::abs(e.pos.x - lx) > kLinedUp) return {lx, turn_y};
    return {lx, from < 0 ? a.river_min_y + kOntoBank : a.river_max_y - kOntoBank};
}

void Battle::update_troop(Entity& e, int dt) {
    if (e.stun_ms > 0) return;  // stunned or frozen: neither walks nor hits
    const UnitDef* u = e.def ? &e.def->unit : nullptr;
    if (e.host != kNoEntity) {  // a rider: shoots what comes into range, never walks (its unit carries it)
        Entity* t = find_mut(e.target);
        if (!(t && can_attack(e, *t) && edge_gap(e, *t) <= e.attack.range + (e.locked ? kRangeSlack : 0.0))) {
            e.locked = false;
            e.target = kNoEntity;
            t = nullptr;
            double best = kInf;
            for (auto& o : entities_) {
                if (!can_attack(e, o)) continue;
                const double g = edge_gap(e, o);
                if (g <= e.attack.range && g < best) {
                    best = g;
                    t = &o;
                }
            }
            if (t) e.target = t->id;
        }
        if (t) attack_tick(e, *t, dt);
        return;
    }
    if (on_own_bank(e)) update_lane(e);
    Entity* target = find_mut(e.target);
    if (e.locked &&
        !(target && can_attack(e, *target) && edge_gap(e, *target) <= e.attack.range + kRangeSlack)) {
        e.locked = false;
    }
    if (!e.locked && !e.dashing && e.hook_ms < 0) {
        e.target = acquire_troop_target(e);
        target = find_mut(e.target);
    }
    if (!target || !target->alive()) {
        e.dashing = false;
        e.hook_ms = -1;
        return;
    }

    if (u && u->dash.max > 0.0 && !target->air) {  // Bandit: a target within reach of a dash is dashed at
        const double gap = edge_gap(e, *target);
        if (!e.dashing && e.dash_cd_ms <= 0 && !e.locked && gap >= u->dash.min && gap <= u->dash.max) e.dashing = true;
        if (e.dashing) {
            const double step = u->dash.speed * dt / 1000.0;
            if (gap <= e.attack.range + step) {  // there: the dash's hit
                e.pos = move_towards(e.pos, target->pos, std::max(0.0, gap - e.attack.range * 0.5));
                e.dashing = false;
                e.dash_cd_ms = u->dash.cooldown_ms;
                hit_source_ = e.def_id;
                apply_damage(*target, u->dash.damage, e.attack.tower_damage_pct, e.id);
                e.locked = true;
                e.attack_cd_ms = e.attack.hit_speed_ms;
            } else {
                e.pos = move_towards(e.pos, target->pos, step);
            }
            return;
        }
    }
    if (u && u->hook.max > 0.0 && !target->air && !e.locked) {  // Fisherman: hooks a target out of his reach
        const double gap = edge_gap(e, *target);
        if (e.hook_ms < 0 && e.hook_cd_ms <= 0 && gap >= u->hook.min && gap <= u->hook.max) e.hook_ms = u->hook.load_ms;
        if (e.hook_ms >= 0) {
            e.hook_ms -= static_cast<int>(std::lround(dt * attack_factor(e)));
            if (e.hook_ms <= 0) {
                e.hook_ms = -1;
                e.hook_cd_ms = 3000;  // guess: one hook per target he goes for
                const Vec2 d = target->pos - e.pos;
                const double len = std::max(1e-9, d.length());
                const double contact = e.radius + target->radius + 0.1;
                if (target->is_building()) {
                    e.pos = target->pos - d * (contact / len);  // a building pulls him to it
                } else {
                    target->pos = e.pos + d * (contact / len);
                    target->locked = false;
                    keep_out_of_river(*target);
                }
            }
            return;
        }
    }

    const double reach = e.attack.range + (e.locked ? kRangeSlack : 0.0);
    if (edge_gap(e, *target) <= reach) {
        attack_tick(e, *target, dt);
    } else {
        // Marching on a building from our own half follows the lane; chasing troops takes the shortest way.
        const int bridge = target->is_building() && !e.air && on_own_bank(e) ? e.lane : -1;
        const Vec2 waypoint = steer_around_buildings(e, next_waypoint(e, target->pos, bridge), target);
        const Vec2 to = waypoint - e.pos;
        if (to.length() > 1e-9) e.heading = to * (1.0 / to.length());
        double mult = speed_factor(e);
        if (e.charging) mult *= u->charge.speed_mult;
        const Vec2 before = e.pos;
        e.pos = move_towards(e.pos, waypoint, follow_step(e, waypoint, e.speed * mult * dt / 1000.0, dt));
        if (u && u->charge.distance > 0.0) {  // walking on uninterrupted builds up a charge
            e.walked += distance(before, e.pos);
            if (e.walked >= u->charge.distance) e.charging = true;
        }
    }
}

Vec2 Battle::steer_around_buildings(const Entity& e, Vec2 waypoint, const Entity* target) const {
    // A building in the way (but the one it walks to) is walked round, on the side the troop is on already, along
    // the tangent to it: pushed straight back out of it a troop right behind a tower would hardly get past (a
    // mini pekka put behind its princess tower goes round its edge in the game, ~2 s - video37 7-11 s).
    if (e.air) return waypoint;
    const Vec2 to = waypoint - e.pos;
    const double len = to.length();
    if (len < 1e-9) return waypoint;
    const Vec2 dir = to * (1.0 / len);
    const Entity* block = nullptr;
    double block_along = kInf;
    for (const auto& b : entities_) {
        if (!b.is_building() || !b.alive() || &b == target) continue;
        const Vec2 v = b.pos - e.pos;
        const double along = v.x * dir.x + v.y * dir.y;
        const double reach = b.radius + e.radius + kSteerMargin;
        if (along <= 0.0 || along - reach > len) continue;  // behind it, or past where it is going
        if (std::abs(dir.x * v.y - dir.y * v.x) >= reach) continue;  // the way passes it by
        if (along < block_along) {
            block = &b;
            block_along = along;
        }
    }
    if (block == nullptr) return waypoint;
    const Vec2 v = block->pos - e.pos;
    const double d = v.length();
    const double reach = block->radius + e.radius + kSteerMargin;
    // The building's centre left of the way: round it on the right, and the other way about; dead ahead: on the
    // side the troop stands.
    double left = dir.x * v.y - dir.y * v.x;
    if (std::abs(left) < 1e-6) left = (e.pos.x - block->pos.x) * dir.y - (e.pos.y - block->pos.y) * dir.x;
    const double turn = left >= 0.0 ? -1.0 : 1.0;  // -1: clockwise
    const double alpha = d > reach ? std::asin(reach / d) : kPi / 2;
    const double c = std::cos(turn * alpha), s = std::sin(turn * alpha);
    const Vec2 u = v * (1.0 / d);
    const Vec2 round{u.x * c - u.y * s, u.x * s + u.y * c};
    return e.pos + round * len;
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
        // Only one walking the same way (one coming at its line at an angle slides off its side and gets past)
        const bool walked = l.heading.x != 0.0 || l.heading.y != 0.0;  // (one still landing counts as in the way)
        if (walked && l.heading.x * dir.x + l.heading.y * dir.y < kFollowAlign) continue;
        const Vec2 rel = l.pos - e.pos;
        const double along = rel.x * dir.x + rel.y * dir.y;
        const double side = std::abs(rel.x * dir.y - rel.y * dir.x);
        const double contact = e.radius + l.radius;
        // ... and ahead of it, not beside it (one level with it walks on past its side)
        if (along <= 0.0 || along > contact + data_.rules.follow_gap || side >= contact * data_.rules.follow_width ||
            side > along * kFollowCone) {
            continue;
        }
        const double front_moves = l.deploy_ms_left > 0 ? 0.0 : l.speed * dt / 1000.0;
        step = std::min(step, std::max(0.0, along - contact) + front_moves);
    }
    return step;
}

void Battle::update_tower(Entity& e, int dt) {
    if (!e.active || e.deploy_ms_left > 0 || e.stun_ms > 0) return;
    if (e.attack.damage <= 0) return;  // a building that only waits or makes units (Tombstone, Goblin Cage)
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
        e.ramp_ms = 0;                // an inferno starts cold on a new target
    }
    const double f = attack_factor(e);
    e.attack_cd_ms -= f == 1.0 ? dt : static_cast<int>(std::lround(dt * f));
    e.ramp_ms += dt;
    if (e.attack_cd_ms <= 0) {
        fire(e, target);
        e.attack_cd_ms += e.attack.hit_speed_ms;
    }
}

void Battle::fire(Entity& e, Entity& target) {
    const AttackDef& a = e.attack;
    const UnitDef* u = e.def ? &e.def->unit : nullptr;
    hit_source_ = e.def_id;
    int damage = a.damage;
    if (u && u->ramp.stage_ms > 0) {  // inferno: hotter the longer it stays on one target
        if (e.ramp_ms >= 2 * u->ramp.stage_ms) damage = u->ramp.damage3;
        else if (e.ramp_ms >= u->ramp.stage_ms) damage = u->ramp.damage2;
    }
    if (e.charging) {  // the charge's hit
        damage = u->charge.damage;
        e.charging = false;
        e.walked = 0.0;
    }
    if (u && u->pellets > 0) {
        // Hunter: the pellets fan out; a target catches the share its width covers - all of them right in front of
        // him. The fan: ~0.93 tiles wide per tile out, so that a princess tower at his full range catches 5 of 10
        // (420 off its HP, 3 times in the recordings c003, c007).
        const double width = 0.93 * std::max(0.5, distance(e.pos, target.pos));
        const double share = std::min(1.0, 2.0 * (target.radius + 0.3) / width);
        damage = static_cast<int>(std::lround(u->pellets * share)) * u->pellet_damage;
    }
    const BuffDef* buff = u && u->hit_buff.kind != BuffKind::None ? &u->hit_buff : nullptr;
    const double pushback = u ? u->attack_pushback : 0.0;
    Entity* hits[4] = {&target, nullptr, nullptr, nullptr};
    int n = 1;
    if (u && u->multi_targets > 1) {  // Electro Wizard: the next nearest enemies in range as well
        for (int k = 1; k < std::min(4, u->multi_targets); ++k) {
            Entity* best = nullptr;
            double best_gap = kInf;
            for (auto& o : entities_) {
                if (!can_attack(e, o) || std::find(hits, hits + n, &o) != hits + n) continue;
                const double g = edge_gap(e, o);
                if (g <= a.range && g < best_gap) {
                    best = &o;
                    best_gap = g;
                }
            }
            if (!best) break;
            hits[n++] = best;
        }
        // Fewer targets than bolts: the rest hit the target too (a lone tower took 2 x 117 = 234 per Electro Wizard
        // attack, every 1.8 s, 29 times in the recordings).
        while (n < std::min(4, u->multi_targets)) hits[n++] = &target;
    }
    for (int k = 0; k < n; ++k) {
        Entity& t = *hits[k];
        if (a.projectile_speed <= 0.0) {
            if (a.splash_radius > 0.0) {
                apply_splash(e.team, a.splash_around_self ? e.pos : t.pos, a.splash_radius, damage, a.tower_damage_pct,
                             a.targets, pushback, buff, e.id);
            } else {
                if (buff) apply_buff(t, *buff, e.team);
                apply_damage(t, damage, a.tower_damage_pct, e.id);
                if (pushback > 0.0 && t.alive() && !t.is_building() && !t.ignore_pushback) knock_back(t, e.pos, pushback);
            }
            continue;
        }
        Projectile p;
        p.team = e.team;
        p.target = t.id;
        p.pos = e.pos;
        p.aim = t.pos;
        p.speed = a.projectile_speed;
        p.damage = damage;
        p.tower_damage_pct = a.tower_damage_pct;
        p.splash_radius = a.splash_radius;
        p.pushback = pushback;
        p.targets = a.targets;
        p.source = e.def_id;
        p.shooter = e.id;
        if (buff) p.buff = *buff;
        if (u && u->heal > 0) {
            p.heal = u->heal;
            p.heal_radius = u->heal_radius;
        }
        projectiles_.push_back(p);
    }
    if (u && u->kamikaze) e.hp = 0;  // a spirit is its own shot; the Battle Ram breaks on its hit
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
        const BuffDef* buff = p.buff.kind != BuffKind::None ? &p.buff : nullptr;
        if (p.splash_radius > 0.0) {
            apply_splash(p.team, p.aim, p.splash_radius, p.damage, p.tower_damage_pct, p.targets, p.pushback, buff,
                         p.shooter);
        } else if (target && target->alive()) {
            if (buff) apply_buff(*target, *buff, p.team);  // (before the damage: a curse counts on the hit that kills)
            apply_damage(*target, p.damage, p.tower_damage_pct, p.shooter);
        }
        if (p.heal > 0) heal_around(p.team, p.aim, p.heal_radius, p.heal);
        queue_spawns(p.team, p.spawn, p.aim, Vec2{0.0, p.team == Team::Blue ? 1.0 : -1.0});
        spell_hit_ = false;
    }
    std::erase_if(projectiles_, [](const Projectile& p) { return p.done; });
}

void Battle::apply_damage(Entity& target, int damage, int tower_damage_pct, EntityId shooter, int tower_abs) {
    // Crown towers take a share of some damage, rounded up: the game takes off the reduction rounded down
    // (Fireball 688 -> 159, Arrows 122 -> 25 per wave at level 11, as seen in friendly battles) - or a damage of
    // their own (the files' TowerDamage). Other buildings take it all.
    if (target.is_tower() && tower_abs >= 0) damage = tower_abs;
    else if (target.is_tower() && tower_damage_pct != 100) damage -= damage * (100 - tower_damage_pct) / 100;
    if (damage <= 0 || !target.targetable()) return;  // (a goblin barrel landing on the king doesn't wake him)
    if (target.shield > 0) {  // a shield takes the whole hit, however big, and the rest of it is lost
        target.shield = std::max(0, target.shield - damage);
        if (record_damage_ && record_events_) {
            events_.push_back({time_ms_, EventType::Damage, target.team, target.id, hit_source_ + ":" + std::to_string(damage)});
        }
        return;
    }
    if (spell_hit_ && !target.is_building()) target.spell_damage += std::min(damage, std::max(0, target.hp));
    target.hp -= damage;
    if (record_damage_ && record_events_) {
        events_.push_back({time_ms_, EventType::Damage, target.team, target.id, hit_source_ + ":" + std::to_string(damage)});
    }
    if (target.kind == EntityKind::KingTower && !target.active && target.alive()) activate_king(target);
    if (target.def && target.def->unit.reflect.any() && shooter != kNoEntity && !spell_hit_) {
        // Electro Giant: an attacker within reach is zapped back.
        Entity* s = find_mut(shooter);
        const AreaDamageDef& r = target.def->unit.reflect;
        if (s && s->alive() && s->team != target.team && edge_gap(*s, target) <= r.radius) {
            const std::string by = hit_source_;
            hit_source_ = target.def_id;
            if (r.buff.kind != BuffKind::None) apply_buff(*s, r.buff, target.team);
            apply_damage(*s, r.damage, r.tower_damage_pct, kNoEntity, r.tower_damage);
            hit_source_ = by;
        }
    }
}

void Battle::apply_splash(Team source, Vec2 center, double radius, int damage, int tower_damage_pct,
                          const TargetRules& targets, double pushback, const BuffDef* buff, EntityId shooter,
                          int tower_abs) {
    TargetRules hit = targets;
    hit.buildings_only = false;  // splash damages everything of the right layer around the impact
    for (auto& t : entities_) {
        if (!t.alive() || t.team == source || !t.targetable() || !matches_targets(hit, t)) continue;
        if (distance(center, t.pos) - t.radius > radius) continue;
        if (buff) apply_buff(t, *buff, source);
        apply_damage(t, damage, tower_damage_pct, shooter, tower_abs);
        if (pushback > 0.0 && t.alive() && !t.is_building() && !t.ignore_pushback) knock_back(t, center, pushback);
    }
}

void Battle::knock_back(Entity& e, Vec2 from, double dist) const {
    const Vec2 d = e.pos - from;
    const double len = d.length();
    // Hit dead center: pushed back towards its own side.
    const Vec2 dir = len > 1e-9 ? d * (1.0 / len) : Vec2{0.0, e.team == Team::Blue ? -1.0 : 1.0};
    // Not at once: in the recordings a troop hit by a fireball flies its ~1 tile in ~0.3 s, slowing down (half of
    // it in the first 0.1 s, ~90% by 0.2 s), and walks on only then.
    e.knock = e.knock + dir * dist;
    e.knock_ms_left = kKnockMs;
    e.locked = false;  // the attack restarts from scratch (first_hit_ms wind-up)
    e.charging = false;
    e.walked = 0.0;
}

void Battle::knocked(Entity& e, int dt) const {
    // The speed falls off linearly to zero at the end: of what is left, this tick covers 1 - (after / before)^2.
    const double before = e.knock_ms_left;
    const double after = std::max(0, e.knock_ms_left - dt);
    const double share = 1.0 - (after / before) * (after / before);
    e.pos = e.pos + e.knock * share;
    e.knock = e.knock * (1.0 - share);
    e.knock_ms_left = static_cast<int>(after);
}

void Battle::resolve_collisions() {
    const ArenaDef& a = data_.arena;
    // Troops on the same layer (ground/air) push each other apart, heavier ones move less.
    for (size_t i = 0; i < entities_.size(); ++i) {
        Entity& p = entities_[i];
        if (p.is_building() || !p.alive() || p.host != kNoEntity) continue;
        for (size_t j = i + 1; j < entities_.size(); ++j) {
            Entity& q = entities_[j];
            if (q.is_building() || !q.alive() || q.host != kNoEntity || p.air != q.air) continue;
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
            } else if (p.team == q.team && !p.locked && !q.locked &&
                       std::abs(d.x) >= (p.radius + q.radius) * data_.rules.follow_width) {
                // Two of a side walking, one catching the other up off its line (not following it): in the game
                // it slides along the other's side and gets past in about a second (a mini pekka 0.8 off its
                // giant, video36 61-64 s); pushed apart straight along the line between them it would hardly.
                n = Vec2{n.x * data_.rules.pass_slide, n.y};
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
        if (!(e.def && e.def->unit.jump)) keep_out_of_river(e);  // (a jumper is in mid-air over the water)
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
    for (auto& e : entities_) {  // riders fall off a unit that died
        if (e.host == kNoEntity || !e.alive()) continue;
        const Entity* h = find(e.host);
        if (!h || !h->alive()) e.hp = 0;
    }
    for (auto& e : entities_) {
        if (e.alive()) continue;
        log(EventType::Death, e.team, e.id, e.def_id);
        if (record_events_) {
            events_.back().pos = e.pos;
            events_.back().play = e.play;
            events_.back().unit = e.unit;
        }
        if (e.def) {  // what it leaves behind
            const UnitDef& u = e.def->unit;
            queue_spawns(e.team, u.death_spawn, e.pos, Vec2{0.0, e.team == Team::Blue ? 1.0 : -1.0});
            if (u.death_damage.any()) {
                AreaEffect a;
                a.team = e.team;
                a.pos = e.pos;
                a.radius = u.death_damage.radius;
                a.delay_ms = u.death_damage.delay_ms;
                a.ms_left = data_.rules.tick_ms;
                a.damage = u.death_damage.damage;
                a.pushback = u.death_damage.pushback;
                a.tower_damage_pct = u.death_damage.tower_damage_pct;
                a.tower_damage = u.death_damage.tower_damage;
                a.buff = u.death_damage.buff;
                a.targets = TargetRules{true, true, false};
                a.source = e.def_id;
                areas_.push_back(a);  // (goes off in update_areas: this tick's deaths are all counted first)
            }
            if (u.death_rage.radius > 0.0) {
                AreaEffect a;
                a.team = e.team;
                a.pos = e.pos;
                a.radius = u.death_rage.radius;
                a.delay_ms = u.death_rage.delay_ms;
                a.ms_left = u.death_rage.ms;
                a.rage = u.death_rage.boost;
                a.source = e.def_id;
                areas_.push_back(a);
            }
            if (e.curse_ms > 0 && e.curse_unit && !e.is_building()) {  // cursed: back as a hog for the curser
                spawn_queue_.push_back({e.curse_team, e.curse_unit, e.pos, -1, kNoEntity, {0.0, 0.0}});
            }
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
    flush_spawns();
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
            if (e.is_tower()) weakest[idx(e.team)] = std::min(weakest[idx(e.team)], e.hp);
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
