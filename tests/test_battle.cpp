// Minimal self-contained tests: crsim_tests [data_dir]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "crsim/battle.h"
#include "crsim/bots.h"

namespace {

int g_failed = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
            ++g_failed;                                                   \
        }                                                                 \
    } while (0)

using namespace cr;

const Deck kMixed = {"knight", "archers", "giant", "musketeer", "minions", "mini_pekka", "valkyrie", "skeletons"};
const Deck kKnights = {"knight", "knight", "knight", "knight", "knight", "knight", "knight", "knight"};

void run_ms(Battle& b, int ms) {
    for (int t = 0; t < ms && !b.finished(); t += b.data().rules.tick_ms) b.step();
}

// Steps until every played card has left its blueprint phase and is on the field.
void settle(Battle& b) {
    for (int t = 0; t < 200 && !b.pending().empty(); ++t) b.step();
}

const Entity* princess(const Battle& b, Team team, bool left) {
    for (const auto& e : b.entities()) {
        if (e.kind == EntityKind::PrincessTower && e.team == team && (e.pos.x < b.data().arena.width / 2) == left) {
            return &e;
        }
    }
    return nullptr;
}

int troop_count(const Battle& b, Team team) {
    int n = 0;
    for (const auto& e : b.entities()) n += !e.is_building() && e.team == team;
    return n;
}

void test_elixir(const GameData& data) {
    Battle b(data, kMixed, kMixed, 1);
    const double start = data.rules.start_elixir;
    CHECK(std::abs(b.player(Team::Blue).elixir - start) < 1e-9);
    run_ms(b, data.rules.elixir_regen_ms);
    CHECK(std::abs(b.player(Team::Blue).elixir - (start + 1.0)) < 1e-6);
    run_ms(b, 60000);
    CHECK(b.player(Team::Blue).elixir == data.rules.max_elixir);
}

// x1 for 2:00, x2 until 4:00 (into overtime), x3 for the last minute of overtime.
void test_elixir_schedule(const GameData& data) {
    Battle b(data, kMixed, kMixed, 1);
    CHECK(b.elixir_multiplier() == 1.0);
    run_ms(b, 130000);
    CHECK(b.elixir_multiplier() == 2.0);
    run_ms(b, 60000);
    CHECK(b.phase() == Phase::Overtime);
    CHECK(b.elixir_multiplier() == 2.0);
    run_ms(b, 60000);
    CHECK(b.phase() == Phase::Overtime);
    CHECK(b.elixir_multiplier() == 3.0);
}

void test_card_cycle(const GameData& data) {
    Battle b(data, kMixed, kMixed, 7);
    const PlayerState before = b.player(Team::Blue);
    const int cost = data.card(before.hand[0]).elixir;
    CHECK(b.play(Team::Blue, 0, {9.0, 10.0}));
    const PlayerState& after = b.player(Team::Blue);
    CHECK(after.hand[0].empty());  // the slot waits for the next card
    CHECK(!b.check_play(Team::Blue, 0, {9.0, 10.0}).empty());
    CHECK(after.queue.back() == before.hand[0]);
    CHECK(std::abs(after.elixir - (before.elixir - cost)) < 1e-9);
    run_ms(b, data.rules.card_refill_ms[0] - 50);
    CHECK(b.player(Team::Blue).hand[0].empty());
    run_ms(b, 50);
    CHECK(b.player(Team::Blue).hand[0] == before.queue.front());
    CHECK(b.player(Team::Blue).queue.size() == before.queue.size());
}

void test_deploy_zones(const GameData& data) {
    Battle b(data, kMixed, kMixed, 1);
    CHECK(b.can_deploy_at(Team::Blue, {9.0, 10.0}));
    CHECK(!b.can_deploy_at(Team::Blue, {9.0, 20.0}));  // enemy half
    CHECK(!b.can_deploy_at(Team::Blue, {9.0, 16.0}));  // river
    CHECK(!b.can_deploy_at(Team::Blue, {9.0, 3.0}));   // on own king tower
    CHECK(b.can_deploy_at(Team::Red, {9.0, 22.0}));
    CHECK(!b.can_deploy_at(Team::Red, {9.0, 10.0}));
}

void test_not_enough_elixir(const GameData& data) {
    Battle b(data, kKnights, kKnights, 1);
    for (int slot = 0; slot < kHandSize && b.player(Team::Blue).elixir >= 3.0; ++slot) {
        CHECK(b.play(Team::Blue, slot, {9.0, 10.0}));
    }
    CHECK(!b.check_play(Team::Blue, 0, {9.0, 10.0}).empty());  // under 3 elixir left, knight costs 3
}

// A lone knight crosses the bridge, damages the princess tower and dies to it.
void test_knight_attacks_tower(const GameData& data) {
    Battle b(data, kKnights, kKnights, 1);
    CHECK(b.play(Team::Blue, 0, {3.5, 13.0}));
    run_ms(b, 40000);
    const Entity* tower = princess(b, Team::Red, true);
    CHECK(tower != nullptr);
    if (tower) {
        CHECK(tower->hp < tower->max_hp);
        std::printf("  knight vs princess tower: tower hp %d / %d\n", tower->hp, tower->max_hp);
    }
    CHECK(troop_count(b, Team::Blue) == 0);
}

// Mirrored pushes in opposite lanes must deal identical damage.
void test_mirror_symmetry(const GameData& data) {
    Battle b(data, kKnights, kKnights, 1);
    CHECK(b.play(Team::Blue, 0, {3.5, 13.0}));
    CHECK(b.play(Team::Red, 0, {14.5, data.arena.height - 13.0}));
    run_ms(b, 40000);
    const Entity* red_left = princess(b, Team::Red, true);
    const Entity* blue_right = princess(b, Team::Blue, false);
    CHECK(red_left && blue_right);
    if (red_left && blue_right) {
        CHECK(red_left->hp < red_left->max_hp);
        CHECK(red_left->hp == blue_right->hp);
    }
}

// Two knights dropped on the same tile push each other apart.
void test_collisions_separate_units(const GameData& data) {
    Battle b(data, kKnights, kKnights, 1);
    CHECK(b.play(Team::Blue, 0, {9.0, 5.5}));
    run_ms(b, data.rules.elixir_regen_ms + 100);
    CHECK(b.play(Team::Blue, 0, {9.0, 5.5}));
    run_ms(b, 1500);
    std::vector<const Entity*> knights;
    for (const auto& e : b.entities()) {
        if (!e.is_building()) knights.push_back(&e);
    }
    CHECK(knights.size() == 2);
    if (knights.size() == 2) CHECK(distance(knights[0]->pos, knights[1]->pos) > 0.8);
}

// Over whole bot matches, no ground troop ever stands in the water.
void test_river_is_impassable(const GameData& data) {
    const ArenaDef& a = data.arena;
    for (uint64_t seed = 1; seed <= 3; ++seed) {
        Battle b(data, kMixed, kMixed, seed);
        auto blue = make_bot(BotKind::Random, seed * 2);
        auto red = make_bot(BotKind::Random, seed * 2 + 1);
        bool wet = false;
        for (int tick = 0; !b.finished() && tick < 4000; ++tick) {
            if (tick % 10 == 0) act_simultaneously(b, *blue, *red);
            b.step();
            for (const auto& e : b.entities()) {
                if (e.is_building() || e.air || e.pos.y <= a.river_min_y || e.pos.y >= a.river_max_y) continue;
                bool on_bridge = false;
                for (double bx : a.bridge_x) on_bridge = on_bridge || std::abs(e.pos.x - bx) <= a.bridge_half_width + 1e-9;
                wet = wet || !on_bridge;
            }
        }
        CHECK(!wet);
    }
}

// Follows the troop spawned last until it crosses the river; returns the x where it crossed.
double crossing_x(Battle& b, EntityId* target_after) {
    settle(b);
    EntityId id = kNoEntity;
    for (const auto& e : b.entities()) id = std::max(id, e.id);
    for (int t = 0; t < 600; ++t) {  // 30 s
        b.step();
        const Entity* e = b.find(id);
        if (!e) return -1.0;
        if (e->pos.y > b.data().arena.river_max_y + 0.5) {
            *target_after = e->target;
            return e->pos.x;
        }
    }
    return -1.0;
}

// With Red's right princess tower down, a troop placed on Blue's right goes over the right bridge
// to the king; one on the left still takes the left lane to the left princess.
void test_lane_after_princess_falls(const GameData& data) {
    Battle b(data, kKnights, kKnights, 1);
    const EntityId right_princess = b.princess_tower(Team::Red, 1);
    for (int t = 0; t < 4000 && b.find(right_princess); ++t) {
        if (b.player(Team::Blue).elixir >= 3.0) b.play(Team::Blue, 0, {14.5, 13.5});
        b.step();
    }
    CHECK(b.find(right_princess) == nullptr);
    for (int t = 0; t < 2000 && troop_count(b, Team::Blue) > 0; ++t) b.step();  // let the attackers die
    CHECK(troop_count(b, Team::Blue) == 0);

    run_ms(b, 3000);
    CHECK(b.play(Team::Blue, 0, {15.5, 8.5}));
    EntityId target = kNoEntity;
    const double right_x = crossing_x(b, &target);
    CHECK(right_x > 9.0);
    CHECK(target == b.king_tower(Team::Red));
    std::printf("  lane test: right troop crossed at x=%.1f\n", right_x);

    for (int t = 0; t < 2000 && troop_count(b, Team::Blue) > 0; ++t) b.step();
    run_ms(b, 3000);
    CHECK(b.play(Team::Blue, 0, {2.5, 8.5}));
    const double left_x = crossing_x(b, &target);
    CHECK(left_x > 0.0 && left_x < 9.0);
    CHECK(target == b.princess_tower(Team::Red, 0));
}

// Flyers follow lanes too: with Red's right princess down, minions from Blue's right corner fly to
// the king, minions from the left go for the left princess.
void test_air_lanes_after_princess_falls(const GameData& data) {
    const Deck minions = {"minions", "minions", "minions", "minions", "minions", "minions", "minions", "minions"};
    auto targets_from = [&](Vec2 pos) {
        Battle b(data, minions, minions, 1);
        const EntityId right_princess = b.princess_tower(Team::Red, 1);
        for (int t = 0; t < 6000 && b.find(right_princess); ++t) {
            if (b.player(Team::Blue).elixir >= 3.0) b.play(Team::Blue, 0, {14.5, 13.5});
            b.step();
        }
        for (int t = 0; t < 4000 && troop_count(b, Team::Blue) > 0; ++t) b.step();
        run_ms(b, 4000);
        b.play(Team::Blue, 0, pos);
        settle(b);
        std::vector<EntityId> ids;
        for (const auto& e : b.entities()) {
            if (!e.is_building() && e.team == Team::Blue) ids.push_back(e.id);
        }
        run_ms(b, 8000);  // well on their way, not yet near the towers
        std::vector<EntityId> targets;
        for (EntityId id : ids) {
            if (const Entity* e = b.find(id)) targets.push_back(e->target);
        }
        return std::make_pair(targets, b.king_tower(Team::Red));
    };
    const auto [right, king] = targets_from({16.5, 1.5});
    CHECK(right.size() == 3);
    for (EntityId t : right) CHECK(t == king);
    Battle ref(data, minions, minions, 1);
    const auto [left, king2] = targets_from({2.5, 1.5});
    CHECK(left.size() == 3);
    for (EntityId t : left) CHECK(t == ref.princess_tower(Team::Red, 0));
}

Deck all_of(const char* id) { return {id, id, id, id, id, id, id, id}; }

// Places one troop for `team` and waits until it has deployed; returns its id.
EntityId place(Battle& b, Team team, Vec2 pos) {
    CHECK(b.play(team, 0, pos));
    settle(b);
    EntityId id = kNoEntity;
    for (const auto& e : b.entities()) id = std::max(id, e.id);
    return id;
}

const Entity* entity(const Battle& b, EntityId id) { return b.find(id); }

// Fireball: flies from the king tower at its speed, 570 damage at level 11, a fixed share to towers.
void test_fireball_damage_and_flight(const GameData& data) {
    const SpellDef& fb = data.card("fireball").spell;
    Battle b(data, all_of("musketeer"), all_of("fireball"), 1);
    const EntityId musk = place(b, Team::Blue, {9.5, 12.5});
    CHECK(b.play(Team::Red, 0, {9.5, 12.5}));
    const Vec2 king_red{data.arena.king_pos.x, data.arena.height - data.arena.king_pos.y};
    const double flight_ms = b.deploy_delay_ms() + distance(king_red, {9.5, 12.5}) / fb.projectile_speed * 1000.0;
    run_ms(b, static_cast<int>(flight_ms) - 100);
    CHECK(entity(b, musk) && entity(b, musk)->hp == entity(b, musk)->max_hp);  // still in the air
    run_ms(b, 200);
    CHECK(entity(b, musk) && entity(b, musk)->hp == entity(b, musk)->max_hp - fb.damage);

    Battle t(data, all_of("fireball"), all_of("fireball"), 1);
    const Entity* tower = t.find(t.princess_tower(Team::Red, 0));
    const Vec2 tower_pos = tower->pos;
    CHECK(t.play(Team::Blue, 0, tower_pos));
    run_ms(t, 4000);
    tower = t.find(t.princess_tower(Team::Red, 0));
    CHECK(tower && tower->hp == tower->max_hp - fb.damage * fb.tower_damage_pct / 100);
}

// Arrows: three waves; minions (230 HP) survive the first and die to the full volley.
void test_arrows_waves(const GameData& data) {
    const SpellDef& ar = data.card("arrows").spell;
    CHECK(ar.waves == 3);
    Battle b(data, all_of("minions"), all_of("arrows"), 1);
    CHECK(b.play(Team::Blue, 0, {9.5, 12.5}));
    settle(b);
    const int before = troop_count(b, Team::Blue);
    CHECK(b.play(Team::Red, 0, {9.5, 12.5}));
    int min_hp_after_first = 1 << 30;
    for (int t = 0; t < 120; ++t) {  // 6 s
        b.step();
        for (const auto& e : b.entities()) {
            if (!e.is_building() && e.team == Team::Blue && e.hp < e.max_hp) min_hp_after_first = std::min(min_hp_after_first, e.hp);
        }
        if (min_hp_after_first < (1 << 30)) break;
    }
    CHECK(before == 3);
    CHECK(min_hp_after_first == data.card("minions").unit.hitpoints - ar.damage);  // first wave only
    run_ms(b, 2000);
    CHECK(troop_count(b, Team::Blue) == 0);
}

// Fireball knocks troops back a tile and restarts their attack; the Giant ignores knockback.
void test_fireball_knockback(const GameData& data) {
    Battle b(data, all_of("knight"), all_of("fireball"), 1);
    const EntityId knight = place(b, Team::Blue, {3.5, 13.5});
    for (int t = 0; t < 600 && !(entity(b, knight) && entity(b, knight)->locked); ++t) b.step();  // reaches the tower
    CHECK(entity(b, knight) && entity(b, knight)->locked);
    const Vec2 before = entity(b, knight)->pos;
    const Vec2 impact{before.x, before.y - 1.0};  // just behind it: pushes it towards the tower
    // Red's fireball needs elixir; wait for it, then throw.
    for (int t = 0; t < 200 && b.player(Team::Red).elixir < 4.0; ++t) b.step();
    CHECK(b.play(Team::Red, 0, impact));
    bool moved = false, unlocked = false;
    for (int t = 0; t < 200 && !moved; ++t) {
        b.step();
        const Entity* k = entity(b, knight);
        if (!k) break;
        if (distance(k->pos, before) > 0.6) {
            moved = true;
            unlocked = !k->locked;
        }
    }
    CHECK(moved);
    CHECK(unlocked);

    // Giant: same walk with and without being fireballed.
    Battle g(data, all_of("giant"), all_of("fireball"), 1);
    Battle control(data, all_of("giant"), all_of("fireball"), 1);
    const EntityId giant = place(g, Team::Blue, {9.5, 12.5});
    place(control, Team::Blue, {9.5, 12.5});
    run_ms(g, 1000);
    run_ms(control, 1000);
    const Vec2 gpos = entity(g, giant)->pos;
    CHECK(g.play(Team::Red, 0, {gpos.x + 1.0, gpos.y}));
    run_ms(g, 4000);
    run_ms(control, 4000);
    const Entity* gi = entity(g, giant);
    const Entity* gc = entity(control, giant);
    CHECK(gi && gi->hp < gi->max_hp);
    CHECK(gi && gc && distance(gi->pos, gc->pos) < 1e-6);
}

// Spells hit troops that are still deploying.
void test_spell_hits_deploying_troops(const GameData& data) {
    Battle b(data, all_of("knight"), all_of("arrows"), 1);
    const Vec2 king_red{data.arena.king_pos.x, data.arena.height - data.arena.king_pos.y};
    const double flight_ms = distance(king_red, {9.5, 14.5}) / data.card("arrows").spell.projectile_speed * 1000.0;
    CHECK(flight_ms + 100 < data.card("knight").unit.deploy_time_ms);
    CHECK(b.play(Team::Blue, 0, {9.5, 14.5}));  // both blueprints wait the same delay
    CHECK(b.play(Team::Red, 0, {9.5, 14.5}));
    run_ms(b, b.deploy_delay_ms() + static_cast<int>(flight_ms) + 60);
    const Entity* k = nullptr;
    for (const auto& e : b.entities()) {
        if (!e.is_building() && e.team == Team::Blue) k = &e;
    }
    CHECK(k && k->deploy_ms_left > 0 && k->hp < k->max_hp);
}

// A played card is a blueprint first: elixir is spent and the card cycles at once, but the troop
// appears only after the deploy delay; group units appear one after another.
void test_blueprint_delay(const GameData& data) {
    Battle b(data, all_of("minions"), all_of("minions"), 1);
    const double elixir = b.player(Team::Blue).elixir;
    const int delay = b.deploy_delay_ms();
    CHECK(delay == data.rules.deploy_delay_ms[0]);
    CHECK(b.play(Team::Blue, 0, {9.5, 10.5}));
    CHECK(std::abs(b.player(Team::Blue).elixir - (elixir - 3)) < 1e-9);
    CHECK(b.pending().size() == 3);
    CHECK(b.player(Team::Blue).plays.size() == 1 && b.player(Team::Blue).plays[0].reveal_ms < 0);
    const int stagger = data.card("minions").spawn_stagger_ms;
    std::vector<int> appeared;
    for (int t = 0; t < 60; ++t) {
        b.step();
        while (static_cast<int>(appeared.size()) < troop_count(b, Team::Blue)) appeared.push_back(b.time_ms());
    }
    CHECK(appeared.size() == 3);
    if (appeared.size() == 3) {
        CHECK(appeared[0] == delay);
        CHECK(appeared[1] == delay + stagger);
        CHECK(appeared[2] == delay + 2 * stagger);
    }
    CHECK(b.player(Team::Blue).plays[0].reveal_ms == delay);
}

// A troop on its deploy timer is already on the field: enemies pick it as their target.
void test_deploying_troops_are_targeted(const GameData& data) {
    Battle b(data, all_of("archers"), all_of("knight"), 1);
    CHECK(b.play(Team::Blue, 0, {9.5, 14.5}));
    run_ms(b, 900);
    CHECK(b.play(Team::Red, 0, {9.5, 17.5}));  // appears while the archers finish deploying
    bool targeted_while_deploying = false;
    for (int t = 0; t < 80; ++t) {
        b.step();
        for (const auto& archer : b.entities()) {
            if (archer.is_building() || archer.team != Team::Blue) continue;
            const Entity* k = b.find(archer.target);
            if (k && k->team == Team::Red && !k->is_building() && k->deploy_ms_left > 0) targeted_while_deploying = true;
        }
    }
    CHECK(targeted_while_deploying);
}

// Game data DeployDelay: musketeer's first shot comes later than its later shots' rhythm suggests.
void test_first_attack_delay(const GameData& data) {
    const UnitDef& musk = data.card("musketeer").unit;
    CHECK(musk.first_attack_delay_ms > 0);
    CHECK(data.card("knight").unit.first_attack_delay_ms == 0);
}

void test_heuristic_bot_beats_random(const GameData& data) {
    int wins = 0;
    const int games = 60;
    for (int g = 0; g < games; ++g) {
        Battle b(data, kMixed, kMixed, 100 + g);
        b.set_record_events(false);
        auto blue = make_bot(BotKind::Heuristic, g);
        auto red = make_bot(BotKind::Random, 1000 + g);
        for (int tick = 0; !b.finished(); ++tick) {
            if (tick % 10 == 0) act_simultaneously(b, *blue, *red);
            b.step();
        }
        wins += b.winner() == Team::Blue;
    }
    std::printf("  heuristic vs random: %d / %d wins\n", wins, games);
    CHECK(wins > games * 3 / 4);
}

void test_match_ends(const GameData& data) {
    Battle b(data, kMixed, kMixed, 3);
    run_ms(b, data.rules.regulation_ms + data.rules.overtime_ms + 1000);
    CHECK(b.finished());
    CHECK(!b.winner().has_value());  // nobody played: draw on equal tower HP
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : "data";
    GameData data;
    try {
        data = GameData::load(data_dir);
    } catch (const std::exception& e) {
        std::printf("failed to load game data: %s\n", e.what());
        return 1;
    }

    test_elixir(data);
    test_elixir_schedule(data);
    test_blueprint_delay(data);
    test_deploying_troops_are_targeted(data);
    test_first_attack_delay(data);
    test_card_cycle(data);
    test_deploy_zones(data);
    test_not_enough_elixir(data);
    test_knight_attacks_tower(data);
    test_mirror_symmetry(data);
    test_collisions_separate_units(data);
    test_river_is_impassable(data);
    test_lane_after_princess_falls(data);
    test_air_lanes_after_princess_falls(data);
    test_fireball_damage_and_flight(data);
    test_arrows_waves(data);
    test_fireball_knockback(data);
    test_spell_hits_deploying_troops(data);
    test_heuristic_bot_beats_random(data);
    test_match_ends(data);

    if (g_failed) {
        std::printf("%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("all tests passed\n");
    return 0;
}
