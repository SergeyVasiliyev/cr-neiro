#include "crsim/game_data.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

namespace cr {
namespace {

using nlohmann::json;

json read_json(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    try {
        return json::parse(in);
    } catch (const json::exception& e) {
        throw std::runtime_error(path.string() + ": " + e.what());
    }
}

Vec2 parse_vec2(const json& j) { return {j.at(0).get<double>(), j.at(1).get<double>()}; }

// The game stores movement speed in tiles per minute (Slow 45, Medium 60, Fast 90, Very Fast 120).
double speed_from_game_units(double tiles_per_minute) { return tiles_per_minute / 60.0; }

TargetRules parse_targets(const json& j) {
    TargetRules t{false, false, false};
    for (const auto& item : j) {
        const auto name = item.get<std::string>();
        if (name == "ground") t.ground = true;
        else if (name == "air") t.air = true;
        else if (name == "buildings") t.buildings_only = true;
        else throw std::runtime_error("unknown target type: " + name);
    }
    return t;
}

AttackDef parse_attack(const json& j) {
    AttackDef a;
    a.damage = j.at("damage").get<int>();
    a.hit_speed_ms = j.at("hit_speed_ms").get<int>();
    a.first_hit_ms = j.value("first_hit_ms", a.hit_speed_ms / 2);
    a.range = j.at("range").get<double>();
    a.projectile_speed = j.value("projectile_speed", 0.0);
    a.splash_radius = j.value("splash_radius", 0.0);
    a.splash_around_self = j.value("splash_around_self", false);
    a.tower_damage_pct = j.value("tower_damage_pct", 100);
    a.targets = parse_targets(j.at("targets"));
    return a;
}

TowerDef parse_tower(const json& j) {
    TowerDef t;
    t.hitpoints = j.at("hitpoints").get<int>();
    t.radius = j.at("radius").get<double>();
    t.footprint = j.value("footprint", t.radius);
    t.activation_ms = j.value("activation_ms", 0);
    t.attack = parse_attack(j.at("attack"));
    return t;
}

BuffDef parse_buff(const json& j) {
    BuffDef b;
    const auto kind = j.at("kind").get<std::string>();
    if (kind == "stun") b.kind = BuffKind::Stun;
    else if (kind == "slow") b.kind = BuffKind::Slow;
    else if (kind == "curse") b.kind = BuffKind::Curse;
    else throw std::runtime_error("unknown buff kind: " + kind);
    b.ms = j.at("ms").get<int>();
    b.amount = j.value("amount", 0.0);
    b.unit = j.value("unit", "");
    return b;
}

SpawnDef parse_spawn(const json& j) {
    SpawnDef s;
    s.unit = j.at("unit").get<std::string>();
    s.count = j.at("count").get<int>();
    s.radius = j.value("radius", 0.0);
    s.interval_ms = j.value("interval_ms", 0);
    s.start_ms = j.value("start_ms", 0);
    s.deploy_ms = j.value("deploy_ms", -1);
    return s;
}

AreaDamageDef parse_area(const json& j) {
    AreaDamageDef a;
    a.damage = j.value("damage", 0);
    a.radius = j.at("radius").get<double>();
    a.pushback = j.value("pushback", 0.0);
    a.tower_damage_pct = j.value("tower_damage_pct", 100);
    a.tower_damage = j.value("tower_damage", -1);
    a.delay_ms = j.value("delay_ms", 0);
    if (j.contains("buff")) a.buff = parse_buff(j.at("buff"));
    return a;
}

void parse_unit_mechanics(UnitDef& u, const json& j) {
    u.building = j.value("building", false);
    u.lifetime_ms = j.value("lifetime_ms", 0);
    u.shield = j.value("shield", 0);
    if (j.contains("charge")) {
        const json& c = j.at("charge");
        u.charge.distance = c.at("distance").get<double>();
        u.charge.speed_mult = c.value("speed_mult", 1.0);
        u.charge.damage = c.at("damage").get<int>();
    }
    u.jump = j.value("jump", false);
    u.kamikaze = j.value("kamikaze", false);
    if (j.contains("dash")) {
        const json& d = j.at("dash");
        u.dash.min = d.at("min").get<double>();
        u.dash.max = d.at("max").get<double>();
        u.dash.speed = d.at("speed").get<double>();
        u.dash.damage = d.at("damage").get<int>();
        u.dash.cooldown_ms = d.value("cooldown_ms", 0);
    }
    if (j.contains("ramp")) {
        const json& r = j.at("ramp");
        u.ramp.damage2 = r.at("damage2").get<int>();
        u.ramp.damage3 = r.at("damage3").get<int>();
        u.ramp.stage_ms = r.at("stage_ms").get<int>();
    }
    u.multi_targets = j.value("multi_targets", 1);
    if (j.contains("hit_buff")) u.hit_buff = parse_buff(j.at("hit_buff"));
    if (j.contains("deploy_damage")) u.deploy_damage = parse_area(j.at("deploy_damage"));
    if (j.contains("reflect")) u.reflect = parse_area(j.at("reflect"));
    if (j.contains("spawn")) u.spawn = parse_spawn(j.at("spawn"));
    if (j.contains("death_spawn")) u.death_spawn = parse_spawn(j.at("death_spawn"));
    if (j.contains("riders")) u.riders = parse_spawn(j.at("riders"));
    if (j.contains("death_damage")) u.death_damage = parse_area(j.at("death_damage"));
    if (j.contains("death_rage")) {
        const json& r = j.at("death_rage");
        u.death_rage.radius = r.at("radius").get<double>();
        u.death_rage.boost = r.at("boost").get<double>();
        u.death_rage.ms = r.at("ms").get<int>();
        u.death_rage.delay_ms = r.value("delay_ms", 0);
    }
    if (j.contains("hook")) {
        const json& h = j.at("hook");
        u.hook.min = h.at("min").get<double>();
        u.hook.max = h.at("max").get<double>();
        u.hook.load_ms = h.at("load_ms").get<int>();
    }
    if (j.contains("pellets")) {
        u.pellets = j.at("pellets").at("count").get<int>();
        u.pellet_damage = j.at("pellets").at("damage").get<int>();
    }
    u.attack_pushback = j.value("attack_pushback", 0.0);
    if (j.contains("heal")) {
        u.heal = j.at("heal").at("amount").get<int>();
        u.heal_radius = j.at("heal").at("radius").get<double>();
    }
}

CardDef parse_card(const std::string& id, const json& j) {
    CardDef c;
    c.id = id;
    c.name = j.at("name").get<std::string>();
    c.rarity = j.value("rarity", "");
    c.elixir = j.at("elixir").get<int>();
    c.unsupported = j.value("unsupported", std::vector<std::string>{});
    c.playable = j.value("playable", true);
    c.index_group = j.value("index_group", 0);
    const auto type = j.at("type").get<std::string>();
    if (type == "spell") {
        c.type = CardType::Spell;
        const json& s = j.at("spell");
        c.spell.damage = s.at("damage").get<int>();
        c.spell.radius = s.at("radius").get<double>();
        c.spell.tower_damage_pct = s.value("tower_damage_pct", 100);
        c.spell.waves = s.value("waves", 1);
        c.spell.wave_interval_ms = s.value("wave_interval_ms", 0);
        c.spell.projectile_speed = s.at("projectile_speed").get<double>();
        c.spell.pushback = s.value("pushback", 0.0);
        c.spell.targets = parse_targets(s.at("targets"));
        const auto kind = s.value("kind", std::string("projectile"));
        if (kind == "area") c.spell.kind = SpellKind::Area;
        else if (kind == "roll") c.spell.kind = SpellKind::Roll;
        else if (kind != "projectile") throw std::runtime_error("unknown spell kind: " + kind);
        if (s.contains("buff")) c.spell.buff = parse_buff(s.at("buff"));
        if (s.contains("spawn")) c.spell.spawn = parse_spawn(s.at("spawn"));
        c.spell.tower_damage = s.value("tower_damage", -1);
        c.spell.duration_ms = s.value("duration_ms", 0);
        c.spell.first_ms = s.value("first_ms", 0);
        c.spell.tick_ms = s.value("tick_ms", 0);
        c.spell.tick_count = s.value("tick_count", 0);
        c.spell.tick_damage = s.value("tick_damage", 0);
        c.spell.tower_tick_damage = s.value("tower_tick_damage", -1);
        c.own_half = j.value("own_half", false);
        c.spell.pull = s.value("pull", 0.0);
        c.spell.max_targets = s.value("max_targets", 0);
        c.spell.hit_ms = s.value("hit_ms", 0);
        c.spell.biggest = s.value("biggest", false);
        c.spell.depth = s.value("depth", 0.0);
        c.spell.range = s.value("range", 0.0);
        return c;
    }
    if (type == "building") c.type = CardType::Building;
    else if (type != "troop") throw std::runtime_error("card " + id + ": unsupported type " + type);
    c.count = j.value("count", 1);
    c.spawn_spread = j.value("spawn_spread", 0.6);
    c.spawn_stagger_ms = j.value("spawn_stagger_ms", 0);
    if (j.contains("second")) c.second = parse_spawn(j.at("second"));
    if (j.contains("formation")) {
        const auto kind = j.at("formation").at("kind").get<std::string>();
        if (kind == "line") c.formation = Formation::Line;
        else if (kind == "full_lane") c.formation = Formation::FullLane;
        else throw std::runtime_error("unknown formation: " + kind);
        c.formation_width = j.at("formation").at("width").get<double>();
    }
    c.anywhere = j.value("anywhere", false);
    c.dig_speed = j.value("dig_speed", 0.0);

    const json& u = j.at("unit");
    c.unit.hitpoints = u.at("hitpoints").get<int>();
    c.unit.speed = speed_from_game_units(u.at("speed").get<double>());
    c.unit.sight_range = u.value("sight_range", 5.5);
    c.unit.radius = u.value("radius", 0.5);
    c.unit.mass = u.value("mass", 1.0);
    c.unit.ignore_pushback = u.value("ignore_pushback", false);
    c.unit.air = u.value("air", false);
    c.unit.deploy_time_ms = u.value("deploy_time_ms", 1000);
    c.unit.first_attack_delay_ms = u.value("first_attack_delay_ms", 0);
    c.unit.attack = parse_attack(u.at("attack"));
    parse_unit_mechanics(c.unit, u);
    if (c.type == CardType::Building) c.unit.building = true;
    return c;
}

}  // namespace

GameData GameData::load(const std::filesystem::path& dir) {
    GameData d;

    const json rules = read_json(dir / "rules.json");
    d.rules.tick_ms = rules.at("tick_ms").get<int>();
    d.rules.regulation_ms = rules.at("regulation_ms").get<int>();
    d.rules.double_elixir_from_ms = rules.at("double_elixir_from_ms").get<int>();
    d.rules.overtime_ms = rules.at("overtime_ms").get<int>();
    d.rules.triple_elixir_from_ms = rules.at("triple_elixir_from_ms").get<int>();
    d.rules.elixir_regen_ms = rules.at("elixir_regen_ms").get<int>();
    const json& delay = rules.at("deploy_delay_ms");
    d.rules.deploy_delay_ms[0] = delay.at("x1").get<int>();
    d.rules.deploy_delay_ms[1] = delay.at("x2").get<int>();
    d.rules.deploy_delay_ms[2] = delay.at("x3").get<int>();
    const json& refill = rules.at("card_refill_ms");
    d.rules.card_refill_ms[0] = refill.at("x1").get<int>();
    d.rules.card_refill_ms[1] = refill.at("x2").get<int>();
    d.rules.card_refill_ms[2] = refill.at("x3").get<int>();
    d.rules.start_elixir = rules.at("start_elixir").get<double>();
    d.rules.max_elixir = rules.at("max_elixir").get<double>();
    d.rules.speed_scale = rules.value("speed_scale", 1.0);
    if (rules.contains("movement")) {
        const json& m = rules.at("movement");
        d.rules.follow_width = m.value("follow_width", d.rules.follow_width);
        d.rules.follow_gap = m.value("follow_gap", d.rules.follow_gap);
        d.rules.pass_slide = m.value("pass_slide", d.rules.pass_slide);
    }

    const json arena = read_json(dir / "arena.json");
    d.arena.width = arena.at("width").get<double>();
    d.arena.height = arena.at("height").get<double>();
    d.arena.river_min_y = arena.at("river_y").at(0).get<double>();
    d.arena.river_max_y = arena.at("river_y").at(1).get<double>();
    d.arena.bridge_x = arena.at("bridge_x").get<std::vector<double>>();
    d.arena.bridge_half_width = arena.at("bridge_half_width").get<double>();
    d.arena.king_pos = parse_vec2(arena.at("king_pos"));
    d.arena.princess_pos[0] = parse_vec2(arena.at("princess_pos").at(0));
    d.arena.princess_pos[1] = parse_vec2(arena.at("princess_pos").at(1));
    d.arena.pocket_depth = arena.at("pocket_depth").get<double>();

    // Hand-made corrections on top of the imported game data survive re-imports after a patch:
    // overrides.json = {"cards": {"fireball": {"spell": {"projectile_speed": 15}}}, "towers": {...}}.
    json overrides = json::object();
    if (std::filesystem::exists(dir / "overrides.json")) overrides = read_json(dir / "overrides.json");

    json towers = read_json(dir / "towers.json");
    if (overrides.contains("towers")) towers.merge_patch(overrides["towers"]);
    d.princess_tower = parse_tower(towers.at("princess"));
    d.king_tower = parse_tower(towers.at("king"));

    json cards = read_json(dir / "cards.json");
    if (overrides.contains("cards")) cards.merge_patch(overrides["cards"]);
    for (const auto& [id, card] : cards.items()) {
        if (id.starts_with("_")) continue;  // "_comment" and similar metadata keys
        try {
            d.cards.emplace(id, parse_card(id, card));
        } catch (const json::exception& e) {
            throw std::runtime_error("card " + id + ": " + e.what());
        }
    }
    // Alphabetical within each index group: cards added later (index_group 1, the Classic Decks pool) come after
    // the earlier ones, whose indices - the nets' tokens - stay what they were.
    std::vector<std::pair<int, std::string>> order;
    for (const auto& [id, card] : d.cards) order.emplace_back(card.index_group, id);
    std::sort(order.begin(), order.end());
    for (const auto& [group, id] : order) {
        d.cards.at(id).index = static_cast<int>(d.card_ids.size());
        d.card_ids.push_back(id);
    }
    auto check_unit = [&](const std::string& owner, const std::string& unit) {
        if (!unit.empty() && !d.cards.contains(unit)) throw std::runtime_error("card " + owner + ": unknown unit " + unit);
    };
    const double s = d.rules.speed_scale;
    for (auto& [id, card] : d.cards) {
        card.unit.speed *= s;
        card.unit.attack.projectile_speed *= s;
        card.unit.dash.speed *= s;
        card.spell.projectile_speed *= s;
        card.dig_speed *= s;
        for (const SpawnDef* sp : {&card.unit.spawn, &card.unit.death_spawn, &card.unit.riders, &card.second, &card.spell.spawn}) {
            check_unit(id, sp->unit);
        }
        check_unit(id, card.unit.hit_buff.unit);
    }
    d.princess_tower.attack.projectile_speed *= s;
    d.king_tower.attack.projectile_speed *= s;
    return d;
}

const CardDef& GameData::card(const std::string& id) const {
    auto it = cards.find(id);
    if (it == cards.end()) throw std::out_of_range("unknown card: " + id);
    return it->second;
}

}  // namespace cr
