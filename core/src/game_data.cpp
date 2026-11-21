#include "crsim/game_data.h"

#include <fstream>
#include <stdexcept>

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

CardDef parse_card(const std::string& id, const json& j) {
    CardDef c;
    c.id = id;
    c.name = j.at("name").get<std::string>();
    c.rarity = j.value("rarity", "");
    c.elixir = j.at("elixir").get<int>();
    c.unsupported = j.value("unsupported", std::vector<std::string>{});
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
        return c;
    }
    if (type != "troop") throw std::runtime_error("card " + id + ": unsupported type " + type);
    c.count = j.value("count", 1);
    c.spawn_spread = j.value("spawn_spread", 0.6);
    c.spawn_stagger_ms = j.value("spawn_stagger_ms", 0);

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
    for (auto& [id, card] : d.cards) {  // std::map iterates in sorted order
        card.index = static_cast<int>(d.card_ids.size());
        d.card_ids.push_back(id);
    }
    const double s = d.rules.speed_scale;
    for (auto& [id, card] : d.cards) {
        card.unit.speed *= s;
        card.unit.attack.projectile_speed *= s;
        card.spell.projectile_speed *= s;
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
