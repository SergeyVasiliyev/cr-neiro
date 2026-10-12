#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "crsim/types.h"

namespace cr {

// What an attacker is allowed to hit.
struct TargetRules {
    bool ground = true;
    bool air = false;
    bool buildings_only = false;
};

struct AttackDef {
    int damage = 0;
    int hit_speed_ms = 1000;
    int first_hit_ms = 500;         // delay before the first hit after locking onto a target
    double range = 0.0;             // tiles, edge to edge
    double projectile_speed = 0.0;  // tiles/s; 0 = instant hit (melee)
    double splash_radius = 0.0;     // tiles; 0 = single target
    bool splash_around_self = false;  // e.g. Valkyrie: splash centered on the attacker
    int tower_damage_pct = 100;     // damage dealt to crown towers, % of normal
    TargetRules targets;
};

struct UnitDef {
    int hitpoints = 0;
    double speed = 0.0;        // tiles/s (data files store tiles/min, like the game)
    double sight_range = 5.5;  // tiles: how far the unit looks for targets
    double radius = 0.5;       // collision radius, tiles
    double mass = 1.0;         // heavier units push lighter ones aside
    bool ignore_pushback = false;  // not knocked back by spells (Giant, Golem, ...)
    bool air = false;
    int deploy_time_ms = 1000;
    int first_attack_delay_ms = 0;  // game data "DeployDelay": extra wait before the very first attack
    AttackDef attack;
};

enum class CardType { Troop, Spell };  // building cards come later

// A projectile spell (Arrows, Fireball): flies from the caster's king tower to the chosen tile and
// deals the same damage to every enemy within the radius, in one or more waves.
struct SpellDef {
    int damage = 0;            // per wave
    double radius = 0.0;       // tiles
    int tower_damage_pct = 100;
    int waves = 1;
    int wave_interval_ms = 0;
    double projectile_speed = 10.0;  // tiles/s
    double pushback = 0.0;     // tiles; knocked-back troops also restart their attack
    TargetRules targets;
};

struct CardDef {
    std::string id;
    int index = -1;  // position in GameData::card_ids (stable, alphabetical)
    std::string name;
    std::string rarity;
    int elixir = 0;
    CardType type = CardType::Troop;
    int count = 1;              // how many units one card spawns
    double spawn_spread = 0.6;  // tiles, radius of the spawn area when count > 1
    int spawn_stagger_ms = 0;   // each next unit of a group appears this much later
    UnitDef unit;               // CardType::Troop
    SpellDef spell;             // CardType::Spell
    // Mechanics of this card the engine does not simulate yet (from the importer).
    std::vector<std::string> unsupported;

    bool fully_supported() const { return unsupported.empty(); }
};

struct TowerDef {
    int hitpoints = 0;
    double radius = 1.0;     // collision radius, tiles
    double footprint = 1.5;  // half side of the square nobody can deploy on, tiles
    AttackDef attack;
};

// Positions are given for Blue; Red is mirrored across the horizontal center line.
struct ArenaDef {
    double width = 18.0;
    double height = 32.0;
    double river_min_y = 15.0;
    double river_max_y = 17.0;
    std::vector<double> bridge_x;
    double bridge_half_width = 1.0;
    Vec2 king_pos;
    Vec2 princess_pos[2];       // [0] = left (small x), [1] = right
    double pocket_depth = 4.0;  // how far past the river you can deploy after taking a princess tower
};

struct RulesDef {
    int tick_ms = 50;
    int regulation_ms = 180000;
    int double_elixir_from_ms = 120000;
    int overtime_ms = 120000;
    int triple_elixir_from_ms = 240000;  // elapsed time; 0 = never
    int elixir_regen_ms = 2800;  // time to gain 1 elixir at x1 speed
    // A played card sits on the field as an untouchable "blueprint" this long before the troop
    // appears / the spell is launched; indexed by elixir speed x1, x2, x3.
    int deploy_delay_ms[3] = {1000, 1000, 1000};
    // After a card is played its hand slot stays empty this long before the next card arrives;
    // indexed by elixir speed x1, x2, x3 (NextSpellCooldownMS in battle_timelines.csv).
    int card_refill_ms[3] = {1000, 500, 350};
    double start_elixir = 5.0;
    double max_elixir = 10.0;
};

struct GameData {
    std::map<std::string, CardDef> cards;
    std::vector<std::string> card_ids;  // sorted; card_ids[c.index] == c.id
    TowerDef princess_tower;
    TowerDef king_tower;
    ArenaDef arena;
    RulesDef rules;

    // Loads rules.json, arena.json, towers.json and cards.json from dir.
    // Throws std::runtime_error on missing files or malformed data.
    static GameData load(const std::filesystem::path& dir);

    // Throws std::out_of_range for unknown ids.
    const CardDef& card(const std::string& id) const;
};

}  // namespace cr
