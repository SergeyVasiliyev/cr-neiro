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

// What a hit, a spell or an aura does to a unit besides damage (game data BUFF): a stun (Zap, the Electro Wizard,
// the Ice Spirit's freeze: no walking, no attacking, the attack's wind-up starts over), a slow (Giant Snowball,
// Poison: speed and attack speed down by `amount`), a curse (Mother Witch: a unit that dies cursed comes back as
// `unit` for the curser's side).
enum class BuffKind : uint8_t { None, Stun, Slow, Curse };
struct BuffDef {
    BuffKind kind = BuffKind::None;
    int ms = 0;
    double amount = 0.0;
    std::string unit;
};

// Units made by a unit or a spell: a building's waves (Tombstone), a troop's (Night Witch), what comes out when it
// dies (Golem's golemites), what a spell drops (Goblin Barrel), riders that sit on a unit (Goblin Giant).
struct SpawnDef {
    std::string unit;      // the card whose unit comes out ("" = none)
    int count = 0;
    double radius = 0.0;   // tiles around the spot
    int interval_ms = 0;   // waves: time between them
    int start_ms = 0;      // waves: the first this long after deploying
    int deploy_ms = -1;    // their deploy timer; -1 = the unit's own
    bool any() const { return !unit.empty() && count > 0; }
};

// Damage (and maybe a push and a buff) to every enemy within a radius: a death explosion (Golem), the Electro
// Wizard's zap when it lands, the Electro Giant's reflected zap.
struct AreaDamageDef {
    int damage = 0;
    double radius = 0.0;
    double pushback = 0.0;
    int tower_damage_pct = 100;
    int tower_damage = -1;  // >= 0: what a crown tower takes instead (the game files' TowerDamage; the Electro Giant's
                            // zap: 97 off a tower in the recordings)
    int delay_ms = 0;  // after the death (Balloon's bomb)
    BuffDef buff;
    bool any() const { return radius > 0.0; }
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

    // --- mechanics of the Classic Decks pool (tools/import_pool.py) ---
    bool building = false;  // a building put down by a card (Tombstone, Inferno Tower)
    int lifetime_ms = 0;    // buildings: lose all their HP evenly over this time
    int shield = 0;         // shield HP, taken before the HP; a hit that breaks it does no more (Dark Prince, Recruits)
    struct { double distance = 0.0; double speed_mult = 1.0; int damage = 0; } charge;  // after walking `distance`
                            // uninterrupted it runs speed_mult faster and its next hit does `damage`
    bool jump = false;      // crosses the river anywhere (Royal Hogs, Dark Prince)
    bool kamikaze = false;  // dies by its hit (spirits, Battle Ram)
    struct { double min = 0.0, max = 0.0, speed = 0.0; int damage = 0, cooldown_ms = 0; } dash;  // Bandit: a target
                            // min..max away is dashed at, untouchable on the way, and the dash's hit does `damage`
    struct { int damage2 = 0, damage3 = 0, stage_ms = 0; } ramp;  // inferno: on one target the damage goes up
                            // to damage2 after stage_ms, damage3 after twice that
    int multi_targets = 1;  // hits this many enemies at once (Electro Wizard)
    BuffDef hit_buff;       // what its hits do besides damage
    AreaDamageDef deploy_damage;  // when it lands (Electro Wizard, Goblin Drill)
    AreaDamageDef reflect;  // hit by an attacker within the radius, it zaps it back (Electro Giant)
    SpawnDef spawn;         // waves of units
    SpawnDef death_spawn;   // units it leaves behind
    SpawnDef riders;        // units sitting on it, out of reach, falling off when it dies (Goblin Giant)
    AreaDamageDef death_damage;
    struct { double radius = 0.0, boost = 0.0; int ms = 0, delay_ms = 0; } death_rage;  // Lumberjack: a rage
                            // aura (speed and attack speed up by `boost` for the friends in it)
    struct { double min = 0.0, max = 0.0; int load_ms = 0; } hook;  // Fisherman: a target min..max away is hooked
                            // (after load_ms) and pulled to him; a building pulls him to it instead
    int pellets = 0, pellet_damage = 0;  // Hunter: a spread of pellets, all of them hit close by
    double attack_pushback = 0.0;  // its hits knock back (Sparky)
    int heal = 0;                  // its hit heals friends within heal_radius (Heal Spirit)
    double heal_radius = 0.0;
};

enum class CardType { Troop, Spell, Building };

// Projectile: flies from the caster's king tower to the chosen tile and hits every enemy within the radius, in one
// or more waves (Arrows, Fireball). Area: an effect on the spot at once (Zap), for a while (Poison, Tornado) or
// hitting the biggest targets one by one (Lightning). Roll: rolls forward from the spot over `range` tiles hitting
// everything in its way once (The Log, Barbarian Barrel).
enum class SpellKind : uint8_t { Projectile, Area, Roll };
struct SpellDef {
    int damage = 0;            // per wave (area: once, at the start; roll: once per enemy)
    double radius = 0.0;       // tiles (roll: half its width)
    int tower_damage_pct = 100;
    int waves = 1;
    int wave_interval_ms = 0;
    double projectile_speed = 10.0;  // tiles/s (roll: its rolling speed)
    double pushback = 0.0;     // tiles; knocked-back troops also restart their attack
    TargetRules targets;
    SpellKind kind = SpellKind::Projectile;
    BuffDef buff;              // on everything it hits
    SpawnDef spawn;            // units it leaves where it lands (Goblin Barrel) or stops (Barbarian Barrel)
    int tower_damage = -1;     // >= 0: what a crown tower takes instead of `damage` (the files' TowerDamage)
    int duration_ms = 0;       // area: how long it lasts (pulling, slowing)
    int first_ms = 0;          // area: its first tick or strike this long after it starts (the buff's own interval:
                               // Poison 1.25 s, Tornado 0.6 s; Lightning's first bolt ~1.1 s - the recordings)
    int tick_ms = 0;           // area: then damage every tick_ms ...
    int tick_count = 0;        // ... this many times (0: while it lasts) ...
    int tick_damage = 0;       // ... this much
    int tower_tick_damage = -1;  // >= 0: a crown tower's share of a tick instead (Tornado: 25 in the recordings)
    double pull = 0.0;         // area: enemy troops in it are pulled to its centre at this speed (tiles/s; Tornado)
    int max_targets = 0;       // area: hits the biggest targets in it one by one, every hit_ms (Lightning)
    int hit_ms = 0;
    bool biggest = false;
    double depth = 0.0;        // roll: half its depth
    double range = 0.0;        // roll: how far it rolls
};

// How a group lands: around the spot (default), in a row `width` wide (Royal Hogs), or spread over the whole width
// of the arena at the spot's depth (Royal Recruits).
enum class Formation : uint8_t { Ring, Line, FullLane };

struct CardDef {
    std::string id;
    int index = -1;  // position in GameData::card_ids (stable: alphabetical within index_group)
    std::string name;
    std::string rarity;
    int elixir = 0;
    CardType type = CardType::Troop;
    int count = 1;              // how many units one card spawns
    double spawn_spread = 0.6;  // tiles, radius of the spawn area when count > 1
    int spawn_stagger_ms = 0;   // each next unit of a group appears this much later
    UnitDef unit;               // CardType::Troop and CardType::Building
    SpellDef spell;             // CardType::Spell
    SpawnDef second;            // a second kind of unit in the group (Goblin Gang's spear goblins)
    Formation formation = Formation::Ring;
    double formation_width = 0.0;
    bool playable = true;       // false: only made by other cards (golemites, lava pups, ...)
    bool anywhere = false;      // a building put down anywhere, like a spell (Goblin Drill digs its way there)
    bool own_half = false;      // a spell put down like a troop, on our half (and the pockets) only, on buildings too
                                // (The Log, Barbarian Barrel: SpellAsDeploy in the files; aimed at the enemy's half in
                                // c013 the log came down at our river's edge and rolled up from there)
    double dig_speed = 0.0;     // ... from the king tower, at this speed (tiles/s), after the blueprint delay
    int index_group = 0;        // cards added later sort after the earlier ones: the nets' tokens stay the same
    // Mechanics of this card the engine does not simulate yet (from the importer).
    std::vector<std::string> unsupported;

    bool fully_supported() const { return unsupported.empty(); }
    bool placed_anywhere() const { return (type == CardType::Spell && !own_half) || anywhere; }
};

struct TowerDef {
    int hitpoints = 0;
    double radius = 1.0;     // collision radius, tiles
    double footprint = 1.5;  // half side of the square nobody can deploy on, tiles
    // King tower: after it wakes up (a princess tower fell, or it was hit) the king stands up and takes
    // out his cannon before the first shot can come.
    int activation_ms = 0;
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
    // Everything moves this much faster in the real game than "speed / 60 tiles per second" from the
    // game files (walking and projectiles alike), measured from recordings (tools/measure_speeds.py).
    double speed_scale = 1.0;
    // Same-side crowding (battle.cpp follow_step and resolve_collisions; rules.json "movement"): a troop walking
    // into the back of a slower one of its side follows it when off its line by less than follow_width of the
    // two radii and at most follow_gap tiles past touching; further off it slides along the other's side, the
    // sideways part of the push pass_slide times larger (tools/crowd_check.py measures it on the recordings).
    double follow_width = 0.4;
    double follow_gap = 0.3;
    double pass_slide = 2.0;
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
