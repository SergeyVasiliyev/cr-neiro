#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "crsim/game_data.h"

namespace cr {

constexpr int kHandSize = 4;
constexpr int kDeckSize = 8;

enum class EntityKind : uint8_t { Troop, PrincessTower, KingTower, Building };

struct Entity {
    EntityId id = kNoEntity;
    EntityKind kind = EntityKind::Troop;
    Team team = Team::Blue;
    std::string def_id;  // card id, or "princess_tower" / "king_tower"
    int card_index = -1;  // CardDef::index for troops and buildings, -1 for towers
    const CardDef* def = nullptr;  // troops and buildings: their card (GameData outlives every battle)
    Vec2 pos;
    double radius = 0.5;
    double mass = 1.0;
    double footprint = 0.0;  // towers: half side of the no-deploy square
    bool ignore_pushback = false;
    bool air = false;
    int hp = 0;
    int max_hp = 0;
    double speed = 0.0;  // tiles/s
    double sight_range = 0.0;
    AttackDef attack;

    int deploy_ms_left = 0;  // > 0: deploy timer running: on the field and targetable, but not acting yet
    int first_attack_delay_ms = 0;  // extra wind-up for the first attack after deploying
    EntityId target = kNoEntity;
    int lane = -1;        // troops on their own half: index of the bridge (lane) they march along
    bool locked = false;  // in range and attacking `target`; won't retarget until it's lost
    Vec2 heading{0.0, 0.0};  // troops: the way it walked last (unit length; zero before it first walks)
    Vec2 knock{0.0, 0.0};    // the part of a knockback still to come; while knock_ms_left > 0 it neither walks nor hits
    int knock_ms_left = 0;
    int attack_cd_ms = 0;
    bool active = true;  // king tower sleeps until a princess tower falls or it takes damage
    int play = -1;       // troops: which of its team's plays (PlayerState::plays index) it came from, -1 = none
    int unit = -1;       // ... and which unit of that play's group
    int spell_damage = 0;  // troops: HP taken from spells (rewards count it at the troop's full worth)

    // --- Classic Decks pool mechanics (UnitDef) ---
    int shield = 0;
    double decay_per_ms = 0.0;  // buildings: HP lost per ms (lifetime)
    double decay_acc = 0.0;
    int spawn_ms = -1;          // waves: time to the next one (-1: makes none)
    int stun_ms = 0;
    int slow_ms = 0;
    double slow = 0.0;
    int rage_ms = 0;
    double rage = 0.0;
    int curse_ms = 0;
    Team curse_team = Team::Blue;
    const CardDef* curse_unit = nullptr;
    double walked = 0.0;   // charge: tiles walked since it last stopped
    bool charging = false;
    int ramp_ms = 0;       // inferno: time on the current target
    EntityId host = kNoEntity;  // riders: the unit they sit on (out of reach while on it)
    Vec2 host_offset{0.0, 0.0};
    bool dashing = false;
    int dash_cd_ms = 0;
    int hook_ms = -1;      // >= 0: the hook is being thrown (ms left)
    int hook_cd_ms = 0;

    bool is_building() const { return kind != EntityKind::Troop; }
    bool is_tower() const { return kind == EntityKind::PrincessTower || kind == EntityKind::KingTower; }
    bool targetable() const { return host == kNoEntity && !dashing; }
    bool alive() const { return hp > 0; }
};

// A spell's lasting effect on the ground (Zap, Poison, Tornado, Lightning), a death explosion that comes late
// (Balloon's bomb), a rage aura (Lumberjack).
struct AreaEffect {
    Team team = Team::Blue;
    Vec2 pos;
    double radius = 0.0;
    int delay_ms = 0;       // before it starts
    int ms_left = 0;        // how long it lasts once started
    bool started = false;
    int damage = 0;         // once, when it starts
    double pushback = 0.0;
    int tower_damage_pct = 100;
    int tower_damage = -1;       // >= 0: a crown tower takes this instead
    int tower_tick_damage = -1;  // ... of a tick
    int elapsed_ms = 0;
    int duration_ms = 0;    // pulling and auras for this long; ticks and strikes may come later
    int ticks_left = -1;    // -1: tick while it lasts
    int tick_ms = 0, tick_left = 0, tick_damage = 0;
    double pull = 0.0;
    int max_targets = 0, hits = 0, hit_ms = 0, hit_left = 0;  // Lightning: the biggest targets one by one
    std::vector<EntityId> hit;
    BuffDef buff;
    double rage = 0.0;      // > 0: an aura speeding up friendly troops
    TargetRules targets;
    int card_index = -1;    // the spell card, for the observation (-1: not shown)
    std::string source;
    bool done = false;
};

// A spell rolling forward over the ground from where it was put down (The Log, Barbarian Barrel).
struct Roller {
    Team team = Team::Blue;
    Vec2 pos;
    double dir = 1.0;       // +1 up the arena (Blue), -1 down (Red)
    double half_width = 0.0, half_depth = 0.0;
    double left = 0.0;      // tiles still to roll
    double speed = 0.0;
    int damage = 0;
    int tower_damage_pct = 100;
    double pushback = 0.0;
    std::vector<EntityId> hit;
    SpawnDef spawn;         // left where it stops
    int card_index = -1;
    std::string source;
    bool done = false;
};

struct Projectile {
    Team team = Team::Blue;
    EntityId target = kNoEntity;  // kNoEntity for spells: they fly to a fixed point
    Vec2 pos;
    Vec2 aim;  // follows the target while it is alive
    double speed = 0.0;
    int damage = 0;
    int tower_damage_pct = 100;
    double splash_radius = 0.0;
    double pushback = 0.0;  // tiles
    int delay_ms = 0;       // spell waves wait at the king tower before launching
    int card_index = -1;    // spells: CardDef::index
    std::string source;     // who fired it (card id / tower), for the damage log
    TargetRules targets;
    bool done = false;
    EntityId shooter = kNoEntity;  // the unit that fired it (an Electro Giant zaps it back)
    BuffDef buff;           // what it does besides damage
    int heal = 0;           // heals friends within heal_radius where it lands (Heal Spirit)
    double heal_radius = 0.0;
    SpawnDef spawn;         // units it leaves where it lands (Goblin Barrel)

    bool is_spell() const { return card_index >= 0; }
};

enum class EventType : uint8_t {
    CardPlayed,
    Death,
    TowerDestroyed,
    KingActivated,
    OvertimeStarted,
    MatchEnded,
    Damage,  // only with set_record_damage(true): entity = who was hit, info = "<source>:<amount>"
};

const char* event_name(EventType type);

struct Event {
    int time_ms = 0;
    EventType type = EventType::CardPlayed;
    Team team = Team::Blue;
    EntityId entity = kNoEntity;
    std::string info;
    Vec2 pos;         // deaths: where it died
    int play = -1;    // deaths of troops: Entity::play / Entity::unit of the one that died
    int unit = -1;
};

using Deck = std::array<std::string, kDeckSize>;

// A played card waiting to materialise: paid for, but not on the field yet (nothing can hit it) and
// visible only to the player who played it. Troops get one entry per unit of the group.
struct PendingDeploy {
    Team team = Team::Blue;
    int card_index = -1;
    Vec2 pos;
    int unit = -1;  // index of the unit within the group; -1 for a spell
    int ms_left = 0;
    int play_index = -1;  // into PlayerState::plays
    int unit_card = -1;   // the card whose unit comes out, if not the played card's (Goblin Gang's spear goblins)
};

// A unit made during a tick (a wave, a death spawn, a spell's goblins), put on the field when the tick's updates
// are done (the entity list must not grow while they run).
struct QueuedSpawn {
    Team team = Team::Blue;
    const CardDef* def = nullptr;
    Vec2 pos;
    int deploy_ms = -1;      // -1: the unit's own deploy time
    EntityId host = kNoEntity;
    Vec2 host_offset{0.0, 0.0};
};

struct CardPlay {
    int card_index = -1;
    int time_ms = 0;       // when it was played
    int reveal_ms = -1;    // when the opponent could first see it (troop appeared / spell launched); -1 = not yet
};

struct PlayerState {
    double elixir = 0.0;
    double elixir_leaked = 0.0;  // total elixir lost while sitting at max
    std::array<std::string, kHandSize> hand;  // "" = empty slot waiting for the next card
    std::array<int, kHandSize> refill_ms_left{};  // > 0 while that slot is empty
    std::deque<std::string> queue;  // front = next card to enter the hand
    std::vector<CardPlay> plays;    // every card played, oldest first (public once revealed)
    int crowns = 0;
};

enum class Phase : uint8_t { Regulation, Overtime, Finished };

// One match. Fully deterministic for a given GameData, decks, seed and sequence of plays.
class Battle {
public:
    Battle(const GameData& data, const Deck& blue, const Deck& red, uint64_t seed = 0);

    // Empty string if the play is legal, otherwise the reason it is not.
    std::string check_play(Team team, int hand_slot, Vec2 pos) const;
    bool play(Team team, int hand_slot, Vec2 pos);
    bool can_deploy_at(Team team, Vec2 pos) const;  // troops
    bool can_cast_at(Vec2 pos) const;               // spells: anywhere on the arena
    bool can_dig_at(Vec2 pos) const;                // buildings put down anywhere (Goblin Drill): not on a building
    bool can_roll_from(Team team, Vec2 pos) const;  // own-half spells (The Log): our half and pockets, buildings too

    // Digital twin: repeats a play seen in the real game. The card (from the team's deck) is swapped
    // into the hand if the simulated cycle disagrees, a troop position the simulator would refuse is
    // moved to the nearest legal spot, and missing elixir is forgiven. Returns the elixir that was
    // missing (0 when the simulator agrees the play was affordable); throws if the card is not in the deck.
    // Without a position (seen played, but not where) only the cycle and the elixir change.
    // ago_ms: the play happened that long ago (an opponent's unit is only seen once it is on the field,
    // a spell once it lands), so its blueprint delay is already partly or fully over - and past that, a
    // troop's deploy timer has been running and a spell has been flying for the rest.
    double force_play(Team team, const std::string& card_id, std::optional<Vec2> pos, int ago_ms = 0);
    // Digital twin: make the simulated player hold what the screen shows. hand: 4 card ids ("" = empty
    // slot), next: the card shown as next ("" = unknown). The rest of the cycle keeps its simulated order.
    void set_hand(Team team, const std::array<std::string, kHandSize>& hand, const std::string& next);
    void set_elixir(Team team, double elixir);
    // What the other player can know: the team's cards still blueprints (none of their units out yet, a spell
    // not launched) are taken out, their elixir given back. For a lookahead from the opponent's view in the
    // simulator, as fair as the live twin, which can't see them either. Returns the elixir given back.
    double forget_blueprints(Team team);
    // Digital twin corrections from what the screen shows. Return false if there is no such entity.
    bool twin_move(EntityId id, Vec2 pos);         // a troop is seen somewhere else
    bool twin_set_hp(EntityId id, int hp);         // a tower (or troop) HP read from the screen
    bool twin_remove(EntityId id);                 // a troop the game no longer has
    // One unit the simulator missed (or killed too early): play / unit tag it as that play's unit.
    EntityId twin_spawn(Team team, const std::string& card_id, Vec2 pos, int play = -1, int unit = -1);
    // Digital twin: the real game decides when the match ends; the copy never finishes on its own
    // (its crowns can be off), it just keeps the clock, the elixir speed and the overtime.
    void set_follow_real(bool on) { follow_real_ = on; }

    // Advances the simulation by one tick (rules.tick_ms).
    void step();

    const GameData& data() const { return data_; }
    int time_ms() const { return time_ms_; }
    Phase phase() const { return phase_; }
    bool finished() const { return phase_ == Phase::Finished; }
    std::optional<Team> winner() const { return winner_; }  // nullopt on a draw
    double elixir_multiplier() const;

    const PlayerState& player(Team team) const { return players_[idx(team)]; }
    const std::vector<Entity>& entities() const { return entities_; }
    const std::vector<Projectile>& projectiles() const { return projectiles_; }
    const std::vector<AreaEffect>& areas() const { return areas_; }
    const std::vector<Roller>& rollers() const { return rollers_; }
    const std::vector<PendingDeploy>& pending() const { return pending_; }
    int deploy_delay_ms() const;  // blueprint delay for a card played now
    int card_refill_ms() const;   // how long the slot of a card played now stays empty
    const Entity* find(EntityId id) const;
    // Tower entity ids; find() returns nullptr once the tower is destroyed.
    EntityId king_tower(Team team) const { return towers_[idx(team)].king; }
    EntityId princess_tower(Team team, int side) const { return towers_[idx(team)].princess[side]; }

    void set_record_events(bool on) { record_events_ = on; }
    // Log every hit as a Damage event (for checking the simulated fighting against recordings).
    void set_record_damage(bool on) { record_damage_ = on; }
    const std::vector<Event>& events() const { return events_; }
    void clear_events() { events_.clear(); }

private:
    struct TowerIds {
        EntityId king = kNoEntity;
        EntityId princess[2] = {kNoEntity, kNoEntity};
    };

    Entity* find_mut(EntityId id);
    Vec2 to_absolute(Team team, Vec2 blue_pos) const;
    void start_play(Team team, int hand_slot, std::optional<Vec2> pos);  // no legality checks
    void spawn_towers(Team team);
    // A played card's unit `unit` of its group (`as`: the card whose unit it is, if not `card`).
    void spawn_unit(Team team, const CardDef& card, Vec2 pos, int unit, const CardDef* as = nullptr);
    // A unit on the field at pos (the end of entities_), with its riders queued.
    Entity& add_unit(Team team, const CardDef& def, Vec2 pos);
    void queue_spawns(Team team, const SpawnDef& s, Vec2 at, Vec2 forward);
    void flush_spawns();
    void materialize_pending(int dt);
    void cast_spell(Team team, const CardDef& card, Vec2 pos, int elapsed_ms = 0);

    void regen_elixir(int dt);
    void refill_hands(int dt);
    void update_timers(Entity& e, int dt);  // buffs, cooldowns, decay, waves
    void update_troop(Entity& e, int dt);
    void update_tower(Entity& e, int dt);
    void update_projectiles(int dt);
    void update_areas(int dt);
    void update_rollers(int dt);
    void resolve_collisions();
    void keep_out_of_river(Entity& e) const;
    void remove_dead();
    void update_phase();

    bool can_attack(const Entity& attacker, const Entity& target) const;
    EntityId acquire_troop_target(const Entity& e) const;
    bool on_own_bank(const Entity& e) const;
    void update_lane(Entity& e) const;
    // bridge >= 0 forces that bridge when the river has to be crossed.
    Vec2 next_waypoint(const Entity& e, Vec2 dst, int bridge = -1) const;
    // How far a walking troop may go this tick, held behind a slower troop of its side in its way.
    double follow_step(const Entity& e, Vec2 waypoint, double step, int dt) const;
    // Where to head instead, when a building (not the target) is in the way to the waypoint.
    Vec2 steer_around_buildings(const Entity& e, Vec2 waypoint, const Entity* target) const;
    void attack_tick(Entity& e, Entity& target, int dt);
    void fire(Entity& e, Entity& target);
    // shooter: the unit that dealt it (kNoEntity: a spell or an aura), for a reflected zap. tower_abs >= 0: what a crown
    // tower takes instead (tower_damage_pct then unused).
    void apply_damage(Entity& target, int damage, int tower_damage_pct, EntityId shooter = kNoEntity, int tower_abs = -1);
    void apply_splash(Team source, Vec2 center, double radius, int damage, int tower_damage_pct,
                      const TargetRules& targets, double pushback = 0.0, const BuffDef* buff = nullptr,
                      EntityId shooter = kNoEntity, int tower_abs = -1);
    void apply_area(Team source, Vec2 center, const AreaDamageDef& a);  // ground and air, towers too
    void apply_buff(Entity& e, const BuffDef& b, Team source);
    void heal_around(Team team, Vec2 center, double radius, int amount);
    double speed_factor(const Entity& e) const;   // slow and rage on walking
    double attack_factor(const Entity& e) const;  // ... and on the attack's timers
    void knock_back(Entity& e, Vec2 from, double distance) const;
    void knocked(Entity& e, int dt) const;  // a tick of a knockback in progress
    void activate_king(Entity& king);
    void finish(std::optional<Team> winner);

    void log(EventType type, Team team, EntityId entity, std::string info = {});

    const GameData& data_;
    std::mt19937_64 rng_;
    int time_ms_ = 0;
    Phase phase_ = Phase::Regulation;
    std::optional<Team> winner_;
    bool king_destroyed_ = false;
    bool follow_real_ = false;  // digital twin: never finish on our own
    bool record_damage_ = false;
    std::string hit_source_;    // who is dealing the damage being applied
    bool spell_hit_ = false;    // ... and whether it is a spell

    std::array<PlayerState, 2> players_;
    std::array<TowerIds, 2> towers_;
    std::vector<Entity> entities_;
    std::vector<Projectile> projectiles_;
    std::vector<AreaEffect> areas_;
    std::vector<Roller> rollers_;
    std::vector<QueuedSpawn> spawn_queue_;
    std::vector<PendingDeploy> pending_;
    EntityId next_id_ = 0;

    bool record_events_ = true;
    std::vector<Event> events_;
};

}  // namespace cr
