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

enum class EntityKind : uint8_t { Troop, PrincessTower, KingTower };

struct Entity {
    EntityId id = kNoEntity;
    EntityKind kind = EntityKind::Troop;
    Team team = Team::Blue;
    std::string def_id;  // card id, or "princess_tower" / "king_tower"
    int card_index = -1;  // CardDef::index for troops, -1 for towers
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
    int attack_cd_ms = 0;
    bool active = true;  // king tower sleeps until a princess tower falls or it takes damage
    int play = -1;       // troops: which of its team's plays (PlayerState::plays index) it came from, -1 = none
    int unit = -1;       // ... and which unit of that play's group
    int spell_damage = 0;  // troops: HP taken from spells (rewards count it at the troop's full worth)

    bool is_building() const { return kind != EntityKind::Troop; }
    bool alive() const { return hp > 0; }
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
    void spawn_unit(Team team, const CardDef& card, Vec2 pos, int unit);
    void materialize_pending(int dt);
    void cast_spell(Team team, const CardDef& card, Vec2 pos, int elapsed_ms = 0);

    void regen_elixir(int dt);
    void refill_hands(int dt);
    void update_troop(Entity& e, int dt);
    void update_tower(Entity& e, int dt);
    void update_projectiles(int dt);
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
    void attack_tick(Entity& e, Entity& target, int dt);
    void fire(Entity& e, Entity& target);
    void apply_damage(Entity& target, int damage, int tower_damage_pct);
    void apply_splash(Team source, Vec2 center, double radius, int damage, int tower_damage_pct,
                      const TargetRules& targets, double pushback = 0.0);
    void knock_back(Entity& e, Vec2 from, double distance) const;
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
    std::vector<PendingDeploy> pending_;
    EntityId next_id_ = 0;

    bool record_events_ = true;
    std::vector<Event> events_;
};

}  // namespace cr
