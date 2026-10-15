// Python module `crsim`: thin wrapper over the C++ core.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "crsim/batch_env.h"
#include "crsim/battle.h"
#include "crsim/bots.h"
#include "crsim/rl.h"

namespace py = pybind11;
using namespace cr;

namespace {

// numpy view over a BatchEnv buffer; `owner` keeps the env alive while the array exists.
template <typename T>
py::array_t<T> view(std::vector<T>& buf, std::vector<py::ssize_t> shape, py::handle owner) {
    return py::array_t<T>(shape, buf.data(), owner);
}

BotKind bot_from(const py::handle& h) {
    if (py::isinstance<py::str>(h)) return bot_kind_from_string(h.cast<std::string>());
    return h.cast<BotKind>();
}

// Encodes one player's observation of a single Battle into fresh numpy arrays.
py::dict encode_battle(const Battle& b, Team team, bool observe_enemy_elixir, bool key_spots) {
    py::array_t<float> global(rl::kGlobalFeatures);
    py::array_t<float> entities({rl::kMaxEntities, rl::kEntityFeatures});
    py::array_t<int32_t> token(rl::kMaxEntities);
    py::array_t<uint8_t> mask(rl::kMaxEntities);
    py::array_t<int32_t> hand(rl::kHandTokens);
    py::array_t<int32_t> history(rl::kHistory);
    py::array_t<float> history_age(rl::kHistory);
    py::array_t<uint8_t> amask(rl::kNumActions);
    rl::ObsConfig cfg;
    cfg.observe_enemy_elixir = observe_enemy_elixir;
    cfg.key_spots = key_spots;
    rl::encode(b, team, cfg,
               {global.mutable_data(), entities.mutable_data(), token.mutable_data(), mask.mutable_data(),
                hand.mutable_data(), history.mutable_data(), history_age.mutable_data(), amask.mutable_data()});
    py::dict d;
    d["global"] = global;
    d["entities"] = entities;
    d["entity_token"] = token;
    d["entity_mask"] = mask;
    d["hand"] = hand;
    d["enemy_history"] = history;
    d["enemy_history_age"] = history_age;
    d["action_mask"] = amask;
    return d;
}

}  // namespace

PYBIND11_MODULE(crsim, m) {
    m.doc() = "Headless Clash Royale battle simulator";

    py::enum_<Team>(m, "Team").value("BLUE", Team::Blue).value("RED", Team::Red);
    py::enum_<EntityKind>(m, "EntityKind")
        .value("TROOP", EntityKind::Troop)
        .value("PRINCESS_TOWER", EntityKind::PrincessTower)
        .value("KING_TOWER", EntityKind::KingTower);
    py::enum_<Phase>(m, "Phase")
        .value("REGULATION", Phase::Regulation)
        .value("OVERTIME", Phase::Overtime)
        .value("FINISHED", Phase::Finished);
    py::enum_<EventType>(m, "EventType")
        .value("CARD_PLAYED", EventType::CardPlayed)
        .value("DEATH", EventType::Death)
        .value("TOWER_DESTROYED", EventType::TowerDestroyed)
        .value("KING_ACTIVATED", EventType::KingActivated)
        .value("OVERTIME_STARTED", EventType::OvertimeStarted)
        .value("MATCH_ENDED", EventType::MatchEnded)
        .value("DAMAGE", EventType::Damage);

    py::class_<Vec2>(m, "Vec2")
        .def(py::init<>())
        .def(py::init([](double x, double y) { return Vec2{x, y}; }))
        // Lets Python pass plain (x, y) tuples wherever a Vec2 is expected.
        .def(py::init([](const py::tuple& t) {
            if (t.size() != 2) throw py::value_error("Vec2 needs (x, y)");
            return Vec2{t[0].cast<double>(), t[1].cast<double>()};
        }))
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("y", &Vec2::y)
        .def("__repr__", [](const Vec2& v) { return "Vec2(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")"; });
    py::implicitly_convertible<py::tuple, Vec2>();

    py::class_<TargetRules>(m, "TargetRules")
        .def_readonly("ground", &TargetRules::ground)
        .def_readonly("air", &TargetRules::air)
        .def_readonly("buildings_only", &TargetRules::buildings_only);
    py::class_<AttackDef>(m, "AttackDef")
        .def_readonly("damage", &AttackDef::damage)
        .def_readonly("hit_speed_ms", &AttackDef::hit_speed_ms)
        .def_readonly("first_hit_ms", &AttackDef::first_hit_ms)
        .def_readonly("range", &AttackDef::range)
        .def_readonly("projectile_speed", &AttackDef::projectile_speed)
        .def_readonly("splash_radius", &AttackDef::splash_radius)
        .def_readonly("tower_damage_pct", &AttackDef::tower_damage_pct)
        .def_readonly("targets", &AttackDef::targets);
    py::class_<UnitDef>(m, "UnitDef")
        .def_readonly("hitpoints", &UnitDef::hitpoints)
        .def_readonly("speed", &UnitDef::speed)
        .def_readonly("sight_range", &UnitDef::sight_range)
        .def_readonly("radius", &UnitDef::radius)
        .def_readonly("mass", &UnitDef::mass)
        .def_readonly("ignore_pushback", &UnitDef::ignore_pushback)
        .def_readonly("air", &UnitDef::air)
        .def_readonly("deploy_time_ms", &UnitDef::deploy_time_ms)
        .def_readonly("first_attack_delay_ms", &UnitDef::first_attack_delay_ms)
        .def_readonly("attack", &UnitDef::attack);
    py::enum_<CardType>(m, "CardType").value("TROOP", CardType::Troop).value("SPELL", CardType::Spell);
    py::class_<SpellDef>(m, "SpellDef")
        .def_readonly("damage", &SpellDef::damage)
        .def_readonly("radius", &SpellDef::radius)
        .def_readonly("tower_damage_pct", &SpellDef::tower_damage_pct)
        .def_readonly("waves", &SpellDef::waves)
        .def_readonly("wave_interval_ms", &SpellDef::wave_interval_ms)
        .def_readonly("projectile_speed", &SpellDef::projectile_speed)
        .def_readonly("pushback", &SpellDef::pushback)
        .def_readonly("targets", &SpellDef::targets);
    py::class_<CardDef>(m, "CardDef")
        .def_readonly("id", &CardDef::id)
        .def_readonly("type", &CardDef::type)
        .def_readonly("spell", &CardDef::spell)
        .def_property_readonly("is_spell", [](const CardDef& c) { return c.type == CardType::Spell; })
        .def_readonly("index", &CardDef::index)
        .def_readonly("name", &CardDef::name)
        .def_readonly("rarity", &CardDef::rarity)
        .def_readonly("elixir", &CardDef::elixir)
        .def_readonly("count", &CardDef::count)
        .def_readonly("spawn_stagger_ms", &CardDef::spawn_stagger_ms)
        .def_readonly("unit", &CardDef::unit)
        .def_readonly("unsupported", &CardDef::unsupported)
        .def_property_readonly("fully_supported", &CardDef::fully_supported);
    py::class_<ArenaDef>(m, "ArenaDef")
        .def_readonly("width", &ArenaDef::width)
        .def_readonly("height", &ArenaDef::height)
        .def_readonly("river_min_y", &ArenaDef::river_min_y)
        .def_readonly("river_max_y", &ArenaDef::river_max_y)
        .def_readonly("bridge_x", &ArenaDef::bridge_x)
        .def_readonly("bridge_half_width", &ArenaDef::bridge_half_width)
        .def_readonly("pocket_depth", &ArenaDef::pocket_depth);
    py::class_<RulesDef>(m, "RulesDef")
        .def_readonly("tick_ms", &RulesDef::tick_ms)
        .def_readonly("regulation_ms", &RulesDef::regulation_ms)
        .def_readonly("double_elixir_from_ms", &RulesDef::double_elixir_from_ms)
        .def_readonly("overtime_ms", &RulesDef::overtime_ms)
        .def_readonly("elixir_regen_ms", &RulesDef::elixir_regen_ms)
        .def_readonly("max_elixir", &RulesDef::max_elixir);
    py::class_<GameData>(m, "GameData")
        .def_static("load", [](const std::string& dir) { return GameData::load(dir); }, py::arg("dir"))
        .def_readonly("cards", &GameData::cards)
        .def_readonly("card_ids", &GameData::card_ids)
        .def_readonly("arena", &GameData::arena)
        .def_readonly("rules", &GameData::rules)
        .def("card", &GameData::card, py::return_value_policy::reference_internal);

    py::class_<Entity>(m, "Entity")
        .def_readonly("id", &Entity::id)
        .def_readonly("kind", &Entity::kind)
        .def_readonly("team", &Entity::team)
        .def_readonly("def_id", &Entity::def_id)
        .def_readonly("card_index", &Entity::card_index)
        .def_readonly("pos", &Entity::pos)
        .def_readonly("radius", &Entity::radius)
        .def_readonly("mass", &Entity::mass)
        .def_readonly("footprint", &Entity::footprint)
        .def_readonly("air", &Entity::air)
        .def_readonly("hp", &Entity::hp)
        .def_readonly("max_hp", &Entity::max_hp)
        .def_readonly("deploy_ms_left", &Entity::deploy_ms_left)
        .def_readonly("target", &Entity::target)
        .def_readonly("locked", &Entity::locked)
        .def_readonly("active", &Entity::active)
        .def_readonly("play", &Entity::play)
        .def_readonly("unit", &Entity::unit)
        .def_property_readonly("is_building", &Entity::is_building);
    py::class_<Projectile>(m, "Projectile")
        .def_readonly("team", &Projectile::team)
        .def_readonly("pos", &Projectile::pos)
        .def_readonly("aim", &Projectile::aim)
        .def_readonly("splash_radius", &Projectile::splash_radius)
        .def_readonly("delay_ms", &Projectile::delay_ms)
        .def_readonly("card_index", &Projectile::card_index)
        .def_property_readonly("is_spell", &Projectile::is_spell);
    py::class_<PendingDeploy>(m, "PendingDeploy")
        .def_readonly("team", &PendingDeploy::team)
        .def_readonly("card_index", &PendingDeploy::card_index)
        .def_readonly("pos", &PendingDeploy::pos)
        .def_readonly("unit", &PendingDeploy::unit)
        .def_readonly("ms_left", &PendingDeploy::ms_left)
        .def_readonly("play_index", &PendingDeploy::play_index);
    py::class_<CardPlay>(m, "CardPlay")
        .def_readonly("card_index", &CardPlay::card_index)
        .def_readonly("time_ms", &CardPlay::time_ms)
        .def_readonly("reveal_ms", &CardPlay::reveal_ms);
    py::class_<Event>(m, "Event")
        .def_readonly("time_ms", &Event::time_ms)
        .def_readonly("type", &Event::type)
        .def_readonly("team", &Event::team)
        .def_readonly("entity", &Event::entity)
        .def_readonly("info", &Event::info)
        .def_readonly("pos", &Event::pos)
        .def_readonly("play", &Event::play)
        .def_readonly("unit", &Event::unit);
    py::class_<PlayerState>(m, "PlayerState")
        .def_readonly("elixir", &PlayerState::elixir)
        .def_readonly("elixir_leaked", &PlayerState::elixir_leaked)
        .def_readonly("plays", &PlayerState::plays)
        .def_readonly("hand", &PlayerState::hand)
        .def_readonly("refill_ms_left", &PlayerState::refill_ms_left)
        .def_property_readonly("queue", [](const PlayerState& p) {
            return std::vector<std::string>(p.queue.begin(), p.queue.end());
        })
        .def_readonly("crowns", &PlayerState::crowns);

    py::class_<Battle>(m, "Battle")
        // Battle keeps a reference to GameData: keep the Python object alive as long as the battle.
        .def(py::init<const GameData&, const Deck&, const Deck&, uint64_t>(), py::arg("data"), py::arg("blue_deck"),
             py::arg("red_deck"), py::arg("seed") = 0, py::keep_alive<1, 2>())
        .def("check_play", &Battle::check_play, py::arg("team"), py::arg("hand_slot"), py::arg("pos"))
        .def("play", &Battle::play, py::arg("team"), py::arg("hand_slot"), py::arg("pos"))
        .def("can_deploy_at", &Battle::can_deploy_at, py::arg("team"), py::arg("pos"))
        .def("can_cast_at", &Battle::can_cast_at, py::arg("pos"))
        .def("force_play", &Battle::force_play, py::arg("team"), py::arg("card_id"), py::arg("pos"),
             py::arg("ago_ms") = 0,
             "Digital twin: repeat a real play (pos None = position unknown, only cycle and elixir; ago_ms = "
             "it happened that long ago); returns the elixir the simulator thinks was missing")
        .def("set_hand", &Battle::set_hand, py::arg("team"), py::arg("hand"), py::arg("next"),
             "Digital twin: hand = 4 card ids ('' = empty), next = the next card ('' = unknown)")
        .def("set_elixir", &Battle::set_elixir, py::arg("team"), py::arg("elixir"))
        .def("twin_move", &Battle::twin_move, py::arg("id"), py::arg("pos"))
        .def("twin_set_hp", &Battle::twin_set_hp, py::arg("id"), py::arg("hp"))
        .def("twin_remove", &Battle::twin_remove, py::arg("id"))
        .def("twin_spawn", &Battle::twin_spawn, py::arg("team"), py::arg("card_id"), py::arg("pos"),
             py::arg("play") = -1, py::arg("unit") = -1)
        // (No keep_alive: copies would chain each other alive. The GameData must outlive them - the twin keeps it.)
        .def("clone", [](const Battle& b) { return Battle(b); },
             "An independent copy of the battle as it is now (the digital twin keeps these to rewind to)")
        .def("set_follow_real", &Battle::set_follow_real, py::arg("on"))
        .def("step", &Battle::step)
        .def("steps", [](Battle& b, int n) { for (int i = 0; i < n && !b.finished(); ++i) b.step(); }, py::arg("n"))
        .def_property_readonly("time_ms", &Battle::time_ms)
        .def_property_readonly("phase", &Battle::phase)
        .def_property_readonly("finished", &Battle::finished)
        .def_property_readonly("winner", &Battle::winner)
        .def_property_readonly("elixir_multiplier", &Battle::elixir_multiplier)
        .def("player", &Battle::player, py::return_value_policy::reference_internal)
        .def_property_readonly("entities", &Battle::entities)
        .def_property_readonly("projectiles", &Battle::projectiles)
        .def_property_readonly("pending", &Battle::pending)
        .def_property_readonly("deploy_delay_ms", &Battle::deploy_delay_ms)
        .def_property_readonly("card_refill_ms", &Battle::card_refill_ms)
        .def_property_readonly("events", &Battle::events)
        .def("clear_events", &Battle::clear_events)
        .def("set_record_events", &Battle::set_record_events)
        .def("set_record_damage", &Battle::set_record_damage, py::arg("on"))
        .def("find", &Battle::find, py::return_value_policy::reference_internal, py::arg("id"),
             "Entity by id, or None once it is dead")
        .def("king_tower", &Battle::king_tower)
        .def("princess_tower", &Battle::princess_tower);

    // --- bots ---
    py::enum_<BotKind>(m, "BotKind")
        .value("NOOP", BotKind::Noop)
        .value("RANDOM", BotKind::Random)
        .value("HEURISTIC", BotKind::Heuristic);
    py::class_<BotMove>(m, "BotMove")
        .def_readonly("hand_slot", &BotMove::hand_slot)
        .def_readonly("pos", &BotMove::pos);
    py::class_<Bot>(m, "Bot")
        .def("decide", &Bot::decide, py::arg("battle"), py::arg("team"), "BotMove or None; does not play")
        .def("act", &Bot::act, py::arg("battle"), py::arg("team"));
    m.def("make_bot", [](const py::handle& kind, uint64_t seed) { return make_bot(bot_from(kind), seed); },
          py::arg("kind"), py::arg("seed") = 0);

    // --- RL interface ---
    auto rl_mod = m.def_submodule("rl", "Observation/action encoding (own-frame, either side)");
    rl_mod.attr("GRID_W") = rl::kGridW;
    rl_mod.attr("GRID_H") = rl::kGridH;
    rl_mod.attr("CELLS") = rl::kCells;
    rl_mod.attr("NUM_ACTIONS") = rl::kNumActions;
    rl_mod.attr("NOOP") = rl::kNoop;
    rl_mod.attr("BOT_ACTION") = kBotAction;
    rl_mod.attr("MAX_ENTITIES") = rl::kMaxEntities;
    rl_mod.attr("ENTITY_FEATURES") = rl::kEntityFeatures;
    rl_mod.attr("GLOBAL_FEATURES") = rl::kGlobalFeatures;
    rl_mod.attr("HAND_TOKENS") = rl::kHandTokens;
    rl_mod.attr("HAND_SIZE") = kHandSize;
    rl_mod.attr("HISTORY") = rl::kHistory;
    rl_mod.attr("FIRST_CARD_TOKEN") = rl::kFirstCardToken;
    rl_mod.def("vocab_size", &rl::vocab_size, py::arg("data"));
    rl_mod.def("encode", &encode_battle, py::arg("battle"), py::arg("team"), py::arg("observe_enemy_elixir") = true,
               py::arg("key_spots") = false);
    rl_mod.def(
        "key_spot_cells",
        [](const Battle& b, Team team) {
            std::vector<int> troops, spells;
            rl::key_spot_cells(b, team, troops, spells);
            return py::make_tuple(troops, spells);
        },
        py::arg("battle"), py::arg("team"), "(troop cells, spell cells): the key spots, own frame");
    rl_mod.def("apply_action", &rl::apply_action, py::arg("battle"), py::arg("team"), py::arg("action"));
    rl_mod.def("cell_position", &rl::cell_position, py::arg("battle"), py::arg("team"), py::arg("cell"));

    py::class_<RewardConfig>(m, "RewardConfig")
        .def(py::init<>())
        .def_readwrite("win", &RewardConfig::win)
        .def_readwrite("crown", &RewardConfig::crown)
        .def_readwrite("tower_damage", &RewardConfig::tower_damage)
        .def_readwrite("elixir_leak", &RewardConfig::elixir_leak)
        .def_readwrite("elixir_trade", &RewardConfig::elixir_trade)
        .def_readwrite("trade_floor", &RewardConfig::trade_floor);

    py::class_<BatchEnv>(m, "BatchEnv")
        .def(py::init([](const GameData& data, const std::vector<std::pair<Deck, Deck>>& matchups, int num_envs,
                         int decision_ticks, uint64_t seed, const py::handle& blue_bot, const py::handle& red_bot,
                         bool observe_enemy_elixir, const RewardConfig& reward, int num_threads, bool auto_reset,
                         const py::object& key_spots) {
                 BatchEnvConfig cfg;
                 cfg.num_envs = num_envs;
                 cfg.num_threads = num_threads;
                 cfg.auto_reset = auto_reset;
                 cfg.decision_ticks = decision_ticks;
                 cfg.seed = seed;
                 for (const auto& [blue, red] : matchups) cfg.matchups.push_back({blue, red});
                 cfg.bots[0] = bot_from(blue_bot);
                 cfg.bots[1] = bot_from(red_bot);
                 cfg.obs.observe_enemy_elixir = observe_enemy_elixir;
                 if (py::isinstance<py::bool_>(key_spots)) {  // one setting for both sides, or (blue, red)
                     cfg.key_spots[0] = cfg.key_spots[1] = key_spots.cast<bool>();
                 } else {
                     const auto both = key_spots.cast<std::pair<bool, bool>>();
                     cfg.key_spots[0] = both.first;
                     cfg.key_spots[1] = both.second;
                 }
                 cfg.reward = reward;
                 return std::make_unique<BatchEnv>(data, std::move(cfg));
             }),
             py::arg("data"), py::arg("matchups"), py::arg("num_envs") = 1, py::arg("decision_ticks") = 10,
             py::arg("seed") = 0, py::arg("blue_bot") = "heuristic", py::arg("red_bot") = "heuristic",
             py::arg("observe_enemy_elixir") = true, py::arg("reward") = RewardConfig{}, py::arg("num_threads") = 0,
             py::arg("auto_reset") = true, py::arg("key_spots") = false)
        .def("reset", &BatchEnv::reset)
        .def("reset_env", &BatchEnv::reset_env, py::arg("env"))
        .def(
            "step",
            [](BatchEnv& env, py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions) {
                if (actions.ndim() != 2 || actions.shape(0) != env.num_envs() || actions.shape(1) != 2) {
                    throw py::value_error("actions must have shape (num_envs, 2)");
                }
                const int32_t* ptr = actions.data();
                py::gil_scoped_release release;
                env.step(ptr);
            },
            py::arg("actions"))
        .def_property_readonly("num_envs", &BatchEnv::num_envs)
        .def("battle", &BatchEnv::battle, py::return_value_policy::reference_internal, py::arg("env"))
        .def_property_readonly("data", &BatchEnv::data, py::return_value_policy::reference_internal)
#define CRSIM_VIEW(name, ...)                                                          \
    .def_property_readonly(#name, [](py::object self) {                                \
        auto& env = self.cast<BatchEnv&>();                                            \
        const py::ssize_t n = env.num_envs();                                          \
        return view(env.name, {__VA_ARGS__}, self);                                    \
    })
        CRSIM_VIEW(global, n, 2, rl::kGlobalFeatures)
        CRSIM_VIEW(entities, n, 2, rl::kMaxEntities, rl::kEntityFeatures)
        CRSIM_VIEW(entity_token, n, 2, rl::kMaxEntities)
        CRSIM_VIEW(entity_mask, n, 2, rl::kMaxEntities)
        CRSIM_VIEW(hand, n, 2, rl::kHandTokens)
        CRSIM_VIEW(enemy_history, n, 2, rl::kHistory)
        CRSIM_VIEW(enemy_history_age, n, 2, rl::kHistory)
        CRSIM_VIEW(action_mask, n, 2, rl::kNumActions)
        CRSIM_VIEW(reward, n, 2)
        CRSIM_VIEW(taken, n, 2)
        CRSIM_VIEW(done, n)
        CRSIM_VIEW(winner, n)
        CRSIM_VIEW(crowns, n, 2)
        CRSIM_VIEW(episode_steps, n)
        CRSIM_VIEW(episode_time_ms, n)
        CRSIM_VIEW(episode_matchup, n)
        CRSIM_VIEW(matchup, n)
#undef CRSIM_VIEW
        ;
}
