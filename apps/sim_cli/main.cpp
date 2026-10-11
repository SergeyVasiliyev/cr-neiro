// Runs matches between two scripted bots.
//   sim_cli [data_dir] [games] [seed] [blue_bot] [red_bot]     bots: noop | random | heuristic
// games == 1: prints the event log of one match. games > 1: silent single-thread benchmark.
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

#include "crsim/battle.h"
#include "crsim/bots.h"

namespace {

using cr::Battle;
using cr::Team;
using cr::Vec2;

// Bots decide every half second of game time, like the RL environment.
constexpr int kDecisionTicks = 10;

std::string clock(int ms) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%d:%02d.%d", ms / 60000, ms / 1000 % 60, ms / 100 % 10);
    return buf;
}

void print_summary(const Battle& b) {
    const auto winner = b.winner();
    std::printf("\nresult: %s | crowns blue %d - red %d | time %s\n", winner ? cr::team_name(*winner) : "draw",
                b.player(Team::Blue).crowns, b.player(Team::Red).crowns, clock(b.time_ms()).c_str());
    for (const auto& e : b.entities()) {
        if (e.is_building()) {
            std::printf("  %-4s %-15s hp %5d / %d\n", cr::team_name(e.team), e.def_id.c_str(), e.hp, e.max_hp);
        }
    }
}

// The starter deck every new player gets.
const cr::Deck kDeck = {"knight", "archers", "minions", "arrows", "fireball", "giant", "mini_pekka", "musketeer"};

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : "data";
    const int games = argc > 2 ? std::stoi(argv[2]) : 1;
    const uint64_t seed = argc > 3 ? std::stoull(argv[3]) : 42;
    cr::BotKind kinds[2] = {cr::BotKind::Heuristic, cr::BotKind::Heuristic};

    cr::GameData data;
    try {
        if (argc > 4) kinds[0] = cr::bot_kind_from_string(argv[4]);
        if (argc > 5) kinds[1] = cr::bot_kind_from_string(argv[5]);
        data = cr::GameData::load(data_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    if (games == 1) {
        Battle b(data, kDeck, kDeck, seed);
        std::unique_ptr<cr::Bot> bots[2] = {cr::make_bot(kinds[0], seed + 1), cr::make_bot(kinds[1], seed + 2)};
        for (int tick = 0; !b.finished(); ++tick) {
            if (tick % kDecisionTicks == 0) cr::act_simultaneously(b, *bots[0], *bots[1]);
            b.step();
            for (const auto& ev : b.events()) {
                std::printf("[%s] %-4s %-16s %s\n", clock(ev.time_ms).c_str(), cr::team_name(ev.team),
                            cr::event_name(ev.type), ev.info.c_str());
            }
            b.clear_events();
        }
        print_summary(b);
        return 0;
    }

    int wins[2] = {0, 0};
    int draws = 0;
    long long ticks = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int g = 0; g < games; ++g) {
        const uint64_t s = seed + static_cast<uint64_t>(g) * 1000;
        Battle b(data, kDeck, kDeck, s);
        b.set_record_events(false);
        std::unique_ptr<cr::Bot> bots[2] = {cr::make_bot(kinds[0], s + 1), cr::make_bot(kinds[1], s + 2)};
        for (int tick = 0; !b.finished(); ++tick) {
            if (tick % kDecisionTicks == 0) cr::act_simultaneously(b, *bots[0], *bots[1]);
            b.step();
            ++ticks;
        }
        if (auto w = b.winner()) ++wins[cr::idx(*w)];
        else ++draws;
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%d games in %.2fs: %.0f games/s, %.0f ticks/s (%.0fx realtime)\n", games, secs, games / secs,
                ticks / secs, ticks / secs * data.rules.tick_ms / 1000.0);
    std::printf("blue (%s) %d, red (%s) %d, draws %d\n", cr::bot_kind_name(kinds[0]), wins[0],
                cr::bot_kind_name(kinds[1]), wins[1], draws);
    return 0;
}
