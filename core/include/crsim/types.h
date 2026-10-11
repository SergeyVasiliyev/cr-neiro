#pragma once

#include <cmath>
#include <cstdint>

namespace cr {

// Arena coordinates are in tiles. Origin is the bottom-left corner of the arena;
// Blue plays from the bottom (small y), Red from the top (large y).
struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    double length() const { return std::sqrt(x * x + y * y); }
};

inline double distance(Vec2 a, Vec2 b) { return (a - b).length(); }

enum class Team : uint8_t { Blue = 0, Red = 1 };

inline Team opponent(Team t) { return t == Team::Blue ? Team::Red : Team::Blue; }
inline int idx(Team t) { return static_cast<int>(t); }
inline const char* team_name(Team t) { return t == Team::Blue ? "blue" : "red"; }

using EntityId = int32_t;
constexpr EntityId kNoEntity = -1;

}  // namespace cr
