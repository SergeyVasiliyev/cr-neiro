#pragma once

#include "crsim/game_data.h"

namespace cr {

// "Own frame": the arena as a player sees it on their phone — their side at the bottom.
// Red's view is the arena rotated by 180 degrees. The transform is its own inverse.
inline Vec2 to_own_frame(const ArenaDef& a, Team team, Vec2 p) {
    if (team == Team::Blue) return p;
    return {a.width - p.x, a.height - p.y};
}

inline Vec2 from_own_frame(const ArenaDef& a, Team team, Vec2 p) { return to_own_frame(a, team, p); }

// Princess tower side index (Battle::princess_tower) that appears on the left in `team`'s own frame.
inline int own_left_side(Team team) { return team == Team::Blue ? 0 : 1; }

}  // namespace cr
