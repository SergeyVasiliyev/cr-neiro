"""Converts RoyaleAPI's cr-api-data JSON (game CSVs, last updated 2023-12) into data/cards.json and data/towers.json.

    .venv\\Scripts\\python.exe tools\\import_royaleapi.py [--src raw/royaleapi] [--level 11]

Download the inputs from https://github.com/RoyaleAPI/cr-api-data/tree/master/docs/json :
cards.json, cards_stats_characters.json, cards_stats_troop.json, cards_stats_projectile.json,
cards_stats_building.json, rarities.json.

Game units: distances in 1/1000 tile, speeds in tiles/min, times in ms. Stats are stored for the
card's level 1 and scale by the rarity's power_level_multiplier (integer math, rounded down).
Only troop cards are imported; mechanics the engine does not simulate yet are listed in
"unsupported" so environments can restrict decks to fully supported cards.
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Character fields that mean a mechanic the engine does not model yet.
UNSUPPORTED_IF_SET = {
    "charge_range": "charge",
    "death_damage": "death_damage",
    "death_spawn_character": "death_spawn",
    "death_spawn_projectile": "death_projectile",
    "spawn_character": "spawner",
    "spawn_projectile": "spawner",
    "dash_max_range": "dash",
    "dash_damage": "dash",
    "kamikaze": "kamikaze",
    "shield_hitpoints": "shield",
    "multiple_targets": "multi_target",
    "buff_on_damage": "buff_on_hit",
    "starting_buff": "starting_buff",
    "variable_damage2": "ramping_damage",
    "minimum_range": "minimum_range",
    "jump_enabled": "river_jump",
    "target_only_troops": "special_targeting",
    "target_only_towers": "special_targeting",
    "target_only_king_tower": "special_targeting",
    "life_time": "limited_lifetime",
    "morph_character": "morph",
    "burst": "burst",
    "attached_character": "attached_unit",
    "projectile_special": "special_attack",
    "damage_special": "special_attack",
    "area_effect_on_dash": "dash",
    "reflected_attack_damage": "reflect",
    "hides_when_not_attacking": "hiding",
    "target_lowest_hp": "special_targeting",
    "retarget_each_tick": "special_targeting",
    "ability": "champion_ability",
    "area_effect_on_hit": "effect_on_hit",
    "buff_when_not_attacking": "buff_when_idle",
    "buff_after_hits": "buff_after_hits",
    "buff_on_kill": "buff_on_kill",
    "spawn_area_object": "area_effect",
    "attack_push_back": "knockback",
    "melee_pushback": "knockback",
    # Newer game files script special behaviour through actions/abilities.
    "on_attack_action": "scripted_action",
    "on_death_action": "scripted_action",
    "on_killed_action": "scripted_action",
    "on_starting_action": "scripted_action",
    "on_starting_attack_action": "scripted_action",
    "death_area_effect": "death_effect",
    "attack_sequence": "attack_sequence",
    "projectile2": "attack_sequence",
    "special_range": "special_attack",
    "mana_on_death_for_opponent": "elixir_on_death",
    "spawn_number": "spawner",
}
# Card (troop row) fields the engine does not model yet.
UNSUPPORTED_CARD = {
    "can_deploy_on_enemy_side": "deploy_anywhere",
    "spell_as_deploy": "deploy_effect",
    "custom_deploy_time": "custom_deploy",
}
# Projectile fields the engine does not model yet.
UNSUPPORTED_PROJECTILE = {
    "spawn_character": "projectile_spawn",
    "pushback": "knockback",
    "chained_hit_radius": "chain",
    "projectile_range": "piercing",
    "spawn_projectile": "projectile_spawn",
    "target_buff": "projectile_buff",
}


def load(src, name):
    path = os.path.join(src, name + ".json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def by_name(rows):
    return {r["name"]: r for r in rows if r.get("name")}


class Scaler:
    def __init__(self, rarities, level):
        self.level = level
        self.rarities = {r["name"]: r for r in rarities}

    def rarity_level(self, rarity):
        return self.level - self.rarities[rarity]["relative_level"]

    def multiplier(self, rarity):
        r = self.rarities[rarity]
        rl = self.rarity_level(rarity)
        if rl < 1 or rl > r["level_count"]:
            return None  # the card does not exist at this level
        return 100 if rl == 1 else r["power_level_multiplier"][rl - 2]

    def scale(self, value, rarity):
        m = self.multiplier(rarity)
        return None if m is None else value * m // 100


def targets(ch):
    if ch.get("target_only_buildings"):
        return ["buildings"]
    t = []
    if ch.get("attacks_ground"):
        t.append("ground")
    if ch.get("attacks_air"):
        t.append("air")
    return t


def unit_spread(count, radius_milli):
    if count <= 1:
        return 0.0
    if radius_milli:
        return radius_milli / 1000
    return round(0.3 + 0.35 * math.sqrt(count), 2)


def convert_attack(ch, projectiles, scaler, unsupported):
    attack = {
        "hit_speed_ms": ch["hit_speed"],
        "first_hit_ms": ch.get("load_time") or 0,
        "range": ch["range"] / 1000,
        "targets": targets(ch),
    }
    pct = ch.get("crown_tower_damage_percent") or 0
    if pct:
        attack["tower_damage_pct"] = 100 + pct
    if ch.get("multiple_projectiles", 0) > 1:
        unsupported.add("multi_projectile")

    proj_name = ch.get("projectile")
    if proj_name:
        p = projectiles.get(proj_name)
        if p is None:
            unsupported.add("unknown_projectile")
            return None
        attack["damage"] = scaler.scale(p.get("damage") or 0, p.get("rarity") or ch["rarity"])
        attack["projectile_speed"] = round(p["speed"] / 60, 3)
        if p.get("radius"):
            attack["splash_radius"] = p["radius"] / 1000
        if not p.get("homing", False) and not p.get("radius"):
            unsupported.add("non_homing_projectile")
        for field, tag in UNSUPPORTED_PROJECTILE.items():
            if p.get(field):
                unsupported.add(tag)
        pct = p.get("crown_tower_damage_percent") or 0
        if pct:
            attack["tower_damage_pct"] = 100 + pct
    else:
        attack["damage"] = scaler.scale(ch.get("damage") or 0, ch["rarity"])
        if ch.get("area_damage_radius"):
            attack["splash_radius"] = ch["area_damage_radius"] / 1000
            if ch.get("self_as_aoe_center"):
                attack["splash_around_self"] = True
    if not attack["targets"] or attack["damage"] == 0:
        unsupported.add("no_attack")
    return attack


def convert_card(card, troop, ch, projectiles, scaler):
    # Stats scale with the rarity of the unit itself, not the card: Skeleton Army (Epic) spawns
    # Common skeletons with the same stats as the Skeletons card at the same card level.
    if scaler.multiplier(card["rarity"]) is None:
        return None  # the card does not exist at this level
    unsupported = set()
    for field, tag in UNSUPPORTED_IF_SET.items():
        if ch.get(field):
            unsupported.add(tag)
    for field, tag in UNSUPPORTED_CARD.items():
        if troop.get(field):
            unsupported.add(tag)
    if troop.get("summon_character_second"):
        unsupported.add("second_unit")
    hp = scaler.scale(ch["hitpoints"], ch["rarity"])
    attack = convert_attack(ch, projectiles, scaler, unsupported)
    if attack is None:
        return None
    count = max(1, troop.get("summon_number") or 1)
    out = {
        "name": card["name"],
        "elixir": card["elixir"],
        "type": "troop",
        "rarity": card["rarity"],
        "count": count,
        "spawn_spread": unit_spread(count, troop.get("summon_radius")),
        **({"spawn_stagger_ms": troop["summon_deploy_delay"]} if count > 1 and troop.get("summon_deploy_delay") else {}),
        "unit": {
            "hitpoints": hp,
            "speed": ch.get("speed") or 0,
            "sight_range": (ch.get("sight_range") or 5500) / 1000,
            "radius": (ch.get("collision_radius") or 500) / 1000,
            "mass": ch.get("mass") or 1,
            **({"ignore_pushback": True} if ch.get("ignore_pushback") else {}),
            "air": bool(ch.get("flying_height")),
            "deploy_time_ms": ch.get("deploy_time") or 1000,
            **({"first_attack_delay_ms": ch["deploy_delay"]} if ch.get("deploy_delay") else {}),
            "attack": attack,
        },
    }
    if out["unit"]["speed"] == 0:
        unsupported.add("immobile")
    if unsupported:
        out["unsupported"] = sorted(unsupported)
    return out


def convert_tower(building, projectiles, scaler):
    proj = projectiles[building["projectile"]]
    return {
        "hitpoints": scaler.scale(building["hitpoints"], building["rarity"]),
        "radius": building["collision_radius"] / 1000,
        "attack": {
            "damage": scaler.scale(proj["damage"], proj["rarity"]),
            "hit_speed_ms": building["hit_speed"],
            "first_hit_ms": building.get("load_time") or building["hit_speed"],
            "range": building["range"] / 1000,
            "projectile_speed": round(proj["speed"] / 60, 3),
            "targets": targets(building),
        },
    }


def convert(cards, troops, characters, projectiles, buildings, scaler, source, generator):
    """RoyaleAPI-shaped inputs (snake_case dicts) -> (cards.json dict, towers.json dict, skipped list)."""
    out_cards = {"_comment": f"Generated by {generator} from {source}. Do not edit by hand."}
    skipped = []
    for card in cards:
        if card["type"] != "Troop":
            continue
        troop = troops.get(card["sc_key"])
        if troop is None or troop.get("not_in_use"):
            skipped.append((card["key"], "no troop stats"))
            continue
        ch = characters.get(troop.get("summon_character"))
        if ch is None:
            skipped.append((card["key"], "no character stats"))
            continue
        converted = convert_card(card, troop, ch, projectiles, scaler)
        if converted is None:
            skipped.append((card["key"], "not available at this level / unknown projectile"))
            continue
        out_cards[card["key"].replace("-", "_")] = converted

    towers = {
        "_comment": f"Generated by {generator} from {source}. footprint = half of the no-deploy square.",
        "princess": {**convert_tower(buildings["PrincessTower"], projectiles, scaler), "footprint": 1.5},
        "king": {**convert_tower(buildings["KingTower"], projectiles, scaler), "footprint": 2.0},
    }
    return out_cards, towers, skipped


def write_outputs(out_dir, out_cards, towers, skipped):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cards.json"), "w", encoding="utf-8") as f:
        json.dump(out_cards, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(os.path.join(out_dir, "towers.json"), "w", encoding="utf-8") as f:
        json.dump(towers, f, ensure_ascii=False, indent=2)
        f.write("\n")

    imported = [k for k in out_cards if not k.startswith("_")]
    full = [k for k in imported if "unsupported" not in out_cards[k]]
    print(f"imported {len(imported)} cards, {len(full)} fully supported:")
    print("  " + ", ".join(sorted(full)))
    for key, why in skipped:
        print(f"  skipped {key}: {why}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=os.path.join(ROOT, "raw", "royaleapi"))
    parser.add_argument("--out", default=os.path.join(ROOT, "data"))
    parser.add_argument("--level", type=int, default=11, help="card and tower level (tournament standard = 11)")
    args = parser.parse_args()

    cards = load(args.src, "cards")
    troops = by_name(load(args.src, "cards_stats_troop"))
    characters = by_name(load(args.src, "cards_stats_characters"))
    projectiles = by_name(load(args.src, "cards_stats_projectile"))
    buildings = by_name(load(args.src, "cards_stats_building"))
    scaler = Scaler(load(args.src, "rarities"), args.level)

    source = f"RoyaleAPI cr-api-data (game CSVs, 2023-12), level {args.level}"
    out_cards, towers, skipped = convert(cards, troops, characters, projectiles, buildings, scaler, source,
                                         "tools/import_royaleapi.py")
    write_outputs(args.out, out_cards, towers, skipped)


if __name__ == "__main__":
    main()
