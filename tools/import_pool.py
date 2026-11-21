"""Adds what the Classic Decks pool (data/classic_decks.json) needs beyond tools/import_game_files.py: the mechanics
of its cards (charge, shields, spawners, death spawns, buffs, area and rolling spells, ...), the cards that importer
skips (buildings, area spells, Heal Spirit) and the units the pool's cards make (golemites, lava pups, ...), all read
from the game's own files at level 11. Run it after tools/import_game_files.py, which writes data/cards.json anew.

    .venv\\Scripts\\python.exe tools\\import_pool.py [--logic raw/game_update/extracted/csv_logic]

Fields it adds (battle.cpp reads them; data/README-like notes in core/include/crsim/game_data.h):
  unit: building, lifetime_ms, shield, charge, jump, kamikaze, dash, ramp, multi_targets, hit_buff, deploy_damage,
        reflect, spawn, death_spawn, death_damage, death_rage, riders, hook, pellets, attack_pushback, heal
  card: second (a second kind of unit, Goblin Gang), formation (line / full_lane), playable, anywhere, dig_speed
  spell: kind (projectile / area / roll), buff, spawn, area and roll parameters
Numbers the files do not give are guesses marked "guess" in the comment of the card ("_notes").
"""
import argparse
import json
import os
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import import_game_files as ig  # noqa: E402
import import_royaleapi as ra  # noqa: E402

LEVEL = 11


def read_files(logic_dir):
    files = {}
    for dirpath, _, names in os.walk(logic_dir):
        for name in names:
            if name.endswith((".toml", ".csv")):
                path = os.path.join(dirpath, name)
                files[os.path.relpath(path, logic_dir).replace(os.sep, "/")] = open(path, encoding="utf-8").read()
    return files


def toml_sections(files, kinds=("AEO", "BUFF", "DAMAGE_TYPE")):
    out = {k: {} for k in kinds}
    for path, text in files.items():
        if not path.endswith(".toml"):
            continue
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            continue
        for kind in kinds:
            for name, fields in (data.get(kind) or {}).items():
                if isinstance(fields, dict):
                    out[kind].setdefault(name, fields)
    return out


class Pool:
    def __init__(self, files):
        self.cards_ra, self.troops, self.characters, self.projectiles, self.buildings, rarities = ig.build_inputs(
            files, json.load(open(os.path.join(ROOT, "raw", "royaleapi", "cards.json"), encoding="utf-8")),
            json.load(open(os.path.join(ROOT, "raw", "royaleapi", "cards_stats_troop.json"), encoding="utf-8")))
        self.scaler = ra.Scaler(rarities, LEVEL)
        sec = toml_sections(files)
        self.aeo, self.buff, self.damage_type = sec["AEO"], sec["BUFF"], sec["DAMAGE_TYPE"]
        self.spell_rows = ig.parse_csv(files["spells_other.csv"])
        rules = json.load(open(os.path.join(ROOT, "data", "rules.json"), encoding="utf-8"))
        self.speed_scale = rules.get("speed_scale", 1.0)

    def lvl(self, value, rarity="Common"):
        return self.scaler.scale(value or 0, rarity)

    def ch(self, name):
        return self.characters[name]

    def proj(self, name):
        return self.projectiles[name]

    def aeo_damage(self, name):
        """(damage, crown tower damage) of an area effect at our level."""
        a = self.aeo[name]
        d = a.get("Damage")
        if isinstance(d, str):
            d = self.damage_type[d]
        d = d or {}
        rarity = a.get("Rarity", "Common")
        return self.lvl(d.get("BaseDamage"), rarity), self.lvl(d.get("TowerDamage"), rarity)

    def unit(self, name, elixir=0, rarity="Common", card_name=None, playable=False):
        """A card dict for one character, as import_royaleapi.convert_card makes them."""
        ch = dict(self.ch(name))
        ch.setdefault("range", 0)  # (buildings that only spawn: no attack fields)
        ch.setdefault("hit_speed", 1000)
        card = {"name": card_name or ig.english_name(name), "elixir": elixir, "rarity": rarity}
        out = ra.convert_card(card, {"summon_number": 1}, ch, self.projectiles, self.scaler)
        if out is None:
            raise SystemExit(f"cannot convert {name}")
        out.pop("unsupported", None)
        if not playable:
            out["playable"] = False
        if ch.get("is_building"):
            out["type"] = "building"
            u = out["unit"]
            u["building"] = True
            u["lifetime_ms"] = ch.get("life_time") or 0
            u["speed"] = 0
            if not ch.get("damage") and not ch.get("projectile"):
                u["attack"]["damage"] = 0
        return out


def stun(ms):
    return {"kind": "stun", "ms": ms}


def build(pool, cards):
    P = pool
    lvl = P.lvl
    notes = {}

    def troop(key):
        c = cards[key]
        c.pop("unsupported", None)
        return c, c["unit"]

    # --- troops ---------------------------------------------------------------------------------------------
    c, u = troop("bandit")
    a = P.ch("Assassin")
    u["dash"] = {"min": a["dash_min_range"] / 1000, "max": a["dash_max_range"] / 1000,
                 "damage": lvl(a["dash_damage"]), "speed": round(a["jump_speed"] / 60, 3),
                 "cooldown_ms": a["dash_cooldown"], "immune_ms": a["dash_immune_to_damage_time"]}

    for key, name in (("battle_ram", "BattleRam"), ("dark_prince", "DarkPrince")):
        c, u = troop(key)
        a = P.ch(name)
        # ChargeRange 300 in the files; the wiki has the charge start after 2 tiles of walking (guess).
        u["charge"] = {"distance": 2.0, "speed_mult": a["charge_speed_multiplier"] / 100,
                       "damage": lvl(a["damage_special"])}
        notes.setdefault(key, []).append("charge distance 2 tiles: guess (ChargeRange=300 in the files)")
    c, u = troop("battle_ram")
    u["kamikaze"] = True
    u["death_spawn"] = {"unit": "barbarians", "count": 2, "radius": 0.6, "deploy_ms": 1000}
    c, u = troop("dark_prince")
    u["shield"] = lvl(P.ch("DarkPrince")["shield_hitpoints"])
    u["jump"] = True

    c, u = troop("electro_giant")
    a = P.ch("ElectroGiant")
    # ReflectAttackCrownTowerDamage is the zap's damage on crown towers (level 1), not a percentage: 38 -> 97 at
    # level 11, the drop seen 20 times in the recordings (and 281 = 184 + 97 when its own hit came in the same frame).
    u["reflect"] = {"damage": lvl(a["reflected_attack_damage"]), "radius": a["reflected_attack_radius"] / 1000,
                    "tower_damage": lvl(a["reflect_attack_crown_tower_damage"]),
                    "buff": stun(a["reflected_attack_buff_duration"])}

    c, u = troop("electro_wizard")
    a = P.ch("ElectroWizard")
    u["multi_targets"] = a["multiple_targets"]
    u["hit_buff"] = stun(a["buff_on_damage_time"])
    d, td = P.aeo_damage("ElectroWizardZap")
    u["deploy_damage"] = {"damage": d, "radius": P.aeo["ElectroWizardZap"]["Radius"] / 1000, "tower_damage": td,
                          "buff": stun(P.aeo["ElectroWizardZap"]["BuffTime"])}

    c, u = troop("fisherman")
    a = P.ch("Fisherman")
    u["hook"] = {"min": a["special_min_range"] / 1000, "max": a["special_range"] / 1000,
                 "load_ms": a["special_load_time"]}

    c, u = troop("goblin_gang")
    c["second"] = {"unit": "spear_goblins", "count": 3}

    c, u = troop("goblin_giant")
    u["riders"] = {"unit": "spear_goblin_rider", "count": 2}
    cards["spear_goblin_rider"] = P.unit("SpearGoblinGiant", card_name="Spear Goblin (on the Goblin Giant)")
    cards["spear_goblin_rider"]["unit"]["death_spawn"] = {"unit": "spear_goblins", "count": 1, "radius": 0.0,
                                                          "deploy_ms": 700}

    c, u = troop("golem")
    u["death_spawn"] = {"unit": "golemite", "count": 2, "radius": 1.5, "deploy_ms": 1000}
    d, _ = P.aeo_damage("GolemDeathExplosion")
    g = P.aeo["GolemDeathExplosion"]
    u["death_damage"] = {"damage": d, "radius": g["Radius"] / 1000, "pushback": g.get("Pushback", 0) / 1000}
    cards["golemite"] = P.unit("Golemite")

    c, u = troop("hunter")
    hp = P.proj("HunterProjectile")
    u["pellets"] = {"count": P.ch("Hunter")["multiple_projectiles"], "damage": lvl(hp["damage"])}
    notes.setdefault("hunter", []).append("pellet spread: guess (Scatter=Line, ProjectileRadius 0.3)")

    c, u = troop("ice_spirit")
    u["kamikaze"] = True
    ip = P.proj("IceSpiritsProjectile")
    u["hit_buff"] = {"kind": "stun", "ms": ip["buff_time"]}  # freeze = stun for longer

    c, u = troop("inferno_dragon")
    a = P.ch("InfernoDragon")
    u["ramp"] = {"damage2": lvl(a["variable_damage2"]), "damage3": lvl(a["variable_damage3"]),
                 "stage_ms": a["variable_damage_time1"]}

    c, u = troop("lava_hound")
    u["death_spawn"] = {"unit": "lava_pups", "count": 6, "radius": 2.5, "deploy_ms": 1000}
    cards["lava_pups"] = P.unit("LavaPups", card_name="Lava Pup")

    c, u = troop("lumberjack")
    r = P.aeo["BarbarianRage"]
    u["death_rage"] = {"radius": r["Radius"] / 1000, "ms": r["LifeDuration"], "boost": 0.3, "delay_ms": 500}
    notes.setdefault("lumberjack", []).append("rage +30% from BUFF RageModeRage 130")

    c, u = troop("mother_witch")
    vp = P.proj("VoodooProjectile")
    u["hit_buff"] = {"kind": "curse", "ms": vp["buff_time"], "unit": "voodoo_hog"}
    cards["voodoo_hog"] = P.unit("VoodooHog", card_name="Cursed Hog")
    cards["voodoo_hog"]["unit"]["jump"] = True

    c, u = troop("night_witch")
    a = P.ch("DarkWitch")
    u["spawn"] = {"unit": "bats", "count": a["spawn_number"], "interval_ms": a["spawn_pause_time"],
                  "start_ms": a.get("spawn_start_time") or 0, "radius": a.get("spawn_radius", 0) / 1000}
    u["death_spawn"] = {"unit": "bats", "count": a["death_spawn_count"], "radius": 0.5, "deploy_ms": 0}

    c, u = troop("balloon")
    d, _ = P.aeo_damage("BalloonBombExplosion")
    u["death_damage"] = {"damage": d, "radius": P.aeo["BalloonBombExplosion"]["Radius"] / 1000,
                         "delay_ms": P.buildings["BalloonBomb"].get("deploy_time", 3000)}

    c, u = troop("princess")
    pp = P.proj("PrincessProjectile")
    u["attack"]["damage"] = lvl(pp["damage"])
    u["attack"]["splash_radius"] = pp["radius"] / 1000
    u["attack"]["projectile_speed"] = round(pp["speed"] / 60, 3)

    c, u = troop("royal_hogs")
    u["jump"] = True
    c["formation"] = {"kind": "line", "width": 3.5}

    c, u = troop("royal_recruits")
    u["shield"] = lvl(P.ch("Recruit")["shield_hitpoints"])
    c["formation"] = {"kind": "full_lane", "width": 14.0}

    c, u = troop("sparky")
    u["attack_pushback"] = P.ch("ZapMachine")["attack_push_back"] / 1000

    c, u = troop("zappies")
    u["hit_buff"] = stun(P.ch("MiniZapMachine")["buff_on_damage_time"])

    cards["heal_spirit"] = P.unit("HealSpirit", elixir=1, rarity="Rare", card_name="Heal Spirit", playable=True)
    hs = cards["heal_spirit"]["unit"]
    hs["kamikaze"] = True
    hb = P.buff["HealSpiritBuff"]
    hs["heal"] = {"amount": lvl(hb["HealPerSecond"]), "radius": P.ch("HealSpirit").get("radius", 2500) / 1000}
    notes.setdefault("heal_spirit", []).append("heals amount = 1 s of HealPerSecond: guess")

    # --- buildings ------------------------------------------------------------------------------------------
    for key, name, elixir, rarity in (("tombstone", "Tombstone", 3, "Rare"), ("goblin_cage", "GoblinCage", 4, "Rare"),
                                      ("inferno_tower", "InfernoTower", 5, "Rare"), ("goblin_drill", "GoblinDrill", 4, "Epic")):
        cards[key] = P.unit(name, elixir=elixir, rarity=rarity, card_name=ig.english_name(name), playable=True)
    t = cards["tombstone"]["unit"]
    a = P.ch("Tombstone")
    t["spawn"] = {"unit": "skeletons", "count": a["spawn_number"], "interval_ms": a["spawn_pause_time"],
                  "start_ms": 0, "radius": 0.6}
    t["death_spawn"] = {"unit": "skeletons", "count": a["death_spawn_count"], "radius": 0.7, "deploy_ms": 0}
    g = cards["goblin_cage"]["unit"]
    g["death_spawn"] = {"unit": "goblin_brawler", "count": 1, "radius": 0.0, "deploy_ms": 500}
    cards["goblin_brawler"] = P.unit("GoblinBrawler", card_name="Goblin Brawler")
    i = cards["inferno_tower"]["unit"]
    a = P.ch("InfernoTower")
    i["ramp"] = {"damage2": lvl(a["variable_damage2"]), "damage3": lvl(a["variable_damage3"]),
                 "stage_ms": a["variable_damage_time1"]}
    dr = cards["goblin_drill"]
    a = P.ch("GoblinDrill")
    dr["anywhere"] = True
    dr["dig_speed"] = round(P.ch("GoblinDrillDig").get("spawn_pathfind_speed", 300) / 60, 3)
    du = dr["unit"]
    du["spawn"] = {"unit": "goblins", "count": a["spawn_number"], "interval_ms": a["spawn_pause_time"],
                   "start_ms": a.get("spawn_start_time") or 0, "radius": 0.6}
    du["death_spawn"] = {"unit": "goblins", "count": a["death_spawn_count"], "radius": 0.5, "deploy_ms": 500}
    d, td = P.aeo_damage("GoblinDrillDamageArea")
    du["deploy_damage"] = {"damage": d, "radius": P.aeo["GoblinDrillDamageArea"]["Radius"] / 1000,
                           "pushback": P.aeo["GoblinDrillDamageArea"]["Pushback"] / 1000, "tower_damage": td}

    # --- spells ---------------------------------------------------------------------------------------------
    def spell(key):
        c = cards[key]
        c.pop("unsupported", None)
        return c, c["spell"]

    c, s = spell("giant_snowball")
    sp = P.proj("SnowballSpell")
    s["buff"] = {"kind": "slow", "ms": sp["buff_time"], "amount": 0.3}

    c, s = spell("goblin_barrel")
    gb = P.proj("GoblinBarrelSpell")
    s["spawn"] = {"unit": "goblins", "count": gb["spawn_character_count"], "radius": 0.7,
                  "deploy_ms": gb["spawn_character_deploy_time"]}

    for key, name, row in (("the_log", "LogProjectileRolling", "Log"), ("barbarian_barrel", "BarbLogProjectileRolling", "BarbLog")):
        c, s = spell(key)
        rp = P.proj(name)
        pct = rp.get("crown_tower_damage_percent") or 0
        # Rolling speed: the file's speed / 60 without the walking speed scale (rules.json speed_scale 1.175): the log
        # in c013 rolled ~9 tiles in ~2.8 s, about 3.2 tiles/s (scaled it would be 3.9). Stored divided by the scale,
        # which GameData multiplies back.
        s.update({"kind": "roll", "damage": lvl(rp["damage"]), "radius": rp["projectile_radius"] / 1000,
                  "depth": rp["projectile_radius_y"] / 1000, "range": rp["projectile_range"] / 1000,
                  "projectile_speed": round(rp["speed"] / 60 / P.speed_scale, 3),
                  "pushback": (rp.get("pushback") or 0) / 1000, "tower_damage_pct": 100 + pct, "targets": ["ground"]})
        # SpellAsDeploy without CanDeployOnEnemySide: put down like a troop, on our half only.
        r = P.spell_rows.get(row, {})
        if r.get("SpellAsDeploy") and not r.get("CanDeployOnEnemySide"):
            c["own_half"] = True
        if rp.get("spawn_character"):
            s["spawn"] = {"unit": "barbarians", "count": 1, "radius": 0.0, "deploy_ms": rp["spawn_character_deploy_time"]}

    def area_card(key, name, elixir, rarity, **spell_fields):
        cards[key] = {"name": name, "elixir": elixir, "type": "spell", "rarity": rarity,
                      "spell": {"kind": "area", "waves": 1, "wave_interval_ms": 0, "projectile_speed": 0.0,
                                "pushback": 0.0, "targets": ["ground", "air"], **spell_fields}}

    d, td = P.aeo_damage("Zap")
    z = P.aeo["Zap"]
    area_card("zap", "Zap", 2, "Common", damage=d, radius=z["Radius"] / 1000, tower_damage=td,
              buff=stun(z["BuffTime"]))
    lt = P.aeo["Lightning"]
    lp = P.proj("LighningSpell")
    # The first bolt ~1.1 s after it starts (HitSpeedOffset 0.5 s, then the bolt's fall; a tower hit 2.59 s after the
    # tap in c012 = 0.5 s tap-to-screen + 1 s blueprint + 1.09 s), then one every HitSpeed.
    area_card("lightning", "Lightning", 6, "Epic", damage=lvl(lp["damage"]), radius=lt["Radius"] / 1000,
              tower_damage_pct=100 + lp["crown_tower_damage_percent"], buff=stun(lp["buff_time"]),
              max_targets=lt["LifeDuration"] // lt["HitSpeed"], hit_ms=lt["HitSpeed"], first_ms=1100, biggest=True)
    # Area spells tick through a buff: its first damage comes one HitFrequency after the area's first hit
    # (HitSpeedOffset). The recordings: Poison's first tick on a tower 1.3-1.4 s after it lands, then 8 ticks a
    # second apart; Tornado's one tick of 25 on a tower ~0.7 s after it lands (7 casts).
    po = P.aeo["Poison"]
    pb = P.buff["Poison"]
    tick = pb["HitFrequency"]
    area_card("poison", "Poison", 4, "Epic", damage=0, radius=po["Radius"] / 1000,
              tower_damage_pct=100 + pb["CrownTowerDamagePercent"], duration_ms=po["LifeDuration"], tick_ms=tick,
              first_ms=po.get("HitSpeedOffset", 0) + tick, tick_count=po["LifeDuration"] // tick,
              tick_damage=lvl(pb["DamagePerSecond"]) * tick // 1000,
              buff={"kind": "slow", "ms": 1000, "amount": -pb["SpeedMultiplier"] / 100})
    to = P.aeo["Tornado"]
    tb = P.buff["Tornado"]
    tick = tb["HitFrequency"]
    dps = lvl(tb["DamagePerSecond"])
    area_card("tornado", "Tornado", 3, "Epic", damage=0, radius=to["Radius"] / 1000,
              tower_damage_pct=100 + tb["CrownTowerDamagePercent"], duration_ms=to["LifeDuration"], tick_ms=tick,
              first_ms=to.get("HitSpeedOffset", 0) + tick, tick_count=max(1, (to["LifeDuration"] - to.get("HitSpeedOffset", 0)) // tick),
              tick_damage=dps * tick // 1000,
              tower_tick_damage=int(dps * (100 + tb["CrownTowerDamagePercent"]) / 100 * tick / 1000), pull=3.5)
    notes.setdefault("tornado", []).append("pull 3.5 tiles/s toward the centre: guess")
    for key, n in notes.items():
        cards[key]["_notes"] = n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logic", default=os.path.join(ROOT, "raw", "game_update", "extracted", "csv_logic"))
    p.add_argument("--cards", default=os.path.join(ROOT, "data", "cards.json"))
    args = p.parse_args()
    files = read_files(args.logic)
    pool = Pool(files)
    cards = json.load(open(args.cards, encoding="utf-8"))
    before = {k for k in cards if not k.startswith("_")}
    build(pool, cards)
    for key in cards:
        if not key.startswith("_") and key not in before:
            cards[key]["index_group"] = 1  # after the earlier cards: their indices (the nets' tokens) stay
    if "tools/import_pool.py" not in cards["_comment"]:
        cards["_comment"] += " Then tools/import_pool.py added the Classic Decks pool's mechanics and cards."
    deck_cards = json.load(open(os.path.join(ROOT, "data", "classic_decks.json"), encoding="utf-8"))["cards"]
    missing = [c for c in deck_cards if c not in cards]
    still = [c for c in deck_cards if cards.get(c, {}).get("unsupported")]
    with open(args.cards, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"{len(cards) - 1} cards; pool cards missing: {missing or 'none'}; still flagged unsupported: {still or 'none'}")


if __name__ == "__main__":
    main()
