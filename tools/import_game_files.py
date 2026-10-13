"""Imports card and tower stats straight from the game's own data files.

    .venv\\Scripts\\python.exe tools\\import_game_files.py [--apk raw/game_update/asset_pack.apk] [--level 11]

Where the files come from: the Play Store install of Clash Royale ships an install-time asset pack
(split_install_time_asset_pack.apk, ~900 MB) whose assets/csv_logic/ holds the game logic as
LZMA-compressed CSV/TOML files. Pull it from a rooted emulator:
    adb shell pm path com.supercell.clashroyale        (find the folder)
    adb pull <folder>/split_install_time_asset_pack.apk raw/game_update/asset_pack.apk
Server-pushed patches land in /data/data/com.supercell.clashroyale/update/csv_logic (needs root);
pass that folder with --update to let its files override the asset pack.

The files are converted into the same shape as RoyaleAPI's JSON (snake_case keys) and run through
the converter in import_royaleapi.py, so both sources produce identical data/cards.json layouts.
"""
import argparse
import csv
import io
import json
import lzma
import os
import re
import sys
import tomllib
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_royaleapi as ra  # noqa: E402

ROOT = ra.ROOT
PREFIX = "assets/csv_logic/"


def decompress(data):
    """Supercell LZMA: 5-byte props + 4-byte size; standard .lzma wants an 8-byte size."""
    if data[:1] == b"]":
        return lzma.decompress(data[:9] + b"\x00" * 4 + data[9:], format=lzma.FORMAT_ALONE)
    return data


def read_logic_files(apk_path, update_dir=None):
    """{relative path under csv_logic/: text}"""
    files = {}
    with zipfile.ZipFile(apk_path) as z:
        for info in z.infolist():
            if info.filename.startswith(PREFIX) and not info.is_dir():
                files[info.filename[len(PREFIX):]] = decompress(z.read(info.filename)).decode("utf-8")
    if update_dir:
        for dirpath, _, names in os.walk(update_dir):
            for name in names:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, update_dir).replace(os.sep, "/")
                with open(path, "rb") as f:
                    files[rel] = decompress(f.read()).decode("utf-8")
    return files


def snake(name):
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    return s.lower()


def snake_dict(d):
    return {snake(k): v for k, v in d.items() if not isinstance(v, dict)}


def parse_csv(text):
    """Supercell CSV: header row, type row, then rows; a row with an empty name continues the
    previous entry (used for per-level arrays). Returns {name: {column: value or [values]}}."""
    rows = list(csv.reader(io.StringIO(text)))
    header, types = rows[0], rows[1]
    out, current = {}, None
    for row in rows[2:]:
        row = row + [""] * (len(header) - len(row))
        if row[0]:
            current = {}
            out[row[0]] = current
        if current is None:
            continue
        for col, typ, val in zip(header, types, row):
            if not col or val == "":
                continue
            t = typ.lower()
            v = int(val) if t == "int" else (val.lower() == "true" if t == "boolean" else val)
            if col in current and row[0] == "":
                prev = current[col]
                current[col] = (prev if isinstance(prev, list) else [prev]) + [v]
            else:
                current[col] = v
    return out


def collect_sections(files):
    """All CHARACTER / BUILDING / PROJECTILE sections from the per-object TOML files and the
    top-level tables of characters_base.toml, projectiles.toml and buildings.toml."""
    sections = {"CHARACTER": {}, "BUILDING": {}, "PROJECTILE": {}, "SPELL_CHARACTER": {}, "SPELL_OTHER": {}}
    top_level = {"characters_base.toml": "CHARACTER", "projectiles.toml": "PROJECTILE",
                 "buildings.toml": "BUILDING", "spells_characters.toml": "SPELL_CHARACTER",
                 "spells_other.toml": "SPELL_OTHER"}
    for path, text in files.items():
        if not path.endswith(".toml"):
            continue
        if not (path.startswith("characters/") or path in top_level):
            continue
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            continue
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            if key in sections:
                for name, fields in value.items():
                    if isinstance(fields, dict):
                        sections[key][name] = fields
            elif path in top_level and not key.isupper():
                sections[top_level[path]].setdefault(key, value)
    return sections


# Spell (row/projectile) fields for mechanics the engine does not model yet.
SPELL_UNSUPPORTED = {
    "area_effect_object": "area_effect",
    "summon_character": "summon",
    "buff_type": "buff",
    "custom_first_projectile": "special_projectile",
    "spawn_character": "spawn",
    "spawn_projectile": "spawn",
    "projectile_range": "rolling",
    "chained_hit_radius": "chain",
    "target_buff": "buff",
}


def convert_spell(card, row, projectiles, scaler):
    """Projectile spells (Arrows, Fireball, Rocket, ...): same damage to every enemy in the radius."""
    proj = projectiles.get(row.get("projectile"))
    if proj is None or scaler.multiplier(card["rarity"]) is None:
        return None
    unsupported = {tag for field, tag in SPELL_UNSUPPORTED.items() if row.get(field) or proj.get(field)}
    radius = (row.get("radius") or proj.get("radius") or 0) / 1000  # the card's radius wins (Arrows)
    # Scaled by the projectile's own rarity, like a troop's stats by the troop's: the Fireball card is Rare but
    # its projectile is Common (level-11 tower damage 159 in the game = 23% of the common-scaled ~690).
    damage = scaler.scale(proj.get("damage") or 0, proj.get("rarity") or card["rarity"])
    if not radius or not damage:
        unsupported.add("no_area_damage")
    targets = [t for t, key in (("ground", "aoe_to_ground"), ("air", "aoe_to_air")) if proj.get(key)]
    spell = {
        "damage": damage,
        "radius": radius,
        "tower_damage_pct": 100 + (proj.get("crown_tower_damage_percent") or 0),
        "waves": row.get("projectile_waves") or 1,
        "wave_interval_ms": row.get("projectile_wave_interval") or 0,
        "projectile_speed": round(proj["speed"] / 60, 3),
        "pushback": (proj.get("pushback") or 0) / 1000,
        "targets": targets or ["ground", "air"],
    }
    out = {"name": card["name"], "elixir": card["elixir"], "type": "spell", "rarity": card["rarity"], "spell": spell}
    if unsupported:
        out["unsupported"] = sorted(unsupported)
    return out


def build_spells(files, sections, projectiles, scaler, royaleapi_cards=None):
    rows = parse_csv(files["spells_other.csv"])
    for name, fields in sections["SPELL_OTHER"].items():
        rows.setdefault(name, {}).update({k: v for k, v in fields.items() if not isinstance(v, dict)})
    known = {c["sc_key"]: c for c in royaleapi_cards or []}
    out = {}
    for name, raw in rows.items():
        if raw.get("NotInUse") or raw.get("NotVisible") or not raw.get("TID") or not raw.get("Projectile"):
            continue
        if name.startswith(("CHAR_DISABLED", "Super")) or "_" in name:
            continue
        k = known.get(name)
        card = {"key": k["key"] if k else snake(name).replace("_", "-"), "name": k["name"] if k else english_name(name),
                "elixir": raw.get("ManaCost", 0), "rarity": raw.get("Rarity", "Common")}
        converted = convert_spell(card, snake_dict(raw), projectiles, scaler)
        if converted:
            out[card["key"].replace("-", "_")] = converted
    return out


def english_name(sc_key):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", sc_key)


def build_inputs(files, royaleapi_cards=None, royaleapi_troops=None):
    """Game files -> RoyaleAPI-shaped (cards, troops, characters, projectiles, buildings, rarities)."""
    sections = collect_sections(files)
    characters = {}
    for name, fields in sections["CHARACTER"].items():
        characters[name] = {"name": name, **snake_dict(fields)}
    buildings = {}
    for name, fields in sections["BUILDING"].items():
        buildings[name] = {"name": name, **snake_dict(fields)}
    characters.update({k: v for k, v in buildings.items() if k not in characters})
    projectiles = {name: {"name": name, **snake_dict(f)} for name, f in sections["PROJECTILE"].items()}

    rarities = []
    for name, r in parse_csv(files["rarities.csv"]).items():
        mult = r.get("PowerLevelMultiplier", [])
        rarities.append({"name": name, "level_count": r["LevelCount"], "relative_level": r.get("RelativeLevel", 0),
                         "power_level_multiplier": mult if isinstance(mult, list) else [mult]})

    known = {c["sc_key"]: c for c in royaleapi_cards or []}
    known_troops = {t["name"]: t for t in royaleapi_troops or [] if t.get("name")}
    cards, troops = [], {}
    rows = parse_csv(files["spells_characters.csv"])
    # Newer cards are (partly) defined in TOML; those fields override the CSV row.
    for name, fields in sections["SPELL_CHARACTER"].items():
        rows.setdefault(name, {}).update({k: v for k, v in fields.items() if not isinstance(v, dict)})
    for name, row in rows.items():
        if row.get("NotInUse") or row.get("NotVisible") or not row.get("TID"):
            continue
        if name.startswith(("CHAR_DISABLED", "Super")) or "_" in name:
            continue  # disabled cards, event-only "Super" variants, chess/event copies
        troop = {"name": name, **snake_dict(row)}
        if not troop.get("summon_character"):
            # Some cards no longer name their unit: it is the character of the same name
            # (ElectroWizard), or the older dataset knows it (ThreeMusketeers -> 3x ThreeMusketeer).
            old = known_troops.get(name, {})
            if old.get("summon_character") in characters:
                for field in ("summon_character", "summon_number", "summon_radius"):
                    troop.setdefault(field, old.get(field))
            elif name in characters:
                troop["summon_character"] = name
            else:
                continue  # a spell or building card
        troops[name] = troop
        k = known.get(name)
        cards.append({
            "key": k["key"] if k else snake(name).replace("_", "-"),
            "name": k["name"] if k else english_name(name),
            "sc_key": name,
            "elixir": row.get("ManaCost", 0),
            "type": "Troop",
            "rarity": row.get("Rarity", "Common"),
        })
    return cards, troops, characters, projectiles, buildings, rarities


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apk", default=os.path.join(ROOT, "raw", "game_update", "asset_pack.apk"))
    parser.add_argument("--update", default=None, help="extra csv_logic folder from /data/data/.../update")
    parser.add_argument("--out", default=os.path.join(ROOT, "data"))
    parser.add_argument("--level", type=int, default=11, help="card and tower level (tournament standard = 11)")
    parser.add_argument("--version", default="", help="game version for the generated comment")
    args = parser.parse_args()

    files = read_logic_files(args.apk, args.update)
    ra_cards_path = os.path.join(ROOT, "raw", "royaleapi", "cards.json")
    ra_cards = json.load(open(ra_cards_path, encoding="utf-8")) if os.path.exists(ra_cards_path) else None
    ra_troops_path = os.path.join(ROOT, "raw", "royaleapi", "cards_stats_troop.json")
    ra_troops = json.load(open(ra_troops_path, encoding="utf-8")) if os.path.exists(ra_troops_path) else None
    cards, troops, characters, projectiles, buildings, rarities = build_inputs(files, ra_cards, ra_troops)
    scaler = ra.Scaler(rarities, args.level)
    version = f" {args.version}" if args.version else ""
    source = f"game files{version} (assets/csv_logic), level {args.level}"
    out_cards, towers, skipped = ra.convert(cards, troops, characters, projectiles, buildings, scaler, source,
                                            "tools/import_game_files.py")
    spells = build_spells(files, collect_sections(files), projectiles, scaler, ra_cards)
    out_cards.update(spells)
    ra.write_outputs(args.out, out_cards, towers, skipped)
    print(f"spells: {len(spells)}, fully supported: "
          + ", ".join(sorted(k for k, v in spells.items() if "unsupported" not in v)))


if __name__ == "__main__":
    main()
