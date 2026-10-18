"""The finishing blow the policy misses: an enemy tower with less HP left than a spell in hand takes off.

The policies never learned it (a near-dead tower with a spell in hand hardly ever happens in their games): in
overtime, with a tower at 120 HP and a Fireball in hand, both v6 and v7b just wait. In overtime that tower is
the match; earlier it is a crown. So before asking the policy: if a spell in hand kills a tower for sure and
there is elixir for it, cast it on the tower.
"""


def tower_damage(spell):
    """HP a spell takes off a crown tower in all its waves (the game's rounding, as in Battle::apply_damage)."""
    s = spell.spell
    per_wave = s.damage - s.damage * (100 - s.tower_damage_pct) // 100
    return per_wave * max(1, s.waves)


def finishing_spell(battle, team, data):
    """(hand slot, (x, y)) of a spell that destroys an enemy tower for sure, or None."""
    me = battle.player(team)
    best = None
    for slot, card_id in enumerate(me.hand):
        if not card_id:
            continue
        card = data.card(card_id)
        if not card.is_spell or me.elixir + 1e-6 < card.elixir:
            continue
        damage = tower_damage(card)
        for e in battle.entities:
            if e.team == team or not e.is_building or e.hp <= 0 or e.hp > damage:
                continue
            king = e.def_id == "king_tower"  # the king is the match: take it over a princess
            if best is None or (king and not best[2]) or (king == best[2] and card.elixir < best[3]):
                best = (slot, (e.pos.x, e.pos.y), king, card.elixir)
    return None if best is None else (best[0], best[1])
