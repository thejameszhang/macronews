"""Per-relation type signatures for the macro KG (ex-post gate).

A triplet is a type violation when its subject_type or object_type falls outside
the legal set for its relation. Hand-authored prior; calibrated against the grader
in the refinement task. See docs/superpowers/specs/2026-06-24-kg-type-signatures-design.md.
"""
from kg.schemas import ENTITY_TYPES_TUPLE

# Buckets that expand to concrete entity types.
QUANT = frozenset({
    "ASSET_METRIC", "ECON_INDICATOR", "INTEREST_RATE", "CURRENCY",
    "COMMODITY", "EQUITY_INDEX", "GOV_BOND", "FIN_INSTRUMENT",
})
ACTOR = frozenset({
    "PERSON", "SOVEREIGN", "GPE", "CENTRAL_BANK", "GOV_BODY",
    "REG_BODY", "COMPANY", "ORG", "US_GICS_SECTOR",
})
ANY = frozenset(ENTITY_TYPES_TUPLE)
C = frozenset({"CONCEPT"})

# Directional/causal relations (relations expressing active change in a quantity): the only
# relations the self-reference filter's Tier-2 (cosine) considers.
DIRECTIONAL_RELATIONS = frozenset({
    "CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT",
    "RAISES", "DECREASES", "LEAVES_UNCHANGED",
})

# Object sets calibrated against the relv3 grader (2026-06-24): the vague relations' legal
# OBJECT sets are widened to their high-faithful excluded types. INVARIANT: the directional
# relations (CAUSES_*/IMPACT/RAISES/DECREASES/LEAVES_UNCHANGED) keep quantity-only objects —
# gating their actor/place objects is the gate's whole purpose; never add an actor/place type
# there. Calibration table, the accept-58%-AC5 decision, and the regen command are in the
# design spec (docs/superpowers/specs/2026-06-24-kg-type-signatures-design.md, §3.1 + §3.6).
RELATION_SIGNATURES = {
    "CAUSES_RISE_IN":   {"subject": ANY,          "object": QUANT | C},
    "CAUSES_FALL_IN":   {"subject": ANY,          "object": QUANT | C},
    "IMPACT":           {"subject": ANY,          "object": QUANT | C},
    "RAISES":           {"subject": ACTOR,        "object": frozenset({"INTEREST_RATE", "ASSET_METRIC", "ECON_INDICATOR", "FIN_INSTRUMENT", "EQUITY_INDEX"}) | C},
    "DECREASES":        {"subject": ACTOR | QUANT, "object": QUANT | C},
    "LEAVES_UNCHANGED": {"subject": ACTOR,        "object": frozenset({"INTEREST_RATE", "ASSET_METRIC", "ECON_INDICATOR", "COMMODITY", "CURRENCY"}) | C},
    "REPORTS":          {"subject": ACTOR,        "object": QUANT | C},
    "FORECASTS":        {"subject": ACTOR,        "object": QUANT | frozenset({"EVENT", "GOV_BODY", "SOVEREIGN", "US_GICS_SECTOR"}) | C},
    "PRODUCES":         {"subject": frozenset({"SOVEREIGN", "GPE", "COMPANY", "ORG", "US_GICS_SECTOR"}), "object": frozenset({"COMMODITY", "ASSET_METRIC", "ECON_INDICATOR", "GOV_BOND"}) | C},
    "IS_MEMBER_OF":     {"subject": frozenset({"SOVEREIGN", "GPE", "COMPANY", "ORG", "PERSON"}), "object": frozenset({"ORG", "EQUITY_INDEX", "US_GICS_SECTOR", "GPE", "CENTRAL_BANK", "COMPANY", "CONCEPT", "GOV_BODY", "PERSON", "REG_BODY"})},
    "OPERATES_IN":      {"subject": frozenset({"COMPANY", "ORG", "US_GICS_SECTOR"}), "object": frozenset({"GPE", "SOVEREIGN", "US_GICS_SECTOR", "COMMODITY", "CENTRAL_BANK", "EQUITY_INDEX"}) | C},
    "CONTROLS":         {"subject": ACTOR,        "object": ACTOR | QUANT | C},
    "REGULATES":        {"subject": frozenset({"REG_BODY", "GOV_BODY", "CENTRAL_BANK", "SOVEREIGN", "ORG"}), "object": frozenset({"COMPANY", "ORG", "US_GICS_SECTOR", "FIN_INSTRUMENT", "COMMODITY"}) | C},
    "ANNOUNCES":        {"subject": ACTOR,        "object": frozenset({"EVENT", "COMPANY", "ORG", "PERSON"}) | QUANT | C},
    "IMPOSES":          {"subject": frozenset({"SOVEREIGN", "GOV_BODY", "GPE", "REG_BODY", "CENTRAL_BANK", "ORG"}), "object": frozenset({"EVENT", "ASSET_METRIC", "SOVEREIGN", "GPE", "COMPANY"}) | C},
}


def type_violation(subject_type: str | None, relation: str, object_type: str | None) -> str | None:
    """Return a reason string if (subject_type, relation, object_type) violates the
    relation's signature, else None. Fail-open on unknown relations.

    None-behavior distinction: an unknown *relation* fails OPEN (returns None — never
    gated), but a None subject/object *type* fails CLOSED (returns a violation, since
    None is not in any legal frozenset).
    """
    sig = RELATION_SIGNATURES.get(relation)
    if sig is None:
        return None
    if subject_type not in sig["subject"]:
        return f"subject {subject_type} illegal for {relation}"
    if object_type not in sig["object"]:
        return f"object {object_type} illegal for {relation}"
    return None
