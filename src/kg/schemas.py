"""Pydantic schemas for KG v2 fact extraction (FINDKG-style).

Each fact is a 6-field record carrying its own type tags inline.
There is no separate `entities` list — entities are implicit via the
union of subject/object across facts. Coreference within an article
is handled by the prompt's "consolidate variants" instruction.

Field order in KGFact is chain-of-thought:
  evidence_paragraphs (ground) -> subject/subject_type -> relation
  -> object/object_type
"""

from typing import Literal

from pydantic import BaseModel, Field

# 19 entity codes. SCREAMING_SNAKE_CASE per FINDKG convention; short
# codes (GPE, ORG, GOV_BODY, REG_BODY) for the types whose long names
# add no information.
ENTITY_TYPES_TUPLE = (
    "PERSON", "SOVEREIGN", "GPE",
    "CENTRAL_BANK", "GOV_BODY", "REG_BODY",
    "COMPANY", "ORG", "US_GICS_SECTOR",
    "CURRENCY", "COMMODITY", "INTEREST_RATE", "EQUITY_INDEX",
    "GOV_BOND", "FIN_INSTRUMENT",
    "ECON_INDICATOR", "ASSET_METRIC", "CONCEPT", "EVENT",
)
ENTITY_TYPES = Literal[
    "PERSON", "SOVEREIGN", "GPE",
    "CENTRAL_BANK", "GOV_BODY", "REG_BODY",
    "COMPANY", "ORG", "US_GICS_SECTOR",
    "CURRENCY", "COMMODITY", "INTEREST_RATE", "EQUITY_INDEX",
    "GOV_BOND", "FIN_INSTRUMENT",
    "ECON_INDICATOR", "ASSET_METRIC", "CONCEPT", "EVENT",
]

# 15 relations. RAISES/DECREASES/LEAVES_UNCHANGED = actor-with-authority
# action on a variable they control (up / down / held). CAUSES_RISE_IN/
# CAUSES_FALL_IN/IMPACT = downstream causal effects observed in the article.
# FORECASTS = a forward-looking view (predicts / expects / projects).
RELATION_TYPES_TUPLE = (
    "IS_MEMBER_OF", "OPERATES_IN",
    "ANNOUNCES", "REPORTS", "FORECASTS", "PRODUCES",
    "CONTROLS", "REGULATES", "IMPOSES",
    "RAISES", "DECREASES", "LEAVES_UNCHANGED",
    "CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT",
)
RELATION_TYPES = Literal[
    "IS_MEMBER_OF", "OPERATES_IN",
    "ANNOUNCES", "REPORTS", "FORECASTS", "PRODUCES",
    "CONTROLS", "REGULATES", "IMPOSES",
    "RAISES", "DECREASES", "LEAVES_UNCHANGED",
    "CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT",
]

# Synthetic node type for asset-group anchors, materialized by build_graph from the
# resolution layer. NOT an extractor entity type — deliberately kept out of
# ENTITY_TYPES/ENTITY_TYPES_TUPLE (the extractor never emits it). The viz palette
# appends it so existing type->color assignments don't shift.
ASSET_GROUP_NODE_TYPE = "ASSET_GROUP"
# Synthetic membership relation; NOT in RELATION_TYPES_TUPLE (structural, not a news
# claim) so it is excluded from the grader fact set.
ASSET_GROUP_RELATION = "RELATED_TO_ASSET_GROUP"


class KGFact(BaseModel):
    """One macroeconomic fact grounded in paragraph evidence.

    Field order is chain-of-thought: ground (evidence_paragraphs) before
    claim (subject -> relation -> object). Types are inline per-endpoint
    so the model commits to a typed entity at each side of the relation.
    """

    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices that justify this fact.",
    )
    subject: str
    subject_type: ENTITY_TYPES
    relation: RELATION_TYPES
    object: str
    object_type: ENTITY_TYPES


class KGArticleResult(BaseModel):
    """What the LLM emits per article. The runner wraps this with
    `article_id` and `date` on JSONL write."""

    facts: list[KGFact] = Field(default_factory=list)
