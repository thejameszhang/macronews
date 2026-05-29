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

# 18 entity codes. SCREAMING_SNAKE_CASE per FINDKG convention; short
# codes (GPE, ORG, GOV_BODY, REG_BODY) for the types whose long names
# add no information.
ENTITY_TYPES_TUPLE = (
    "PERSON", "SOVEREIGN", "GPE",
    "CENTRAL_BANK", "GOV_BODY", "REG_BODY",
    "COMPANY", "ORG", "SECTOR",
    "CURRENCY", "COMMODITY", "INTEREST_RATE", "INDEX",
    "GOV_BOND", "FIN_INSTRUMENT",
    "ECON_INDICATOR", "CONCEPT", "EVENT",
)
ENTITY_TYPES = Literal[
    "PERSON", "SOVEREIGN", "GPE",
    "CENTRAL_BANK", "GOV_BODY", "REG_BODY",
    "COMPANY", "ORG", "SECTOR",
    "CURRENCY", "COMMODITY", "INTEREST_RATE", "INDEX",
    "GOV_BOND", "FIN_INSTRUMENT",
    "ECON_INDICATOR", "CONCEPT", "EVENT",
]

# 18 relations. RAISES/DECREASES = actor-with-authority action on a
# variable they control. POSITIVE_IMPACT_ON/NEGATIVE_IMPACT_ON/IMPACT =
# downstream causal effects observed in the article.
RELATION_TYPES_TUPLE = (
    "OWNS", "HOLDS_POSITION", "IS_MEMBER_OF", "OPERATES_IN",
    "ANNOUNCES", "REPORTS", "PRODUCES", "ACQUIRES", "INVESTS_IN",
    "CONTROLS", "REGULATES", "IMPOSES",
    "RAISES", "DECREASES",
    "POSITIVE_IMPACT_ON", "NEGATIVE_IMPACT_ON", "IMPACT",
    "RELATED_TO",
)
RELATION_TYPES = Literal[
    "OWNS", "HOLDS_POSITION", "IS_MEMBER_OF", "OPERATES_IN",
    "ANNOUNCES", "REPORTS", "PRODUCES", "ACQUIRES", "INVESTS_IN",
    "CONTROLS", "REGULATES", "IMPOSES",
    "RAISES", "DECREASES",
    "POSITIVE_IMPACT_ON", "NEGATIVE_IMPACT_ON", "IMPACT",
    "RELATED_TO",
]


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
