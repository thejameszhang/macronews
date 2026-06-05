"""Pydantic schema for the KG grader's per-fact verdict.

Schema-BLIND by design: the grader is never handed the entity/relation taxonomy,
so it judges each typed fact from an outside reader's point of view and this
verdict never has to know the valid codes. Field order is chain-of-thought — the
judge first cites the paragraphs it reads, decides macro-relevance, then for each
slot suggests a better label only when one genuinely fits, and finally snaps the
headline ``correct`` verdict as a conclusion.

A ``*_suggestion`` is blank when the extractor's label is a fine fit; non-blank it
carries a SHORT better label. Downstream, mapping a non-blank suggestion against
the extractor's code list recovers both "mis-pick" (suggestion is a listed code)
and "schema-gap" (suggestion is out-of-schema) — without coupling the grader to
the schema. There is intentionally NO critique field (the reasoning model thinks
in its own scratchpad; a written critique only burns output tokens).
"""

from pydantic import BaseModel, Field


class KGFactVerdict(BaseModel):
    """One judge verdict on one extracted KG fact (schema-blind)."""

    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices the judge independently reads as bearing "
        "on this fact. May differ from the extractor's.",
    )
    macro_relevant: bool = Field(
        default=True,
        description="False if the fact does not bear on the broad macroeconomy.",
    )
    subject_type_suggestion: str = Field(
        default="",
        description="Blank if the subject's type label is a reasonable fit; "
        "otherwise a SHORT better label (1-3 words or a code), never a sentence.",
    )
    relation_suggestion: str = Field(
        default="",
        description="Blank if the relation is a reasonable fit; otherwise a SHORT "
        "better relation word/label, never a sentence.",
    )
    object_type_suggestion: str = Field(
        default="",
        description="Blank if the object's type label is a reasonable fit; "
        "otherwise a SHORT better label (1-3 words or a code), never a sentence.",
    )
    correct: bool = Field(
        default=True,
        description="True if the stated relationship between the two entities is "
        "true to the article, in the stated direction. False for a fabricated, "
        "inverted, or otherwise unsupported claim.",
    )
