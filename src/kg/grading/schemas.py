"""Pydantic schemas for the KG grader's statement-level verdict.

Schema-BLIND by design: the grader is never handed the entity/relation taxonomy,
so it judges from an outside reader's point of view and this verdict never has to
know the valid codes. One verdict per statement; field order is chain-of-thought
(read -> macro-relevance -> is the statement true to the article -> does it assert
a direction -> per-triplet faithfulness).

`supported` = the statement is true to the article. Per-triplet `faithful` = the
triplet encodes the statement in the right DIRECTION (false if reversed, fabricated,
or it drops a direction the statement asserts). `asserts_direction` is judged from
the statement wording alone (NOT the relation code) so the directional A/B slice is
model-agnostic. A `*_suggestion` is blank when the extractor's label is a fine fit;
non-blank it carries a SHORT better label.
"""

from pydantic import BaseModel, Field


class TripletVerdict(BaseModel):
    """Judge verdict on one (subject, relation, object) triplet of a statement."""

    faithful: bool = Field(
        default=True,
        description="True if the triplet encodes the statement in the right "
        "direction. False only if reversed, fabricated, or it drops a direction "
        "the statement asserts.",
    )
    relation_suggestion: str = Field(
        default="",
        description="Blank if the relation is a reasonable fit; otherwise a SHORT "
        "better relation word/label, never a sentence.",
    )
    subject_type_suggestion: str = Field(
        default="",
        description="Blank if the subject's type label is a reasonable fit; "
        "otherwise a SHORT better label (1-3 words or a code).",
    )
    object_type_suggestion: str = Field(
        default="",
        description="Blank if the object's type label is a reasonable fit; "
        "otherwise a SHORT better label (1-3 words or a code).",
    )


class KGStatementVerdict(BaseModel):
    """One judge verdict on one extracted statement (schema-blind)."""

    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices the judge independently reads as bearing "
        "on this statement.",
    )
    macro_relevant: bool = Field(
        default=True,
        description="False if the statement does not bear on the broad macroeconomy.",
    )
    supported: bool = Field(
        default=True,
        description="True if the statement is true to the article. For a PREDICTION "
        "or OPINION, true means the article really attributes that expectation/view.",
    )
    asserts_direction: bool = Field(
        default=False,
        description="True if the statement itself asserts a directional move "
        "(rise/fall/increase/decrease/widen/tighten), judged from its wording.",
    )
    triplets: list[TripletVerdict] = Field(
        default_factory=list,
        description="Per-triplet verdicts, aligned to the input triplet order.",
    )
