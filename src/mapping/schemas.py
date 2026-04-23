"""
Pydantic schemas for mapping articles to macro assets.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SingleAssetResult(BaseModel):
    """Result of a single-asset mapping call (one article × one asset).

    Field order is deliberately evidence -> reasoning -> signal -> score ->
    decision. Under guided JSON decoding, fields are emitted in declaration
    order; putting `relevant` last forces the model to enumerate evidence
    and reason BEFORE committing to a yes/no. This is the chain-of-thought-
    via-field-order technique and is required for reliable detection of
    localized relevance in otherwise off-topic articles.
    """
    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices where a force acting on this asset "
        "appears. Scan the article paragraph by paragraph and list every "
        "paragraph that names a rule-triggering force. Empty if not relevant.",
    )
    reasoning: str = Field(
        default="",
        description="Rule citation and quoted phrase from the article.",
    )
    signal: Literal["strong", "weak"] = Field(
        default="weak",
        description='Signal strength: "strong" if the asset is NAMED in the '
        'text (rule 1), else "weak".',
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score from 0.0 to 1.0. 0.0 if not relevant.",
    )
    relevant: bool = Field(
        default=False,
        description="True if evidence_paragraphs is non-empty AND the "
        "evidence triggers a rule for this asset. Commit to this decision "
        "AFTER enumerating evidence and reasoning above.",
    )

    @field_validator("signal", mode="before")
    @classmethod
    def normalize_signal(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip().rstrip(",").lower()
            if v.startswith("strong"):
                return "strong"
        return "weak"
