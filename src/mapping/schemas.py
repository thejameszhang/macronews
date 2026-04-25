"""
Pydantic schemas for mapping articles to macro assets.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SingleAssetResult(BaseModel):
    """Result of a single-asset mapping call (one article × one asset).

    Experimental no-reasoning schema: the `reasoning` field is dropped to
    cut output tokens, and the prompt carries an instruction asking the
    model to reason before committing. Field order: evidence -> signal ->
    score -> decision. Compare against HEAD (with `reasoning`) on gold.
    """
    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices where a force acting on this asset "
        "appears. Scan the article paragraph by paragraph and list every "
        "paragraph that names a rule-triggering force. Empty if not relevant.",
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
        "evidence triggers a rule for this asset.",
    )

    @field_validator("signal", mode="before")
    @classmethod
    def normalize_signal(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip().rstrip(",").lower()
            if v.startswith("strong"):
                return "strong"
        return "weak"
