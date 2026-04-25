"""
Pydantic schemas for mapping articles to macro assets.
"""

from pydantic import BaseModel, Field


class SingleAssetResult(BaseModel):
    """Result of a single-asset mapping call (one article × one asset).

    Field order is the chain-of-thought:
    evidence -> score -> relevant.
    """
    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices where a force acting on this asset "
        "appears. Scan the article paragraph by paragraph and list every "
        "paragraph that names a rule-triggering force. Empty if not relevant.",
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
