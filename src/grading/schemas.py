"""Pydantic schemas for the grader's per-mapping verdicts."""

from typing import Literal

from pydantic import BaseModel, Field


class GraderResult(BaseModel):
    """Result of a single (article, mapping) grader call.

    Field order is the chain-of-thought, parallel to
    ``mapping.schemas.SingleAssetResult``: the model first cites which
    paragraphs it thinks support the mapper's claim, then commits to a
    continuous score, then snaps to a binary verdict.

    The verdict is binary by design. The mapper's continuous
    ``relevance_score`` already encodes uncertainty; encoding it again
    in the verdict (via a ``borderline`` bucket) double-encodes the
    same signal and tempts the model to default to "borderline" on
    every uncertain call. Forcing a binary commit makes the verdict a
    cleaner agreement signal: does the grader agree the mapper got
    this tag right (including the score), or not.
    """

    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices the grader judges as supporting "
        "(or failing to support) the mapper's claim. May overlap with the "
        "mapper's evidence_paragraphs or differ entirely.",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Independent [0, 1] relevance score on the same scale "
        "as the mapper. Carries all of the grader's uncertainty about the "
        "tag — the verdict is intentionally binary so this score is the "
        "only place uncertainty lives.",
    )
    verdict: Literal["correct", "incorrect"] = Field(
        default="correct",
        description="Binary judgment on whether the mapper's tag holds up. "
        "`correct` covers strong agreement, weak-but-real agreement, and "
        "mid-range agreement (mapper made a valid call). `incorrect` "
        "covers either a tag that should not have been made at all, or a "
        "mapper score that is significantly off from the grader's reading. "
        "Default to `correct` if uncertain.",
    )


# Score-gap thresholds for the self-consistency check.
_AGREE_GAP = 0.40   # verdict=correct but scores differ by more than this -> flag
_DISAGREE_GAP = 0.15  # verdict=incorrect but scores differ by less than this -> flag


def is_self_inconsistent(
    mapper_score: float,
    grader_score: float,
    verdict: str,
) -> bool:
    """True when verdict and score-gap tell different stories.

    Under the binary-verdict schema:
      * ``correct`` + |mapper - grader| > 0.40  → the verdict says the
        mapper got it right but the scores disagree substantially.
      * ``incorrect`` + |mapper - grader| < 0.15 → the verdict says the
        mapper made a real mistake but the scores barely differ.

    Either pattern is internally inconsistent and worth flagging for
    review. Flagged rows are kept in the sidecar, not dropped.
    """
    gap = abs(mapper_score - grader_score)
    if verdict == "correct" and gap > _AGREE_GAP:
        return True
    if verdict == "incorrect" and gap < _DISAGREE_GAP:
        return True
    return False
