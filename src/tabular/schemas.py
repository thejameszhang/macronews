"""Sidecar row schema for the tabular-body detector.

One TabularResult per article in the raw NML, joined to the cleaned-text
shards via ``accession_number``. Consumed by ``src/loaders.py`` to add
the ``tabular_body`` filter reason.
"""

from pydantic import BaseModel, Field, computed_field


class TabularResult(BaseModel):
    """Per-article token accounting from the raw NML.

    Stored fields: ``accession_number``, ``p_tokens``, ``pre_tabular_tokens``,
    ``pre_narrative_tokens``.

    Derived:
      * ``pre_total_tokens`` — sum of pre token buckets (informational)
      * ``pre_aligned_pct`` — pre_tabular / pre_total, 0 if pre_total == 0
        (informational; not load-bearing for the filter rule)
      * ``is_tabular_body`` — final filter decision (serialized in JSONL):
            "tabular bucket is the largest, and it's nonzero" — i.e.
            pre_tabular_tokens >= p_tokens AND >= pre_narrative_tokens AND > 0.

    Equivalent to the prior rule (p == 0 AND pre_aligned_pct >= 0.5) with
    the strict ``p == 0`` constraint relaxed to ``pre_tab >= p``: a small
    <p> intro/dateline/trailer no longer kills the filter as long as the
    tabular bucket dominates. The pct >= 0.5 condition is restated as
    ``pre_tab >= pre_nar`` (mathematically identical for pre_total > 0).
    """

    accession_number: str
    p_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens in non-trailer <p> body blocks.",
    )
    pre_tabular_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens in column-aligned <pre> lines, summed across all <pre> blocks.",
    )
    pre_narrative_tokens: int = Field(
        default=0,
        ge=0,
        description="Tokens in non-aligned <pre> lines, summed across all <pre> blocks.",
    )

    @property
    def pre_total_tokens(self) -> int:
        return self.pre_tabular_tokens + self.pre_narrative_tokens

    @property
    def pre_aligned_pct(self) -> float:
        if self.pre_total_tokens == 0:
            return 0.0
        return self.pre_tabular_tokens / self.pre_total_tokens

    @computed_field
    @property
    def is_tabular_body(self) -> bool:
        return (
            self.pre_tabular_tokens > 0
            and self.pre_tabular_tokens >= self.p_tokens
            and self.pre_tabular_tokens >= self.pre_narrative_tokens
        )
