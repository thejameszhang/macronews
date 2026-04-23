"""
Pydantic schemas for mapping headlines to macro assets.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator
from utils.config import load_asset_universe
from config.paths import ASSET_UNIVERSE_YAML

# Load full universe from YAML
_UNIVERSE = load_asset_universe(ASSET_UNIVERSE_YAML)

ASSET_SYMBOLS = list(_UNIVERSE.keys())
ASSET_SYMBOLS_SET = set(ASSET_SYMBOLS)

# Build NAME_TO_SYMBOL dynamically
NAME_TO_SYMBOL = {}
for sym, info in _UNIVERSE.items():
    name = info.get("name")
    if name:
        NAME_TO_SYMBOL[name.lower()] = sym
    # Also add the symbol itself as a fallback (lowercase)
    NAME_TO_SYMBOL[sym.lower()] = sym

# Add specific aliases for better extraction
NAME_TO_SYMBOL.update({
    "10y treasury": "ZN",
    "jgb": "160120001",
    "aud": "6A",
    "gbp": "6B",
    "cad": "6C",
    "eur": "6E",
    "jpy": "6J",
    "chf": "6S",
    "nok": "NOK",
    "s&p 500": "ES",
    "ftse": "Z",
    "tsx 60": "SXF",
    "euro stoxx": "FESX",
    "nikkei": "164120019",
    "kospi": "A01"
})



class AssetMapping(BaseModel):
    """A single asset tagged by a mapper, with its transmission channel reasoning."""
    asset: str = Field(
        description="Futures symbol from the asset universe.",
    )
    signal: Literal["strong", "weak"] = Field(
        default="strong",
        description='Signal strength: "strong" (direct transmission channel, '
        'few reasoning steps) or "weak" (indirect but plausible, requires '
        'a longer chain of reasoning). Both are valuable for asset pricing.',
    )

    @field_validator("signal", mode="before")
    @classmethod
    def normalize_signal(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip().rstrip(",").lower()
            if v.startswith("strong"):
                return "strong"
        return "weak"
    reasoning: str = Field(
        default="",
        description="Evidence-grounded reasoning: quote or cite the specific "
        "text that triggers this mapping, then explain the transmission "
        "channel to this asset's price. Spell out multi-step chains.",
    )


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


class SummarizeResult(BaseModel):
    """Result of the summarize call (one per article)."""
    company_specific: bool = Field(
        default=False,
        description="True if the article is primarily about a specific company.",
    )
    macro_summary: str = Field(
        default="",
        description="Macro analysis summary of the article.",
    )


class MappingResult(BaseModel):
    relevant_assets: list[AssetMapping] = Field(
        default_factory=list,
        description="Assets with per-asset transmission channel reasoning. "
        "Empty list if no assets are relevant.",
    )
    company_specific: bool = Field(
        default=False,
        description="True if the article is primarily about a specific company "
        "(earnings, M&A, management changes, etc.). When true, downstream "
        "paragraph-level and validation stages are skipped.",
    )
    macro_summary: str = Field(
        default="",
        description="2-3 sentence macro summary of the article. "
        "Empty string if macro_themes is ['NONE']. "
        "Only populated by article-level v2 prompt.",
    )

    @property
    def asset_symbols(self) -> list[str]:
        """Extract just the symbol strings for downstream code."""
        return [a.asset for a in self.relevant_assets]


