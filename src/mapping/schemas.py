"""
Pydantic schemas for mapping headlines to macro assets.
"""

from pydantic import BaseModel, Field
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
    signal: str = Field(
        default="strong",
        description='Signal strength: "strong" (direct transmission channel, '
        'few reasoning steps) or "weak" (indirect but plausible, requires '
        'a longer chain of reasoning). Both are valuable for asset pricing.',
    )
    reasoning: str = Field(
        default="",
        description="Evidence-grounded reasoning: quote or cite the specific "
        "text that triggers this mapping, then explain the transmission "
        "channel to this asset's price. Spell out multi-step chains.",
    )


class ValidationResult(BaseModel):
    valid: bool = Field(
        description="True if you agree the asset is affected via "
        "the described transmission channel.",
    )
    signal: str = Field(
        default="strong",
        description='Your final determination of signal strength: "strong" '
        '(direct channel) or "weak" (indirect but plausible). You may '
        'upgrade or downgrade the mappers\' proposed signal.',
    )
    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="Paragraph indices containing the strongest evidence. "
        "Required regardless of valid=true/false.",
    )
    reasoning: str = Field(
        default="",
        description="If valid=true: the transmission channel you agree with. "
        "If valid=false: why the mappers' reasoning is flawed.",
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


