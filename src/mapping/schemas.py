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

MACRO_THEMES = [
    "MONETARY_POLICY", "INFLATION_DATA", "LABOR_DATA", "GDP_GROWTH",
    "FISCAL_POLICY", "ENERGY_SUPPLY", "AGRICULTURAL_SUPPLY", "METALS_MARKET",
    "GEOPOLITICAL_RISK", "TRADE_POLICY", "CURRENCY_POLICY", "NONE",
]
MACRO_THEMES_SET = set(MACRO_THEMES)

REGIONS = ["US", "EUROPE", "UK", "JAPAN", "CANADA", "AUSTRALIA", "GLOBAL", "NONE"]
REGIONS_SET = set(REGIONS)


class ValidationResult(BaseModel):
    valid: bool = Field(
        description="True if the asset's relevance is grounded in specific textual evidence.",
    )
    evidence_paragraphs: list[int] = Field(
        default_factory=list,
        description="0-indexed paragraph indices containing the grounding evidence. "
        "Empty if valid=false.",
    )
    rationale: str = Field(
        default="",
        description="1-2 sentence explanation of the transmission channel "
        "from the article's content to this specific asset. Empty if valid=false.",
    )


class MappingResult(BaseModel):
    macro_themes: list[str] = Field(
        default_factory=lambda: ["NONE"],
        description="1 or 2 macro themes from the allowed set. "
        "Use ['NONE'] if no macroeconomic relevance.",
    )
    regions: list[str] = Field(
        default_factory=lambda: ["NONE"],
        description="1 or 2 regions from the allowed set. "
        "Use ['NONE'] if macro_themes is ['NONE'].",
    )
    relevant_assets: list[str] = Field(
        default_factory=list,
        description="List of asset symbols this article is relevant to. "
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


