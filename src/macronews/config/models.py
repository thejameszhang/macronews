from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum
import polars as pl


class AssetClass(Enum):
    """Enumeration of asset classes for futures contracts."""
    COMMODITY = "commodity"
    CURRENCY = "currency"
    BOND = "government bond"
    EQUITY = "equity index"
    VOLATILITY = "volatility"
    STIR = "short-term interest rate"
    SECTOR = "US equity sector"


@dataclass
class Future:
    """Data class representing a futures contract configuration."""
    symbol: str
    exchange_name: str
    calcseriesname: str
    name: str
    asset_class: AssetClass
    curcdd: str
    ric: str
    
    @classmethod
    def from_dict(cls, symbol: str, data: dict) -> 'Future':
        """Create a Future instance from a symbol and data dictionary."""
        return cls(
            symbol=symbol,
            exchange_name=data['exchange_name'],
            calcseriesname=data['calcseriesname'],
            name=data['name'],
            asset_class=AssetClass(data['asset_class']),
            curcdd=data['curcdd'],
            ric=data['ric'],
        )
