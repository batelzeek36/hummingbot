"""
Coinglass API Data Models

All dataclasses and enums for Coinglass API responses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CVDDirection(Enum):
    """CVD trend direction."""
    STRONG_BULLISH = "strong_bullish"    # CVD rising strongly
    BULLISH = "bullish"                   # CVD rising
    NEUTRAL = "neutral"                   # CVD flat
    BEARISH = "bearish"                   # CVD falling
    STRONG_BEARISH = "strong_bearish"    # CVD falling strongly


@dataclass
class CVDSnapshot:
    """Snapshot of CVD data."""
    timestamp: float
    symbol: str
    buy_volume: float          # Taker buy volume
    sell_volume: float         # Taker sell volume
    cvd: float                 # Cumulative delta (buy - sell)
    cvd_change: float          # Change from previous snapshot
    direction: CVDDirection


@dataclass
class SpotPerpDivergence:
    """
    Spot vs Perpetual CVD divergence - THE killer signal.

    When perp CVD diverges from spot CVD, leverage traders are getting trapped.
    """
    symbol: str
    spot_cvd: float
    perp_cvd: float
    spot_direction: CVDDirection
    perp_direction: CVDDirection
    divergence_type: str       # "bullish_divergence", "bearish_divergence", "aligned"
    signal_strength: float     # 0-100
    interpretation: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class LiquidationCluster:
    """A cluster of liquidations at a price level."""
    price: float
    side: str                  # "long" or "short"
    estimated_size_usd: float
    distance_pct: float        # Distance from current price
    risk_level: str            # "low", "medium", "high"


@dataclass
class LiquidationHeatmap:
    """Liquidation heatmap data for a symbol."""
    symbol: str
    current_price: float
    long_clusters: List[LiquidationCluster]    # Liquidations below price
    short_clusters: List[LiquidationCluster]   # Liquidations above price
    nearest_long_liq: Optional[float]          # Nearest long liquidation price
    nearest_short_liq: Optional[float]         # Nearest short liquidation price
    magnetic_direction: str                     # "up" (toward short liqs) or "down" (toward long liqs)
    interpretation: str


@dataclass
class LongShortRatio:
    """Long/Short ratio data."""
    symbol: str
    exchange: str
    long_ratio: float          # Percentage of longs (0-100)
    short_ratio: float         # Percentage of shorts (0-100)
    long_short_ratio: float    # Ratio (>1 = more longs)
    sentiment: str             # "crowded_long", "crowded_short", "balanced"
    crowding_score: float      # 0-100, how crowded
    contrarian_signal: str     # "long", "short", "neutral"
