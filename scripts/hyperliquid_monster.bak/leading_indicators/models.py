"""
Data models for Leading Indicators.

Contains all dataclasses and enums used by the leading indicators module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class OIMomentum(Enum):
    """Open Interest momentum classification."""
    BULLISH_CONFIRM = "bullish_confirm"      # OI rising + price rising
    BEARISH_CONFIRM = "bearish_confirm"      # OI rising + price falling
    BULLISH_EXHAUSTION = "bullish_exhaustion"  # OI falling + price rising (squeeze)
    BEARISH_EXHAUSTION = "bearish_exhaustion"  # OI falling + price falling (capitulation)
    NEUTRAL = "neutral"


@dataclass
class OISnapshot:
    """Snapshot of Open Interest data."""
    timestamp: float
    coin: str
    open_interest: float      # Total OI in USD
    mark_price: float
    funding_rate: float       # Current hourly funding
    premium: float            # Mark vs oracle diff
    day_volume: float         # 24h notional volume


@dataclass
class OIAnalysis:
    """Analysis result for OI-based signals."""
    coin: str
    momentum: OIMomentum
    oi_change_pct: float      # % change in OI over lookback
    price_change_pct: float   # % change in price over lookback
    conviction_score: float   # 0-100, how strong the signal is
    should_confirm_long: bool
    should_confirm_short: bool
    warning: Optional[str] = None


@dataclass
class LiquidationCluster:
    """Detected liquidation cluster."""
    coin: str
    side: str                 # "long" or "short"
    estimated_price: float    # Price level where liquidations cluster
    estimated_size: float     # Estimated USD at risk
    distance_pct: float       # Distance from current price


@dataclass
class PremiumAnalysis:
    """Premium/Discount analysis for directional pressure."""
    coin: str
    premium_pct: float        # Premium as percentage of price
    direction_bias: str       # "bullish", "bearish", or "neutral"
    pressure_score: float     # 0-100, how strong the pressure
    is_extreme: bool          # True if premium is at extreme levels


@dataclass
class FundingVelocity:
    """Funding rate velocity analysis."""
    coin: str
    current_rate: float       # Current hourly funding rate
    velocity: float           # Rate of change (acceleration)
    direction: str            # "accelerating_bullish", "accelerating_bearish", "decelerating", "flipping"
    flip_warning: bool        # True if funding about to flip sign
    annualized_apr: float     # Current rate annualized


@dataclass
class VolumeSurge:
    """Volume surge detection."""
    coin: str
    current_volume: float     # Current 24h volume
    avg_volume: float         # Average volume over lookback
    surge_ratio: float        # Current / Average
    is_surging: bool          # True if volume significantly above average
    surge_direction: str      # "bullish", "bearish", or "unknown" based on price action


@dataclass
class DirectionalSignal:
    """
    The "HolyGrail" combined directional signal.

    Combines all leading indicators into a single direction recommendation.
    """
    coin: str
    direction: str            # "strong_long", "long", "neutral", "short", "strong_short"
    confidence: float         # 0-100 confidence score

    # Component scores (each -100 to +100, positive = bullish)
    oi_score: float
    premium_score: float
    funding_velocity_score: float
    volume_score: float

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Detailed reasoning
    reasoning: str = ""


# =============================================================================
# PHASE 1 GOD MODE: SPIKE DETECTION & MARKET REGIME
# =============================================================================

@dataclass
class OISpikeResult:
    """Result of OI spike detection - instant violent OI changes."""
    coin: str
    detected: bool
    spike_type: str           # "spike_up", "spike_down", or "none"
    change_pct: float         # Percentage change in single snapshot
    interpretation: str       # What this means for trading
    timestamp: float


@dataclass
class PremiumSpikeResult:
    """Result of premium spike detection - sudden premium changes."""
    coin: str
    detected: bool
    spike_type: str           # "bullish_spike", "bearish_spike", or "none"
    change_pct: float         # Percentage change in premium
    previous_premium: float
    current_premium: float
    interpretation: str
    timestamp: float


@dataclass
class MarketRegime:
    """
    Market-wide regime detection across all monitored coins.

    Detects when the entire market is crowded in one direction,
    which often precedes market-wide squeezes.
    """
    regime: str               # "crowded_long", "crowded_short", "mixed", "neutral"
    confidence: float         # 0-100 how confident we are in the regime
    bullish_coins: int        # Number of coins with bullish funding
    bearish_coins: int        # Number of coins with bearish funding
    neutral_coins: int        # Number of coins with neutral funding
    total_coins: int
    avg_funding_rate: float   # Average funding across all coins
    squeeze_risk: str         # "high", "medium", "low"
    dominant_direction: str   # "long", "short", or "none"
    warnings: List[str] = field(default_factory=list)
