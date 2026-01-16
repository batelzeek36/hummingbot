"""
Volatility-based leverage classification for Hyperliquid Monster Bot.

Based on 30-day historical volatility analysis (2026-01-11).
Determines safe leverage levels per coin to avoid liquidation.
"""

from enum import Enum


class CoinVolatility(Enum):
    """Volatility classification for leverage decisions."""
    SAFE = "safe"           # Max daily <10%, can handle 8x
    MEDIUM = "medium"       # Max daily <15%, can handle 5x
    HIGH = "high"           # Max daily <25%, max 3x
    EXTREME = "extreme"     # Max daily 25%+, max 2x


# Coin classifications based on volatility analysis
# CRITICAL: These must be verified against liquidation thresholds!
# 8x = 12.5% liq, 5x = 20% liq, 3x = 33% liq, 2x = 50% liq
# NOTE: Hyperliquid uses kPEPE/kBONK (1000x multiplier) for small-unit tokens
COIN_VOLATILITY = {
    # SAFE - Low volatility, high leverage OK (max daily <10%)
    "BTC-USD": CoinVolatility.SAFE,      # Max 6.0% daily, buffer +6.5%

    # MEDIUM - Moderate volatility (max daily <15%)
    "ETH-USD": CoinVolatility.MEDIUM,    # Max 9.8% daily, buffer +10.2% - MOVED from SAFE for safety
    "SOL-USD": CoinVolatility.MEDIUM,    # Max 10.4% daily, buffer +9.6%
    "TAO-USD": CoinVolatility.MEDIUM,    # Max 13.1% daily, buffer +6.9%

    # HIGH - Elevated volatility, reduce leverage (max daily <25%)
    "DOGE-USD": CoinVolatility.HIGH,     # Max 15.0% daily, buffer +18.3%
    "HYPE-USD": CoinVolatility.HIGH,     # Max 17.4% daily, buffer +15.9%
    # PURR-USD REMOVED - Max 55.8% exceeds 50% threshold, no safe leverage
    "AVNT-USD": CoinVolatility.EXTREME,  # Max 25.8% daily - borderline, moved to 2x for safety

    # EXTREME - Very high volatility, minimal leverage (max daily 25%+)
    "kPEPE-USD": CoinVolatility.EXTREME, # Max 28.1% daily (Hyperliquid uses kPEPE)
    "kBONK-USD": CoinVolatility.EXTREME, # Max 39.7% daily (Hyperliquid uses kBONK)
    "WIF-USD": CoinVolatility.EXTREME,   # Max 26.6% daily, buffer +23.4%
    "HYPER-USD": CoinVolatility.EXTREME, # Max 33.4% daily, buffer +16.6%
    "IP-USD": CoinVolatility.EXTREME,    # Max 30.7% daily, buffer +19.3% - MOVED from HIGH for safety
    # MOG-USD, JEFF-USD - NOT on Hyperliquid
    # ANIME-USD REMOVED - Max 51.9% exceeds 50% liq threshold, no safe leverage exists
    # VVV-USD REMOVED - Max 40.5% daily, only 9.5% buffer to 50% liq threshold - too risky
}


# Leverage limits by volatility class
VOLATILITY_LEVERAGE = {
    CoinVolatility.SAFE: 8,
    CoinVolatility.MEDIUM: 5,
    CoinVolatility.HIGH: 3,
    CoinVolatility.EXTREME: 2,
}


# Minimum loss threshold (%) for max loss exit by volatility class
# These represent typical 10-15 minute price swings that are "noise"
# Exit only triggers if loss exceeds this AND exceeds funding multiplier
VOLATILITY_MIN_LOSS_THRESHOLD = {
    CoinVolatility.SAFE: 0.3,      # BTC: 6% daily max → ~0.1-0.3% in 10min
    CoinVolatility.MEDIUM: 0.5,    # SOL/ETH: 10% daily → ~0.2-0.5% in 10min
    CoinVolatility.HIGH: 1.0,      # DOGE/HYPE: 17% daily → ~0.5-1% in 10min
    CoinVolatility.EXTREME: 3.0,   # kBONK/HYPER: 30-40% daily → ~1-3% in 10min (raised from 2% to reduce false exits)
}


# Grid trend pause threshold (%) by volatility class
# EMA separation must exceed this to consider it a "strong trend" worth pausing for
# Scaled to ~10% of daily max move to avoid false pauses on normal intraday swings
VOLATILITY_GRID_TREND_THRESHOLD = {
    CoinVolatility.SAFE: 0.5,      # BTC: 6% daily → 0.5% is ~8% of daily range
    CoinVolatility.MEDIUM: 1.0,    # SOL/ETH: 10% daily → 1.0% is 10% of daily range
    CoinVolatility.HIGH: 1.5,      # DOGE/HYPE: 17% daily → 1.5% is ~9% of daily range
    CoinVolatility.EXTREME: 2.5,   # kBONK/VVV: 40% daily → 2.5% is ~6% of daily range
}


def get_safe_leverage(pair: str, max_leverage: int = 8) -> int:
    """
    Get safe leverage for a trading pair based on volatility.

    Args:
        pair: Trading pair (e.g., "BTC-USD")
        max_leverage: Maximum allowed leverage cap

    Returns:
        Safe leverage level for the pair
    """
    volatility = COIN_VOLATILITY.get(pair, CoinVolatility.HIGH)  # Default to HIGH if unknown
    safe_lev = VOLATILITY_LEVERAGE.get(volatility, 3)
    return min(safe_lev, max_leverage)


def get_min_loss_threshold(pair: str) -> float:
    """
    Get minimum loss threshold (%) for max loss exit based on volatility.

    This prevents false exits from normal market noise on volatile coins.

    Args:
        pair: Trading pair (e.g., "BTC-USD")

    Returns:
        Minimum loss percentage threshold (e.g., 0.5 for 0.5%)
    """
    volatility = COIN_VOLATILITY.get(pair, CoinVolatility.HIGH)  # Default to HIGH if unknown
    return VOLATILITY_MIN_LOSS_THRESHOLD.get(volatility, 1.0)


def get_volatility_class(pair: str) -> CoinVolatility:
    """
    Get volatility classification for a trading pair.

    Args:
        pair: Trading pair (e.g., "BTC-USD")

    Returns:
        CoinVolatility enum value
    """
    return COIN_VOLATILITY.get(pair, CoinVolatility.HIGH)


def get_grid_trend_threshold(pair: str) -> float:
    """
    Get grid trend pause threshold (%) based on volatility.

    This determines how much EMA separation is needed to consider
    a trend "strong enough" to pause grid trading.

    Args:
        pair: Trading pair (e.g., "SOL-USD")

    Returns:
        Trend threshold percentage (e.g., 1.0 for 1%)
    """
    volatility = COIN_VOLATILITY.get(pair, CoinVolatility.HIGH)  # Default to HIGH if unknown
    return VOLATILITY_GRID_TREND_THRESHOLD.get(volatility, 1.5)
