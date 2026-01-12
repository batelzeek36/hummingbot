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
    "VVV-USD": CoinVolatility.EXTREME,   # Max 40.5% daily - MOVED from HIGH!
    "IP-USD": CoinVolatility.EXTREME,    # Max 30.7% daily, buffer +19.3% - MOVED from HIGH for safety
    # MOG-USD, JEFF-USD - NOT on Hyperliquid
    # ANIME-USD REMOVED - Max 51.9% exceeds 50% liq threshold, no safe leverage exists
}


# Leverage limits by volatility class
VOLATILITY_LEVERAGE = {
    CoinVolatility.SAFE: 8,
    CoinVolatility.MEDIUM: 5,
    CoinVolatility.HIGH: 3,
    CoinVolatility.EXTREME: 2,
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
