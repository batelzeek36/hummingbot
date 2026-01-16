"""
Premium/Discount Analysis for Leading Indicators.

Premium = (Mark Price - Oracle Price) / Oracle Price
- Positive premium = Longs paying to stay in = Bullish pressure
- Negative premium (discount) = Shorts paying = Bearish pressure
"""

from typing import Deque, Optional

from .models import OISnapshot, PremiumAnalysis


def analyze_premium(history: Deque[OISnapshot]) -> Optional[PremiumAnalysis]:
    """
    Analyze premium/discount for directional pressure.

    Premium = (Mark Price - Oracle Price) / Oracle Price
    - Positive premium = Longs paying to stay in = Bullish pressure
    - Negative premium (discount) = Shorts paying = Bearish pressure

    Args:
        history: Deque of OI snapshots containing premium data

    Returns:
        PremiumAnalysis or None if insufficient data
    """
    if history is None or len(history) < 1:
        return None

    newest = history[-1]

    # Premium is already in the data as mark vs oracle difference
    # Hyperliquid returns premium as a decimal (e.g., 0.001 = 0.1%)
    premium_pct = newest.premium * 100  # Convert to percentage

    # Thresholds for direction bias
    BULLISH_THRESHOLD = 0.02   # 0.02% premium = bullish pressure
    BEARISH_THRESHOLD = -0.02  # -0.02% discount = bearish pressure
    EXTREME_THRESHOLD = 0.1   # 0.1% = extreme pressure

    if premium_pct > BULLISH_THRESHOLD:
        direction_bias = "bullish"
        pressure_score = min(100, abs(premium_pct) * 500)  # Scale to 0-100
    elif premium_pct < BEARISH_THRESHOLD:
        direction_bias = "bearish"
        pressure_score = min(100, abs(premium_pct) * 500)
    else:
        direction_bias = "neutral"
        pressure_score = 0

    is_extreme = abs(premium_pct) > EXTREME_THRESHOLD

    return PremiumAnalysis(
        coin=newest.coin,
        premium_pct=premium_pct,
        direction_bias=direction_bias,
        pressure_score=pressure_score,
        is_extreme=is_extreme,
    )
