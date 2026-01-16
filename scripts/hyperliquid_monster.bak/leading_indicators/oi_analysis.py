"""
Open Interest (OI) Analysis for Leading Indicators.

OI is a TRUE leading indicator that predicts price moves:
- OI Rising + Price Rising = New longs entering, move has conviction
- OI Rising + Price Falling = New shorts entering, move has conviction
- OI Falling + Price Rising = Short squeeze (shorts closing), exhaustion soon
- OI Falling + Price Falling = Long capitulation, bounce setup
"""

from typing import Deque, Optional, Tuple

from .models import OIAnalysis, OIMomentum, OISnapshot


def analyze_oi_momentum(
    history: Deque[OISnapshot],
    oi_threshold: float,
    price_threshold: float,
) -> Optional[OIAnalysis]:
    """
    Analyze OI momentum for a trading pair.

    This is the KEY leading indicator logic:
    - OI Rising + Price Rising = Bullish confirmation (new longs)
    - OI Rising + Price Falling = Bearish confirmation (new shorts)
    - OI Falling + Price Rising = Bullish exhaustion (short squeeze ending)
    - OI Falling + Price Falling = Bearish exhaustion (capitulation ending)

    Args:
        history: Deque of OI snapshots for the pair
        oi_threshold: OI change % to consider significant
        price_threshold: Price change % to consider significant

    Returns:
        OIAnalysis or None if insufficient data
    """
    if history is None or len(history) < 3:
        return None

    # Get oldest and newest for comparison
    oldest = history[0]
    newest = history[-1]

    # Calculate changes
    if oldest.open_interest == 0 or oldest.mark_price == 0:
        return None

    oi_change_pct = ((newest.open_interest - oldest.open_interest)
                     / oldest.open_interest * 100)
    price_change_pct = ((newest.mark_price - oldest.mark_price)
                        / oldest.mark_price * 100)

    # Classify momentum
    oi_rising = oi_change_pct > oi_threshold
    oi_falling = oi_change_pct < -oi_threshold
    price_rising = price_change_pct > price_threshold
    price_falling = price_change_pct < -price_threshold

    if oi_rising and price_rising:
        momentum = OIMomentum.BULLISH_CONFIRM
        should_long = True
        should_short = False
        conviction = min(100, abs(oi_change_pct) * 10)
    elif oi_rising and price_falling:
        momentum = OIMomentum.BEARISH_CONFIRM
        should_long = False
        should_short = True
        conviction = min(100, abs(oi_change_pct) * 10)
    elif oi_falling and price_rising:
        momentum = OIMomentum.BULLISH_EXHAUSTION
        should_long = False  # Don't chase the squeeze
        should_short = False  # Don't short into squeeze either
        conviction = min(100, abs(oi_change_pct) * 8)
    elif oi_falling and price_falling:
        momentum = OIMomentum.BEARISH_EXHAUSTION
        should_long = True   # Capitulation = bounce setup
        should_short = False
        conviction = min(100, abs(oi_change_pct) * 8)
    else:
        momentum = OIMomentum.NEUTRAL
        should_long = True   # Neutral = no blocking
        should_short = True
        conviction = 0

    # Generate warning if needed
    warning = None
    if momentum == OIMomentum.BULLISH_EXHAUSTION:
        warning = f"Short squeeze detected - OI falling {oi_change_pct:.1f}% while price up {price_change_pct:.1f}%"
    elif momentum == OIMomentum.BEARISH_EXHAUSTION:
        warning = f"Long capitulation detected - potential bounce setup"

    return OIAnalysis(
        coin=newest.coin,
        momentum=momentum,
        oi_change_pct=oi_change_pct,
        price_change_pct=price_change_pct,
        conviction_score=conviction,
        should_confirm_long=should_long,
        should_confirm_short=should_short,
        warning=warning,
    )


def get_oi_direction_signal(analysis: Optional[OIAnalysis]) -> Tuple[Optional[str], float]:
    """
    Get simple OI direction signal for momentum strategy.

    Args:
        analysis: OIAnalysis result

    Returns:
        Tuple of (signal, conviction) where signal is:
        - "bullish" if OI confirms long entry
        - "bearish" if OI confirms short entry
        - "neutral" if no clear signal
        - None if insufficient data
    """
    if analysis is None:
        return None, 0

    if analysis.momentum == OIMomentum.BULLISH_CONFIRM:
        return "bullish", analysis.conviction_score
    elif analysis.momentum == OIMomentum.BEARISH_CONFIRM:
        return "bearish", analysis.conviction_score
    elif analysis.momentum == OIMomentum.BEARISH_EXHAUSTION:
        # Capitulation = contrarian bullish
        return "bullish", analysis.conviction_score * 0.7
    elif analysis.momentum == OIMomentum.BULLISH_EXHAUSTION:
        # Don't give direction signal during squeeze
        return "neutral", 0
    else:
        return "neutral", 0


def should_block_entry(analysis: Optional[OIAnalysis], direction: str) -> Tuple[bool, str]:
    """
    Check if an entry should be blocked based on OI analysis.

    Args:
        analysis: OIAnalysis result
        direction: "long" or "short"

    Returns:
        Tuple of (should_block, reason)
    """
    if analysis is None:
        return False, ""

    if direction == "long":
        if analysis.momentum == OIMomentum.BULLISH_EXHAUSTION:
            return True, f"OI exhaustion: squeeze ending (OI {analysis.oi_change_pct:+.1f}%)"
        if analysis.momentum == OIMomentum.BEARISH_CONFIRM:
            return True, f"OI bearish: new shorts entering (OI {analysis.oi_change_pct:+.1f}%)"

    elif direction == "short":
        if analysis.momentum == OIMomentum.BEARISH_EXHAUSTION:
            return True, f"OI exhaustion: capitulation (bounce likely)"
        if analysis.momentum == OIMomentum.BULLISH_CONFIRM:
            return True, f"OI bullish: new longs entering (OI {analysis.oi_change_pct:+.1f}%)"

    return False, ""


def get_oi_funding_combined_signal(
    history: Deque[OISnapshot],
    oi_threshold: float,
) -> Optional[str]:
    """
    Combined OI + Funding signal for squeeze detection.

    This is the most powerful signal:
    - High positive funding + Rising OI = Long squeeze setup (too crowded)
    - High negative funding + Rising OI = Short squeeze setup

    Args:
        history: Deque of OI snapshots for the pair
        oi_threshold: OI change % to consider significant

    Returns:
        "long_squeeze_warning", "short_squeeze_warning", or None
    """
    if history is None or len(history) < 3:
        return None

    newest = history[-1]
    oldest = history[0]

    if oldest.open_interest == 0:
        return None

    oi_change_pct = ((newest.open_interest - oldest.open_interest)
                     / oldest.open_interest * 100)

    # High funding thresholds (hourly rate)
    HIGH_POSITIVE_FUNDING = 0.0005  # 0.05% hourly = ~438% APR
    HIGH_NEGATIVE_FUNDING = -0.0005

    oi_rising = oi_change_pct > oi_threshold

    if newest.funding_rate > HIGH_POSITIVE_FUNDING and oi_rising:
        return "long_squeeze_warning"  # Longs crowded + more piling in
    elif newest.funding_rate < HIGH_NEGATIVE_FUNDING and oi_rising:
        return "short_squeeze_warning"  # Shorts crowded + more piling in

    return None
