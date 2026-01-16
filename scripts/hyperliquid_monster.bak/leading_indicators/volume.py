"""
Volume Surge Detection for Leading Indicators.

Volume surges often precede big moves:
- Volume surge with price up = bullish confirmation
- Volume surge with price down = bearish confirmation
- Volume surge with price flat = breakout imminent
"""

from typing import Deque, Optional

from .models import OISnapshot, VolumeSurge


def detect_volume_surge(history: Deque[OISnapshot]) -> Optional[VolumeSurge]:
    """
    Detect volume surges that often precede big moves.

    A volume surge with price moving up = bullish confirmation
    A volume surge with price moving down = bearish confirmation
    A volume surge with price flat = breakout imminent (direction unknown)

    Args:
        history: Deque of OI snapshots containing volume data

    Returns:
        VolumeSurge or None if insufficient data
    """
    if history is None or len(history) < 3:
        return None

    newest = history[-1]
    current_volume = newest.day_volume

    # Calculate average volume over history
    volumes = [h.day_volume for h in history]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0

    if avg_volume == 0:
        return None

    surge_ratio = current_volume / avg_volume

    # Threshold for surge detection
    SURGE_THRESHOLD = 1.5  # 50% above average

    is_surging = surge_ratio > SURGE_THRESHOLD

    # Determine surge direction based on price action
    oldest = history[0]
    if oldest.mark_price == 0:
        return None

    price_change_pct = ((newest.mark_price - oldest.mark_price)
                       / oldest.mark_price * 100)

    if is_surging:
        if price_change_pct > 0.5:
            surge_direction = "bullish"
        elif price_change_pct < -0.5:
            surge_direction = "bearish"
        else:
            surge_direction = "breakout_imminent"
    else:
        surge_direction = "normal"

    return VolumeSurge(
        coin=newest.coin,
        current_volume=current_volume,
        avg_volume=avg_volume,
        surge_ratio=surge_ratio,
        is_surging=is_surging,
        surge_direction=surge_direction,
    )
