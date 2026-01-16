"""
Funding Velocity Analysis for Leading Indicators.

Funding velocity detects:
- Funding accelerating in bullish direction (more positive)
- Funding accelerating in bearish direction (more negative)
- Funding decelerating (moving toward neutral)
- Funding about to flip sign (reversal warning)
"""

from typing import Deque, Optional

from .models import FundingVelocity, OISnapshot


def analyze_funding_velocity(history: Deque[OISnapshot]) -> Optional[FundingVelocity]:
    """
    Analyze funding rate velocity (acceleration/deceleration).

    This detects:
    - Funding accelerating in bullish direction (more positive)
    - Funding accelerating in bearish direction (more negative)
    - Funding decelerating (moving toward neutral)
    - Funding about to flip sign (reversal warning)

    Args:
        history: Deque of OI snapshots containing funding rate data

    Returns:
        FundingVelocity or None if insufficient data
    """
    if history is None or len(history) < 3:
        return None

    # Get funding rates over time
    newest = history[-1]
    oldest = history[0]

    current_rate = newest.funding_rate
    old_rate = oldest.funding_rate

    # Calculate velocity (change per period)
    velocity = (current_rate - old_rate) / len(history)

    # Annualized APR (hourly rate * 24 * 365)
    annualized_apr = current_rate * 24 * 365 * 100

    # Determine direction
    VELOCITY_THRESHOLD = 0.00001  # Minimum velocity to be significant

    # Check for flip warning (funding close to zero and has momentum toward flip)
    flip_warning = False
    if abs(current_rate) < 0.0001:  # Near zero
        if (current_rate > 0 and velocity < -VELOCITY_THRESHOLD) or \
           (current_rate < 0 and velocity > VELOCITY_THRESHOLD):
            flip_warning = True

    # Classify direction
    if velocity > VELOCITY_THRESHOLD:
        if current_rate > 0:
            direction = "accelerating_bullish"  # Getting more bullish
        else:
            direction = "recovering"  # Was bearish, becoming less so
    elif velocity < -VELOCITY_THRESHOLD:
        if current_rate < 0:
            direction = "accelerating_bearish"  # Getting more bearish
        else:
            direction = "weakening"  # Was bullish, becoming less so
    else:
        direction = "stable"

    if flip_warning:
        direction = "flipping"

    return FundingVelocity(
        coin=newest.coin,
        current_rate=current_rate,
        velocity=velocity,
        direction=direction,
        flip_warning=flip_warning,
        annualized_apr=annualized_apr,
    )
