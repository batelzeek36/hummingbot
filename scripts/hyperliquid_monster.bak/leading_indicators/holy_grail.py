"""
Holy Grail Combined Directional Signal.

THE HOLY GRAIL: Combined directional signal from ALL leading indicators.

Combines:
1. OI Momentum (-100 to +100)
2. Premium pressure (-100 to +100)
3. Funding velocity (-100 to +100)
4. Volume surge confirmation (-100 to +100)

Weights:
- OI Momentum: 35% (strongest predictor)
- Premium: 25% (real-time pressure)
- Funding Velocity: 25% (momentum shift)
- Volume: 15% (confirmation)
"""

from typing import List, Optional, Tuple

from .models import (
    DirectionalSignal,
    FundingVelocity,
    OIAnalysis,
    OIMomentum,
    PremiumAnalysis,
    VolumeSurge,
)


# Weights for combining signals
OI_WEIGHT = 0.35
PREMIUM_WEIGHT = 0.25
FUNDING_WEIGHT = 0.25
VOLUME_WEIGHT = 0.15


def calculate_holy_grail_signal(
    coin: str,
    oi_analysis: Optional[OIAnalysis],
    premium_analysis: Optional[PremiumAnalysis],
    funding_vel: Optional[FundingVelocity],
    volume_surge: Optional[VolumeSurge],
) -> Optional[DirectionalSignal]:
    """
    THE HOLY GRAIL: Combined directional signal from ALL leading indicators.

    Args:
        coin: Trading pair symbol
        oi_analysis: OI momentum analysis result
        premium_analysis: Premium analysis result
        funding_vel: Funding velocity analysis result
        volume_surge: Volume surge detection result

    Returns:
        DirectionalSignal with combined recommendation
    """
    warnings: List[str] = []
    reasoning_parts: List[str] = []

    # =====================================================================
    # 1. OI MOMENTUM SCORE
    # =====================================================================
    oi_score = 0.0
    if oi_analysis:
        if oi_analysis.momentum == OIMomentum.BULLISH_CONFIRM:
            oi_score = oi_analysis.conviction_score  # 0 to +100
            reasoning_parts.append(f"OI: Bullish confirm (+{oi_score:.0f})")
        elif oi_analysis.momentum == OIMomentum.BEARISH_CONFIRM:
            oi_score = -oi_analysis.conviction_score  # 0 to -100
            reasoning_parts.append(f"OI: Bearish confirm ({oi_score:.0f})")
        elif oi_analysis.momentum == OIMomentum.BULLISH_EXHAUSTION:
            oi_score = -30  # Penalize - squeeze ending
            warnings.append("Short squeeze ending - avoid longs")
            reasoning_parts.append(f"OI: Bullish exhaustion ({oi_score:.0f})")
        elif oi_analysis.momentum == OIMomentum.BEARISH_EXHAUSTION:
            oi_score = 40  # Favor longs - bounce setup
            reasoning_parts.append(f"OI: Bearish exhaustion/bounce (+{oi_score:.0f})")
        else:
            reasoning_parts.append("OI: Neutral (0)")

    # =====================================================================
    # 2. PREMIUM SCORE
    # =====================================================================
    premium_score = 0.0
    if premium_analysis:
        if premium_analysis.direction_bias == "bullish":
            premium_score = premium_analysis.pressure_score  # 0 to +100
            if premium_analysis.is_extreme:
                warnings.append(f"Extreme premium ({premium_analysis.premium_pct:.3f}%) - crowded long")
                premium_score *= 0.5  # Reduce score if too crowded
            reasoning_parts.append(f"Premium: Bullish (+{premium_score:.0f})")
        elif premium_analysis.direction_bias == "bearish":
            premium_score = -premium_analysis.pressure_score  # 0 to -100
            if premium_analysis.is_extreme:
                warnings.append(f"Extreme discount ({premium_analysis.premium_pct:.3f}%) - crowded short")
                premium_score *= 0.5  # Reduce score if too crowded
            reasoning_parts.append(f"Premium: Bearish ({premium_score:.0f})")
        else:
            reasoning_parts.append("Premium: Neutral (0)")

    # =====================================================================
    # 3. FUNDING VELOCITY SCORE
    # =====================================================================
    funding_velocity_score = 0.0
    if funding_vel:
        if funding_vel.direction == "accelerating_bullish":
            funding_velocity_score = min(80, abs(funding_vel.velocity) * 100000)
            reasoning_parts.append(f"Funding: Accelerating bullish (+{funding_velocity_score:.0f})")
        elif funding_vel.direction == "accelerating_bearish":
            funding_velocity_score = -min(80, abs(funding_vel.velocity) * 100000)
            reasoning_parts.append(f"Funding: Accelerating bearish ({funding_velocity_score:.0f})")
        elif funding_vel.direction == "recovering":
            funding_velocity_score = 30  # Recovering from bearish
            reasoning_parts.append(f"Funding: Recovering (+{funding_velocity_score:.0f})")
        elif funding_vel.direction == "weakening":
            funding_velocity_score = -30  # Weakening from bullish
            reasoning_parts.append(f"Funding: Weakening ({funding_velocity_score:.0f})")
        elif funding_vel.direction == "flipping":
            warnings.append("Funding rate about to flip - trend reversal possible")
            # Direction depends on which way it's flipping
            if funding_vel.current_rate > 0 and funding_vel.velocity < 0:
                funding_velocity_score = -50  # Was bullish, flipping bearish
            else:
                funding_velocity_score = 50  # Was bearish, flipping bullish
            reasoning_parts.append(f"Funding: FLIPPING ({funding_velocity_score:.0f})")
        else:
            reasoning_parts.append("Funding: Stable (0)")

    # =====================================================================
    # 4. VOLUME SCORE
    # =====================================================================
    volume_score = 0.0
    if volume_surge and volume_surge.is_surging:
        if volume_surge.surge_direction == "bullish":
            volume_score = min(60, (volume_surge.surge_ratio - 1) * 40)
            reasoning_parts.append(f"Volume: Bullish surge (+{volume_score:.0f})")
        elif volume_surge.surge_direction == "bearish":
            volume_score = -min(60, (volume_surge.surge_ratio - 1) * 40)
            reasoning_parts.append(f"Volume: Bearish surge ({volume_score:.0f})")
        elif volume_surge.surge_direction == "breakout_imminent":
            warnings.append(f"Volume surge {volume_surge.surge_ratio:.1f}x - breakout imminent")
            reasoning_parts.append("Volume: Breakout imminent (0)")
    else:
        reasoning_parts.append("Volume: Normal (0)")

    # =====================================================================
    # COMBINE SCORES WITH WEIGHTS
    # =====================================================================
    combined_score = (
        oi_score * OI_WEIGHT +
        premium_score * PREMIUM_WEIGHT +
        funding_velocity_score * FUNDING_WEIGHT +
        volume_score * VOLUME_WEIGHT
    )

    # Calculate confidence based on agreement between indicators
    scores = [oi_score, premium_score, funding_velocity_score, volume_score]
    positive_count = sum(1 for s in scores if s > 10)
    negative_count = sum(1 for s in scores if s < -10)

    # Higher confidence when indicators agree
    if positive_count >= 3 or negative_count >= 3:
        confidence = min(95, abs(combined_score) + 20)
    elif positive_count >= 2 or negative_count >= 2:
        confidence = min(80, abs(combined_score) + 10)
    else:
        confidence = min(60, abs(combined_score))

    # =====================================================================
    # DETERMINE DIRECTION
    # =====================================================================
    if combined_score > 40:
        direction = "strong_long"
    elif combined_score > 15:
        direction = "long"
    elif combined_score < -40:
        direction = "strong_short"
    elif combined_score < -15:
        direction = "short"
    else:
        direction = "neutral"

    reasoning = " | ".join(reasoning_parts)

    return DirectionalSignal(
        coin=coin,
        direction=direction,
        confidence=confidence,
        oi_score=oi_score,
        premium_score=premium_score,
        funding_velocity_score=funding_velocity_score,
        volume_score=volume_score,
        warnings=warnings,
        reasoning=reasoning,
    )


def get_direction_recommendation(signal: Optional[DirectionalSignal]) -> Tuple[str, float, List[str]]:
    """
    Simple interface to get direction recommendation.

    Args:
        signal: DirectionalSignal from calculate_holy_grail_signal

    Returns:
        Tuple of (direction, confidence, warnings)
        direction: "long", "short", or "neutral"
        confidence: 0-100
        warnings: List of warning strings
    """
    if signal is None:
        return "neutral", 0, []

    # Simplify direction
    if signal.direction in ("strong_long", "long"):
        direction = "long"
    elif signal.direction in ("strong_short", "short"):
        direction = "short"
    else:
        direction = "neutral"

    return direction, signal.confidence, signal.warnings
