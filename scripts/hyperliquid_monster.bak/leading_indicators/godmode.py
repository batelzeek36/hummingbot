"""
GOD MODE Phase 1: Spike Detection & Market Regime.

Advanced detection features:
- OI Spike Detection: Instant detection of violent OI changes (>5-10% in 30s)
- Premium Spike Detection: Sudden premium changes indicating rapid sentiment shift
- Market Regime Detection: Cross-market crowding analysis for squeeze prediction
"""

from typing import Deque, Dict, List, Optional, Tuple

from .models import (
    MarketRegime,
    OIAnalysis,
    OISpikeResult,
    OISnapshot,
    PremiumSpikeResult,
)
from .oi_analysis import should_block_entry as oi_should_block_entry


def detect_oi_spike(history: Deque[OISnapshot]) -> Optional[OISpikeResult]:
    """
    Detect instant violent OI changes in a single snapshot.

    Your 12-period lookback smooths out spikes - this catches them instantly.

    Signals:
    - OI spike UP (>10% in 30s) = Big player entering, expect momentum
    - OI spike DOWN (>10% in 30s) = Liquidation cascade starting

    Args:
        history: Deque of OI snapshots for the pair

    Returns:
        OISpikeResult or None if insufficient data
    """
    if history is None or len(history) < 2:
        return None

    # Compare last two snapshots (instant detection)
    prev = history[-2]
    curr = history[-1]

    if prev.open_interest == 0:
        return None

    change_pct = ((curr.open_interest - prev.open_interest)
                  / prev.open_interest * 100)

    # Spike thresholds
    SPIKE_UP_THRESHOLD = 10.0    # 10% increase in single snapshot
    SPIKE_DOWN_THRESHOLD = -10.0  # 10% decrease in single snapshot

    # Also detect smaller but still significant spikes
    MINOR_SPIKE_UP = 5.0
    MINOR_SPIKE_DOWN = -5.0

    detected = False
    spike_type = "none"
    interpretation = ""

    if change_pct >= SPIKE_UP_THRESHOLD:
        detected = True
        spike_type = "spike_up"
        interpretation = (
            f"MAJOR OI SPIKE UP: {change_pct:.1f}% in ~30s. "
            "Big player entering - expect momentum continuation. "
            "Consider following the direction if price confirms."
        )
    elif change_pct >= MINOR_SPIKE_UP:
        detected = True
        spike_type = "spike_up"
        interpretation = (
            f"OI spike up: {change_pct:.1f}% in ~30s. "
            "Significant new positions opening - momentum building."
        )
    elif change_pct <= SPIKE_DOWN_THRESHOLD:
        detected = True
        spike_type = "spike_down"
        interpretation = (
            f"MAJOR OI SPIKE DOWN: {change_pct:.1f}% in ~30s. "
            "Liquidation cascade or mass exit. "
            "Expect volatility spike then potential reversal."
        )
    elif change_pct <= MINOR_SPIKE_DOWN:
        detected = True
        spike_type = "spike_down"
        interpretation = (
            f"OI spike down: {change_pct:.1f}% in ~30s. "
            "Positions closing rapidly - watch for reversal setup."
        )

    return OISpikeResult(
        coin=curr.coin,
        detected=detected,
        spike_type=spike_type,
        change_pct=change_pct,
        interpretation=interpretation,
        timestamp=curr.timestamp,
    )


def detect_premium_spike(history: Deque[OISnapshot]) -> Optional[PremiumSpikeResult]:
    """
    Detect sudden premium changes that often precede violent moves.

    A premium spike >50% change indicates rapid sentiment shift.

    Args:
        history: Deque of OI snapshots for the pair

    Returns:
        PremiumSpikeResult or None if insufficient data
    """
    if history is None or len(history) < 2:
        return None

    prev = history[-2]
    curr = history[-1]

    # Handle near-zero premiums carefully
    if abs(prev.premium) < 0.00001:
        # If previous was near zero, use absolute change
        if abs(curr.premium) > 0.0005:  # Significant new premium
            detected = True
            change_pct = 100.0  # Treat as 100% change
        else:
            return PremiumSpikeResult(
                coin=curr.coin,
                detected=False,
                spike_type="none",
                change_pct=0.0,
                previous_premium=prev.premium,
                current_premium=curr.premium,
                interpretation="Premium stable near zero",
                timestamp=curr.timestamp,
            )
    else:
        change_pct = ((curr.premium - prev.premium) / abs(prev.premium) * 100)

    # Spike thresholds
    SPIKE_THRESHOLD = 50.0  # 50% change in premium

    detected = abs(change_pct) >= SPIKE_THRESHOLD
    spike_type = "none"
    interpretation = ""

    if detected:
        if curr.premium > prev.premium:
            spike_type = "bullish_spike"
            interpretation = (
                f"PREMIUM SPIKE UP: {change_pct:+.1f}% in ~30s. "
                f"Premium went from {prev.premium*100:.4f}% to {curr.premium*100:.4f}%. "
                "Aggressive long pressure building - potential pump incoming."
            )
        else:
            spike_type = "bearish_spike"
            interpretation = (
                f"PREMIUM SPIKE DOWN: {change_pct:+.1f}% in ~30s. "
                f"Premium went from {prev.premium*100:.4f}% to {curr.premium*100:.4f}%. "
                "Aggressive short pressure building - potential dump incoming."
            )

    return PremiumSpikeResult(
        coin=curr.coin,
        detected=detected,
        spike_type=spike_type,
        change_pct=change_pct,
        previous_premium=prev.premium,
        current_premium=curr.premium,
        interpretation=interpretation,
        timestamp=curr.timestamp,
    )


def get_market_regime(oi_histories: Dict[str, Deque[OISnapshot]]) -> Optional[MarketRegime]:
    """
    Detect market-wide regime across all monitored coins.

    When 8+ coins have same-direction funding, the market is crowded
    and a squeeze in the opposite direction becomes likely.

    Args:
        oi_histories: Dict of pair -> OI history deque

    Returns:
        MarketRegime with cross-market analysis
    """
    pairs_tracked = list(oi_histories.keys())
    if len(pairs_tracked) < 3:
        return None

    # Thresholds for classifying funding direction
    BULLISH_THRESHOLD = 0.0001   # 0.01% hourly = ~88% APR
    BEARISH_THRESHOLD = -0.0001

    bullish_coins = 0
    bearish_coins = 0
    neutral_coins = 0
    funding_rates: List[float] = []
    warnings: List[str] = []

    for pair in pairs_tracked:
        history = oi_histories.get(pair)
        if history is None or len(history) < 1:
            continue

        funding = history[-1].funding_rate
        funding_rates.append(funding)

        if funding > BULLISH_THRESHOLD:
            bullish_coins += 1
        elif funding < BEARISH_THRESHOLD:
            bearish_coins += 1
        else:
            neutral_coins += 1

    total_coins = bullish_coins + bearish_coins + neutral_coins
    if total_coins == 0:
        return None

    avg_funding = sum(funding_rates) / len(funding_rates) if funding_rates else 0

    # Determine regime
    bullish_pct = bullish_coins / total_coins * 100
    bearish_pct = bearish_coins / total_coins * 100

    # Crowded thresholds
    CROWDED_THRESHOLD = 70  # 70% of coins same direction = crowded

    if bullish_pct >= CROWDED_THRESHOLD:
        regime = "crowded_long"
        squeeze_risk = "high"
        dominant_direction = "long"
        warnings.append(
            f"MARKET CROWDED LONG: {bullish_coins}/{total_coins} coins have positive funding. "
            "Long squeeze risk is LOW, but short squeeze risk is HIGH if sentiment shifts."
        )
    elif bearish_pct >= CROWDED_THRESHOLD:
        regime = "crowded_short"
        squeeze_risk = "high"
        dominant_direction = "short"
        warnings.append(
            f"MARKET CROWDED SHORT: {bearish_coins}/{total_coins} coins have negative funding. "
            "Short squeeze imminent if any catalyst appears."
        )
    elif bullish_pct >= 50 or bearish_pct >= 50:
        regime = "leaning"
        squeeze_risk = "medium"
        dominant_direction = "long" if bullish_pct > bearish_pct else "short"
    else:
        regime = "mixed"
        squeeze_risk = "low"
        dominant_direction = "none"

    # Additional warning for extreme average funding
    if abs(avg_funding) > 0.0005:  # 0.05% hourly = ~438% APR average
        direction = "LONG" if avg_funding > 0 else "SHORT"
        warnings.append(
            f"EXTREME AVG FUNDING: {avg_funding*100:.4f}% hourly across market. "
            f"Market-wide {direction} crowding detected."
        )

    # Calculate confidence
    max_pct = max(bullish_pct, bearish_pct)
    confidence = min(95, max_pct + (10 if squeeze_risk == "high" else 0))

    return MarketRegime(
        regime=regime,
        confidence=confidence,
        bullish_coins=bullish_coins,
        bearish_coins=bearish_coins,
        neutral_coins=neutral_coins,
        total_coins=total_coins,
        avg_funding_rate=avg_funding,
        squeeze_risk=squeeze_risk,
        dominant_direction=dominant_direction,
        warnings=warnings,
    )


def get_all_spikes(oi_histories: Dict[str, Deque[OISnapshot]]) -> Dict[str, Dict]:
    """
    Check all tracked pairs for OI and premium spikes.

    Args:
        oi_histories: Dict of pair -> OI history deque

    Returns:
        Dict of pair -> spike info (only pairs with detected spikes)
    """
    results = {}
    for pair, history in oi_histories.items():
        oi_spike = detect_oi_spike(history)
        premium_spike = detect_premium_spike(history)

        if (oi_spike and oi_spike.detected) or (premium_spike and premium_spike.detected):
            results[pair] = {
                "oi_spike": {
                    "detected": oi_spike.detected if oi_spike else False,
                    "type": oi_spike.spike_type if oi_spike else "none",
                    "change": f"{oi_spike.change_pct:+.1f}%" if oi_spike else "N/A",
                },
                "premium_spike": {
                    "detected": premium_spike.detected if premium_spike else False,
                    "type": premium_spike.spike_type if premium_spike else "none",
                    "change": f"{premium_spike.change_pct:+.1f}%" if premium_spike else "N/A",
                },
            }
    return results


def should_block_entry_godmode(
    pair: str,
    direction: str,
    history: Deque[OISnapshot],
    oi_analysis: Optional[OIAnalysis],
    oi_histories: Dict[str, Deque[OISnapshot]],
    check_market_regime: bool = True
) -> Tuple[bool, str]:
    """
    Enhanced entry blocking with GOD MODE checks.

    Combines original OI blocking with:
    - OI spike detection
    - Premium spike detection
    - Market regime awareness

    Args:
        pair: Trading pair
        direction: "long" or "short"
        history: OI history for this pair
        oi_analysis: Pre-computed OI analysis for this pair
        oi_histories: All OI histories (for market regime)
        check_market_regime: Whether to factor in market-wide regime

    Returns:
        Tuple of (should_block, reason)
    """
    reasons = []

    # Original OI momentum check
    blocked, reason = oi_should_block_entry(oi_analysis, direction)
    if blocked:
        reasons.append(reason)

    # OI Spike check
    oi_spike = detect_oi_spike(history)
    if oi_spike and oi_spike.detected:
        if direction == "long" and oi_spike.spike_type == "spike_down":
            reasons.append(
                f"OI SPIKE DOWN ({oi_spike.change_pct:.1f}%): "
                "Liquidation cascade - wait for dust to settle"
            )
        elif direction == "short" and oi_spike.spike_type == "spike_up":
            reasons.append(
                f"OI SPIKE UP ({oi_spike.change_pct:.1f}%): "
                "Big player entering long - don't short into momentum"
            )

    # Premium spike check
    premium_spike = detect_premium_spike(history)
    if premium_spike and premium_spike.detected:
        if direction == "long" and premium_spike.spike_type == "bearish_spike":
            reasons.append(
                f"PREMIUM SPIKE DOWN ({premium_spike.change_pct:.1f}%): "
                "Aggressive shorting - wait for stabilization"
            )
        elif direction == "short" and premium_spike.spike_type == "bullish_spike":
            reasons.append(
                f"PREMIUM SPIKE UP ({premium_spike.change_pct:.1f}%): "
                "Aggressive longing - don't short into FOMO"
            )

    # Market regime check
    if check_market_regime:
        regime = get_market_regime(oi_histories)
        if regime and regime.squeeze_risk == "high":
            if direction == "long" and regime.regime == "crowded_long":
                # Don't block longs in crowded long - they might still work
                # But warn
                pass
            elif direction == "short" and regime.regime == "crowded_short":
                # Shorting into crowded short is DANGEROUS
                reasons.append(
                    f"MARKET CROWDED SHORT ({regime.bearish_coins}/{regime.total_coins}): "
                    "Short squeeze imminent - extremely dangerous to short"
                )
            elif direction == "long" and regime.regime == "crowded_short":
                # This is actually good for longs
                pass
            elif direction == "short" and regime.regime == "crowded_long":
                # Careful shorting crowded longs - they can squeeze higher
                if regime.confidence > 80:
                    reasons.append(
                        f"MARKET CROWDED LONG ({regime.bullish_coins}/{regime.total_coins}): "
                        "Be cautious shorting - longs can squeeze higher before reversing"
                    )

    if reasons:
        return True, " | ".join(reasons)
    return False, ""
