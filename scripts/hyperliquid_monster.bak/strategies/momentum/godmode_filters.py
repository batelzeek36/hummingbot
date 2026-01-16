"""
GOD MODE filters for Momentum Strategy.

Contains MTF confluence checks, liquidation magnet analysis, and CVD divergence filters.
These are Phase 3 features that provide advanced entry blocking logic.
"""

import logging
from typing import Optional, Tuple, TYPE_CHECKING

from ...coinglass import LiquidationHeatmap, SpotPerpDivergence
from ...multi_timeframe import MTFConfluence, MultiTimeframeAnalyzer
from ...leading_indicators import DirectionalSignal

if TYPE_CHECKING:
    from ...config import HyperliquidMonsterV2Config


def check_liq_proximity_warning(
    heatmap: LiquidationHeatmap,
    current_price: float,
    warning_pct: float,
    logger: Optional[logging.Logger] = None,
):
    """
    Check if price is near a liquidation cluster and log warning.

    Args:
        heatmap: Liquidation heatmap data
        current_price: Current market price
        warning_pct: Warning threshold percentage
        logger: Logger instance
    """
    logger = logger or logging.getLogger(__name__)

    # Check nearest long liquidations (below price)
    if heatmap.nearest_long_liq:
        distance_pct = ((current_price - heatmap.nearest_long_liq) / current_price) * 100
        if distance_pct < warning_pct:
            logger.warning(
                f"LIQ WARNING: Long liquidations {distance_pct:.1f}% below at ${heatmap.nearest_long_liq:.2f}"
            )

    # Check nearest short liquidations (above price)
    if heatmap.nearest_short_liq:
        distance_pct = ((heatmap.nearest_short_liq - current_price) / current_price) * 100
        if distance_pct < warning_pct:
            logger.warning(
                f"LIQ WARNING: Short liquidations {distance_pct:.1f}% above at ${heatmap.nearest_short_liq:.2f}"
            )


def check_liq_magnet_blocks(
    heatmap: LiquidationHeatmap,
    direction: str,
    min_imbalance: float,
) -> Tuple[bool, str]:
    """
    Check if liquidation magnet direction blocks the entry.

    Price is "magnetically attracted" to liquidation clusters.
    Don't go long if strong magnetic pull is down (toward long liquidations).
    Don't go short if strong magnetic pull is up (toward short liquidations).

    Args:
        heatmap: Liquidation heatmap data
        direction: "long" or "short"
        min_imbalance: Minimum imbalance ratio to block

    Returns:
        Tuple of (should_block, reason)
    """
    # Calculate imbalance
    long_total = sum(c.estimated_size_usd for c in heatmap.long_clusters[:3]) if heatmap.long_clusters else 0
    short_total = sum(c.estimated_size_usd for c in heatmap.short_clusters[:3]) if heatmap.short_clusters else 0

    if long_total == 0 and short_total == 0:
        return False, ""

    # Check imbalance ratio
    if direction == "long":
        # Block long if strong magnet pulling down (more long liquidations)
        if long_total > 0 and short_total > 0:
            imbalance = long_total / short_total
            if imbalance >= min_imbalance:
                return True, f"Strong downward magnet ({imbalance:.1f}x more long liqs)"
        elif long_total > 0 and short_total == 0:
            return True, "All liquidity is long liquidations below"

    elif direction == "short":
        # Block short if strong magnet pulling up (more short liquidations)
        if short_total > 0 and long_total > 0:
            imbalance = short_total / long_total
            if imbalance >= min_imbalance:
                return True, f"Strong upward magnet ({imbalance:.1f}x more short liqs)"
        elif short_total > 0 and long_total == 0:
            return True, "All liquidity is short liquidations above"

    return False, ""


def check_mtf_entry_block(
    mtf_analyzer: Optional[MultiTimeframeAnalyzer],
    pair: str,
    direction: str,
    mtf_confluence: Optional[MTFConfluence],
    config: "HyperliquidMonsterV2Config",
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Check if MTF confluence blocks entry.

    Args:
        mtf_analyzer: Multi-timeframe analyzer
        pair: Trading pair
        direction: "long" or "short"
        mtf_confluence: Current MTF confluence data
        config: Bot configuration
        logger: Logger instance

    Returns:
        Tuple of (should_block, reason)
    """
    logger = logger or logging.getLogger(__name__)

    if not mtf_confluence or not config.mtf_block_contrary:
        return False, ""

    # Check if MTF blocks entry
    if mtf_analyzer:
        should_block, reason = mtf_analyzer.should_block_entry(pair, direction)
        if should_block:
            return True, reason

    # Require 1H alignment if configured
    if config.mtf_require_htf_alignment and "1h" in mtf_confluence.signals:
        htf_signal = mtf_confluence.signals["1h"]
        if direction == "long" and htf_signal.trend.value.startswith("bear"):
            return True, "1H timeframe bearish"
        elif direction == "short" and htf_signal.trend.value.startswith("bull"):
            return True, "1H timeframe bullish"

    return False, ""


def check_cvd_entry_block(
    cvd_divergence: Optional[SpotPerpDivergence],
    direction: str,
    config: "HyperliquidMonsterV2Config",
) -> Tuple[bool, str]:
    """
    Check if CVD divergence blocks entry.

    Args:
        cvd_divergence: Spot vs perp divergence data
        direction: "long" or "short"
        config: Bot configuration

    Returns:
        Tuple of (should_block, reason)
    """
    if not cvd_divergence or not config.cvd_block_contrary_entries:
        return False, ""

    threshold = float(config.cvd_divergence_threshold)

    if direction == "long":
        if cvd_divergence.divergence_type == "bearish_divergence":
            if cvd_divergence.signal_strength >= threshold:
                return True, f"Perp longs trapped ({cvd_divergence.signal_strength:.0f}%)"

    elif direction == "short":
        if cvd_divergence.divergence_type == "bullish_divergence":
            if cvd_divergence.signal_strength >= threshold:
                return True, f"Perp shorts trapped ({cvd_divergence.signal_strength:.0f}%)"

    return False, ""


def check_holy_grail_entry_block(
    holy_grail: Optional[DirectionalSignal],
    direction: str,
    config: "HyperliquidMonsterV2Config",
) -> Tuple[bool, str]:
    """
    Check if Holy Grail signal blocks entry.

    Args:
        holy_grail: Holy Grail combined signal
        direction: "long" or "short"
        config: Bot configuration

    Returns:
        Tuple of (should_block, reason)
    """
    if not holy_grail or not config.use_holy_grail_signal:
        return False, ""

    min_confidence = float(config.holy_grail_min_confidence)

    if direction == "long":
        # Block long if signal says short with high confidence
        if config.holy_grail_block_contrary:
            if holy_grail.direction in ("short", "strong_short") and holy_grail.confidence >= min_confidence:
                return True, f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"

        # Require strong signal if configured
        if config.holy_grail_require_strong:
            if holy_grail.direction != "strong_long" or holy_grail.confidence < min_confidence:
                return True, "HOLY GRAIL not strong_long"

    elif direction == "short":
        # Block short if signal says long with high confidence
        if config.holy_grail_block_contrary:
            if holy_grail.direction in ("long", "strong_long") and holy_grail.confidence >= min_confidence:
                return True, f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"

        # Require strong signal if configured
        if config.holy_grail_require_strong:
            if holy_grail.direction != "strong_short" or holy_grail.confidence < min_confidence:
                return True, "HOLY GRAIL not strong_short"

    return False, ""


def check_oi_entry_block(
    oi_tracker,
    pair: str,
    direction: str,
    config: "HyperliquidMonsterV2Config",
) -> Tuple[bool, str]:
    """
    Check if OI leading indicator blocks entry during exhaustion.

    Args:
        oi_tracker: HyperliquidLeadingIndicators instance
        pair: Trading pair
        direction: "long" or "short"
        config: Bot configuration

    Returns:
        Tuple of (should_block, reason)
    """
    if not config.oi_block_exhaustion_entries or not oi_tracker:
        return False, ""

    return oi_tracker.should_block_entry(pair, direction)


def check_all_godmode_filters(
    direction: str,
    config: "HyperliquidMonsterV2Config",
    mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None,
    mtf_confluence: Optional[MTFConfluence] = None,
    liq_heatmap: Optional[LiquidationHeatmap] = None,
    cvd_divergence: Optional[SpotPerpDivergence] = None,
    holy_grail: Optional[DirectionalSignal] = None,
    oi_tracker=None,
    pair: str = "",
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """
    Check all GOD MODE filters for entry blocking.

    Args:
        direction: "long" or "short"
        config: Bot configuration
        mtf_analyzer: Multi-timeframe analyzer
        mtf_confluence: MTF confluence data
        liq_heatmap: Liquidation heatmap data
        cvd_divergence: CVD divergence data
        holy_grail: Holy Grail signal
        oi_tracker: OI tracker
        pair: Trading pair
        logger: Logger instance

    Returns:
        Tuple of (should_block, reason)
    """
    logger = logger or logging.getLogger(__name__)

    # Check MTF confluence
    should_block, reason = check_mtf_entry_block(
        mtf_analyzer, pair, direction, mtf_confluence, config, logger
    )
    if should_block:
        return True, f"MTF: {reason}"

    # Check liquidation magnet
    if liq_heatmap and config.liq_magnet_block_contrary:
        should_block, reason = check_liq_magnet_blocks(
            liq_heatmap, direction, float(config.liq_magnet_min_imbalance)
        )
        if should_block:
            return True, f"LIQ MAGNET: {reason}"

    # Check CVD divergence
    should_block, reason = check_cvd_entry_block(cvd_divergence, direction, config)
    if should_block:
        return True, f"CVD DIVERGENCE: {reason}"

    # Check Holy Grail
    should_block, reason = check_holy_grail_entry_block(holy_grail, direction, config)
    if should_block:
        return True, f"HOLY GRAIL: {reason}"

    # Check OI exhaustion
    should_block, reason = check_oi_entry_block(oi_tracker, pair, direction, config)
    if should_block:
        return True, f"OI: {reason}"

    return False, ""
