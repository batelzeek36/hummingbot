"""
Signal evaluation logic for Momentum Strategy.

Contains functions to evaluate long/short entry signals and check confluence.
"""

import logging
from decimal import Decimal
from typing import Dict, Optional, TYPE_CHECKING

from ...indicators import MACDResult, TrendInfo, VolumeAnalysis
from ...leading_indicators import DirectionalSignal, OIAnalysis, OIMomentum
from ...multi_timeframe import MTFConfluence
from ...coinglass import LiquidationHeatmap

if TYPE_CHECKING:
    from ...config import HyperliquidMonsterV2Config


def evaluate_long_signals(
    price: float,
    trend: Optional[TrendInfo],
    macd: Optional[MACDResult],
    volume: Optional[VolumeAnalysis],
    funding: Optional[str],
    config: "HyperliquidMonsterV2Config",
    oi: Optional[OIAnalysis] = None,
    mtf: Optional[MTFConfluence] = None,
    liq: Optional[LiquidationHeatmap] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, bool]:
    """
    Evaluate confirming signals for a long entry.

    Args:
        price: Current price
        trend: Trend analysis info
        macd: MACD indicator results
        volume: Volume analysis results
        funding: Funding sentiment ("bullish", "bearish", "neutral")
        config: Bot configuration
        oi: OI analysis (leading indicator)
        mtf: Multi-timeframe confluence
        liq: Liquidation heatmap
        logger: Logger instance

    Returns:
        Dict mapping signal names to boolean values
    """
    logger = logger or logging.getLogger(__name__)
    signals = {"RSI": True}  # RSI already confirmed

    # Trend: must be in uptrend or neutral
    if config.use_trend_filter and trend:
        signals["Trend"] = trend.direction != "downtrend"
        # Bonus: price above EMA 200
        if price > trend.ema_200:
            signals["Trend"] = True

    # MACD: MACD line above signal line (bullish)
    if config.use_macd_filter and macd:
        signals["MACD"] = macd.macd_line > macd.signal_line

    # Volume: at least minimum ratio
    if config.use_volume_filter and volume:
        signals["Volume"] = volume.volume_ratio >= float(config.min_volume_ratio)

    # Funding: bullish sentiment (shorts crowded)
    if config.use_funding_sentiment and funding:
        signals["Funding"] = funding == "bullish"

    # OI: LEADING INDICATOR - confirms long conviction
    # Bullish: OI rising with price (new longs) OR capitulation (bounce setup)
    if config.use_oi_filter and oi:
        signals["OI"] = oi.should_confirm_long
        if oi.momentum == OIMomentum.BULLISH_CONFIRM:
            logger.debug(f"OI confirms LONG: new longs entering (OI +{oi.oi_change_pct:.1f}%)")
        elif oi.momentum == OIMomentum.BEARISH_EXHAUSTION:
            logger.debug(f"OI contrarian LONG: capitulation bounce setup")

    # MTF Confluence: Phase 3 - bullish confluence confirms long
    if config.use_mtf_confluence and mtf:
        signals["MTF"] = mtf.confluence_direction == "bullish" and mtf.has_confluence
        if signals["MTF"]:
            logger.debug(f"MTF confirms LONG: {mtf.bullish_count} bullish TFs ({mtf.weighted_score:+.1f})")

    # Liquidation Magnet: Phase 3 - upward magnet confirms long
    if config.use_liq_magnet and liq:
        signals["LiqMagnet"] = liq.magnetic_direction == "up"
        if signals["LiqMagnet"]:
            logger.debug(f"LIQ MAGNET confirms LONG: upward pull toward short liquidations")

    return signals


def evaluate_short_signals(
    price: float,
    trend: Optional[TrendInfo],
    macd: Optional[MACDResult],
    volume: Optional[VolumeAnalysis],
    funding: Optional[str],
    config: "HyperliquidMonsterV2Config",
    oi: Optional[OIAnalysis] = None,
    mtf: Optional[MTFConfluence] = None,
    liq: Optional[LiquidationHeatmap] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, bool]:
    """
    Evaluate confirming signals for a short entry.

    Args:
        price: Current price
        trend: Trend analysis info
        macd: MACD indicator results
        volume: Volume analysis results
        funding: Funding sentiment ("bullish", "bearish", "neutral")
        config: Bot configuration
        oi: OI analysis (leading indicator)
        mtf: Multi-timeframe confluence
        liq: Liquidation heatmap
        logger: Logger instance

    Returns:
        Dict mapping signal names to boolean values
    """
    logger = logger or logging.getLogger(__name__)
    signals = {"RSI": True}  # RSI already confirmed

    # Trend: must be in downtrend or neutral
    if config.use_trend_filter and trend:
        signals["Trend"] = trend.direction != "uptrend"
        # Bonus: price below EMA 200
        if price < trend.ema_200:
            signals["Trend"] = True

    # MACD: MACD line below signal line (bearish)
    if config.use_macd_filter and macd:
        signals["MACD"] = macd.macd_line < macd.signal_line

    # Volume: at least minimum ratio
    if config.use_volume_filter and volume:
        signals["Volume"] = volume.volume_ratio >= float(config.min_volume_ratio)

    # Funding: bearish sentiment (longs crowded)
    if config.use_funding_sentiment and funding:
        signals["Funding"] = funding == "bearish"

    # OI: LEADING INDICATOR - confirms short conviction
    # Bearish: OI rising with price falling (new shorts entering)
    if config.use_oi_filter and oi:
        signals["OI"] = oi.should_confirm_short
        if oi.momentum == OIMomentum.BEARISH_CONFIRM:
            logger.debug(f"OI confirms SHORT: new shorts entering (OI +{oi.oi_change_pct:.1f}%)")

    # MTF Confluence: Phase 3 - bearish confluence confirms short
    if config.use_mtf_confluence and mtf:
        signals["MTF"] = mtf.confluence_direction == "bearish" and mtf.has_confluence
        if signals["MTF"]:
            logger.debug(f"MTF confirms SHORT: {mtf.bearish_count} bearish TFs ({mtf.weighted_score:+.1f})")

    # Liquidation Magnet: Phase 3 - downward magnet confirms short
    if config.use_liq_magnet and liq:
        signals["LiqMagnet"] = liq.magnetic_direction == "down"
        if signals["LiqMagnet"]:
            logger.debug(f"LIQ MAGNET confirms SHORT: downward pull toward long liquidations")

    return signals


def has_enough_signals(
    signals: Dict[str, bool],
    direction: str,
    config: "HyperliquidMonsterV2Config",
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Check if we have enough confirming signals.

    RSI is always required. Additionally need min_signals_required
    from the other indicators (including OI, MTF, and LiqMagnet).

    Args:
        signals: Dict of signal names to boolean values
        direction: "long" or "short"
        config: Bot configuration
        logger: Logger instance

    Returns:
        True if enough signals are present
    """
    logger = logger or logging.getLogger(__name__)

    # RSI must be true
    if not signals.get("RSI", False):
        return False

    # If OI confirmation is required (stricter mode), check it
    if config.oi_require_confirmation and config.use_oi_filter:
        if not signals.get("OI", False):
            logger.debug(f"MOMENTUM: {direction} blocked - OI confirmation required but not present")
            return False

    # Count other confirming signals (now includes OI, MTF, LiqMagnet)
    other_signals = [
        signals.get("Trend", False),
        signals.get("MACD", False),
        signals.get("Volume", False),
        signals.get("Funding", False),
        signals.get("OI", False),        # LEADING INDICATOR
        signals.get("MTF", False),       # Phase 3: Multi-Timeframe
        signals.get("LiqMagnet", False), # Phase 3: Liquidation Magnet
    ]

    confirming_count = sum(1 for s in other_signals if s)
    required = config.min_signals_required

    if confirming_count >= required:
        active = [k for k, v in signals.items() if v]
        # Highlight leading indicators
        extras = []
        if signals.get("OI", False):
            extras.append("OI+")
        if signals.get("MTF", False):
            extras.append("MTF+")
        if signals.get("LiqMagnet", False):
            extras.append("LIQ+")
        extras_str = " ".join(extras) + " " if extras else ""

        logger.info(
            f"MOMENTUM: {direction.upper()} confluence met {extras_str}"
            f"({confirming_count + 1}/{len(signals)} signals: {', '.join(active)})"
        )
        return True

    return False


def build_entry_info(
    holy_grail: Optional[DirectionalSignal],
    mtf: Optional[MTFConfluence],
    liq: Optional[LiquidationHeatmap],
    holy_grail_min_confidence: float,
) -> str:
    """
    Build extra info string for entry logging.

    Args:
        holy_grail: Holy Grail signal
        mtf: MTF confluence
        liq: Liquidation heatmap
        holy_grail_min_confidence: Minimum confidence for holy grail

    Returns:
        Formatted info string
    """
    parts = []

    if holy_grail and holy_grail.confidence >= holy_grail_min_confidence:
        parts.append(f"HG:{holy_grail.direction[:3]}{holy_grail.confidence:.0f}%")

    if mtf and mtf.has_confluence:
        parts.append(f"MTF:{mtf.bullish_count}B/{mtf.bearish_count}Be")

    if liq:
        parts.append(f"LIQ:{liq.magnetic_direction}")

    return f" [{', '.join(parts)}]" if parts else ""
