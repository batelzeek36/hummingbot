"""
Indicator calculation helpers for Momentum Strategy.

Contains functions to calculate technical indicators and funding sentiment.
"""

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from ...indicators import MACDResult, TechnicalIndicators, TrendInfo, VolumeAnalysis
from ...leading_indicators import DirectionalSignal, OIAnalysis
from ...multi_timeframe import MTFConfluence
from ...coinglass import LiquidationHeatmap, SpotPerpDivergence
from ...ml_models import MLPrediction

if TYPE_CHECKING:
    from hummingbot.connector.connector_base import ConnectorBase
    from ...config import HyperliquidMonsterV2Config


def calculate_all_indicators(
    prices: List[float],
    volumes: List[float],
    config: "HyperliquidMonsterV2Config",
) -> Optional[Tuple[float, Optional[TrendInfo], Optional[MACDResult], Optional[VolumeAnalysis]]]:
    """
    Calculate all technical indicators.

    Args:
        prices: List of historical prices
        volumes: List of historical volumes
        config: Bot configuration

    Returns:
        Tuple of (rsi, trend, macd, volume) or None if RSI unavailable
    """
    # RSI (always required)
    rsi = TechnicalIndicators.calculate_rsi(prices, config.momentum_lookback)
    if rsi is None:
        return None

    # Trend filter (optional)
    trend = None
    if config.use_trend_filter:
        trend = TechnicalIndicators.analyze_trend(
            prices,
            config.ema_short_period,
            config.ema_long_period
        )

    # MACD (optional)
    macd = None
    if config.use_macd_filter:
        macd = TechnicalIndicators.calculate_macd(
            prices,
            config.macd_fast,
            config.macd_slow,
            config.macd_signal
        )

    # Volume analysis (optional)
    volume = None
    if config.use_volume_filter and len(volumes) >= config.volume_lookback:
        volume = TechnicalIndicators.analyze_volume(
            volumes,
            config.volume_lookback,
            float(config.strong_volume_ratio)
        )

    return (rsi, trend, macd, volume)


def get_funding_sentiment(
    connector: "ConnectorBase",
    pair: str,
    threshold: float,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """
    Get funding rate sentiment for the trading pair.

    Positive funding = longs paying shorts = too many longs = bearish
    Negative funding = shorts paying longs = too many shorts = bullish

    Args:
        connector: Exchange connector
        pair: Trading pair
        threshold: Sentiment threshold
        logger: Logger instance

    Returns:
        "bullish", "bearish", or "neutral"
    """
    logger = logger or logging.getLogger(__name__)

    try:
        funding_info = connector.get_funding_info(pair)
        if funding_info is None:
            return None

        rate = float(funding_info.rate)

        if rate > threshold:
            return "bearish"  # Crowded long, expect pullback
        elif rate < -threshold:
            return "bullish"  # Crowded short, expect squeeze
        else:
            return "neutral"

    except Exception as e:
        logger.debug(f"Could not get funding sentiment: {e}")
        return None


def build_indicator_status(
    rsi: Optional[float] = None,
    trend: Optional[TrendInfo] = None,
    macd: Optional[MACDResult] = None,
    volume: Optional[VolumeAnalysis] = None,
    funding_sentiment: Optional[str] = None,
    oi_analysis: Optional[OIAnalysis] = None,
    holy_grail: Optional[DirectionalSignal] = None,
    mtf_confluence: Optional[MTFConfluence] = None,
    liq_heatmap: Optional[LiquidationHeatmap] = None,
    cvd_divergence: Optional[SpotPerpDivergence] = None,
    ml_model=None,
    ml_prediction: Optional[MLPrediction] = None,
) -> Dict[str, str]:
    """
    Build indicator status dictionary for display.

    Args:
        rsi: RSI value
        trend: Trend analysis info
        macd: MACD results
        volume: Volume analysis
        funding_sentiment: Funding sentiment string
        oi_analysis: OI analysis from leading indicators
        holy_grail: Holy Grail combined signal
        mtf_confluence: MTF confluence data
        liq_heatmap: Liquidation heatmap
        cvd_divergence: CVD divergence data
        ml_model: ML model instance
        ml_prediction: ML prediction

    Returns:
        Dict mapping indicator names to status strings
    """
    status = {}

    if rsi is not None:
        status["RSI"] = f"{rsi:.1f}"

    if trend:
        status["Trend"] = f"{trend.direction} (EMA50/200: {trend.strength:.1f}%)"

    if macd:
        direction = "bullish" if macd.histogram > 0 else "bearish"
        status["MACD"] = f"{direction} (hist: {macd.histogram:.4f})"

    if volume:
        vol_str = "HIGH" if volume.is_high_volume else "normal"
        status["Volume"] = f"{volume.volume_ratio:.2f}x ({vol_str})"

    if funding_sentiment:
        status["Funding"] = funding_sentiment

    # OI: LEADING INDICATOR from Hyperliquid API
    if oi_analysis:
        oi = oi_analysis
        momentum_str = oi.momentum.value.replace("_", " ").title()
        status["OI"] = f"{momentum_str} (OI: {oi.oi_change_pct:+.1f}%, Price: {oi.price_change_pct:+.1f}%)"

    # HOLY GRAIL: Combined leading indicator signal
    if holy_grail:
        hg = holy_grail
        status["HolyGrail"] = (
            f"{hg.direction.upper()} ({hg.confidence:.0f}%) "
            f"[OI:{hg.oi_score:+.0f} Prem:{hg.premium_score:+.0f} "
            f"FVel:{hg.funding_velocity_score:+.0f} Vol:{hg.volume_score:+.0f}]"
        )

    # === Phase 3: MTF Confluence ===
    if mtf_confluence:
        mtf = mtf_confluence
        confluence_str = "YES" if mtf.has_confluence else "NO"
        status["MTF"] = (
            f"{mtf.confluence_direction.upper()} ({confluence_str}) "
            f"[{mtf.bullish_count}B/{mtf.bearish_count}Be, score:{mtf.weighted_score:+.1f}]"
        )

    # === Phase 3: Liquidation Heatmap ===
    if liq_heatmap:
        liq = liq_heatmap
        status["LiqMagnet"] = (
            f"{liq.magnetic_direction.upper()} "
            f"[Long:{liq.nearest_long_liq or 'N/A'} Short:{liq.nearest_short_liq or 'N/A'}]"
        )

    # === Phase 3: CVD Divergence ===
    if cvd_divergence:
        cvd = cvd_divergence
        status["CVD"] = (
            f"{cvd.divergence_type or 'aligned'} "
            f"({cvd.signal_strength:.0f}%) [Spot:{cvd.spot_direction} Perp:{cvd.perp_direction}]"
        )

    # === Phase 4: ML Signal Confirmation ===
    if ml_model:
        ml_status = ml_model.get_status()
        trained_str = "TRAINED" if ml_status["is_trained"] else "LEARNING"
        status["ML"] = (
            f"{trained_str} ({ml_status['training_samples']} samples) "
            f"[Win:{ml_status['win_rate']:.0f}% Acc:{ml_status['accuracy']:.0f}%]"
        )

    if ml_prediction:
        pred = ml_prediction
        status["MLPred"] = (
            f"{pred.confidence:.0f}% conf ({pred.reasoning[:30]}...)"
        )

    return status
