"""
Enhanced Momentum Trading Strategy for Hyperliquid Monster Bot v2.11

Multi-indicator momentum strategy with:
- RSI (primary signal)
- Trend filter (EMA 50/200)
- Volume confirmation
- Funding rate sentiment
- MACD confirmation
- **Open Interest (OI) - TRUE LEADING INDICATOR** (v2.6)
- **HOLY GRAIL Signal** - Combined leading indicators (v2.7)
- **Coinglass CVD & Liquidation Heatmap** (v2.9 - GOD MODE Phase 2)
- **Multi-Timeframe Confluence** (v2.10 - GOD MODE Phase 3)
- **ML Signal Confirmation** (v2.11 - GOD MODE Phase 4)

GOD MODE Phase 4 (v2.11):
- XGBoost Signal Confirmation: ML model as final quality gate
- Feature Engineering: 24 normalized features from all indicators
- Online Learning: Model improves from each trade outcome
- Trade Outcome Tracking: Records wins/losses for continuous learning

Based on GOD_MODE_ENHANCEMENTS.md recommendations.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import PositionAction, PriceType, TradeType

from ..config import HyperliquidMonsterV2Config
from ..indicators import MACDResult, TechnicalIndicators, TrendInfo, VolumeAnalysis
from ..leading_indicators import DirectionalSignal, HyperliquidLeadingIndicators, OIAnalysis, OIMomentum
from ..models import StrategyMetrics, StrategyMode
from ..multi_timeframe import MTFConfluence, MultiTimeframeAnalyzer
from ..coinglass_api import CoinglassAPI, LiquidationHeatmap, SpotPerpDivergence
from ..ml_models import MLFeatures, MLPrediction, SignalConfirmationModel


class MomentumStrategy:
    """
    Enhanced Momentum Trading Strategy with Leading Indicators.

    Uses multi-indicator confluence for higher quality entries:
    1. RSI oversold/overbought (primary trigger)
    2. Trend filter - only trade WITH the trend
    3. Volume confirmation - higher volume = stronger signal
    4. Funding rate sentiment - use market positioning data
    5. MACD confirmation - momentum direction alignment
    6. **Open Interest (OI)** - TRUE LEADING INDICATOR from Hyperliquid API
    7. **Coinglass CVD & Liquidation Heatmap** - Aggregated market data
    8. **Multi-Timeframe Confluence** - 1m/5m/15m/1h weighted signals
    9. **ML Signal Confirmation** - XGBoost model as final quality gate

    GOD MODE Phase 4 Features:
    - ML Confirmation: XGBoost model filters false positives
    - Online Learning: Model improves from trade outcomes
    - Feature Engineering: 24 normalized features
    """

    def __init__(
        self,
        config: HyperliquidMonsterV2Config,
        connector: ConnectorBase,
        metrics: StrategyMetrics,
        place_order_fn: Callable,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the enhanced momentum strategy.

        Args:
            config: Bot configuration
            connector: Exchange connector
            metrics: Strategy metrics tracker
            place_order_fn: Function to place orders
            logger: Logger instance
        """
        self.config = config
        self.connector = connector
        self.metrics = metrics
        self.place_order = place_order_fn
        self.logger = logger or logging.getLogger(__name__)

        # Price history for indicator calculations
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.volume_history: Dict[str, List[float]] = {}

        # Position state
        self.position: Optional[str] = None  # "long", "short", or None
        self.entry_price: Optional[Decimal] = None
        self.entry_signals: List[str] = []  # Signals that triggered entry

        # Cached indicator values for status display
        self._last_rsi: Optional[float] = None
        self._last_trend: Optional[TrendInfo] = None
        self._last_macd: Optional[MACDResult] = None
        self._last_volume: Optional[VolumeAnalysis] = None
        self._last_funding_sentiment: Optional[str] = None
        self._last_oi_analysis: Optional[OIAnalysis] = None
        self._last_holy_grail: Optional[DirectionalSignal] = None
        self._last_mtf_confluence: Optional[MTFConfluence] = None
        self._last_liq_heatmap: Optional[LiquidationHeatmap] = None
        self._last_cvd_divergence: Optional[SpotPerpDivergence] = None
        self._last_ml_prediction: Optional[MLPrediction] = None
        self._current_trade_id: Optional[str] = None  # For ML outcome tracking

        # Initialize Open Interest tracker (TRUE LEADING INDICATOR)
        self._oi_tracker: Optional[HyperliquidLeadingIndicators] = None
        if self.config.use_oi_filter:
            self._oi_tracker = HyperliquidLeadingIndicators(
                oi_lookback_periods=self.config.oi_lookback_periods,
                fetch_interval_seconds=self.config.oi_fetch_interval,
                oi_change_threshold=float(self.config.oi_change_threshold),
                logger=self.logger,
            )
            self.logger.info("OI LEADING INDICATOR: Initialized - will fetch from Hyperliquid API")

        # Initialize Coinglass API (GOD MODE Phase 2)
        self._coinglass: Optional[CoinglassAPI] = None
        if self.config.coinglass_enabled:
            try:
                api_key = self.config.coinglass_api_key or None
                self._coinglass = CoinglassAPI(
                    api_key=api_key,
                    request_interval=float(self.config.coinglass_request_interval),
                    logger=self.logger,
                )
                self.logger.info("COINGLASS API: Initialized - CVD, Liquidations, Long/Short ratios")
            except Exception as e:
                self.logger.warning(f"COINGLASS API: Failed to initialize - {e}")
                self._coinglass = None

        # Initialize Multi-Timeframe Analyzer (GOD MODE Phase 3)
        self._mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None
        if self.config.use_mtf_confluence:
            timeframe_weights = {
                "1m": float(self.config.mtf_weight_1m),
                "5m": float(self.config.mtf_weight_5m),
                "15m": float(self.config.mtf_weight_15m),
                "1h": float(self.config.mtf_weight_1h),
            }
            self._mtf_analyzer = MultiTimeframeAnalyzer(
                timeframe_weights=timeframe_weights,
                ema_short_period=self.config.mtf_ema_short,
                ema_long_period=self.config.mtf_ema_long,
                min_confluence_timeframes=self.config.mtf_min_timeframes,
                strong_trend_threshold=float(self.config.mtf_strong_trend_threshold),
                logger=self.logger,
            )
            self.logger.info(
                f"MTF CONFLUENCE: Initialized - 1m/5m/15m/1h with {self.config.mtf_min_timeframes}+ TF agreement"
            )

        # Initialize ML Signal Confirmation Model (GOD MODE Phase 4)
        self._ml_model: Optional[SignalConfirmationModel] = None
        if self.config.use_ml_confirmation:
            try:
                self._ml_model = SignalConfirmationModel(
                    min_confidence=float(self.config.ml_min_confidence),
                    min_training_samples=self.config.ml_min_training_samples,
                    model_path=self.config.ml_model_path,
                    data_path=self.config.ml_data_path,
                    logger=self.logger,
                )
                status = self._ml_model.get_status()
                self.logger.info(
                    f"ML CONFIRMATION: Initialized - "
                    f"{'Trained' if status['is_trained'] else 'Learning'} "
                    f"({status['training_samples']} samples)"
                )
            except Exception as e:
                self.logger.warning(f"ML CONFIRMATION: Failed to initialize - {e}")
                self._ml_model = None

    @property
    def pair(self) -> str:
        """Get the momentum trading pair."""
        return self.config.momentum_pair

    def run(self, global_pnl_updater: Callable[[Decimal], None]):
        """
        Run the enhanced momentum strategy.

        Args:
            global_pnl_updater: Callback to update global P&L when closing positions
        """
        # Get current price
        price = self.connector.get_price_by_type(self.pair, PriceType.MidPrice)
        if price is None:
            return

        now = datetime.now()
        price_float = float(price)

        # Initialize history if needed
        if self.pair not in self.price_history:
            self.price_history[self.pair] = []
        if self.pair not in self.volume_history:
            self.volume_history[self.pair] = []

        # Add current price to history
        self.price_history[self.pair].append((now, price_float))

        # Try to get volume data (may not be available on all exchanges)
        try:
            # Estimate volume from order book spread as proxy
            bid = self.connector.get_price_by_type(self.pair, PriceType.BestBid)
            ask = self.connector.get_price_by_type(self.pair, PriceType.BestAsk)
            if bid and ask:
                spread = float(ask - bid)
                # Inverse spread as volume proxy (tighter spread = more volume)
                volume_proxy = 1.0 / max(spread / float(price), 0.0001)
                self.volume_history[self.pair].append(volume_proxy)
        except Exception:
            pass

        # Keep data for trend analysis (need 200+ periods for EMA 200)
        cutoff = now - timedelta(hours=4)  # 4 hours of data
        self.price_history[self.pair] = [
            (t, p) for t, p in self.price_history[self.pair] if t > cutoff
        ]
        if self.pair in self.volume_history:
            # Keep same length as price history
            max_len = len(self.price_history[self.pair])
            self.volume_history[self.pair] = self.volume_history[self.pair][-max_len:]

        # Check if we have enough data for basic analysis
        min_required = max(
            self.config.momentum_lookback,
            self.config.ema_long_period if self.config.use_trend_filter else 0,
            self.config.macd_slow + self.config.macd_signal if self.config.use_macd_filter else 0
        )

        if len(self.price_history[self.pair]) < min_required:
            if self.metrics.status == StrategyMode.WARMING_UP:
                pct = len(self.price_history[self.pair]) / min_required * 100
                if len(self.price_history[self.pair]) % 20 == 0:
                    self.logger.info(f"MOMENTUM: Warming up... {pct:.0f}% ({len(self.price_history[self.pair])}/{min_required} periods)")
            return

        # Activate strategy once warmed up
        if self.metrics.status == StrategyMode.WARMING_UP:
            self.metrics.status = StrategyMode.ACTIVE
            self.logger.info("MOMENTUM: Strategy activated with enhanced indicators")

        if self.metrics.status != StrategyMode.ACTIVE:
            return

        # Calculate all indicators
        prices = [p for _, p in self.price_history[self.pair]]
        indicators = self._calculate_all_indicators(prices)

        if indicators is None:
            return

        rsi, trend, macd, volume, funding_sentiment = indicators

        # Fetch OI data from Hyperliquid API (TRUE LEADING INDICATOR)
        oi_analysis = None
        holy_grail_signal = None
        if self._oi_tracker:
            self._oi_tracker.update([self.pair])
            oi_analysis = self._oi_tracker.analyze_oi_momentum(self.pair)
            if oi_analysis and oi_analysis.warning:
                self.logger.info(f"OI WARNING: {oi_analysis.warning}")

            # Get HOLY GRAIL combined signal
            if self.config.use_holy_grail_signal:
                holy_grail_signal = self._oi_tracker.get_holy_grail_signal(self.pair)
                if holy_grail_signal and holy_grail_signal.warnings:
                    for warning in holy_grail_signal.warnings:
                        self.logger.info(f"HOLY GRAIL WARNING: {warning}")

        # === GOD MODE Phase 3: MTF Confluence ===
        mtf_confluence = None
        if self._mtf_analyzer:
            # Feed price to MTF analyzer
            self._mtf_analyzer.update(self.pair, price_float, now)
            mtf_confluence = self._mtf_analyzer.get_confluence(self.pair)
            if mtf_confluence and mtf_confluence.warnings:
                for warning in mtf_confluence.warnings:
                    self.logger.debug(f"MTF WARNING: {warning}")

        # === GOD MODE Phase 3: Liquidation Heatmap + CVD Divergence ===
        liq_heatmap = None
        cvd_divergence = None
        if self._coinglass:
            try:
                # Fetch liquidation heatmap (shows where liquidations cluster)
                if self.config.use_liq_magnet:
                    liq_heatmap = self._coinglass.get_liquidation_heatmap(self.pair)
                    if liq_heatmap:
                        # Check for imminent liquidation warning
                        self._check_liq_proximity_warning(liq_heatmap, price_float)

                # Fetch CVD divergence (spot vs perp)
                if self.config.use_cvd_signals:
                    cvd_divergence = self._coinglass.get_spot_perp_divergence(self.pair)
                    if cvd_divergence and cvd_divergence.warnings:
                        for warning in cvd_divergence.warnings:
                            self.logger.info(f"CVD DIVERGENCE: {warning}")
            except Exception as e:
                self.logger.debug(f"Coinglass fetch error: {e}")

        # Cache for status display
        self._last_rsi = rsi
        self._last_trend = trend
        self._last_macd = macd
        self._last_volume = volume
        self._last_funding_sentiment = funding_sentiment
        self._last_oi_analysis = oi_analysis
        self._last_holy_grail = holy_grail_signal
        self._last_mtf_confluence = mtf_confluence
        self._last_liq_heatmap = liq_heatmap
        self._last_cvd_divergence = cvd_divergence

        # Check exit conditions if in position
        if self.position and self.entry_price:
            should_exit, reason = self._check_exit(price, trend)
            if should_exit:
                self._close_position(price, reason, global_pnl_updater)
                return

        # Check entry conditions if not in position
        if self.position is None:
            self._check_entry(
                price_float, rsi, trend, macd, volume, funding_sentiment,
                oi_analysis, holy_grail_signal, mtf_confluence, liq_heatmap, cvd_divergence
            )

    def _calculate_all_indicators(
        self,
        prices: List[float]
    ) -> Optional[Tuple[float, Optional[TrendInfo], Optional[MACDResult], Optional[VolumeAnalysis], Optional[str]]]:
        """
        Calculate all technical indicators.

        Returns:
            Tuple of (rsi, trend, macd, volume, funding_sentiment) or None
        """
        # RSI (always required)
        rsi = TechnicalIndicators.calculate_rsi(prices, self.config.momentum_lookback)
        if rsi is None:
            return None

        # Trend filter (optional)
        trend = None
        if self.config.use_trend_filter:
            trend = TechnicalIndicators.analyze_trend(
                prices,
                self.config.ema_short_period,
                self.config.ema_long_period
            )

        # MACD (optional)
        macd = None
        if self.config.use_macd_filter:
            macd = TechnicalIndicators.calculate_macd(
                prices,
                self.config.macd_fast,
                self.config.macd_slow,
                self.config.macd_signal
            )

        # Volume analysis (optional)
        volume = None
        if self.config.use_volume_filter and self.pair in self.volume_history:
            volumes = self.volume_history[self.pair]
            if len(volumes) >= self.config.volume_lookback:
                volume = TechnicalIndicators.analyze_volume(
                    volumes,
                    self.config.volume_lookback,
                    float(self.config.strong_volume_ratio)
                )

        # Funding rate sentiment (optional)
        funding_sentiment = None
        if self.config.use_funding_sentiment:
            funding_sentiment = self._get_funding_sentiment()

        return (rsi, trend, macd, volume, funding_sentiment)

    def _get_funding_sentiment(self) -> Optional[str]:
        """
        Get funding rate sentiment for the momentum pair.

        Positive funding = longs paying shorts = too many longs = bearish
        Negative funding = shorts paying longs = too many shorts = bullish

        Returns:
            "bullish", "bearish", or "neutral"
        """
        try:
            funding_info = self.connector.get_funding_info(self.pair)
            if funding_info is None:
                return None

            rate = float(funding_info.rate)
            threshold = float(self.config.funding_sentiment_threshold)

            if rate > threshold:
                return "bearish"  # Crowded long, expect pullback
            elif rate < -threshold:
                return "bullish"  # Crowded short, expect squeeze
            else:
                return "neutral"

        except Exception as e:
            self.logger.debug(f"Could not get funding sentiment: {e}")
            return None

    def _check_entry(
        self,
        current_price: float,
        rsi: float,
        trend: Optional[TrendInfo],
        macd: Optional[MACDResult],
        volume: Optional[VolumeAnalysis],
        funding_sentiment: Optional[str],
        oi_analysis: Optional[OIAnalysis] = None,
        holy_grail: Optional[DirectionalSignal] = None,
        mtf_confluence: Optional[MTFConfluence] = None,
        liq_heatmap: Optional[LiquidationHeatmap] = None,
        cvd_divergence: Optional[SpotPerpDivergence] = None,
    ):
        """
        Check entry conditions with multi-indicator confluence.

        Entry requires:
        1. RSI signal (oversold for long, overbought for short)
        2. At least N confirming signals from: trend, volume, funding, MACD, OI
        3. OI must not block the entry (exhaustion detection)
        4. Holy Grail signal must not block the entry (if enabled)
        5. MTF Confluence must not block the entry (Phase 3)
        6. Liquidation magnet must not strongly oppose (Phase 3)
        7. CVD divergence must not indicate trapped traders (Phase 3)
        """
        # HOLY GRAIL: Check combined leading indicator signal first
        if holy_grail and self.config.use_holy_grail_signal:
            min_confidence = float(self.config.holy_grail_min_confidence)

            # Log the holy grail signal
            if holy_grail.confidence >= min_confidence:
                self.logger.debug(
                    f"HOLY GRAIL: {holy_grail.direction.upper()} "
                    f"({holy_grail.confidence:.0f}% confidence) - {holy_grail.reasoning}"
                )

        # Check for long signal
        if rsi < float(self.config.rsi_oversold):
            # === GOD MODE Phase 3: MTF Confluence Check ===
            if mtf_confluence and self.config.mtf_block_contrary:
                should_block, reason = self._mtf_analyzer.should_block_entry(self.pair, "long") if self._mtf_analyzer else (False, "")
                if should_block:
                    self.logger.info(f"MOMENTUM: Long blocked by MTF - {reason}")
                    return

                # Require 1H alignment if configured
                if self.config.mtf_require_htf_alignment and "1h" in mtf_confluence.signals:
                    htf_signal = mtf_confluence.signals["1h"]
                    if htf_signal.trend.value.startswith("bear"):
                        self.logger.info(f"MOMENTUM: Long blocked - 1H timeframe bearish")
                        return

            # === GOD MODE Phase 3: Liquidation Magnet Check ===
            if liq_heatmap and self.config.liq_magnet_block_contrary:
                should_block, reason = self._check_liq_magnet_blocks(liq_heatmap, "long")
                if should_block:
                    self.logger.info(f"MOMENTUM: Long blocked by LIQ MAGNET - {reason}")
                    return

            # === GOD MODE Phase 3: CVD Divergence Check ===
            if cvd_divergence and self.config.cvd_block_contrary_entries:
                if cvd_divergence.divergence_type == "bearish_divergence":
                    if cvd_divergence.signal_strength >= float(self.config.cvd_divergence_threshold):
                        self.logger.info(
                            f"MOMENTUM: Long blocked by CVD DIVERGENCE - "
                            f"Perp longs trapped ({cvd_divergence.signal_strength:.0f}%)"
                        )
                        return

            # HOLY GRAIL: Block long if signal says short with high confidence
            if holy_grail and self.config.use_holy_grail_signal and self.config.holy_grail_block_contrary:
                min_confidence = float(self.config.holy_grail_min_confidence)
                if holy_grail.direction in ("short", "strong_short") and holy_grail.confidence >= min_confidence:
                    self.logger.info(
                        f"MOMENTUM: Long blocked by HOLY GRAIL - "
                        f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"
                    )
                    return

                # Require strong signal if configured
                if self.config.holy_grail_require_strong:
                    if holy_grail.direction != "strong_long" or holy_grail.confidence < min_confidence:
                        self.logger.info(f"MOMENTUM: Long blocked - HOLY GRAIL not strong_long")
                        return

            # OI LEADING INDICATOR: Block entry during exhaustion
            if self.config.oi_block_exhaustion_entries and oi_analysis:
                should_block, reason = self._oi_tracker.should_block_entry(self.pair, "long") if self._oi_tracker else (False, "")
                if should_block:
                    self.logger.info(f"MOMENTUM: Long blocked by OI - {reason}")
                    return

            signals = self._evaluate_long_signals(
                current_price, trend, macd, volume, funding_sentiment, oi_analysis, mtf_confluence, liq_heatmap
            )
            if self._has_enough_signals(signals, "long"):
                # === GOD MODE Phase 4: ML Signal Confirmation ===
                ml_features = None
                if self._ml_model and self.config.ml_block_low_confidence:
                    should_take, ml_reason, ml_features = self._ml_model.should_confirm_trade(
                        symbol=self.pair,
                        direction="long",
                        rsi=rsi,
                        trend_info=trend,
                        macd_info=macd,
                        volume_info=volume,
                        oi_analysis=oi_analysis,
                        holy_grail=holy_grail,
                        mtf_confluence=mtf_confluence,
                        liq_heatmap=liq_heatmap,
                        cvd_divergence=cvd_divergence,
                    )
                    if not should_take:
                        self.logger.info(f"MOMENTUM: Long blocked by ML - {ml_reason}")
                        return

                # Build entry info string
                extra_info = self._build_entry_info(holy_grail, mtf_confluence, liq_heatmap)
                self._open_position("long", Decimal(str(current_price)), rsi, signals, extra_info, ml_features)
                return

        # Check for short signal
        if rsi > float(self.config.rsi_overbought):
            # === GOD MODE Phase 3: MTF Confluence Check ===
            if mtf_confluence and self.config.mtf_block_contrary:
                should_block, reason = self._mtf_analyzer.should_block_entry(self.pair, "short") if self._mtf_analyzer else (False, "")
                if should_block:
                    self.logger.info(f"MOMENTUM: Short blocked by MTF - {reason}")
                    return

                # Require 1H alignment if configured
                if self.config.mtf_require_htf_alignment and "1h" in mtf_confluence.signals:
                    htf_signal = mtf_confluence.signals["1h"]
                    if htf_signal.trend.value.startswith("bull"):
                        self.logger.info(f"MOMENTUM: Short blocked - 1H timeframe bullish")
                        return

            # === GOD MODE Phase 3: Liquidation Magnet Check ===
            if liq_heatmap and self.config.liq_magnet_block_contrary:
                should_block, reason = self._check_liq_magnet_blocks(liq_heatmap, "short")
                if should_block:
                    self.logger.info(f"MOMENTUM: Short blocked by LIQ MAGNET - {reason}")
                    return

            # === GOD MODE Phase 3: CVD Divergence Check ===
            if cvd_divergence and self.config.cvd_block_contrary_entries:
                if cvd_divergence.divergence_type == "bullish_divergence":
                    if cvd_divergence.signal_strength >= float(self.config.cvd_divergence_threshold):
                        self.logger.info(
                            f"MOMENTUM: Short blocked by CVD DIVERGENCE - "
                            f"Perp shorts trapped ({cvd_divergence.signal_strength:.0f}%)"
                        )
                        return

            # HOLY GRAIL: Block short if signal says long with high confidence
            if holy_grail and self.config.use_holy_grail_signal and self.config.holy_grail_block_contrary:
                min_confidence = float(self.config.holy_grail_min_confidence)
                if holy_grail.direction in ("long", "strong_long") and holy_grail.confidence >= min_confidence:
                    self.logger.info(
                        f"MOMENTUM: Short blocked by HOLY GRAIL - "
                        f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"
                    )
                    return

                # Require strong signal if configured
                if self.config.holy_grail_require_strong:
                    if holy_grail.direction != "strong_short" or holy_grail.confidence < min_confidence:
                        self.logger.info(f"MOMENTUM: Short blocked - HOLY GRAIL not strong_short")
                        return

            # OI LEADING INDICATOR: Block entry during exhaustion
            if self.config.oi_block_exhaustion_entries and oi_analysis:
                should_block, reason = self._oi_tracker.should_block_entry(self.pair, "short") if self._oi_tracker else (False, "")
                if should_block:
                    self.logger.info(f"MOMENTUM: Short blocked by OI - {reason}")
                    return

            signals = self._evaluate_short_signals(
                current_price, trend, macd, volume, funding_sentiment, oi_analysis, mtf_confluence, liq_heatmap
            )
            if self._has_enough_signals(signals, "short"):
                # === GOD MODE Phase 4: ML Signal Confirmation ===
                ml_features = None
                if self._ml_model and self.config.ml_block_low_confidence:
                    should_take, ml_reason, ml_features = self._ml_model.should_confirm_trade(
                        symbol=self.pair,
                        direction="short",
                        rsi=rsi,
                        trend_info=trend,
                        macd_info=macd,
                        volume_info=volume,
                        oi_analysis=oi_analysis,
                        holy_grail=holy_grail,
                        mtf_confluence=mtf_confluence,
                        liq_heatmap=liq_heatmap,
                        cvd_divergence=cvd_divergence,
                    )
                    if not should_take:
                        self.logger.info(f"MOMENTUM: Short blocked by ML - {ml_reason}")
                        return

                # Build entry info string
                extra_info = self._build_entry_info(holy_grail, mtf_confluence, liq_heatmap)
                self._open_position("short", Decimal(str(current_price)), rsi, signals, extra_info, ml_features)
                return

    def _check_liq_proximity_warning(self, heatmap: LiquidationHeatmap, current_price: float):
        """Check if price is near a liquidation cluster and log warning."""
        warning_pct = float(self.config.liq_cluster_warning_pct)

        # Check nearest long liquidations (below price)
        if heatmap.nearest_long_liq:
            distance_pct = ((current_price - heatmap.nearest_long_liq) / current_price) * 100
            if distance_pct < warning_pct:
                self.logger.warning(
                    f"LIQ WARNING: Long liquidations {distance_pct:.1f}% below at ${heatmap.nearest_long_liq:.2f}"
                )

        # Check nearest short liquidations (above price)
        if heatmap.nearest_short_liq:
            distance_pct = ((heatmap.nearest_short_liq - current_price) / current_price) * 100
            if distance_pct < warning_pct:
                self.logger.warning(
                    f"LIQ WARNING: Short liquidations {distance_pct:.1f}% above at ${heatmap.nearest_short_liq:.2f}"
                )

    def _check_liq_magnet_blocks(self, heatmap: LiquidationHeatmap, direction: str) -> Tuple[bool, str]:
        """
        Check if liquidation magnet direction blocks the entry.

        Price is "magnetically attracted" to liquidation clusters.
        Don't go long if strong magnetic pull is down (toward long liquidations).
        Don't go short if strong magnetic pull is up (toward short liquidations).
        """
        min_imbalance = float(self.config.liq_magnet_min_imbalance)

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

    def _build_entry_info(
        self,
        holy_grail: Optional[DirectionalSignal],
        mtf: Optional[MTFConfluence],
        liq: Optional[LiquidationHeatmap]
    ) -> str:
        """Build extra info string for entry logging."""
        parts = []

        if holy_grail and holy_grail.confidence >= float(self.config.holy_grail_min_confidence):
            parts.append(f"HG:{holy_grail.direction[:3]}{holy_grail.confidence:.0f}%")

        if mtf and mtf.has_confluence:
            parts.append(f"MTF:{mtf.bullish_count}B/{mtf.bearish_count}Be")

        if liq:
            parts.append(f"LIQ:{liq.magnetic_direction}")

        return f" [{', '.join(parts)}]" if parts else ""

    def _evaluate_long_signals(
        self,
        price: float,
        trend: Optional[TrendInfo],
        macd: Optional[MACDResult],
        volume: Optional[VolumeAnalysis],
        funding: Optional[str],
        oi: Optional[OIAnalysis] = None,
        mtf: Optional[MTFConfluence] = None,
        liq: Optional[LiquidationHeatmap] = None,
    ) -> Dict[str, bool]:
        """Evaluate confirming signals for a long entry."""
        signals = {"RSI": True}  # RSI already confirmed

        # Trend: must be in uptrend or neutral
        if self.config.use_trend_filter and trend:
            signals["Trend"] = trend.direction != "downtrend"
            # Bonus: price above EMA 200
            if price > trend.ema_200:
                signals["Trend"] = True

        # MACD: MACD line above signal line (bullish)
        if self.config.use_macd_filter and macd:
            signals["MACD"] = macd.macd_line > macd.signal_line

        # Volume: at least minimum ratio
        if self.config.use_volume_filter and volume:
            signals["Volume"] = volume.volume_ratio >= float(self.config.min_volume_ratio)

        # Funding: bullish sentiment (shorts crowded)
        if self.config.use_funding_sentiment and funding:
            signals["Funding"] = funding == "bullish"

        # OI: LEADING INDICATOR - confirms long conviction
        # Bullish: OI rising with price (new longs) OR capitulation (bounce setup)
        if self.config.use_oi_filter and oi:
            signals["OI"] = oi.should_confirm_long
            if oi.momentum == OIMomentum.BULLISH_CONFIRM:
                self.logger.debug(f"OI confirms LONG: new longs entering (OI +{oi.oi_change_pct:.1f}%)")
            elif oi.momentum == OIMomentum.BEARISH_EXHAUSTION:
                self.logger.debug(f"OI contrarian LONG: capitulation bounce setup")

        # MTF Confluence: Phase 3 - bullish confluence confirms long
        if self.config.use_mtf_confluence and mtf:
            signals["MTF"] = mtf.confluence_direction == "bullish" and mtf.has_confluence
            if signals["MTF"]:
                self.logger.debug(f"MTF confirms LONG: {mtf.bullish_count} bullish TFs ({mtf.weighted_score:+.1f})")

        # Liquidation Magnet: Phase 3 - upward magnet confirms long
        if self.config.use_liq_magnet and liq:
            signals["LiqMagnet"] = liq.magnetic_direction == "up"
            if signals["LiqMagnet"]:
                self.logger.debug(f"LIQ MAGNET confirms LONG: upward pull toward short liquidations")

        return signals

    def _evaluate_short_signals(
        self,
        price: float,
        trend: Optional[TrendInfo],
        macd: Optional[MACDResult],
        volume: Optional[VolumeAnalysis],
        funding: Optional[str],
        oi: Optional[OIAnalysis] = None,
        mtf: Optional[MTFConfluence] = None,
        liq: Optional[LiquidationHeatmap] = None,
    ) -> Dict[str, bool]:
        """Evaluate confirming signals for a short entry."""
        signals = {"RSI": True}  # RSI already confirmed

        # Trend: must be in downtrend or neutral
        if self.config.use_trend_filter and trend:
            signals["Trend"] = trend.direction != "uptrend"
            # Bonus: price below EMA 200
            if price < trend.ema_200:
                signals["Trend"] = True

        # MACD: MACD line below signal line (bearish)
        if self.config.use_macd_filter and macd:
            signals["MACD"] = macd.macd_line < macd.signal_line

        # Volume: at least minimum ratio
        if self.config.use_volume_filter and volume:
            signals["Volume"] = volume.volume_ratio >= float(self.config.min_volume_ratio)

        # Funding: bearish sentiment (longs crowded)
        if self.config.use_funding_sentiment and funding:
            signals["Funding"] = funding == "bearish"

        # OI: LEADING INDICATOR - confirms short conviction
        # Bearish: OI rising with price falling (new shorts entering)
        if self.config.use_oi_filter and oi:
            signals["OI"] = oi.should_confirm_short
            if oi.momentum == OIMomentum.BEARISH_CONFIRM:
                self.logger.debug(f"OI confirms SHORT: new shorts entering (OI +{oi.oi_change_pct:.1f}%)")

        # MTF Confluence: Phase 3 - bearish confluence confirms short
        if self.config.use_mtf_confluence and mtf:
            signals["MTF"] = mtf.confluence_direction == "bearish" and mtf.has_confluence
            if signals["MTF"]:
                self.logger.debug(f"MTF confirms SHORT: {mtf.bearish_count} bearish TFs ({mtf.weighted_score:+.1f})")

        # Liquidation Magnet: Phase 3 - downward magnet confirms short
        if self.config.use_liq_magnet and liq:
            signals["LiqMagnet"] = liq.magnetic_direction == "down"
            if signals["LiqMagnet"]:
                self.logger.debug(f"LIQ MAGNET confirms SHORT: downward pull toward long liquidations")

        return signals

    def _has_enough_signals(self, signals: Dict[str, bool], direction: str) -> bool:
        """
        Check if we have enough confirming signals.

        RSI is always required. Additionally need min_signals_required
        from the other indicators (including OI, MTF, and LiqMagnet).
        """
        # RSI must be true
        if not signals.get("RSI", False):
            return False

        # If OI confirmation is required (stricter mode), check it
        if self.config.oi_require_confirmation and self.config.use_oi_filter:
            if not signals.get("OI", False):
                self.logger.debug(f"MOMENTUM: {direction} blocked - OI confirmation required but not present")
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
        required = self.config.min_signals_required

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

            self.logger.info(
                f"MOMENTUM: {direction.upper()} confluence met {extras_str}"
                f"({confirming_count + 1}/{len(signals)} signals: {', '.join(active)})"
            )
            return True

        return False

    def _open_position(
        self,
        direction: str,
        price: Decimal,
        rsi: float,
        signals: Dict[str, bool],
        hg_info: str = "",
        ml_features: Optional[MLFeatures] = None,
    ):
        """Open a momentum position with enhanced logging and ML tracking."""
        import uuid

        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        # Notional = margin * leverage, then convert to asset quantity
        notional = position_size * leverage
        amount = notional / price
        trade_type = TradeType.BUY if direction == "long" else TradeType.SELL

        order_id = self.place_order(
            pair=self.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.OPEN,
            strategy="momentum"
        )

        if order_id:
            self.position = direction
            self.entry_price = price
            self.entry_signals = [k for k, v in signals.items() if v]

            # Generate trade ID for ML tracking
            self._current_trade_id = str(uuid.uuid4())[:8]

            # Record entry for ML online learning
            if self._ml_model and ml_features:
                self._ml_model.record_entry(
                    trade_id=self._current_trade_id,
                    features=ml_features,
                    entry_price=float(price),
                )

            # Build signal summary
            signal_str = ", ".join(self.entry_signals)
            ml_info = " [ML+]" if ml_features else ""
            self.logger.info(
                f"MOMENTUM: Opened {direction.upper()} @ {price:.2f} "
                f"(RSI: {rsi:.1f}, ${position_size} x {leverage}x) "
                f"Signals: [{signal_str}]{hg_info}{ml_info}"
            )

    def _check_exit(self, current_price: Decimal, trend: Optional[TrendInfo]) -> Tuple[bool, str]:
        """
        Check if momentum position should exit.

        Includes standard TP/SL plus trend reversal exit.
        """
        if not self.entry_price:
            return False, ""

        if self.position == "long":
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        # Take profit
        if pnl_pct >= self.config.momentum_take_profit:
            return True, "take_profit"

        # Stop loss
        if pnl_pct <= -self.config.momentum_stop_loss:
            return True, "stop_loss"

        # Trend reversal exit (optional enhancement)
        if self.config.use_trend_filter and trend:
            if self.position == "long" and trend.direction == "downtrend":
                # Exit long if trend turns down
                if pnl_pct > 0:  # Only if in profit
                    return True, "trend_reversal"
            elif self.position == "short" and trend.direction == "uptrend":
                # Exit short if trend turns up
                if pnl_pct > 0:
                    return True, "trend_reversal"

        return False, ""

    def _close_position(self, price: Decimal, reason: str, global_pnl_updater: Callable[[Decimal], None]):
        """Close momentum position with enhanced P&L tracking and ML learning."""
        if not self.position:
            return

        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        # Notional = margin * leverage, then convert to asset quantity
        notional = position_size * leverage
        amount = notional / price
        trade_type = TradeType.SELL if self.position == "long" else TradeType.BUY

        order_id = self.place_order(
            pair=self.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.CLOSE,
            strategy="momentum"
        )

        if order_id and self.entry_price:
            if self.position == "long":
                pnl = (price - self.entry_price) * amount * leverage
            else:
                pnl = (self.entry_price - price) * amount * leverage

            self.metrics.realized_pnl += pnl
            global_pnl_updater(pnl)

            # Track win/loss
            if pnl > 0:
                self.metrics.winning_trades += 1

            # === GOD MODE Phase 4: Record exit for ML online learning ===
            ml_learned = ""
            if self._ml_model and self._current_trade_id:
                self._ml_model.record_exit(
                    trade_id=self._current_trade_id,
                    exit_price=float(price),
                    exit_reason=reason,
                )
                ml_learned = " [ML📚]"  # Learning indicator

            pnl_pct = pnl / position_size * 100
            signals_str = ", ".join(self.entry_signals) if self.entry_signals else "N/A"

            self.logger.info(
                f"MOMENTUM: Closed {self.position.upper()} @ {price:.2f} "
                f"({reason}) P&L: ${pnl:.4f} ({pnl_pct:.2f}%) "
                f"Entry signals: [{signals_str}]{ml_learned}"
            )

        self.position = None
        self.entry_price = None
        self.entry_signals = []
        self._current_trade_id = None  # Reset trade ID

    def close_position_for_shutdown(self, global_pnl_updater: Callable[[Decimal], None]):
        """Close position for shutdown."""
        if self.position:
            price = self.connector.get_price_by_type(self.pair, PriceType.MidPrice)
            if price:
                self._close_position(price, "shutdown", global_pnl_updater)

    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.position is not None

    def get_position_info(self) -> Optional[Tuple[str, Decimal]]:
        """Get current position info (direction, entry_price)."""
        if self.position and self.entry_price:
            return (self.position, self.entry_price)
        return None

    def get_indicator_status(self) -> Dict[str, str]:
        """Get current indicator values for status display."""
        status = {}

        if self._last_rsi is not None:
            status["RSI"] = f"{self._last_rsi:.1f}"

        if self._last_trend:
            status["Trend"] = f"{self._last_trend.direction} (EMA50/200: {self._last_trend.strength:.1f}%)"

        if self._last_macd:
            direction = "bullish" if self._last_macd.histogram > 0 else "bearish"
            status["MACD"] = f"{direction} (hist: {self._last_macd.histogram:.4f})"

        if self._last_volume:
            vol_str = "HIGH" if self._last_volume.is_high_volume else "normal"
            status["Volume"] = f"{self._last_volume.volume_ratio:.2f}x ({vol_str})"

        if self._last_funding_sentiment:
            status["Funding"] = self._last_funding_sentiment

        # OI: LEADING INDICATOR from Hyperliquid API
        if self._last_oi_analysis:
            oi = self._last_oi_analysis
            momentum_str = oi.momentum.value.replace("_", " ").title()
            status["OI"] = f"{momentum_str} (OI: {oi.oi_change_pct:+.1f}%, Price: {oi.price_change_pct:+.1f}%)"

        # HOLY GRAIL: Combined leading indicator signal
        if self._last_holy_grail:
            hg = self._last_holy_grail
            status["HolyGrail"] = (
                f"{hg.direction.upper()} ({hg.confidence:.0f}%) "
                f"[OI:{hg.oi_score:+.0f} Prem:{hg.premium_score:+.0f} "
                f"FVel:{hg.funding_velocity_score:+.0f} Vol:{hg.volume_score:+.0f}]"
            )

        # === Phase 3: MTF Confluence ===
        if self._last_mtf_confluence:
            mtf = self._last_mtf_confluence
            confluence_str = "YES" if mtf.has_confluence else "NO"
            status["MTF"] = (
                f"{mtf.confluence_direction.upper()} ({confluence_str}) "
                f"[{mtf.bullish_count}B/{mtf.bearish_count}Be, score:{mtf.weighted_score:+.1f}]"
            )

        # === Phase 3: Liquidation Heatmap ===
        if self._last_liq_heatmap:
            liq = self._last_liq_heatmap
            status["LiqMagnet"] = (
                f"{liq.magnetic_direction.upper()} "
                f"[Long:{liq.nearest_long_liq or 'N/A'} Short:{liq.nearest_short_liq or 'N/A'}]"
            )

        # === Phase 3: CVD Divergence ===
        if self._last_cvd_divergence:
            cvd = self._last_cvd_divergence
            status["CVD"] = (
                f"{cvd.divergence_type or 'aligned'} "
                f"({cvd.signal_strength:.0f}%) [Spot:{cvd.spot_direction} Perp:{cvd.perp_direction}]"
            )

        # === Phase 4: ML Signal Confirmation ===
        if self._ml_model:
            ml_status = self._ml_model.get_status()
            trained_str = "TRAINED" if ml_status["is_trained"] else "LEARNING"
            status["ML"] = (
                f"{trained_str} ({ml_status['training_samples']} samples) "
                f"[Win:{ml_status['win_rate']:.0f}% Acc:{ml_status['accuracy']:.0f}%]"
            )

        if self._last_ml_prediction:
            pred = self._last_ml_prediction
            status["MLPred"] = (
                f"{pred.confidence:.0f}% conf ({pred.reasoning[:30]}...)"
            )

        return status

    def get_oi_tracker(self) -> Optional[HyperliquidLeadingIndicators]:
        """Get the OI tracker for external access (e.g., status display)."""
        return self._oi_tracker

    def get_mtf_analyzer(self) -> Optional[MultiTimeframeAnalyzer]:
        """Get the MTF analyzer for external access (e.g., status display)."""
        return self._mtf_analyzer

    def get_coinglass(self) -> Optional[CoinglassAPI]:
        """Get the Coinglass API client for external access."""
        return self._coinglass

    def get_ml_model(self) -> Optional[SignalConfirmationModel]:
        """Get the ML Signal Confirmation model for external access."""
        return self._ml_model
