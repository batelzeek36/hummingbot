"""
Enhanced Momentum Trading Strategy for Hyperliquid Monster Bot v2.12

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
from hummingbot.core.data_type.common import PriceType

from ...config import HyperliquidMonsterV2Config
from ...indicators import MACDResult, TrendInfo, VolumeAnalysis
from ...leading_indicators import DirectionalSignal, HyperliquidLeadingIndicators, OIAnalysis
from ...models import StrategyMetrics, StrategyMode
from ...multi_timeframe import MTFConfluence, MultiTimeframeAnalyzer
from ...coinglass import CoinglassAPI, LiquidationHeatmap, SpotPerpDivergence
from ...ml_models import MLPrediction, SignalConfirmationModel
from ...news_events import NewsEventDetector, EventCheckResult
from ...attribution import AttributionTracker

from .signals import evaluate_long_signals, evaluate_short_signals, has_enough_signals, build_entry_info
from .exits import check_exit
from .positions import PositionManager
from .godmode_filters import (
    check_liq_proximity_warning,
    check_all_godmode_filters,
)
from .indicators import calculate_all_indicators, get_funding_sentiment, build_indicator_status


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

        # Initialize News Event Detector (GOD MODE Phase 5)
        self._news_detector: Optional[NewsEventDetector] = None
        if self.config.news_events_enabled:
            try:
                self._news_detector = NewsEventDetector(
                    pause_on_high_impact=self.config.news_pause_on_high_impact,
                    pause_on_critical_only=self.config.news_pause_on_critical_only,
                    close_positions_on_critical=self.config.news_close_on_critical,
                    warning_minutes_ahead=self.config.news_warning_minutes,
                    logger=self.logger,
                )
                status = self._news_detector.get_status()
                self.logger.info(
                    f"NEWS EVENTS: Initialized - "
                    f"{status.get('total_events', 0)} events loaded, "
                    f"next: {status.get('next_event', 'None')}"
                )
            except Exception as e:
                self.logger.warning(f"NEWS EVENTS: Failed to initialize - {e}")
                self._news_detector = None

        # Initialize Attribution Tracker (GOD MODE Phase 5)
        self._attribution_tracker: Optional[AttributionTracker] = None
        if self.config.attribution_enabled:
            try:
                self._attribution_tracker = AttributionTracker(
                    data_path=self.config.attribution_data_path,
                    auto_save=self.config.attribution_auto_save,
                    logger=self.logger,
                )
                status = self._attribution_tracker.get_status()
                self.logger.info(
                    f"ATTRIBUTION: Initialized - "
                    f"{status['total_trades']} historical trades loaded"
                )
            except Exception as e:
                self.logger.warning(f"ATTRIBUTION: Failed to initialize - {e}")
                self._attribution_tracker = None

        # Store last news check result
        self._last_news_check: Optional[EventCheckResult] = None

        # Initialize position manager
        self._position_manager = PositionManager(
            config=config,
            connector=connector,
            metrics=metrics,
            place_order_fn=place_order_fn,
            ml_model=self._ml_model,
            logger=self.logger,
        )

    @property
    def pair(self) -> str:
        """Get the momentum trading pair."""
        return self.config.momentum_pair

    @property
    def position(self) -> Optional[str]:
        """Get current position direction."""
        return self._position_manager.position

    @property
    def entry_price(self) -> Optional[Decimal]:
        """Get current position entry price."""
        return self._position_manager.entry_price

    @property
    def entry_signals(self) -> List[str]:
        """Get signals that triggered current position."""
        return self._position_manager.entry_signals

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

        # === GOD MODE Phase 5: News Event Detection ===
        if self._news_detector:
            news_check = self._news_detector.check_trading_allowed()
            self._last_news_check = news_check

            if news_check.should_pause:
                # Log warning about event (only once per event)
                warning = self._news_detector.get_warning_message()
                if warning:
                    self.logger.warning(f"MOMENTUM: {warning}")

                # If we have a position, we might want to close it for CRITICAL events
                if self.position and self.config.news_close_on_critical:
                    should_close, reason = self._news_detector.should_close_positions()
                    if should_close:
                        self.logger.warning(f"MOMENTUM: Closing position due to {reason}")
                        # Position will be closed in normal exit check below

                # Block new entries during news events
                if not self.position:
                    return  # Skip rest of strategy, don't enter new positions
            else:
                # Log upcoming events as info
                if news_check.upcoming_events:
                    next_event = news_check.upcoming_events[0]
                    minutes_until = next_event.time_until() / 60
                    if minutes_until <= 60 and int(minutes_until) % 15 == 0:
                        self.logger.info(
                            f"MOMENTUM: {next_event.name} in {minutes_until:.0f}min"
                        )

        # Calculate all indicators
        prices = [p for _, p in self.price_history[self.pair]]
        volumes = self.volume_history.get(self.pair, [])

        indicators = calculate_all_indicators(prices, volumes, self.config)
        if indicators is None:
            return

        rsi, trend, macd, volume = indicators

        # Get funding sentiment
        funding_sentiment = None
        if self.config.use_funding_sentiment:
            funding_sentiment = get_funding_sentiment(
                self.connector, self.pair,
                float(self.config.funding_sentiment_threshold),
                self.logger
            )

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
                        check_liq_proximity_warning(
                            liq_heatmap, price_float,
                            float(self.config.liq_cluster_warning_pct),
                            self.logger
                        )

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
        if self._position_manager.has_position() and self._position_manager.entry_price:
            should_exit, reason = check_exit(
                self._position_manager.position,
                self._position_manager.entry_price,
                price, trend, self.config
            )
            if should_exit:
                self._position_manager.close_position(price, reason, global_pnl_updater)
                return

        # Check entry conditions if not in position
        if not self._position_manager.has_position():
            self._check_entry(
                price_float, rsi, trend, macd, volume, funding_sentiment,
                oi_analysis, holy_grail_signal, mtf_confluence, liq_heatmap, cvd_divergence
            )

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
            self._try_long_entry(
                current_price, rsi, trend, macd, volume, funding_sentiment,
                oi_analysis, holy_grail, mtf_confluence, liq_heatmap, cvd_divergence
            )
            return

        # Check for short signal
        if rsi > float(self.config.rsi_overbought):
            self._try_short_entry(
                current_price, rsi, trend, macd, volume, funding_sentiment,
                oi_analysis, holy_grail, mtf_confluence, liq_heatmap, cvd_divergence
            )
            return

    def _try_long_entry(
        self,
        current_price: float,
        rsi: float,
        trend: Optional[TrendInfo],
        macd: Optional[MACDResult],
        volume: Optional[VolumeAnalysis],
        funding_sentiment: Optional[str],
        oi_analysis: Optional[OIAnalysis],
        holy_grail: Optional[DirectionalSignal],
        mtf_confluence: Optional[MTFConfluence],
        liq_heatmap: Optional[LiquidationHeatmap],
        cvd_divergence: Optional[SpotPerpDivergence],
    ):
        """Try to open a long position with GOD MODE filter checks."""
        # Check all GOD MODE filters
        should_block, reason = check_all_godmode_filters(
            direction="long",
            config=self.config,
            mtf_analyzer=self._mtf_analyzer,
            mtf_confluence=mtf_confluence,
            liq_heatmap=liq_heatmap,
            cvd_divergence=cvd_divergence,
            holy_grail=holy_grail,
            oi_tracker=self._oi_tracker,
            pair=self.pair,
            logger=self.logger,
        )
        if should_block:
            self.logger.info(f"MOMENTUM: Long blocked by {reason}")
            return

        # Evaluate signals
        signals = evaluate_long_signals(
            current_price, trend, macd, volume, funding_sentiment,
            self.config, oi_analysis, mtf_confluence, liq_heatmap, self.logger
        )

        if has_enough_signals(signals, "long", self.config, self.logger):
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
            extra_info = build_entry_info(
                holy_grail, mtf_confluence, liq_heatmap,
                float(self.config.holy_grail_min_confidence)
            )
            self._position_manager.open_position(
                "long", Decimal(str(current_price)), rsi, signals, extra_info, ml_features
            )

    def _try_short_entry(
        self,
        current_price: float,
        rsi: float,
        trend: Optional[TrendInfo],
        macd: Optional[MACDResult],
        volume: Optional[VolumeAnalysis],
        funding_sentiment: Optional[str],
        oi_analysis: Optional[OIAnalysis],
        holy_grail: Optional[DirectionalSignal],
        mtf_confluence: Optional[MTFConfluence],
        liq_heatmap: Optional[LiquidationHeatmap],
        cvd_divergence: Optional[SpotPerpDivergence],
    ):
        """Try to open a short position with GOD MODE filter checks."""
        # Check all GOD MODE filters
        should_block, reason = check_all_godmode_filters(
            direction="short",
            config=self.config,
            mtf_analyzer=self._mtf_analyzer,
            mtf_confluence=mtf_confluence,
            liq_heatmap=liq_heatmap,
            cvd_divergence=cvd_divergence,
            holy_grail=holy_grail,
            oi_tracker=self._oi_tracker,
            pair=self.pair,
            logger=self.logger,
        )
        if should_block:
            self.logger.info(f"MOMENTUM: Short blocked by {reason}")
            return

        # Evaluate signals
        signals = evaluate_short_signals(
            current_price, trend, macd, volume, funding_sentiment,
            self.config, oi_analysis, mtf_confluence, liq_heatmap, self.logger
        )

        if has_enough_signals(signals, "short", self.config, self.logger):
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
            extra_info = build_entry_info(
                holy_grail, mtf_confluence, liq_heatmap,
                float(self.config.holy_grail_min_confidence)
            )
            self._position_manager.open_position(
                "short", Decimal(str(current_price)), rsi, signals, extra_info, ml_features
            )

    def close_position_for_shutdown(self, global_pnl_updater: Callable[[Decimal], None]):
        """Close position for shutdown."""
        self._position_manager.close_position_for_shutdown(global_pnl_updater)

    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self._position_manager.has_position()

    def get_position_info(self) -> Optional[Tuple[str, Decimal]]:
        """Get current position info (direction, entry_price)."""
        return self._position_manager.get_position_info()

    def get_indicator_status(self) -> Dict[str, str]:
        """Get current indicator values for status display."""
        return build_indicator_status(
            rsi=self._last_rsi,
            trend=self._last_trend,
            macd=self._last_macd,
            volume=self._last_volume,
            funding_sentiment=self._last_funding_sentiment,
            oi_analysis=self._last_oi_analysis,
            holy_grail=self._last_holy_grail,
            mtf_confluence=self._last_mtf_confluence,
            liq_heatmap=self._last_liq_heatmap,
            cvd_divergence=self._last_cvd_divergence,
            ml_model=self._ml_model,
            ml_prediction=self._last_ml_prediction,
        )

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
