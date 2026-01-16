"""
Attribution Tracker - Records signal states for each trade.

This is called from the strategy when trades are opened/closed.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    SignalSnapshot,
    SignalType,
    TradeAttribution,
)


class AttributionTracker:
    """
    Tracks signal attribution for all trades.

    Usage:
        tracker = AttributionTracker()

        # On trade open
        snapshot = tracker.capture_signals(...)
        trade_id = tracker.record_entry(symbol, direction, price, snapshot)

        # On trade close
        tracker.record_exit(trade_id, exit_price, exit_reason, pnl_usd)
    """

    DEFAULT_DATA_PATH = "hyperliquid_monster_attribution.json"

    def __init__(
        self,
        data_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        auto_save: bool = True,
    ):
        """
        Initialize Attribution Tracker.

        Args:
            data_path: Path to save attribution data
            logger: Logger instance
            auto_save: Automatically save after each trade
        """
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self.logger = logger or logging.getLogger(__name__)
        self.auto_save = auto_save

        # Storage
        self.trades: Dict[str, TradeAttribution] = {}
        self.completed_trades: List[TradeAttribution] = []

        # Load existing data
        self._load_data()

    def _load_data(self):
        """Load existing attribution data."""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r') as f:
                    data = json.load(f)

                    for trade_data in data.get("completed_trades", []):
                        try:
                            trade = TradeAttribution.from_dict(trade_data)
                            self.completed_trades.append(trade)
                        except Exception as e:
                            self.logger.debug(f"Failed to load trade: {e}")

                self.logger.info(
                    f"ATTRIBUTION: Loaded {len(self.completed_trades)} historical trades"
                )
        except Exception as e:
            self.logger.warning(f"ATTRIBUTION: Could not load data - {e}")

    def _save_data(self):
        """Save attribution data to disk."""
        try:
            data = {
                "completed_trades": [t.to_dict() for t in self.completed_trades],
                "pending_trades": [t.to_dict() for t in self.trades.values()],
                "total_trades": len(self.completed_trades),
            }
            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"ATTRIBUTION: Could not save data - {e}")

    def capture_signals(
        self,
        # Core indicators
        rsi: Optional[float] = None,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        trend_info: Optional[Any] = None,
        macd_info: Optional[Any] = None,
        volume_info: Optional[Any] = None,
        # Leading indicators
        oi_analysis: Optional[Any] = None,
        holy_grail: Optional[Any] = None,
        # Coinglass
        cvd_divergence: Optional[Any] = None,
        liq_heatmap: Optional[Any] = None,
        # MTF
        mtf_confluence: Optional[Any] = None,
        # ML
        ml_prediction: Optional[Any] = None,
    ) -> SignalSnapshot:
        """
        Capture current signal states as a snapshot.

        Args:
            All indicator objects from the strategy

        Returns:
            SignalSnapshot with active signals recorded
        """
        active_signals: List[SignalType] = []
        signal_values: Dict[str, float] = {}

        # RSI signals
        if rsi is not None:
            signal_values["rsi"] = rsi
            if rsi <= rsi_oversold:
                active_signals.append(SignalType.RSI_OVERSOLD)
            elif rsi >= rsi_overbought:
                active_signals.append(SignalType.RSI_OVERBOUGHT)

        # Trend signals
        trend_strength = None
        if trend_info:
            trend_strength = getattr(trend_info, 'strength', None)
            if trend_strength:
                signal_values["trend_strength"] = trend_strength
            direction = getattr(trend_info, 'direction', '')
            if 'uptrend' in str(direction).lower():
                active_signals.append(SignalType.TREND_BULLISH)
            elif 'downtrend' in str(direction).lower():
                active_signals.append(SignalType.TREND_BEARISH)

        # MACD signals
        macd_histogram = None
        if macd_info:
            macd_histogram = getattr(macd_info, 'histogram', None)
            if macd_histogram is not None:
                signal_values["macd_histogram"] = macd_histogram
                if macd_histogram > 0:
                    active_signals.append(SignalType.MACD_BULLISH)
                elif macd_histogram < 0:
                    active_signals.append(SignalType.MACD_BEARISH)

        # Volume signals
        volume_ratio = None
        if volume_info:
            volume_ratio = getattr(volume_info, 'volume_ratio', None)
            if volume_ratio is not None:
                signal_values["volume_ratio"] = volume_ratio
                if volume_ratio >= 1.5:
                    active_signals.append(SignalType.VOLUME_SURGE)

        # OI signals
        oi_change_pct = None
        if oi_analysis:
            oi_change_pct = getattr(oi_analysis, 'oi_change_pct', None)
            if oi_change_pct is not None:
                signal_values["oi_change_pct"] = oi_change_pct

            momentum = getattr(oi_analysis, 'momentum', None)
            if momentum:
                momentum_str = str(momentum.value) if hasattr(momentum, 'value') else str(momentum)
                if 'bullish' in momentum_str.lower():
                    active_signals.append(SignalType.OI_BULLISH)
                elif 'bearish' in momentum_str.lower():
                    active_signals.append(SignalType.OI_BEARISH)
                elif 'exhaustion' in momentum_str.lower():
                    active_signals.append(SignalType.OI_EXHAUSTION)

        # Holy Grail signals
        hg_confidence = None
        hg_direction = None
        premium_pct = None
        funding_rate = None
        if holy_grail:
            hg_confidence = getattr(holy_grail, 'confidence', None)
            hg_direction = getattr(holy_grail, 'direction', None)
            premium_pct = getattr(holy_grail, 'premium_score', None)
            funding_rate = getattr(holy_grail, 'funding_velocity_score', None)

            if hg_confidence:
                signal_values["holy_grail_confidence"] = hg_confidence
            if premium_pct:
                signal_values["premium_pct"] = premium_pct
            if funding_rate:
                signal_values["funding_rate"] = funding_rate

            if hg_direction:
                if 'strong_long' in str(hg_direction):
                    active_signals.append(SignalType.HOLY_GRAIL_LONG)
                    active_signals.append(SignalType.HOLY_GRAIL_STRONG)
                elif 'long' in str(hg_direction):
                    active_signals.append(SignalType.HOLY_GRAIL_LONG)
                elif 'strong_short' in str(hg_direction):
                    active_signals.append(SignalType.HOLY_GRAIL_SHORT)
                    active_signals.append(SignalType.HOLY_GRAIL_STRONG)
                elif 'short' in str(hg_direction):
                    active_signals.append(SignalType.HOLY_GRAIL_SHORT)

        # CVD signals
        cvd_dir = None
        cvd_strength = None
        if cvd_divergence:
            div_type = getattr(cvd_divergence, 'divergence_type', '')
            cvd_strength = getattr(cvd_divergence, 'signal_strength', None)

            if cvd_strength:
                signal_values["cvd_divergence_strength"] = cvd_strength

            if 'bullish' in str(div_type).lower():
                cvd_dir = "bullish"
                active_signals.append(SignalType.CVD_BULLISH)
            elif 'bearish' in str(div_type).lower():
                cvd_dir = "bearish"
                active_signals.append(SignalType.CVD_BEARISH)

            if cvd_strength and cvd_strength >= 70:
                active_signals.append(SignalType.CVD_DIVERGENCE)

        # Liquidation magnet signals
        liq_imbalance = None
        if liq_heatmap:
            long_total = sum(c.estimated_size_usd for c in getattr(liq_heatmap, 'long_clusters', [])[:3])
            short_total = sum(c.estimated_size_usd for c in getattr(liq_heatmap, 'short_clusters', [])[:3])
            total = long_total + short_total

            if total > 0:
                liq_imbalance = (short_total - long_total) / total
                signal_values["liq_imbalance"] = liq_imbalance

                if liq_imbalance > 0.3:
                    active_signals.append(SignalType.LIQ_MAGNET_UP)
                elif liq_imbalance < -0.3:
                    active_signals.append(SignalType.LIQ_MAGNET_DOWN)

        # MTF signals
        mtf_score = None
        mtf_count = None
        if mtf_confluence:
            mtf_score = getattr(mtf_confluence, 'weighted_score', None)
            mtf_count = getattr(mtf_confluence, 'bullish_count', 0) + getattr(mtf_confluence, 'bearish_count', 0)

            if mtf_score is not None:
                signal_values["mtf_weighted_score"] = mtf_score

            if mtf_score and mtf_score > 0.3:
                active_signals.append(SignalType.MTF_CONFLUENCE_BULLISH)
            elif mtf_score and mtf_score < -0.3:
                active_signals.append(SignalType.MTF_CONFLUENCE_BEARISH)

            signals = getattr(mtf_confluence, 'signals', {})
            if '1h' in signals:
                trend_val = signals['1h'].trend.value if hasattr(signals['1h'].trend, 'value') else str(signals['1h'].trend)
                if 'bull' in trend_val.lower():
                    active_signals.append(SignalType.MTF_1H_BULLISH)
                elif 'bear' in trend_val.lower():
                    active_signals.append(SignalType.MTF_1H_BEARISH)

        # ML signals
        ml_confidence = None
        ml_win_prob = None
        if ml_prediction:
            ml_confidence = getattr(ml_prediction, 'confidence', None)
            ml_win_prob = getattr(ml_prediction, 'win_probability', None)

            if ml_confidence is not None:
                signal_values["ml_confidence"] = ml_confidence
                if ml_confidence >= 70:
                    active_signals.append(SignalType.ML_HIGH_CONFIDENCE)
                elif ml_confidence < 50:
                    active_signals.append(SignalType.ML_LOW_CONFIDENCE)

        return SignalSnapshot(
            timestamp=datetime.utcnow(),
            active_signals=active_signals,
            signal_values=signal_values,
            rsi=rsi,
            trend_strength=trend_strength,
            macd_histogram=macd_histogram,
            volume_ratio=volume_ratio,
            oi_change_pct=oi_change_pct,
            premium_pct=premium_pct,
            funding_rate=funding_rate,
            holy_grail_confidence=hg_confidence,
            holy_grail_direction=hg_direction,
            cvd_direction=cvd_dir,
            cvd_divergence_strength=cvd_strength,
            liq_imbalance=liq_imbalance,
            mtf_weighted_score=mtf_score,
            mtf_confluence_count=mtf_count,
            ml_confidence=ml_confidence,
            ml_win_probability=ml_win_prob,
        )

    def record_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        signals: SignalSnapshot,
        leverage: Optional[float] = None,
        position_size_usd: Optional[float] = None,
    ) -> str:
        """
        Record a new trade entry.

        Args:
            symbol: Trading pair
            direction: "long" or "short"
            entry_price: Entry price
            signals: Signal snapshot at entry
            leverage: Position leverage
            position_size_usd: Position size in USD

        Returns:
            Trade ID for tracking
        """
        trade_id = str(uuid.uuid4())[:8]

        trade = TradeAttribution(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_time=datetime.utcnow(),
            entry_price=entry_price,
            entry_signals=signals,
            leverage=leverage,
            position_size_usd=position_size_usd,
        )

        self.trades[trade_id] = trade

        self.logger.info(
            f"ATTRIBUTION: Recorded entry {trade_id} - {symbol} {direction} "
            f"@ ${entry_price:.2f} with {len(signals.active_signals)} active signals"
        )

        return trade_id

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        pnl_usd: Optional[float] = None,
    ) -> Optional[TradeAttribution]:
        """
        Record trade exit and complete attribution.

        Args:
            trade_id: Trade ID from entry
            exit_price: Exit price
            exit_reason: Why trade was closed
            pnl_usd: Actual PnL in USD

        Returns:
            Completed TradeAttribution or None if not found
        """
        if trade_id not in self.trades:
            self.logger.debug(f"ATTRIBUTION: Trade {trade_id} not found")
            return None

        trade = self.trades.pop(trade_id)
        trade.complete_exit(
            exit_time=datetime.utcnow(),
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_usd=pnl_usd,
        )

        self.completed_trades.append(trade)

        self.logger.info(
            f"ATTRIBUTION: Completed {trade_id} - {'WIN' if trade.is_win else 'LOSS'} "
            f"{trade.pnl_pct:+.2f}% ({exit_reason})"
        )

        if self.auto_save:
            self._save_data()

        return trade

    def get_pending_trades(self) -> List[TradeAttribution]:
        """Get all pending (open) trades."""
        return list(self.trades.values())

    def get_completed_trades(
        self,
        limit: Optional[int] = None,
        symbol: Optional[str] = None,
    ) -> List[TradeAttribution]:
        """Get completed trades with optional filtering."""
        trades = self.completed_trades

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]

        if limit:
            trades = trades[-limit:]

        return trades

    def get_status(self) -> dict:
        """Get tracker status for display."""
        completed = self.completed_trades
        wins = [t for t in completed if t.is_win]
        losses = [t for t in completed if not t.is_win]

        total_pnl_pct = sum(t.pnl_pct or 0 for t in completed)
        total_pnl_usd = sum(t.pnl_usd or 0 for t in completed if t.pnl_usd)

        return {
            "total_trades": len(completed),
            "pending_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(1, len(completed)) * 100,
            "total_pnl_pct": total_pnl_pct,
            "total_pnl_usd": total_pnl_usd,
            "data_path": self.data_path,
        }
