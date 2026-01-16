"""
Backtest Engine - Core strategy replay engine.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import logging
import time
from collections import deque
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from .models import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    Candle,
    EquityCurvePoint,
    HistoricalDataPoint,
    TradeDirection,
    TradeExitReason,
)
from .data_fetcher import HyperliquidHistoricalData
from .metrics import BacktestMetrics


class BacktestEngine:
    """
    Backtest engine for strategy replay.

    Usage:
        engine = BacktestEngine(config)
        await engine.load_data()
        result = engine.run()
        print(result.get_summary())
    """

    def __init__(
        self,
        config: BacktestConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Data
        self.data: List[HistoricalDataPoint] = []
        self.data_fetcher = HyperliquidHistoricalData(logger=self.logger)

        # State
        self.equity = config.initial_capital
        self.peak_equity = config.initial_capital
        self.current_position: Optional[BacktestTrade] = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[EquityCurvePoint] = []

        # Indicators (simple moving window)
        self.price_history: deque = deque(maxlen=200)
        self.volume_history: deque = deque(maxlen=20)

        # Trade counter
        self._trade_id = 0

    async def load_data(
        self,
        data: Optional[List[HistoricalDataPoint]] = None,
    ):
        """
        Load historical data for backtesting.

        Args:
            data: Pre-loaded data, or None to fetch from API
        """
        if data:
            self.data = data
        else:
            self.data = await self.data_fetcher.fetch_historical_data(
                symbol=self.config.symbol,
                start_time=self.config.start_date,
                end_time=self.config.end_date,
                interval=self.config.timeframe,
            )

        self.logger.info(f"BACKTEST: Loaded {len(self.data)} data points")

    def _calculate_rsi(self, period: int = 14) -> Optional[float]:
        """Calculate RSI from price history."""
        if len(self.price_history) < period + 1:
            return None

        prices = list(self.price_history)[-period - 1:]
        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1] if prices else 0

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period

        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_macd(self) -> Optional[Tuple[float, float, float]]:
        """Calculate MACD (line, signal, histogram)."""
        if len(self.price_history) < 26:
            return None

        prices = list(self.price_history)

        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        macd_line = ema12 - ema26

        # Signal line (9-period EMA of MACD) - simplified
        signal = macd_line * 0.9  # Approximation

        histogram = macd_line - signal

        return macd_line, signal, histogram

    def _check_entry_signal(
        self,
        dp: HistoricalDataPoint,
    ) -> Optional[Tuple[TradeDirection, str, List[str]]]:
        """
        Check if current data point triggers an entry signal.

        Returns:
            Tuple of (direction, reason, signals) or None
        """
        rsi = self._calculate_rsi()
        if rsi is None:
            return None

        macd_result = self._calculate_macd()

        signals = []
        direction = None
        reason = ""

        # RSI signals
        if rsi <= self.config.rsi_oversold:
            signals.append("rsi_oversold")
            direction = TradeDirection.LONG
            reason = f"RSI={rsi:.1f}"
        elif rsi >= self.config.rsi_overbought:
            signals.append("rsi_overbought")
            direction = TradeDirection.SHORT
            reason = f"RSI={rsi:.1f}"
        else:
            return None  # RSI must trigger

        # MACD confirmation
        if macd_result:
            _, _, histogram = macd_result
            if direction == TradeDirection.LONG and histogram > 0:
                signals.append("macd_bullish")
            elif direction == TradeDirection.SHORT and histogram < 0:
                signals.append("macd_bearish")

        # Volume confirmation
        if len(self.volume_history) >= 5:
            avg_vol = sum(self.volume_history) / len(self.volume_history)
            current_vol = dp.candle.volume
            if current_vol > avg_vol * 1.5:
                signals.append("volume_surge")

        # Funding rate sentiment
        if dp.funding_rate is not None:
            if direction == TradeDirection.LONG and dp.funding_rate < 0:
                signals.append("funding_bullish")
            elif direction == TradeDirection.SHORT and dp.funding_rate > 0:
                signals.append("funding_bearish")

        # Check minimum signals
        if len(signals) >= self.config.min_signals:
            return direction, reason, signals

        return None

    def _check_exit_signal(
        self,
        trade: BacktestTrade,
        dp: HistoricalDataPoint,
    ) -> Optional[Tuple[TradeExitReason, float]]:
        """
        Check if current position should be exited.

        Returns:
            Tuple of (exit_reason, exit_price) or None
        """
        current_price = dp.candle.close

        # Calculate current PnL
        if trade.direction == TradeDirection.LONG:
            current_pnl_pct = (current_price - trade.entry_price) / trade.entry_price * 100
        else:
            current_pnl_pct = (trade.entry_price - current_price) / trade.entry_price * 100

        # Update high water mark
        if current_pnl_pct > trade.max_profit_pct:
            trade.max_profit_pct = current_pnl_pct
        if current_pnl_pct < -trade.max_drawdown_pct:
            trade.max_drawdown_pct = abs(current_pnl_pct)

        # Take profit
        if current_pnl_pct >= self.config.take_profit_pct:
            return TradeExitReason.TAKE_PROFIT, current_price

        # Stop loss
        if current_pnl_pct <= -self.config.stop_loss_pct:
            return TradeExitReason.STOP_LOSS, current_price

        # Trailing stop
        if self.config.trailing_stop_enabled and trade.max_profit_pct > self.config.trailing_stop_pct:
            trailing_stop_level = trade.max_profit_pct - self.config.trailing_stop_pct
            if current_pnl_pct <= trailing_stop_level:
                return TradeExitReason.TRAILING_STOP, current_price

        return None

    def _open_position(
        self,
        dp: HistoricalDataPoint,
        direction: TradeDirection,
        reason: str,
        signals: List[str],
    ) -> BacktestTrade:
        """Open a new position."""
        self._trade_id += 1

        position_size = self.equity * (self.config.position_size_pct / 100)
        notional = position_size * self.config.leverage

        trade = BacktestTrade(
            trade_id=self._trade_id,
            symbol=self.config.symbol,
            direction=direction,
            entry_time=dp.timestamp,
            entry_price=dp.candle.close,
            entry_reason=reason,
            position_size_usd=position_size,
            leverage=self.config.leverage,
            notional_value=notional,
            entry_signals=signals,
        )

        self.current_position = trade

        self.logger.debug(
            f"BACKTEST: Opened {direction.value} @ ${trade.entry_price:.2f} "
            f"({reason})"
        )

        return trade

    def _close_position(
        self,
        dp: HistoricalDataPoint,
        reason: TradeExitReason,
        exit_price: float,
    ):
        """Close current position."""
        if not self.current_position:
            return

        trade = self.current_position
        trade.complete_exit(
            exit_time=dp.timestamp,
            exit_price=exit_price,
            exit_reason=reason,
            fee_rate=self.config.taker_fee,
        )

        # Update equity
        self.equity += trade.pnl_usd or 0

        # Update peak
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.trades.append(trade)
        self.current_position = None

        self.logger.debug(
            f"BACKTEST: Closed {trade.direction.value} @ ${exit_price:.2f} "
            f"| PnL: {trade.pnl_pct:+.2f}% (${trade.pnl_usd:+.2f}) "
            f"| {reason.value}"
        )

    def run(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            progress_callback: Optional callback(current, total) for progress

        Returns:
            BacktestResult with all metrics
        """
        if not self.data:
            raise ValueError("No data loaded. Call load_data() first.")

        start_time = time.time()

        self.logger.info(
            f"BACKTEST: Starting {self.config.symbol} backtest "
            f"({len(self.data)} data points)"
        )

        # Reset state
        self.equity = self.config.initial_capital
        self.peak_equity = self.config.initial_capital
        self.current_position = None
        self.trades = []
        self.equity_curve = []
        self.price_history.clear()
        self.volume_history.clear()
        self._trade_id = 0

        # Run through data
        for i, dp in enumerate(self.data):
            # Update history
            self.price_history.append(dp.candle.close)
            self.volume_history.append(dp.candle.volume)

            # Calculate unrealized PnL
            unrealized_pnl = 0.0
            if self.current_position:
                pos = self.current_position
                if pos.direction == TradeDirection.LONG:
                    unrealized_pnl = pos.notional_value * (
                        (dp.candle.close - pos.entry_price) / pos.entry_price
                    )
                else:
                    unrealized_pnl = pos.notional_value * (
                        (pos.entry_price - dp.candle.close) / pos.entry_price
                    )

            # Record equity curve
            current_equity = self.equity + unrealized_pnl
            drawdown_pct = (self.peak_equity - current_equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0

            self.equity_curve.append(EquityCurvePoint(
                timestamp=dp.timestamp,
                equity=current_equity,
                unrealized_pnl=unrealized_pnl,
                drawdown_pct=max(0, drawdown_pct),
            ))

            # Check for exit first
            if self.current_position:
                exit_signal = self._check_exit_signal(self.current_position, dp)
                if exit_signal:
                    reason, exit_price = exit_signal
                    self._close_position(dp, reason, exit_price)

            # Check for entry (only if no position)
            if not self.current_position:
                entry_signal = self._check_entry_signal(dp)
                if entry_signal:
                    direction, reason, signals = entry_signal
                    self._open_position(dp, direction, reason, signals)

            # Progress callback
            if progress_callback and i % 1000 == 0:
                progress_callback(i, len(self.data))

        # Close any remaining position at end
        if self.current_position and self.data:
            last_dp = self.data[-1]
            self._close_position(
                last_dp,
                TradeExitReason.END_OF_DATA,
                last_dp.candle.close,
            )

        # Calculate metrics
        metrics = BacktestMetrics(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_capital=self.config.initial_capital,
        )

        all_metrics = metrics.get_all_metrics()
        duration = time.time() - start_time

        result = BacktestResult(
            config=self.config,
            trades=self.trades,
            total_trades=all_metrics["total_trades"],
            winning_trades=all_metrics["winning_trades"],
            losing_trades=all_metrics["losing_trades"],
            total_return_pct=all_metrics["total_return_pct"],
            total_return_usd=all_metrics["total_return_usd"],
            final_equity=self.equity,
            max_drawdown_pct=all_metrics["max_drawdown_pct"],
            max_drawdown_usd=all_metrics["max_drawdown_usd"],
            sharpe_ratio=all_metrics["sharpe_ratio"],
            sortino_ratio=all_metrics["sortino_ratio"],
            calmar_ratio=all_metrics["calmar_ratio"],
            profit_factor=all_metrics["profit_factor"],
            win_rate=all_metrics["win_rate"],
            avg_win_pct=all_metrics["avg_win_pct"],
            avg_loss_pct=all_metrics["avg_loss_pct"],
            avg_trade_pct=all_metrics["avg_trade_pct"],
            largest_win_pct=all_metrics["largest_win_pct"],
            largest_loss_pct=all_metrics["largest_loss_pct"],
            avg_hold_time_hours=all_metrics["avg_hold_time_hours"],
            equity_curve=self.equity_curve,
            backtest_duration_seconds=duration,
        )

        self.logger.info(
            f"BACKTEST: Completed in {duration:.1f}s "
            f"| {result.total_trades} trades "
            f"| {result.total_return_pct:+.2f}% return "
            f"| {result.sharpe_ratio:.2f} Sharpe"
        )

        return result

    def run_parameter_sweep(
        self,
        param_name: str,
        values: List[float],
    ) -> List[Tuple[float, BacktestResult]]:
        """
        Run multiple backtests with different parameter values.

        Args:
            param_name: Parameter to sweep (e.g., "rsi_oversold")
            values: List of values to test

        Returns:
            List of (value, result) tuples
        """
        results = []

        for value in values:
            # Update config parameter
            if hasattr(self.config, param_name):
                setattr(self.config, param_name, value)
            else:
                self.logger.warning(f"Unknown parameter: {param_name}")
                continue

            # Run backtest
            result = self.run()
            results.append((value, result))

            self.logger.info(
                f"SWEEP: {param_name}={value} -> "
                f"{result.total_return_pct:+.2f}% | "
                f"Sharpe {result.sharpe_ratio:.2f}"
            )

        return results
