"""
Technical Indicators for Hyperliquid Monster Bot v2.

Reusable indicator calculations for trading strategies.
All indicators use standard financial formulas.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Tuple


@dataclass
class MACDResult:
    """MACD calculation result."""
    macd_line: float
    signal_line: float
    histogram: float


@dataclass
class TrendInfo:
    """Trend analysis result."""
    direction: str  # "uptrend", "downtrend", "neutral"
    ema_50: float
    ema_200: float
    strength: float  # How far apart the EMAs are (percentage)


@dataclass
class VolumeAnalysis:
    """Volume analysis result."""
    current_volume: float
    average_volume: float
    volume_ratio: float  # current / average
    is_high_volume: bool


class TechnicalIndicators:
    """
    Technical indicator calculations.

    All methods are static and work with price/volume lists.
    """

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        """
        Calculate Exponential Moving Average.

        Args:
            prices: List of prices (oldest first)
            period: EMA period

        Returns:
            EMA value or None if insufficient data
        """
        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)

        # Start with SMA for the first EMA value
        sma = sum(prices[:period]) / period
        ema = sma

        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        """
        Calculate Simple Moving Average.

        Args:
            prices: List of prices (oldest first)
            period: SMA period

        Returns:
            SMA value or None if insufficient data
        """
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index.

        Args:
            prices: List of prices (oldest first)
            period: RSI period (default 14)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(prices) < period + 1:
            return None

        changes = []
        for i in range(1, len(prices)):
            changes.append(prices[i] - prices[i-1])

        if not changes:
            return None

        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]

        avg_gain = sum(gains) / len(changes) if gains else 0
        avg_loss = sum(losses) / len(changes) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(
        prices: List[float],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Optional[MACDResult]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: List of prices (oldest first)
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)

        Returns:
            MACDResult or None if insufficient data
        """
        if len(prices) < slow_period + signal_period:
            return None

        # Calculate MACD line (fast EMA - slow EMA)
        fast_ema = TechnicalIndicators.calculate_ema(prices, fast_period)
        slow_ema = TechnicalIndicators.calculate_ema(prices, slow_period)

        if fast_ema is None or slow_ema is None:
            return None

        macd_line = fast_ema - slow_ema

        # Calculate MACD history for signal line
        macd_history = []
        for i in range(slow_period, len(prices) + 1):
            subset = prices[:i]
            fast = TechnicalIndicators.calculate_ema(subset, fast_period)
            slow = TechnicalIndicators.calculate_ema(subset, slow_period)
            if fast is not None and slow is not None:
                macd_history.append(fast - slow)

        if len(macd_history) < signal_period:
            return None

        # Signal line is EMA of MACD line
        signal_line = TechnicalIndicators.calculate_ema(macd_history, signal_period)

        if signal_line is None:
            return None

        histogram = macd_line - signal_line

        return MACDResult(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram
        )

    @staticmethod
    def analyze_trend(
        prices: List[float],
        short_period: int = 50,
        long_period: int = 200
    ) -> Optional[TrendInfo]:
        """
        Analyze trend using EMA crossover.

        Args:
            prices: List of prices (oldest first)
            short_period: Short-term EMA period (default 50)
            long_period: Long-term EMA period (default 200)

        Returns:
            TrendInfo or None if insufficient data
        """
        if len(prices) < long_period:
            return None

        ema_short = TechnicalIndicators.calculate_ema(prices, short_period)
        ema_long = TechnicalIndicators.calculate_ema(prices, long_period)

        if ema_short is None or ema_long is None:
            return None

        # Calculate trend strength as percentage difference
        strength = abs(ema_short - ema_long) / ema_long * 100

        # Determine trend direction
        if ema_short > ema_long * 1.001:  # 0.1% buffer to avoid noise
            direction = "uptrend"
        elif ema_short < ema_long * 0.999:
            direction = "downtrend"
        else:
            direction = "neutral"

        return TrendInfo(
            direction=direction,
            ema_50=ema_short,
            ema_200=ema_long,
            strength=strength
        )

    @staticmethod
    def analyze_volume(
        volumes: List[float],
        period: int = 20,
        high_volume_threshold: float = 1.5
    ) -> Optional[VolumeAnalysis]:
        """
        Analyze volume relative to average.

        Args:
            volumes: List of volume values (oldest first)
            period: Period for average volume calculation
            high_volume_threshold: Ratio above which volume is considered high

        Returns:
            VolumeAnalysis or None if insufficient data
        """
        if len(volumes) < period:
            return None

        current_volume = volumes[-1]
        average_volume = sum(volumes[-period:]) / period

        if average_volume == 0:
            return None

        volume_ratio = current_volume / average_volume
        is_high_volume = volume_ratio >= high_volume_threshold

        return VolumeAnalysis(
            current_volume=current_volume,
            average_volume=average_volume,
            volume_ratio=volume_ratio,
            is_high_volume=is_high_volume
        )

    @staticmethod
    def get_signal_confluence(
        rsi_signal: bool,
        macd_signal: bool,
        trend_signal: bool,
        volume_signal: bool,
        funding_signal: bool,
        min_signals: int = 2
    ) -> Tuple[bool, int, List[str]]:
        """
        Check if enough signals agree for a trade.

        Args:
            rsi_signal: RSI indicates entry
            macd_signal: MACD confirms direction
            trend_signal: Trend aligns with direction
            volume_signal: Volume is adequate
            funding_signal: Funding rate sentiment agrees
            min_signals: Minimum signals required

        Returns:
            Tuple of (should_trade, signal_count, active_signals)
        """
        signals = {
            "RSI": rsi_signal,
            "MACD": macd_signal,
            "Trend": trend_signal,
            "Volume": volume_signal,
            "Funding": funding_signal,
        }

        active = [name for name, active in signals.items() if active]
        count = len(active)

        return (count >= min_signals, count, active)
