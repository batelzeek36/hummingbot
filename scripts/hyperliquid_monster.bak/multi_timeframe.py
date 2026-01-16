"""
Multi-Timeframe Confluence Engine for Hyperliquid Monster Bot v2.10

GOD MODE Phase 3: Multi-Timeframe Signal Aggregation

Combines signals across multiple timeframes (1m, 5m, 15m, 1h) with weighted scoring.
Higher timeframes carry more weight. Only enters when 3+ timeframes agree on direction.

Key Features:
- Multi-TF trend analysis (EMA crossovers per timeframe)
- Weighted signal aggregation (1h > 15m > 5m > 1m)
- Confluence scoring (requires N timeframes to agree)
- Integration with existing Holy Grail signals

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from .indicators import TechnicalIndicators


class TimeframeTrend(Enum):
    """Trend direction for a single timeframe."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: str              # "1m", "5m", "15m", "1h"
    trend: TimeframeTrend
    ema_short: float            # Short EMA value
    ema_long: float             # Long EMA value
    ema_separation_pct: float   # % separation between EMAs
    rsi: Optional[float]        # RSI if available
    weight: float               # Weight for this timeframe
    signal_score: float         # -100 to +100 (negative = bearish, positive = bullish)


@dataclass
class MTFConfluence:
    """
    Multi-Timeframe Confluence result.

    Aggregates signals across all timeframes into a single recommendation.
    """
    timestamp: float
    symbol: str

    # Individual timeframe signals
    signals: Dict[str, TimeframeSignal]  # keyed by timeframe

    # Aggregated results
    bullish_count: int          # Number of bullish timeframes
    bearish_count: int          # Number of bearish timeframes
    neutral_count: int          # Number of neutral timeframes

    # Weighted score
    weighted_score: float       # -100 to +100

    # Confluence
    confluence_direction: str   # "bullish", "bearish", "mixed"
    confluence_strength: float  # 0-100
    has_confluence: bool        # True if min_timeframes agree

    # Final recommendation
    recommendation: str         # "LONG", "SHORT", "NEUTRAL"
    confidence: float           # 0-100
    reasoning: str

    # Warnings
    warnings: List[str] = field(default_factory=list)


class MultiTimeframeAnalyzer:
    """
    Multi-Timeframe Confluence Engine.

    Tracks price data across multiple timeframes and calculates confluence signals.
    Higher timeframes are weighted more heavily in the final decision.

    Timeframe Weights (default):
    - 1h:  40% (king timeframe - don't fight it)
    - 15m: 30%
    - 5m:  20%
    - 1m:  10% (noise filter only)
    """

    # Default timeframe weights (should sum to 1.0)
    DEFAULT_WEIGHTS = {
        "1m": 0.10,
        "5m": 0.20,
        "15m": 0.30,
        "1h": 0.40,
    }

    # Timeframe in seconds for resampling
    TIMEFRAME_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
    }

    def __init__(
        self,
        timeframe_weights: Optional[Dict[str, float]] = None,
        ema_short_period: int = 9,
        ema_long_period: int = 21,
        rsi_period: int = 14,
        min_confluence_timeframes: int = 3,
        strong_trend_threshold: float = 1.0,  # % EMA separation for "strong" trend
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Multi-Timeframe Analyzer.

        Args:
            timeframe_weights: Custom weights per timeframe (must sum to 1.0)
            ema_short_period: Short EMA period for trend detection
            ema_long_period: Long EMA period for trend detection
            rsi_period: RSI calculation period
            min_confluence_timeframes: Minimum timeframes that must agree for confluence
            strong_trend_threshold: EMA separation % to consider trend "strong"
            logger: Logger instance
        """
        self.weights = timeframe_weights or self.DEFAULT_WEIGHTS.copy()
        self.ema_short = ema_short_period
        self.ema_long = ema_long_period
        self.rsi_period = rsi_period
        self.min_confluence = min_confluence_timeframes
        self.strong_threshold = strong_trend_threshold
        self.logger = logger or logging.getLogger(__name__)

        # Validate weights sum to ~1.0
        weight_sum = sum(self.weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            self.logger.warning(f"MTF weights sum to {weight_sum}, normalizing...")
            for tf in self.weights:
                self.weights[tf] /= weight_sum

        # Price data storage per symbol
        # Raw 1-minute data that gets resampled to higher timeframes
        self._raw_prices: Dict[str, Deque[Tuple[datetime, float]]] = {}

        # Resampled OHLC data per timeframe per symbol
        self._ohlc_data: Dict[str, Dict[str, Deque[Dict]]] = {}

        # Cache for computed signals
        self._signal_cache: Dict[str, MTFConfluence] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = 10  # 10 second cache

        # Stats
        self._updates = 0

    def update(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """
        Update price data for a symbol.

        Call this frequently (every tick) with the latest price.
        The analyzer will resample to higher timeframes automatically.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            price: Current price
            timestamp: Optional timestamp (defaults to now)
        """
        ts = timestamp or datetime.now()

        # Initialize storage if needed
        if symbol not in self._raw_prices:
            self._raw_prices[symbol] = deque(maxlen=7200)  # 5 days of 1-min data
            self._ohlc_data[symbol] = {
                tf: deque(maxlen=200) for tf in self.weights.keys()
            }

        self._raw_prices[symbol].append((ts, price))
        self._updates += 1

        # Resample to all timeframes
        self._resample_to_timeframes(symbol)

    def _resample_to_timeframes(self, symbol: str):
        """Resample raw price data to OHLC for each timeframe."""
        raw = self._raw_prices.get(symbol)
        if not raw or len(raw) < 2:
            return

        for tf, seconds in self.TIMEFRAME_SECONDS.items():
            self._resample_single_timeframe(symbol, tf, seconds)

    def _resample_single_timeframe(self, symbol: str, timeframe: str, period_seconds: int):
        """Resample to a single timeframe."""
        raw = self._raw_prices[symbol]
        ohlc = self._ohlc_data[symbol][timeframe]

        # Get the latest complete period
        now = datetime.now()
        period_start = now.replace(
            second=0,
            microsecond=0
        )

        # Align to period boundary
        if timeframe == "5m":
            period_start = period_start.replace(minute=(period_start.minute // 5) * 5)
        elif timeframe == "15m":
            period_start = period_start.replace(minute=(period_start.minute // 15) * 15)
        elif timeframe == "1h":
            period_start = period_start.replace(minute=0)

        # Collect prices in current period
        period_prices = [
            p for ts, p in raw
            if period_start <= ts < period_start + timedelta(seconds=period_seconds)
        ]

        if not period_prices:
            return

        # Create OHLC candle
        candle = {
            "timestamp": period_start,
            "open": period_prices[0],
            "high": max(period_prices),
            "low": min(period_prices),
            "close": period_prices[-1],
        }

        # Update or append
        if ohlc and ohlc[-1]["timestamp"] == period_start:
            ohlc[-1] = candle  # Update current candle
        elif not ohlc or ohlc[-1]["timestamp"] < period_start:
            ohlc.append(candle)  # New candle

    def get_confluence(self, symbol: str) -> Optional[MTFConfluence]:
        """
        Get multi-timeframe confluence signal for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            MTFConfluence with aggregated signals or None if insufficient data
        """
        import time

        # Check cache
        cache_key = f"{symbol}_mtf"
        if cache_key in self._cache_timestamps:
            if time.time() - self._cache_timestamps[cache_key] < self._cache_ttl:
                return self._signal_cache.get(cache_key)

        # Calculate signals for each timeframe
        signals = {}
        for tf in self.weights.keys():
            signal = self._calculate_timeframe_signal(symbol, tf)
            if signal:
                signals[tf] = signal

        if len(signals) < 2:
            return None  # Need at least 2 timeframes

        # Aggregate signals
        confluence = self._aggregate_signals(symbol, signals)

        # Cache result
        self._signal_cache[cache_key] = confluence
        self._cache_timestamps[cache_key] = time.time()

        return confluence

    def _calculate_timeframe_signal(self, symbol: str, timeframe: str) -> Optional[TimeframeSignal]:
        """Calculate signal for a single timeframe."""
        if symbol not in self._ohlc_data:
            return None

        ohlc = self._ohlc_data[symbol].get(timeframe)
        if not ohlc or len(ohlc) < self.ema_long + 5:
            return None

        # Extract close prices
        closes = [c["close"] for c in ohlc]

        # Calculate EMAs
        ema_short = TechnicalIndicators.calculate_ema(closes, self.ema_short)
        ema_long = TechnicalIndicators.calculate_ema(closes, self.ema_long)

        if ema_short is None or ema_long is None:
            return None

        # Calculate EMA separation percentage
        if ema_long != 0:
            ema_sep_pct = ((ema_short - ema_long) / ema_long) * 100
        else:
            ema_sep_pct = 0

        # Calculate RSI
        rsi = TechnicalIndicators.calculate_rsi(closes, self.rsi_period)

        # Determine trend
        trend = self._determine_trend(ema_sep_pct, rsi)

        # Calculate signal score (-100 to +100)
        signal_score = self._calculate_signal_score(ema_sep_pct, rsi, trend)

        return TimeframeSignal(
            timeframe=timeframe,
            trend=trend,
            ema_short=ema_short,
            ema_long=ema_long,
            ema_separation_pct=ema_sep_pct,
            rsi=rsi,
            weight=self.weights[timeframe],
            signal_score=signal_score,
        )

    def _determine_trend(self, ema_sep_pct: float, rsi: Optional[float]) -> TimeframeTrend:
        """Determine trend direction from EMA separation and RSI."""
        # Strong trends
        if ema_sep_pct > self.strong_threshold:
            if rsi and rsi > 60:
                return TimeframeTrend.STRONG_BULLISH
            return TimeframeTrend.BULLISH
        elif ema_sep_pct < -self.strong_threshold:
            if rsi and rsi < 40:
                return TimeframeTrend.STRONG_BEARISH
            return TimeframeTrend.BEARISH

        # Weak trends
        if ema_sep_pct > 0.2:
            return TimeframeTrend.BULLISH
        elif ema_sep_pct < -0.2:
            return TimeframeTrend.BEARISH

        return TimeframeTrend.NEUTRAL

    def _calculate_signal_score(
        self,
        ema_sep_pct: float,
        rsi: Optional[float],
        trend: TimeframeTrend
    ) -> float:
        """Calculate signal score from -100 to +100."""
        score = 0.0

        # EMA contribution (60% weight)
        # Cap at +/- 3% separation
        ema_score = max(-100, min(100, ema_sep_pct * 33.33))
        score += ema_score * 0.6

        # RSI contribution (40% weight)
        if rsi is not None:
            # RSI 50 = neutral, 30 = bullish, 70 = bearish
            rsi_score = (50 - rsi) * 2.5  # -50 to +50 range, then scale
            score += rsi_score * 0.4

        return max(-100, min(100, score))

    def _aggregate_signals(self, symbol: str, signals: Dict[str, TimeframeSignal]) -> MTFConfluence:
        """Aggregate signals from all timeframes into a single confluence result."""
        import time

        bullish = 0
        bearish = 0
        neutral = 0
        weighted_score = 0.0
        warnings = []

        for tf, sig in signals.items():
            # Count directions
            if sig.trend in (TimeframeTrend.BULLISH, TimeframeTrend.STRONG_BULLISH):
                bullish += 1
            elif sig.trend in (TimeframeTrend.BEARISH, TimeframeTrend.STRONG_BEARISH):
                bearish += 1
            else:
                neutral += 1

            # Weighted score
            weighted_score += sig.signal_score * sig.weight

        # Normalize weighted score by actual weights used
        total_weight = sum(s.weight for s in signals.values())
        if total_weight > 0:
            weighted_score /= total_weight

        # Determine confluence direction
        total_tf = len(signals)
        if bullish >= self.min_confluence:
            confluence_dir = "bullish"
            confluence_strength = (bullish / total_tf) * 100
            has_confluence = True
        elif bearish >= self.min_confluence:
            confluence_dir = "bearish"
            confluence_strength = (bearish / total_tf) * 100
            has_confluence = True
        else:
            confluence_dir = "mixed"
            confluence_strength = max(bullish, bearish) / total_tf * 100
            has_confluence = False

        # Check for conflicting signals (warning)
        if bullish > 0 and bearish > 0:
            if "1h" in signals:
                htf = signals["1h"]
                if htf.trend in (TimeframeTrend.BULLISH, TimeframeTrend.STRONG_BULLISH) and bearish > 0:
                    warnings.append("Lower timeframes bearish but 1H bullish - be cautious on shorts")
                elif htf.trend in (TimeframeTrend.BEARISH, TimeframeTrend.STRONG_BEARISH) and bullish > 0:
                    warnings.append("Lower timeframes bullish but 1H bearish - be cautious on longs")

        # Final recommendation
        if has_confluence and confluence_dir == "bullish" and weighted_score > 20:
            recommendation = "LONG"
            confidence = min(95, 50 + weighted_score / 2)
        elif has_confluence and confluence_dir == "bearish" and weighted_score < -20:
            recommendation = "SHORT"
            confidence = min(95, 50 + abs(weighted_score) / 2)
        else:
            recommendation = "NEUTRAL"
            confidence = 50 - abs(weighted_score) / 4

        # Build reasoning
        tf_summary = ", ".join([
            f"{tf}:{sig.trend.value[:4]}" for tf, sig in signals.items()
        ])
        reasoning = (
            f"MTF: {bullish}B/{bearish}Be/{neutral}N across {total_tf} TFs. "
            f"Weighted score: {weighted_score:+.1f}. [{tf_summary}]"
        )

        return MTFConfluence(
            timestamp=time.time(),
            symbol=symbol,
            signals=signals,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            weighted_score=weighted_score,
            confluence_direction=confluence_dir,
            confluence_strength=confluence_strength,
            has_confluence=has_confluence,
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            warnings=warnings,
        )

    def should_block_entry(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Check if MTF confluence blocks an entry.

        Args:
            symbol: Trading pair
            direction: "long" or "short"

        Returns:
            Tuple of (should_block, reason)
        """
        confluence = self.get_confluence(symbol)
        if not confluence:
            return False, ""

        # Block if strong confluence in opposite direction
        if direction == "long":
            if confluence.confluence_direction == "bearish" and confluence.confluence_strength >= 75:
                return True, f"MTF BEARISH ({confluence.bearish_count} TFs, {confluence.weighted_score:+.1f})"
            if confluence.weighted_score < -50:
                return True, f"MTF weighted score strongly bearish ({confluence.weighted_score:+.1f})"

        elif direction == "short":
            if confluence.confluence_direction == "bullish" and confluence.confluence_strength >= 75:
                return True, f"MTF BULLISH ({confluence.bullish_count} TFs, {confluence.weighted_score:+.1f})"
            if confluence.weighted_score > 50:
                return True, f"MTF weighted score strongly bullish ({confluence.weighted_score:+.1f})"

        return False, ""

    def get_status(self) -> Dict:
        """Get analyzer status."""
        return {
            "updates": self._updates,
            "symbols_tracked": list(self._raw_prices.keys()),
            "timeframes": list(self.weights.keys()),
            "weights": self.weights,
            "min_confluence": self.min_confluence,
        }

    def get_data_status(self, symbol: str) -> Dict[str, int]:
        """Get data availability status for a symbol."""
        if symbol not in self._ohlc_data:
            return {}

        return {
            tf: len(self._ohlc_data[symbol][tf])
            for tf in self.weights.keys()
        }
