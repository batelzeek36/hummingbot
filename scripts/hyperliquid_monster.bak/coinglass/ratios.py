"""
Coinglass Long/Short Ratio Module

Handles fetching and processing of long/short ratio data from Coinglass API.
"""

import time
from typing import TYPE_CHECKING, Optional

from .models import LongShortRatio

if TYPE_CHECKING:
    from .client import CoinglassClient


class RatioMixin:
    """Mixin providing long/short ratio methods for CoinglassAPI."""

    def get_long_short_ratio(
        self: "CoinglassClient",
        pair: str,
        exchange: str = "Binance"
    ) -> Optional[LongShortRatio]:
        """
        Get long/short ratio for a trading pair.

        Extreme ratios indicate crowding and potential squeezes.

        Args:
            pair: Trading pair
            exchange: Exchange name

        Returns:
            LongShortRatio or None
        """
        symbol = self._to_coinglass_symbol(pair)
        cache_key = f"ls_ratio_{symbol}_{exchange}"

        if self._is_cache_valid(cache_key) and cache_key in self._long_short_cache:
            return self._long_short_cache[cache_key]

        data = self._request(
            "/api/futures/global-long-short-account-ratio",
            params={"symbol": symbol, "interval": "h1", "limit": 1}
        )

        if not data or not isinstance(data, list) or len(data) == 0:
            return None

        item = data[0] if isinstance(data, list) else data

        long_ratio = float(item.get("longRate", item.get("longAccount", 50)))
        short_ratio = float(item.get("shortRate", item.get("shortAccount", 50)))

        # Calculate ratio and sentiment
        ls_ratio_value = _calculate_ls_ratio(long_ratio, short_ratio)
        sentiment, crowding_score, contrarian_signal = _classify_sentiment(
            long_ratio, short_ratio
        )

        ratio = LongShortRatio(
            symbol=symbol,
            exchange=exchange,
            long_ratio=long_ratio,
            short_ratio=short_ratio,
            long_short_ratio=ls_ratio_value,
            sentiment=sentiment,
            crowding_score=crowding_score,
            contrarian_signal=contrarian_signal,
        )

        self._long_short_cache[cache_key] = ratio
        self._cache_timestamps[cache_key] = time.time()

        return ratio


def _calculate_ls_ratio(long_ratio: float, short_ratio: float) -> float:
    """Calculate long/short ratio value."""
    if short_ratio > 0:
        return long_ratio / short_ratio
    return float('inf') if long_ratio > 0 else 1.0


def _classify_sentiment(
    long_ratio: float,
    short_ratio: float
) -> tuple[str, float, str]:
    """
    Classify market sentiment based on long/short ratios.

    Returns:
        Tuple of (sentiment, crowding_score, contrarian_signal)
    """
    if long_ratio > 65:
        sentiment = "crowded_long"
        crowding_score = min(100, (long_ratio - 50) * 2)
        contrarian_signal = "short"
    elif short_ratio > 65:
        sentiment = "crowded_short"
        crowding_score = min(100, (short_ratio - 50) * 2)
        contrarian_signal = "long"
    elif long_ratio > 55:
        sentiment = "leaning_long"
        crowding_score = (long_ratio - 50) * 2
        contrarian_signal = "neutral"
    elif short_ratio > 55:
        sentiment = "leaning_short"
        crowding_score = (short_ratio - 50) * 2
        contrarian_signal = "neutral"
    else:
        sentiment = "balanced"
        crowding_score = 0
        contrarian_signal = "neutral"

    return sentiment, crowding_score, contrarian_signal
