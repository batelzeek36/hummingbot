"""
Coinglass CVD (Cumulative Volume Delta) Module

Handles fetching and processing of CVD data from Coinglass API.
"""

import time
from collections import deque
from typing import TYPE_CHECKING, Optional

from .models import CVDDirection, CVDSnapshot

if TYPE_CHECKING:
    from .client import CoinglassClient


class CVDMixin:
    """Mixin providing CVD-related methods for CoinglassAPI."""

    def get_futures_cvd(self: "CoinglassClient", pair: str, interval: str = "h1") -> Optional[CVDSnapshot]:
        """
        Get aggregated futures CVD (Cumulative Volume Delta).

        CVD = Cumulative(Taker Buy Volume - Taker Sell Volume)
        Rising CVD = Buyers aggressive, Falling CVD = Sellers aggressive

        Args:
            pair: Trading pair (e.g., "BTC-USD")
            interval: Time interval (m1, m5, m15, h1, h4, h12, h24)

        Returns:
            CVDSnapshot or None
        """
        symbol = self._to_coinglass_symbol(pair)
        cache_key = f"futures_cvd_{symbol}"

        # Check cache
        if self._is_cache_valid(cache_key) and symbol in self._cvd_cache:
            if self._cvd_cache[symbol]:
                return self._cvd_cache[symbol][-1]

        data = self._request(
            "/api/futures/aggregated-taker-buy-sell-volume-history",
            params={"symbol": symbol, "interval": interval, "limit": 24}
        )

        if not data:
            return None

        # Initialize cache
        if symbol not in self._cvd_cache:
            self._cvd_cache[symbol] = deque(maxlen=100)

        # Process data
        snapshots = []
        prev_cvd = 0

        for item in data:
            buy_vol = float(item.get("buyVolume", 0))
            sell_vol = float(item.get("sellVolume", 0))
            cvd = buy_vol - sell_vol
            cvd_change = cvd - prev_cvd if prev_cvd != 0 else 0
            prev_cvd = cvd

            # Determine direction
            direction = _classify_cvd_direction(cvd_change, cvd)

            snapshot = CVDSnapshot(
                timestamp=float(item.get("timestamp", time.time())),
                symbol=symbol,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                cvd=cvd,
                cvd_change=cvd_change,
                direction=direction,
            )
            snapshots.append(snapshot)

        # Update cache
        self._cvd_cache[symbol].extend(snapshots)
        self._cache_timestamps[cache_key] = time.time()

        return snapshots[-1] if snapshots else None

    def get_spot_cvd(self: "CoinglassClient", pair: str, interval: str = "h1") -> Optional[CVDSnapshot]:
        """
        Get aggregated spot CVD.

        Spot CVD = Real money flow (no leverage)
        Used to compare against futures CVD for divergence detection.

        Args:
            pair: Trading pair
            interval: Time interval

        Returns:
            CVDSnapshot or None
        """
        symbol = self._to_coinglass_symbol(pair)
        cache_key = f"spot_cvd_{symbol}"

        if self._is_cache_valid(cache_key) and symbol in self._spot_cvd_cache:
            if self._spot_cvd_cache[symbol]:
                return self._spot_cvd_cache[symbol][-1]

        data = self._request(
            "/api/spot/aggregated-taker-buy-sell-history",
            params={"symbol": symbol, "interval": interval, "limit": 24}
        )

        if not data:
            return None

        if symbol not in self._spot_cvd_cache:
            self._spot_cvd_cache[symbol] = deque(maxlen=100)

        snapshots = []
        prev_cvd = 0

        for item in data:
            buy_vol = float(item.get("buyVolume", item.get("buy", 0)))
            sell_vol = float(item.get("sellVolume", item.get("sell", 0)))
            cvd = buy_vol - sell_vol
            cvd_change = cvd - prev_cvd if prev_cvd != 0 else 0
            prev_cvd = cvd

            direction = _classify_cvd_direction(cvd_change, cvd)

            snapshot = CVDSnapshot(
                timestamp=float(item.get("timestamp", time.time())),
                symbol=symbol,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                cvd=cvd,
                cvd_change=cvd_change,
                direction=direction,
            )
            snapshots.append(snapshot)

        self._spot_cvd_cache[symbol].extend(snapshots)
        self._cache_timestamps[cache_key] = time.time()

        return snapshots[-1] if snapshots else None


def _classify_cvd_direction(cvd_change: float, cvd: float) -> CVDDirection:
    """Classify CVD direction based on change magnitude."""
    if cvd_change > 0:
        if cvd_change < abs(cvd) * 0.1:
            return CVDDirection.BULLISH
        return CVDDirection.STRONG_BULLISH
    elif cvd_change < 0:
        if abs(cvd_change) < abs(cvd) * 0.1:
            return CVDDirection.BEARISH
        return CVDDirection.STRONG_BEARISH
    return CVDDirection.NEUTRAL
