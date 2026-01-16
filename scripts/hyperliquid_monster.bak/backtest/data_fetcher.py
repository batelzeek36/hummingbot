"""
Historical Data Fetcher for Hyperliquid.

Fetches OHLCV, OI, Funding Rate, and other data from Hyperliquid API.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import aiohttp

from .models import Candle, HistoricalDataPoint


class DataCache:
    """Simple file-based cache for historical data."""

    def __init__(self, cache_dir: str = ".backtest_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, symbol: str, data_type: str, date: datetime) -> str:
        """Get cache file path for a specific day."""
        date_str = date.strftime("%Y%m%d")
        return os.path.join(self.cache_dir, f"{symbol}_{data_type}_{date_str}.json")

    def get(
        self,
        symbol: str,
        data_type: str,
        date: datetime,
    ) -> Optional[List[Dict]]:
        """Get cached data if available."""
        path = self._get_cache_path(symbol, data_type, date)
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def set(
        self,
        symbol: str,
        data_type: str,
        date: datetime,
        data: List[Dict],
    ):
        """Cache data for a specific day."""
        path = self._get_cache_path(symbol, data_type, date)
        with open(path, 'w') as f:
            json.dump(data, f)

    def clear(self, symbol: Optional[str] = None):
        """Clear cache, optionally for specific symbol."""
        for filename in os.listdir(self.cache_dir):
            if symbol is None or filename.startswith(symbol):
                os.remove(os.path.join(self.cache_dir, filename))


class HyperliquidHistoricalData:
    """
    Fetches historical data from Hyperliquid API.

    Hyperliquid API endpoints:
    - /info: Market info, funding rates
    - /candles: Historical OHLCV data

    Note: Hyperliquid uses millisecond timestamps
    """

    BASE_URL = "https://api.hyperliquid.xyz"
    INFO_URL = f"{BASE_URL}/info"

    # Map common symbols to Hyperliquid format
    SYMBOL_MAP = {
        "BTC-USD": "BTC",
        "ETH-USD": "ETH",
        "SOL-USD": "SOL",
        "DOGE-USD": "DOGE",
        "WIF-USD": "WIF",
        "HYPE-USD": "HYPE",
    }

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        cache: Optional[DataCache] = None,
        rate_limit_delay: float = 0.2,  # 200ms between requests
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.cache = cache or DataCache()
        self.rate_limit_delay = rate_limit_delay
        self._last_request = 0.0

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to Hyperliquid format."""
        return self.SYMBOL_MAP.get(symbol, symbol.replace("-USD", "").replace("-PERP", ""))

    async def _rate_limit(self):
        """Apply rate limiting."""
        import time
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_request = time.time()

    async def _post_request(
        self,
        endpoint: str,
        data: Dict,
    ) -> Optional[Dict]:
        """Make POST request to Hyperliquid API."""
        await self._rate_limit()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.warning(f"API returned {response.status}: {await response.text()}")
                        return None
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return None

    async def fetch_candles(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
    ) -> List[Candle]:
        """
        Fetch historical OHLCV candles.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            start_time: Start of range
            end_time: End of range
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)

        Returns:
            List of Candle objects
        """
        hl_symbol = self._normalize_symbol(symbol)

        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        # Hyperliquid candle request
        data = {
            "type": "candleSnapshot",
            "req": {
                "coin": hl_symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            }
        }

        self.logger.info(f"BACKTEST: Fetching {hl_symbol} candles from {start_time} to {end_time}")

        result = await self._post_request(self.INFO_URL, data)
        if not result:
            return []

        candles = []
        for item in result:
            try:
                # Hyperliquid format: {"t": timestamp_ms, "T": close_time, "s": symbol,
                #                      "i": interval, "o": open, "c": close, "h": high,
                #                      "l": low, "v": volume, "n": trades}
                candle = Candle(
                    timestamp=datetime.fromtimestamp(item["t"] / 1000),
                    open=float(item["o"]),
                    high=float(item["h"]),
                    low=float(item["l"]),
                    close=float(item["c"]),
                    volume=float(item["v"]),
                    trades=item.get("n"),
                )
                candles.append(candle)
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Failed to parse candle: {e}")

        self.logger.info(f"BACKTEST: Fetched {len(candles)} candles")
        return candles

    async def fetch_funding_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Tuple[datetime, float]]:
        """
        Fetch historical funding rates.

        Args:
            symbol: Trading pair
            start_time: Start of range
            end_time: End of range

        Returns:
            List of (timestamp, funding_rate) tuples
        """
        hl_symbol = self._normalize_symbol(symbol)

        # Convert to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)

        data = {
            "type": "fundingHistory",
            "coin": hl_symbol,
            "startTime": start_ms,
            "endTime": end_ms,
        }

        result = await self._post_request(self.INFO_URL, data)
        if not result:
            return []

        funding_rates = []
        for item in result:
            try:
                ts = datetime.fromtimestamp(item["time"] / 1000)
                rate = float(item["fundingRate"])
                funding_rates.append((ts, rate))
            except (KeyError, ValueError) as e:
                self.logger.debug(f"Failed to parse funding: {e}")

        return funding_rates

    async def fetch_current_meta(self, symbol: str) -> Optional[Dict]:
        """
        Fetch current market metadata (for reference).

        Returns mark price, index price, OI, funding rate.
        """
        hl_symbol = self._normalize_symbol(symbol)

        data = {"type": "meta"}
        result = await self._post_request(self.INFO_URL, data)

        if not result or "universe" not in result:
            return None

        for asset in result["universe"]:
            if asset.get("name") == hl_symbol:
                return asset

        return None

    async def fetch_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        use_cache: bool = True,
    ) -> List[HistoricalDataPoint]:
        """
        Fetch complete historical data including OHLCV and available leading indicators.

        This is the main method for backtest data preparation.

        Args:
            symbol: Trading pair
            start_time: Start of range
            end_time: End of range
            interval: Candle interval
            use_cache: Whether to use/save cache

        Returns:
            List of HistoricalDataPoint with all available data
        """
        # Try cache first
        if use_cache:
            cache_key = f"{symbol}_{interval}_{start_time.date()}_{end_time.date()}"
            # Implementation note: could add cache logic here

        # Fetch candles
        candles = await self.fetch_candles(symbol, start_time, end_time, interval)

        # Fetch funding rates
        funding_history = await self.fetch_funding_history(symbol, start_time, end_time)
        funding_dict = {ts: rate for ts, rate in funding_history}

        # Build data points
        data_points = []

        prev_oi = None
        for candle in candles:
            # Find nearest funding rate
            funding = None
            for ts, rate in funding_dict.items():
                if abs((ts - candle.timestamp).total_seconds()) < 3600:  # Within 1 hour
                    funding = rate
                    break

            # Calculate premium (simplified - would need index price)
            # For backtesting, we'll estimate based on funding
            premium_pct = None
            if funding:
                # Rough estimate: premium ~ funding * 24 (annualized) / 365
                premium_pct = funding * 100  # Just use funding as proxy

            dp = HistoricalDataPoint(
                timestamp=candle.timestamp,
                candle=candle,
                funding_rate=funding,
                premium_pct=premium_pct,
            )
            data_points.append(dp)

        self.logger.info(
            f"BACKTEST: Prepared {len(data_points)} data points for {symbol} "
            f"({start_time.date()} to {end_time.date()})"
        )

        return data_points

    async def fetch_batch_candles(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
        batch_size_days: int = 7,
    ) -> List[Candle]:
        """
        Fetch candles in batches to handle large date ranges.

        Args:
            symbol: Trading pair
            start_time: Start of range
            end_time: End of range
            interval: Candle interval
            batch_size_days: Days per batch

        Returns:
            Combined list of candles
        """
        all_candles = []

        current_start = start_time
        while current_start < end_time:
            batch_end = min(current_start + timedelta(days=batch_size_days), end_time)

            candles = await self.fetch_candles(symbol, current_start, batch_end, interval)
            all_candles.extend(candles)

            current_start = batch_end
            await asyncio.sleep(0.5)  # Extra delay between batches

        # Sort by timestamp and remove duplicates
        all_candles.sort(key=lambda c: c.timestamp)

        # Remove duplicates
        seen = set()
        unique_candles = []
        for candle in all_candles:
            key = candle.timestamp.isoformat()
            if key not in seen:
                seen.add(key)
                unique_candles.append(candle)

        return unique_candles

    def get_status(self) -> Dict:
        """Get data fetcher status."""
        return {
            "base_url": self.BASE_URL,
            "rate_limit_delay": self.rate_limit_delay,
            "cache_enabled": self.cache is not None,
        }
