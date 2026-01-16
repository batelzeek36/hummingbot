"""
Coinglass Base API Client

Handles authentication, rate limiting, caching, and HTTP requests.
"""

import logging
import os
import time
from collections import deque
from typing import Deque, Dict, Optional

import requests

from .models import CVDSnapshot, LiquidationHeatmap, LongShortRatio


class CoinglassClient:
    """
    Base Coinglass API client with auth, rate limiting, and caching.

    Rate limit: 30 requests/min for Hobbyist plan.
    """

    BASE_URL = "https://open-api-v4.coinglass.com"

    # Coin symbol mapping (Coinglass uses different symbols)
    SYMBOL_MAP = {
        "BTC-USD": "BTC",
        "ETH-USD": "ETH",
        "SOL-USD": "SOL",
        "DOGE-USD": "DOGE",
        "TAO-USD": "TAO",
        "HYPE-USD": "HYPE",
        "AVNT-USD": "AVNT",
        "kPEPE-USD": "1000PEPE",
        "kBONK-USD": "1000BONK",
        "WIF-USD": "WIF",
        "VVV-USD": "VVV",
        "HYPER-USD": "HYPER",
        "IP-USD": "IP",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        request_interval: float = 2.0,  # Min seconds between requests (30/min limit)
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Coinglass API client.

        Args:
            api_key: Coinglass API key (or set COINGLASS_API_KEY env var)
            request_interval: Minimum seconds between API requests
            logger: Logger instance
        """
        self.api_key = api_key or os.getenv("COINGLASS_API_KEY")
        if not self.api_key:
            raise ValueError("Coinglass API key required. Set COINGLASS_API_KEY env var or pass api_key.")

        self.request_interval = request_interval
        self.logger = logger or logging.getLogger(__name__)

        # Rate limiting
        self._last_request_time: float = 0
        self._request_count = 0

        # Cache
        self._cvd_cache: Dict[str, Deque[CVDSnapshot]] = {}
        self._spot_cvd_cache: Dict[str, Deque[CVDSnapshot]] = {}
        self._liquidation_cache: Dict[str, LiquidationHeatmap] = {}
        self._long_short_cache: Dict[str, LongShortRatio] = {}

        # Cache TTL (seconds)
        self._cache_ttl = 60  # 1 minute cache
        self._cache_timestamps: Dict[str, float] = {}

        # Stats
        self._total_requests = 0
        self._error_count = 0

    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "accept": "application/json",
            "CG-API-KEY": self.api_key,
        }

    def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make an API request with rate limiting and error handling.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response or None on error
        """
        self._rate_limit()

        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=10
            )
            self._total_requests += 1

            if response.status_code == 200:
                data = response.json()
                if data.get("success") or data.get("code") == "0":
                    return data.get("data", data)
                else:
                    self.logger.warning(f"Coinglass API error: {data.get('msg', 'Unknown error')}")
                    return None
            else:
                self._error_count += 1
                self.logger.warning(f"Coinglass API HTTP {response.status_code}: {response.text[:200]}")
                return None

        except requests.RequestException as e:
            self._error_count += 1
            self.logger.error(f"Coinglass API request failed: {e}")
            return None

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[cache_key] < self._cache_ttl

    def _to_coinglass_symbol(self, pair: str) -> str:
        """Convert trading pair to Coinglass symbol."""
        return self.SYMBOL_MAP.get(pair, pair.replace("-USD", ""))

    def get_status(self) -> Dict:
        """Get API status."""
        return {
            "total_requests": self._total_requests,
            "error_count": self._error_count,
            "cache_size": {
                "cvd": sum(len(v) for v in self._cvd_cache.values()),
                "spot_cvd": sum(len(v) for v in self._spot_cvd_cache.values()),
                "liquidation": len(self._liquidation_cache),
                "long_short": len(self._long_short_cache),
            },
            "api_key_set": bool(self.api_key),
        }
