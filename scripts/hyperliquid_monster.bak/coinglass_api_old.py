"""
Coinglass API Integration for Hyperliquid Monster Bot v2.8+

GOD MODE Phase 2: Real aggregated market data from Coinglass API.

Provides:
- CVD (Cumulative Volume Delta) - Aggregated across exchanges
- Spot vs Perp CVD Divergence - The killer reversal signal
- Liquidation Heatmap - Know where liquidation clusters sit
- Long/Short Ratios - Market positioning data
- Aggregated Open Interest - Cross-exchange OI

API: https://open-api-v4.coinglass.com
Docs: https://docs.coinglass.com

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import requests


class CVDDirection(Enum):
    """CVD trend direction."""
    STRONG_BULLISH = "strong_bullish"    # CVD rising strongly
    BULLISH = "bullish"                   # CVD rising
    NEUTRAL = "neutral"                   # CVD flat
    BEARISH = "bearish"                   # CVD falling
    STRONG_BEARISH = "strong_bearish"    # CVD falling strongly


@dataclass
class CVDSnapshot:
    """Snapshot of CVD data."""
    timestamp: float
    symbol: str
    buy_volume: float          # Taker buy volume
    sell_volume: float         # Taker sell volume
    cvd: float                 # Cumulative delta (buy - sell)
    cvd_change: float          # Change from previous snapshot
    direction: CVDDirection


@dataclass
class SpotPerpDivergence:
    """
    Spot vs Perpetual CVD divergence - THE killer signal.

    When perp CVD diverges from spot CVD, leverage traders are getting trapped.
    """
    symbol: str
    spot_cvd: float
    perp_cvd: float
    spot_direction: CVDDirection
    perp_direction: CVDDirection
    divergence_type: str       # "bullish_divergence", "bearish_divergence", "aligned"
    signal_strength: float     # 0-100
    interpretation: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class LiquidationCluster:
    """A cluster of liquidations at a price level."""
    price: float
    side: str                  # "long" or "short"
    estimated_size_usd: float
    distance_pct: float        # Distance from current price
    risk_level: str            # "low", "medium", "high"


@dataclass
class LiquidationHeatmap:
    """Liquidation heatmap data for a symbol."""
    symbol: str
    current_price: float
    long_clusters: List[LiquidationCluster]    # Liquidations below price
    short_clusters: List[LiquidationCluster]   # Liquidations above price
    nearest_long_liq: Optional[float]          # Nearest long liquidation price
    nearest_short_liq: Optional[float]         # Nearest short liquidation price
    magnetic_direction: str                     # "up" (toward short liqs) or "down" (toward long liqs)
    interpretation: str


@dataclass
class LongShortRatio:
    """Long/Short ratio data."""
    symbol: str
    exchange: str
    long_ratio: float          # Percentage of longs (0-100)
    short_ratio: float         # Percentage of shorts (0-100)
    long_short_ratio: float    # Ratio (>1 = more longs)
    sentiment: str             # "crowded_long", "crowded_short", "balanced"
    crowding_score: float      # 0-100, how crowded
    contrarian_signal: str     # "long", "short", "neutral"


class CoinglassAPI:
    """
    Coinglass API client for aggregated market data.

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

    # =========================================================================
    # CVD (Cumulative Volume Delta)
    # =========================================================================

    def get_futures_cvd(self, pair: str, interval: str = "h1") -> Optional[CVDSnapshot]:
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
            if cvd_change > 0:
                direction = CVDDirection.BULLISH if cvd_change < abs(cvd) * 0.1 else CVDDirection.STRONG_BULLISH
            elif cvd_change < 0:
                direction = CVDDirection.BEARISH if abs(cvd_change) < abs(cvd) * 0.1 else CVDDirection.STRONG_BEARISH
            else:
                direction = CVDDirection.NEUTRAL

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

    def get_spot_cvd(self, pair: str, interval: str = "h1") -> Optional[CVDSnapshot]:
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

            if cvd_change > 0:
                direction = CVDDirection.BULLISH if cvd_change < abs(cvd) * 0.1 else CVDDirection.STRONG_BULLISH
            elif cvd_change < 0:
                direction = CVDDirection.BEARISH if abs(cvd_change) < abs(cvd) * 0.1 else CVDDirection.STRONG_BEARISH
            else:
                direction = CVDDirection.NEUTRAL

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

    def get_spot_perp_divergence(self, pair: str) -> Optional[SpotPerpDivergence]:
        """
        Detect Spot vs Perpetual CVD divergence - THE KILLER SIGNAL.

        When perp CVD diverges from spot CVD, leverage traders are trapped:
        - Perp CVD up + Spot CVD flat/down = Longs trapped, SHORT signal
        - Perp CVD down + Spot CVD flat/up = Shorts trapped, LONG signal

        Args:
            pair: Trading pair

        Returns:
            SpotPerpDivergence analysis or None
        """
        futures_cvd = self.get_futures_cvd(pair)
        spot_cvd = self.get_spot_cvd(pair)

        if not futures_cvd or not spot_cvd:
            return None

        symbol = self._to_coinglass_symbol(pair)
        warnings = []

        # Analyze divergence
        perp_direction = futures_cvd.direction
        spot_direction = spot_cvd.direction

        # Detect divergence type
        divergence_type = "aligned"
        signal_strength = 0.0
        interpretation = ""

        # Strong divergence: perp bullish, spot bearish/neutral
        if perp_direction in (CVDDirection.BULLISH, CVDDirection.STRONG_BULLISH):
            if spot_direction in (CVDDirection.BEARISH, CVDDirection.STRONG_BEARISH):
                divergence_type = "bearish_divergence"
                signal_strength = 90.0 if perp_direction == CVDDirection.STRONG_BULLISH else 70.0
                interpretation = (
                    "BEARISH DIVERGENCE: Perp CVD rising but spot CVD falling. "
                    "Leveraged longs are buying, but real money is selling. "
                    "LONGS ARE TRAPPED - expect reversal down."
                )
                warnings.append("HIGH CONFIDENCE SHORT SIGNAL")
            elif spot_direction == CVDDirection.NEUTRAL:
                divergence_type = "bearish_divergence"
                signal_strength = 60.0
                interpretation = (
                    "Mild bearish divergence: Perp CVD rising, spot CVD flat. "
                    "Leverage is leading without spot follow-through."
                )

        # Strong divergence: perp bearish, spot bullish/neutral
        elif perp_direction in (CVDDirection.BEARISH, CVDDirection.STRONG_BEARISH):
            if spot_direction in (CVDDirection.BULLISH, CVDDirection.STRONG_BULLISH):
                divergence_type = "bullish_divergence"
                signal_strength = 90.0 if perp_direction == CVDDirection.STRONG_BEARISH else 70.0
                interpretation = (
                    "BULLISH DIVERGENCE: Perp CVD falling but spot CVD rising. "
                    "Leveraged shorts are selling, but real money is buying. "
                    "SHORTS ARE TRAPPED - expect reversal up."
                )
                warnings.append("HIGH CONFIDENCE LONG SIGNAL")
            elif spot_direction == CVDDirection.NEUTRAL:
                divergence_type = "bullish_divergence"
                signal_strength = 60.0
                interpretation = (
                    "Mild bullish divergence: Perp CVD falling, spot CVD flat. "
                    "Leverage is leading without spot follow-through."
                )

        # Aligned - both moving same direction
        else:
            if perp_direction == spot_direction:
                interpretation = (
                    f"Aligned: Both perp and spot CVD are {perp_direction.value}. "
                    "No divergence - follow the trend."
                )
            else:
                interpretation = "Mixed signals - no clear divergence."

        return SpotPerpDivergence(
            symbol=symbol,
            spot_cvd=spot_cvd.cvd,
            perp_cvd=futures_cvd.cvd,
            spot_direction=spot_direction,
            perp_direction=perp_direction,
            divergence_type=divergence_type,
            signal_strength=signal_strength,
            interpretation=interpretation,
            warnings=warnings,
        )

    # =========================================================================
    # LIQUIDATION DATA
    # =========================================================================

    def get_liquidation_heatmap(self, pair: str) -> Optional[LiquidationHeatmap]:
        """
        Get liquidation heatmap showing where liquidation clusters sit.

        Price is "magnetically attracted" to liquidation clusters because
        whales hunt them for liquidity.

        Args:
            pair: Trading pair

        Returns:
            LiquidationHeatmap or None
        """
        symbol = self._to_coinglass_symbol(pair)
        cache_key = f"liq_heatmap_{symbol}"

        if self._is_cache_valid(cache_key) and symbol in self._liquidation_cache:
            return self._liquidation_cache[symbol]

        data = self._request(
            "/api/futures/liquidation-heatmap",
            params={"symbol": symbol, "interval": "h24"}
        )

        if not data:
            return None

        # Parse heatmap data
        current_price = float(data.get("price", 0))
        liq_data = data.get("data", [])

        long_clusters = []
        short_clusters = []

        for level in liq_data:
            price = float(level.get("price", 0))
            liq_amount = float(level.get("liqAmount", level.get("liquidation", 0)))

            if price == 0 or current_price == 0:
                continue

            distance_pct = ((price - current_price) / current_price) * 100

            # Determine risk level based on size and distance
            if liq_amount > 10_000_000:  # $10M+
                risk_level = "high"
            elif liq_amount > 1_000_000:  # $1M+
                risk_level = "medium"
            else:
                risk_level = "low"

            cluster = LiquidationCluster(
                price=price,
                side="long" if price < current_price else "short",
                estimated_size_usd=liq_amount,
                distance_pct=distance_pct,
                risk_level=risk_level,
            )

            if price < current_price:
                long_clusters.append(cluster)
            else:
                short_clusters.append(cluster)

        # Sort by distance
        long_clusters.sort(key=lambda x: abs(x.distance_pct))
        short_clusters.sort(key=lambda x: abs(x.distance_pct))

        # Find nearest clusters
        nearest_long = long_clusters[0].price if long_clusters else None
        nearest_short = short_clusters[0].price if short_clusters else None

        # Determine magnetic direction (price tends toward larger liquidation pool)
        long_total = sum(c.estimated_size_usd for c in long_clusters[:3])
        short_total = sum(c.estimated_size_usd for c in short_clusters[:3])

        if long_total > short_total * 1.5:
            magnetic_direction = "down"
            interpretation = (
                f"More liquidation liquidity below (${long_total/1e6:.1f}M long liqs). "
                "Price may be drawn down to hunt long liquidations."
            )
        elif short_total > long_total * 1.5:
            magnetic_direction = "up"
            interpretation = (
                f"More liquidation liquidity above (${short_total/1e6:.1f}M short liqs). "
                "Price may be drawn up to hunt short liquidations."
            )
        else:
            magnetic_direction = "neutral"
            interpretation = "Liquidation clusters relatively balanced above and below."

        heatmap = LiquidationHeatmap(
            symbol=symbol,
            current_price=current_price,
            long_clusters=long_clusters[:5],  # Top 5
            short_clusters=short_clusters[:5],
            nearest_long_liq=nearest_long,
            nearest_short_liq=nearest_short,
            magnetic_direction=magnetic_direction,
            interpretation=interpretation,
        )

        self._liquidation_cache[symbol] = heatmap
        self._cache_timestamps[cache_key] = time.time()

        return heatmap

    # =========================================================================
    # LONG/SHORT RATIOS
    # =========================================================================

    def get_long_short_ratio(self, pair: str, exchange: str = "Binance") -> Optional[LongShortRatio]:
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

        # Calculate ratio
        if short_ratio > 0:
            ls_ratio = long_ratio / short_ratio
        else:
            ls_ratio = float('inf') if long_ratio > 0 else 1.0

        # Determine sentiment
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

        ratio = LongShortRatio(
            symbol=symbol,
            exchange=exchange,
            long_ratio=long_ratio,
            short_ratio=short_ratio,
            long_short_ratio=ls_ratio,
            sentiment=sentiment,
            crowding_score=crowding_score,
            contrarian_signal=contrarian_signal,
        )

        self._long_short_cache[cache_key] = ratio
        self._cache_timestamps[cache_key] = time.time()

        return ratio

    # =========================================================================
    # COMBINED SIGNALS
    # =========================================================================

    def get_combined_signal(self, pair: str) -> Dict:
        """
        Get all Coinglass signals combined for a trading pair.

        Args:
            pair: Trading pair

        Returns:
            Dict with all signals and overall recommendation
        """
        futures_cvd = self.get_futures_cvd(pair)
        divergence = self.get_spot_perp_divergence(pair)
        liquidation = self.get_liquidation_heatmap(pair)
        ls_ratio = self.get_long_short_ratio(pair)

        signals = {
            "pair": pair,
            "timestamp": time.time(),
            "futures_cvd": {
                "direction": futures_cvd.direction.value if futures_cvd else "unknown",
                "cvd_change": futures_cvd.cvd_change if futures_cvd else 0,
            } if futures_cvd else None,
            "spot_perp_divergence": {
                "type": divergence.divergence_type if divergence else "unknown",
                "strength": divergence.signal_strength if divergence else 0,
                "warnings": divergence.warnings if divergence else [],
            } if divergence else None,
            "liquidation": {
                "magnetic_direction": liquidation.magnetic_direction if liquidation else "unknown",
                "nearest_long_liq": liquidation.nearest_long_liq if liquidation else None,
                "nearest_short_liq": liquidation.nearest_short_liq if liquidation else None,
            } if liquidation else None,
            "long_short_ratio": {
                "sentiment": ls_ratio.sentiment if ls_ratio else "unknown",
                "crowding_score": ls_ratio.crowding_score if ls_ratio else 0,
                "contrarian_signal": ls_ratio.contrarian_signal if ls_ratio else "neutral",
            } if ls_ratio else None,
        }

        # Calculate overall recommendation
        bullish_signals = 0
        bearish_signals = 0

        if futures_cvd:
            if futures_cvd.direction in (CVDDirection.BULLISH, CVDDirection.STRONG_BULLISH):
                bullish_signals += 1
            elif futures_cvd.direction in (CVDDirection.BEARISH, CVDDirection.STRONG_BEARISH):
                bearish_signals += 1

        if divergence:
            if divergence.divergence_type == "bullish_divergence":
                bullish_signals += 2  # Weighted more heavily
            elif divergence.divergence_type == "bearish_divergence":
                bearish_signals += 2

        if liquidation:
            if liquidation.magnetic_direction == "up":
                bullish_signals += 1
            elif liquidation.magnetic_direction == "down":
                bearish_signals += 1

        if ls_ratio:
            if ls_ratio.contrarian_signal == "long":
                bullish_signals += 1
            elif ls_ratio.contrarian_signal == "short":
                bearish_signals += 1

        # Overall recommendation
        if bullish_signals > bearish_signals + 1:
            signals["recommendation"] = "LONG"
            signals["confidence"] = min(95, bullish_signals * 20)
        elif bearish_signals > bullish_signals + 1:
            signals["recommendation"] = "SHORT"
            signals["confidence"] = min(95, bearish_signals * 20)
        else:
            signals["recommendation"] = "NEUTRAL"
            signals["confidence"] = 50

        return signals

    def should_block_entry(self, pair: str, direction: str) -> Tuple[bool, str]:
        """
        Check if entry should be blocked based on Coinglass data.

        Args:
            pair: Trading pair
            direction: "long" or "short"

        Returns:
            Tuple of (should_block, reason)
        """
        divergence = self.get_spot_perp_divergence(pair)
        ls_ratio = self.get_long_short_ratio(pair)

        reasons = []

        # Check divergence
        if divergence and divergence.signal_strength >= 70:
            if direction == "long" and divergence.divergence_type == "bearish_divergence":
                reasons.append(
                    f"SPOT/PERP BEARISH DIVERGENCE ({divergence.signal_strength:.0f}%): "
                    "Perp longs are trapped - don't join them"
                )
            elif direction == "short" and divergence.divergence_type == "bullish_divergence":
                reasons.append(
                    f"SPOT/PERP BULLISH DIVERGENCE ({divergence.signal_strength:.0f}%): "
                    "Perp shorts are trapped - don't join them"
                )

        # Check long/short ratio
        if ls_ratio and ls_ratio.crowding_score >= 70:
            if direction == "long" and ls_ratio.sentiment == "crowded_long":
                reasons.append(
                    f"CROWDED LONG ({ls_ratio.long_ratio:.1f}%): "
                    "Too many longs - squeeze risk"
                )
            elif direction == "short" and ls_ratio.sentiment == "crowded_short":
                reasons.append(
                    f"CROWDED SHORT ({ls_ratio.short_ratio:.1f}%): "
                    "Too many shorts - squeeze risk"
                )

        if reasons:
            return True, " | ".join(reasons)
        return False, ""

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
