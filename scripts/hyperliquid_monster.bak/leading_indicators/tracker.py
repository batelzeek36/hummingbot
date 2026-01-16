"""
Hyperliquid Leading Indicators Tracker.

Main class that fetches data from Hyperliquid API and coordinates analysis
across all leading indicator modules.

Uses the metaAndAssetCtxs endpoint to get:
- Open Interest
- Funding rates
- Premium (mark vs oracle)
- Volume data
"""

import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import requests

from .models import (
    DirectionalSignal,
    FundingVelocity,
    MarketRegime,
    OIAnalysis,
    OISpikeResult,
    OISnapshot,
    PremiumAnalysis,
    PremiumSpikeResult,
    VolumeSurge,
)
from .oi_analysis import (
    analyze_oi_momentum,
    get_oi_direction_signal,
    get_oi_funding_combined_signal,
    should_block_entry,
)
from .premium import analyze_premium
from .funding_velocity import analyze_funding_velocity
from .volume import detect_volume_surge
from .holy_grail import calculate_holy_grail_signal, get_direction_recommendation
from .godmode import (
    detect_oi_spike,
    detect_premium_spike,
    get_all_spikes,
    get_market_regime,
    should_block_entry_godmode,
)


class HyperliquidLeadingIndicators:
    """
    Fetches and analyzes leading indicators from Hyperliquid API.

    Uses the metaAndAssetCtxs endpoint to get:
    - Open Interest
    - Funding rates
    - Premium (mark vs oracle)
    - Volume data
    """

    API_URL = "https://api.hyperliquid.xyz/info"

    # Coin name mapping (Hyperliquid uses different names)
    COIN_MAP = {
        "BTC-USD": "BTC",
        "ETH-USD": "ETH",
        "SOL-USD": "SOL",
        "DOGE-USD": "DOGE",
        "TAO-USD": "TAO",
        "HYPE-USD": "HYPE",
        "AVNT-USD": "AVNT",
        "kPEPE-USD": "kPEPE",
        "kBONK-USD": "kBONK",
        "WIF-USD": "WIF",
        "VVV-USD": "VVV",
        "HYPER-USD": "HYPER",
        "IP-USD": "IP",
    }

    def __init__(
        self,
        oi_lookback_periods: int = 12,       # How many snapshots to keep
        fetch_interval_seconds: int = 30,    # Min time between API calls
        oi_change_threshold: float = 2.0,    # % OI change to be significant
        price_change_threshold: float = 0.5, # % price change to be significant
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the leading indicators fetcher.

        Args:
            oi_lookback_periods: Number of snapshots to keep for analysis
            fetch_interval_seconds: Minimum seconds between API fetches
            oi_change_threshold: OI change % to consider significant
            price_change_threshold: Price change % to consider significant
            logger: Logger instance
        """
        self.oi_lookback = oi_lookback_periods
        self.fetch_interval = fetch_interval_seconds
        self.oi_threshold = oi_change_threshold
        self.price_threshold = price_change_threshold
        self.logger = logger or logging.getLogger(__name__)

        # OI history per coin
        self._oi_history: Dict[str, Deque[OISnapshot]] = {}

        # Cache
        self._last_fetch_time: float = 0
        self._last_raw_data: Optional[Dict] = None
        self._coin_index_map: Dict[str, int] = {}  # coin name -> index in response

        # Stats
        self._fetch_count = 0
        self._error_count = 0

    def _fetch_market_data(self) -> Optional[Dict]:
        """
        Fetch market data from Hyperliquid API.

        Returns:
            Raw API response or None on error
        """
        now = time.time()

        # Rate limiting
        if now - self._last_fetch_time < self.fetch_interval:
            return self._last_raw_data

        try:
            response = requests.post(
                self.API_URL,
                headers={"Content-Type": "application/json"},
                json={"type": "metaAndAssetCtxs"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            self._last_fetch_time = now
            self._last_raw_data = data
            self._fetch_count += 1

            # Build coin index map on first fetch
            if not self._coin_index_map and len(data) >= 2:
                universe = data[0].get("universe", [])
                for i, asset in enumerate(universe):
                    coin_name = asset.get("name", "")
                    self._coin_index_map[coin_name] = i

            return data

        except requests.RequestException as e:
            self._error_count += 1
            self.logger.warning(f"Failed to fetch Hyperliquid data: {e}")
            return self._last_raw_data  # Return cached data on error
        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Error parsing Hyperliquid data: {e}")
            return None

    def _get_coin_data(self, pair: str) -> Optional[OISnapshot]:
        """
        Get current OI snapshot for a trading pair.

        Args:
            pair: Trading pair like "BTC-USD"

        Returns:
            OISnapshot or None
        """
        coin = self.COIN_MAP.get(pair, pair.replace("-USD", ""))

        data = self._fetch_market_data()
        if data is None or len(data) < 2:
            return None

        idx = self._coin_index_map.get(coin)
        if idx is None:
            return None

        asset_ctxs = data[1]
        if idx >= len(asset_ctxs):
            return None

        ctx = asset_ctxs[idx]

        try:
            return OISnapshot(
                timestamp=time.time(),
                coin=coin,
                open_interest=float(ctx.get("openInterest", 0)),
                mark_price=float(ctx.get("markPx", 0)),
                funding_rate=float(ctx.get("funding", 0)),
                premium=float(ctx.get("premium", 0)),
                day_volume=float(ctx.get("dayNtlVlm", 0)),
            )
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Error parsing coin data for {coin}: {e}")
            return None

    def update(self, pairs: List[str]) -> Dict[str, OISnapshot]:
        """
        Update OI history for the given pairs.

        Args:
            pairs: List of trading pairs to update

        Returns:
            Dict of pair -> latest OISnapshot
        """
        results = {}

        for pair in pairs:
            snapshot = self._get_coin_data(pair)
            if snapshot is None:
                continue

            # Initialize history if needed
            if pair not in self._oi_history:
                self._oi_history[pair] = deque(maxlen=self.oi_lookback)

            # Add to history
            self._oi_history[pair].append(snapshot)
            results[pair] = snapshot

        return results

    # =========================================================================
    # OI ANALYSIS - Delegated to oi_analysis module
    # =========================================================================

    def analyze_oi_momentum(self, pair: str) -> Optional[OIAnalysis]:
        """Analyze OI momentum for a trading pair."""
        history = self._oi_history.get(pair)
        return analyze_oi_momentum(history, self.oi_threshold, self.price_threshold)

    def get_oi_direction_signal(self, pair: str) -> Tuple[Optional[str], float]:
        """Get simple OI direction signal for momentum strategy."""
        analysis = self.analyze_oi_momentum(pair)
        return get_oi_direction_signal(analysis)

    def should_block_entry(self, pair: str, direction: str) -> Tuple[bool, str]:
        """Check if an entry should be blocked based on OI analysis."""
        analysis = self.analyze_oi_momentum(pair)
        return should_block_entry(analysis, direction)

    def get_current_oi(self, pair: str) -> Optional[float]:
        """Get current open interest for a pair."""
        history = self._oi_history.get(pair)
        if history and len(history) > 0:
            return history[-1].open_interest
        return None

    def get_current_funding(self, pair: str) -> Optional[float]:
        """Get current funding rate for a pair."""
        history = self._oi_history.get(pair)
        if history and len(history) > 0:
            return history[-1].funding_rate
        return None

    def get_oi_funding_combined_signal(self, pair: str) -> Optional[str]:
        """Combined OI + Funding signal for squeeze detection."""
        history = self._oi_history.get(pair)
        return get_oi_funding_combined_signal(history, self.oi_threshold)

    # =========================================================================
    # PREMIUM ANALYSIS - Delegated to premium module
    # =========================================================================

    def analyze_premium(self, pair: str) -> Optional[PremiumAnalysis]:
        """Analyze premium/discount for directional pressure."""
        history = self._oi_history.get(pair)
        return analyze_premium(history)

    # =========================================================================
    # FUNDING VELOCITY - Delegated to funding_velocity module
    # =========================================================================

    def analyze_funding_velocity(self, pair: str) -> Optional[FundingVelocity]:
        """Analyze funding rate velocity (acceleration/deceleration)."""
        history = self._oi_history.get(pair)
        return analyze_funding_velocity(history)

    # =========================================================================
    # VOLUME SURGE - Delegated to volume module
    # =========================================================================

    def detect_volume_surge(self, pair: str) -> Optional[VolumeSurge]:
        """Detect volume surges that often precede big moves."""
        history = self._oi_history.get(pair)
        return detect_volume_surge(history)

    # =========================================================================
    # HOLY GRAIL - Delegated to holy_grail module
    # =========================================================================

    def get_holy_grail_signal(self, pair: str) -> Optional[DirectionalSignal]:
        """THE HOLY GRAIL: Combined directional signal from ALL leading indicators."""
        oi_analysis = self.analyze_oi_momentum(pair)
        premium_analysis = self.analyze_premium(pair)
        funding_vel = self.analyze_funding_velocity(pair)
        volume_surge = self.detect_volume_surge(pair)

        return calculate_holy_grail_signal(
            coin=pair,
            oi_analysis=oi_analysis,
            premium_analysis=premium_analysis,
            funding_vel=funding_vel,
            volume_surge=volume_surge,
        )

    def get_direction_recommendation(self, pair: str) -> Tuple[str, float, List[str]]:
        """Simple interface to get direction recommendation."""
        signal = self.get_holy_grail_signal(pair)
        return get_direction_recommendation(signal)

    # =========================================================================
    # GOD MODE - Delegated to godmode module
    # =========================================================================

    def detect_oi_spike(self, pair: str) -> Optional[OISpikeResult]:
        """Detect instant violent OI changes in a single snapshot."""
        history = self._oi_history.get(pair)
        return detect_oi_spike(history)

    def detect_premium_spike(self, pair: str) -> Optional[PremiumSpikeResult]:
        """Detect sudden premium changes that often precede violent moves."""
        history = self._oi_history.get(pair)
        return detect_premium_spike(history)

    def get_market_regime(self) -> Optional[MarketRegime]:
        """Detect market-wide regime across all monitored coins."""
        return get_market_regime(self._oi_history)

    def get_all_spikes(self) -> Dict[str, Dict]:
        """Check all tracked pairs for OI and premium spikes."""
        return get_all_spikes(self._oi_history)

    def should_block_entry_godmode(
        self,
        pair: str,
        direction: str,
        check_market_regime: bool = True
    ) -> Tuple[bool, str]:
        """Enhanced entry blocking with GOD MODE checks."""
        history = self._oi_history.get(pair)
        oi_analysis = self.analyze_oi_momentum(pair)
        return should_block_entry_godmode(
            pair=pair,
            direction=direction,
            history=history,
            oi_analysis=oi_analysis,
            oi_histories=self._oi_history,
            check_market_regime=check_market_regime,
        )

    # =========================================================================
    # STATUS DISPLAY
    # =========================================================================

    def get_status(self) -> Dict:
        """Get status summary for display."""
        pairs_tracked = list(self._oi_history.keys())

        oi_data = {}
        for pair in pairs_tracked:
            analysis = self.analyze_oi_momentum(pair)
            if analysis:
                oi_data[pair] = {
                    "momentum": analysis.momentum.value,
                    "oi_change": f"{analysis.oi_change_pct:+.1f}%",
                    "price_change": f"{analysis.price_change_pct:+.1f}%",
                    "conviction": f"{analysis.conviction_score:.0f}",
                }

        return {
            "pairs_tracked": len(pairs_tracked),
            "fetch_count": self._fetch_count,
            "error_count": self._error_count,
            "oi_analysis": oi_data,
        }

    def get_extended_status(self) -> Dict:
        """Get extended status with all leading indicators."""
        pairs_tracked = list(self._oi_history.keys())

        signals = {}
        for pair in pairs_tracked:
            holy_grail = self.get_holy_grail_signal(pair)
            if holy_grail:
                signals[pair] = {
                    "direction": holy_grail.direction,
                    "confidence": f"{holy_grail.confidence:.0f}%",
                    "oi_score": f"{holy_grail.oi_score:+.0f}",
                    "premium_score": f"{holy_grail.premium_score:+.0f}",
                    "funding_vel_score": f"{holy_grail.funding_velocity_score:+.0f}",
                    "volume_score": f"{holy_grail.volume_score:+.0f}",
                    "warnings": holy_grail.warnings,
                }

        # Add market regime
        market_regime = self.get_market_regime()

        return {
            "pairs_tracked": len(pairs_tracked),
            "fetch_count": self._fetch_count,
            "error_count": self._error_count,
            "signals": signals,
            "market_regime": {
                "regime": market_regime.regime if market_regime else "unknown",
                "squeeze_risk": market_regime.squeeze_risk if market_regime else "unknown",
                "dominant_direction": market_regime.dominant_direction if market_regime else "none",
            } if market_regime else None,
        }
