"""
Leading Indicators for Hyperliquid Monster Bot v2.

Fetches REAL leading indicators from Hyperliquid API:
- Open Interest (OI) changes
- OI + Price divergence detection
- Premium/Discount (mark vs oracle) directional pressure
- Funding velocity (rate of change)
- Volume surge detection
- Combined "HolyGrail" directional signal

These are TRUE leading indicators that predict price moves,
unlike RSI/MACD which only react to past price.

Why OI matters:
- OI Rising + Price Rising = New longs entering, move has conviction
- OI Rising + Price Falling = New shorts entering, move has conviction
- OI Falling + Price Rising = Short squeeze (shorts closing), exhaustion soon
- OI Falling + Price Falling = Long capitulation, bounce setup

Why Premium matters:
- High Premium (mark > oracle) = Longs paying to stay in = Bullish pressure
- High Discount (mark < oracle) = Shorts paying to stay in = Bearish pressure

Why Funding Velocity matters:
- Funding accelerating positive = Bullish momentum building
- Funding accelerating negative = Bearish momentum building
- Funding flipping direction = Trend reversal signal

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import requests


class OIMomentum(Enum):
    """Open Interest momentum classification."""
    BULLISH_CONFIRM = "bullish_confirm"      # OI rising + price rising
    BEARISH_CONFIRM = "bearish_confirm"      # OI rising + price falling
    BULLISH_EXHAUSTION = "bullish_exhaustion"  # OI falling + price rising (squeeze)
    BEARISH_EXHAUSTION = "bearish_exhaustion"  # OI falling + price falling (capitulation)
    NEUTRAL = "neutral"


@dataclass
class OISnapshot:
    """Snapshot of Open Interest data."""
    timestamp: float
    coin: str
    open_interest: float      # Total OI in USD
    mark_price: float
    funding_rate: float       # Current hourly funding
    premium: float            # Mark vs oracle diff
    day_volume: float         # 24h notional volume


@dataclass
class OIAnalysis:
    """Analysis result for OI-based signals."""
    coin: str
    momentum: OIMomentum
    oi_change_pct: float      # % change in OI over lookback
    price_change_pct: float   # % change in price over lookback
    conviction_score: float   # 0-100, how strong the signal is
    should_confirm_long: bool
    should_confirm_short: bool
    warning: Optional[str] = None


@dataclass
class LiquidationCluster:
    """Detected liquidation cluster."""
    coin: str
    side: str                 # "long" or "short"
    estimated_price: float    # Price level where liquidations cluster
    estimated_size: float     # Estimated USD at risk
    distance_pct: float       # Distance from current price


@dataclass
class PremiumAnalysis:
    """Premium/Discount analysis for directional pressure."""
    coin: str
    premium_pct: float        # Premium as percentage of price
    direction_bias: str       # "bullish", "bearish", or "neutral"
    pressure_score: float     # 0-100, how strong the pressure
    is_extreme: bool          # True if premium is at extreme levels


@dataclass
class FundingVelocity:
    """Funding rate velocity analysis."""
    coin: str
    current_rate: float       # Current hourly funding rate
    velocity: float           # Rate of change (acceleration)
    direction: str            # "accelerating_bullish", "accelerating_bearish", "decelerating", "flipping"
    flip_warning: bool        # True if funding about to flip sign
    annualized_apr: float     # Current rate annualized


@dataclass
class VolumeSurge:
    """Volume surge detection."""
    coin: str
    current_volume: float     # Current 24h volume
    avg_volume: float         # Average volume over lookback
    surge_ratio: float        # Current / Average
    is_surging: bool          # True if volume significantly above average
    surge_direction: str      # "bullish", "bearish", or "unknown" based on price action


@dataclass
class DirectionalSignal:
    """
    The "HolyGrail" combined directional signal.

    Combines all leading indicators into a single direction recommendation.
    """
    coin: str
    direction: str            # "strong_long", "long", "neutral", "short", "strong_short"
    confidence: float         # 0-100 confidence score

    # Component scores (each -100 to +100, positive = bullish)
    oi_score: float
    premium_score: float
    funding_velocity_score: float
    volume_score: float

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Detailed reasoning
    reasoning: str = ""


# =============================================================================
# PHASE 1 GOD MODE: SPIKE DETECTION & MARKET REGIME
# =============================================================================

@dataclass
class OISpikeResult:
    """Result of OI spike detection - instant violent OI changes."""
    coin: str
    detected: bool
    spike_type: str           # "spike_up", "spike_down", or "none"
    change_pct: float         # Percentage change in single snapshot
    interpretation: str       # What this means for trading
    timestamp: float


@dataclass
class PremiumSpikeResult:
    """Result of premium spike detection - sudden premium changes."""
    coin: str
    detected: bool
    spike_type: str           # "bullish_spike", "bearish_spike", or "none"
    change_pct: float         # Percentage change in premium
    previous_premium: float
    current_premium: float
    interpretation: str
    timestamp: float


@dataclass
class MarketRegime:
    """
    Market-wide regime detection across all monitored coins.

    Detects when the entire market is crowded in one direction,
    which often precedes market-wide squeezes.
    """
    regime: str               # "crowded_long", "crowded_short", "mixed", "neutral"
    confidence: float         # 0-100 how confident we are in the regime
    bullish_coins: int        # Number of coins with bullish funding
    bearish_coins: int        # Number of coins with bearish funding
    neutral_coins: int        # Number of coins with neutral funding
    total_coins: int
    avg_funding_rate: float   # Average funding across all coins
    squeeze_risk: str         # "high", "medium", "low"
    dominant_direction: str   # "long", "short", or "none"
    warnings: List[str] = field(default_factory=list)


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

    def analyze_oi_momentum(self, pair: str) -> Optional[OIAnalysis]:
        """
        Analyze OI momentum for a trading pair.

        This is the KEY leading indicator logic:
        - OI Rising + Price Rising = Bullish confirmation (new longs)
        - OI Rising + Price Falling = Bearish confirmation (new shorts)
        - OI Falling + Price Rising = Bullish exhaustion (short squeeze ending)
        - OI Falling + Price Falling = Bearish exhaustion (capitulation ending)

        Args:
            pair: Trading pair to analyze

        Returns:
            OIAnalysis or None if insufficient data
        """
        history = self._oi_history.get(pair)
        if history is None or len(history) < 3:
            return None

        # Get oldest and newest for comparison
        oldest = history[0]
        newest = history[-1]

        # Calculate changes
        if oldest.open_interest == 0 or oldest.mark_price == 0:
            return None

        oi_change_pct = ((newest.open_interest - oldest.open_interest)
                         / oldest.open_interest * 100)
        price_change_pct = ((newest.mark_price - oldest.mark_price)
                            / oldest.mark_price * 100)

        # Classify momentum
        oi_rising = oi_change_pct > self.oi_threshold
        oi_falling = oi_change_pct < -self.oi_threshold
        price_rising = price_change_pct > self.price_threshold
        price_falling = price_change_pct < -self.price_threshold

        if oi_rising and price_rising:
            momentum = OIMomentum.BULLISH_CONFIRM
            should_long = True
            should_short = False
            conviction = min(100, abs(oi_change_pct) * 10)
        elif oi_rising and price_falling:
            momentum = OIMomentum.BEARISH_CONFIRM
            should_long = False
            should_short = True
            conviction = min(100, abs(oi_change_pct) * 10)
        elif oi_falling and price_rising:
            momentum = OIMomentum.BULLISH_EXHAUSTION
            should_long = False  # Don't chase the squeeze
            should_short = False  # Don't short into squeeze either
            conviction = min(100, abs(oi_change_pct) * 8)
        elif oi_falling and price_falling:
            momentum = OIMomentum.BEARISH_EXHAUSTION
            should_long = True   # Capitulation = bounce setup
            should_short = False
            conviction = min(100, abs(oi_change_pct) * 8)
        else:
            momentum = OIMomentum.NEUTRAL
            should_long = True   # Neutral = no blocking
            should_short = True
            conviction = 0

        # Generate warning if needed
        warning = None
        if momentum == OIMomentum.BULLISH_EXHAUSTION:
            warning = f"Short squeeze detected - OI falling {oi_change_pct:.1f}% while price up {price_change_pct:.1f}%"
        elif momentum == OIMomentum.BEARISH_EXHAUSTION:
            warning = f"Long capitulation detected - potential bounce setup"

        return OIAnalysis(
            coin=pair,
            momentum=momentum,
            oi_change_pct=oi_change_pct,
            price_change_pct=price_change_pct,
            conviction_score=conviction,
            should_confirm_long=should_long,
            should_confirm_short=should_short,
            warning=warning,
        )

    def get_oi_direction_signal(self, pair: str) -> Tuple[Optional[str], float]:
        """
        Get simple OI direction signal for momentum strategy.

        Returns:
            Tuple of (signal, conviction) where signal is:
            - "bullish" if OI confirms long entry
            - "bearish" if OI confirms short entry
            - "neutral" if no clear signal
            - None if insufficient data
        """
        analysis = self.analyze_oi_momentum(pair)
        if analysis is None:
            return None, 0

        if analysis.momentum == OIMomentum.BULLISH_CONFIRM:
            return "bullish", analysis.conviction_score
        elif analysis.momentum == OIMomentum.BEARISH_CONFIRM:
            return "bearish", analysis.conviction_score
        elif analysis.momentum == OIMomentum.BEARISH_EXHAUSTION:
            # Capitulation = contrarian bullish
            return "bullish", analysis.conviction_score * 0.7
        elif analysis.momentum == OIMomentum.BULLISH_EXHAUSTION:
            # Don't give direction signal during squeeze
            return "neutral", 0
        else:
            return "neutral", 0

    def should_block_entry(self, pair: str, direction: str) -> Tuple[bool, str]:
        """
        Check if an entry should be blocked based on OI analysis.

        Args:
            pair: Trading pair
            direction: "long" or "short"

        Returns:
            Tuple of (should_block, reason)
        """
        analysis = self.analyze_oi_momentum(pair)
        if analysis is None:
            return False, ""

        if direction == "long":
            if analysis.momentum == OIMomentum.BULLISH_EXHAUSTION:
                return True, f"OI exhaustion: squeeze ending (OI {analysis.oi_change_pct:+.1f}%)"
            if analysis.momentum == OIMomentum.BEARISH_CONFIRM:
                return True, f"OI bearish: new shorts entering (OI {analysis.oi_change_pct:+.1f}%)"

        elif direction == "short":
            if analysis.momentum == OIMomentum.BEARISH_EXHAUSTION:
                return True, f"OI exhaustion: capitulation (bounce likely)"
            if analysis.momentum == OIMomentum.BULLISH_CONFIRM:
                return True, f"OI bullish: new longs entering (OI {analysis.oi_change_pct:+.1f}%)"

        return False, ""

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
        """
        Combined OI + Funding signal for squeeze detection.

        This is the most powerful signal:
        - High positive funding + Rising OI = Long squeeze setup (too crowded)
        - High negative funding + Rising OI = Short squeeze setup

        Returns:
            "long_squeeze_warning", "short_squeeze_warning", or None
        """
        history = self._oi_history.get(pair)
        if history is None or len(history) < 3:
            return None

        newest = history[-1]
        oldest = history[0]

        if oldest.open_interest == 0:
            return None

        oi_change_pct = ((newest.open_interest - oldest.open_interest)
                         / oldest.open_interest * 100)

        # High funding thresholds (hourly rate)
        HIGH_POSITIVE_FUNDING = 0.0005  # 0.05% hourly = ~438% APR
        HIGH_NEGATIVE_FUNDING = -0.0005

        oi_rising = oi_change_pct > self.oi_threshold

        if newest.funding_rate > HIGH_POSITIVE_FUNDING and oi_rising:
            return "long_squeeze_warning"  # Longs crowded + more piling in
        elif newest.funding_rate < HIGH_NEGATIVE_FUNDING and oi_rising:
            return "short_squeeze_warning"  # Shorts crowded + more piling in

        return None

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

    # =========================================================================
    # PREMIUM ANALYSIS - Directional pressure from mark vs oracle
    # =========================================================================

    def analyze_premium(self, pair: str) -> Optional[PremiumAnalysis]:
        """
        Analyze premium/discount for directional pressure.

        Premium = (Mark Price - Oracle Price) / Oracle Price
        - Positive premium = Longs paying to stay in = Bullish pressure
        - Negative premium (discount) = Shorts paying = Bearish pressure

        Args:
            pair: Trading pair to analyze

        Returns:
            PremiumAnalysis or None if insufficient data
        """
        history = self._oi_history.get(pair)
        if history is None or len(history) < 1:
            return None

        newest = history[-1]

        # Premium is already in the data as mark vs oracle difference
        # Hyperliquid returns premium as a decimal (e.g., 0.001 = 0.1%)
        premium_pct = newest.premium * 100  # Convert to percentage

        # Thresholds for direction bias
        BULLISH_THRESHOLD = 0.02   # 0.02% premium = bullish pressure
        BEARISH_THRESHOLD = -0.02  # -0.02% discount = bearish pressure
        EXTREME_THRESHOLD = 0.1   # 0.1% = extreme pressure

        if premium_pct > BULLISH_THRESHOLD:
            direction_bias = "bullish"
            pressure_score = min(100, abs(premium_pct) * 500)  # Scale to 0-100
        elif premium_pct < BEARISH_THRESHOLD:
            direction_bias = "bearish"
            pressure_score = min(100, abs(premium_pct) * 500)
        else:
            direction_bias = "neutral"
            pressure_score = 0

        is_extreme = abs(premium_pct) > EXTREME_THRESHOLD

        return PremiumAnalysis(
            coin=pair,
            premium_pct=premium_pct,
            direction_bias=direction_bias,
            pressure_score=pressure_score,
            is_extreme=is_extreme,
        )

    # =========================================================================
    # FUNDING VELOCITY - Rate of change in funding
    # =========================================================================

    def analyze_funding_velocity(self, pair: str) -> Optional[FundingVelocity]:
        """
        Analyze funding rate velocity (acceleration/deceleration).

        This detects:
        - Funding accelerating in bullish direction (more positive)
        - Funding accelerating in bearish direction (more negative)
        - Funding decelerating (moving toward neutral)
        - Funding about to flip sign (reversal warning)

        Args:
            pair: Trading pair to analyze

        Returns:
            FundingVelocity or None if insufficient data
        """
        history = self._oi_history.get(pair)
        if history is None or len(history) < 3:
            return None

        # Get funding rates over time
        funding_rates = [h.funding_rate for h in history]
        newest = history[-1]
        oldest = history[0]

        current_rate = newest.funding_rate
        old_rate = oldest.funding_rate

        # Calculate velocity (change per period)
        velocity = (current_rate - old_rate) / len(history)

        # Annualized APR (hourly rate * 24 * 365)
        annualized_apr = current_rate * 24 * 365 * 100

        # Determine direction
        VELOCITY_THRESHOLD = 0.00001  # Minimum velocity to be significant

        # Check for flip warning (funding close to zero and has momentum toward flip)
        flip_warning = False
        if abs(current_rate) < 0.0001:  # Near zero
            if (current_rate > 0 and velocity < -VELOCITY_THRESHOLD) or \
               (current_rate < 0 and velocity > VELOCITY_THRESHOLD):
                flip_warning = True

        # Classify direction
        if velocity > VELOCITY_THRESHOLD:
            if current_rate > 0:
                direction = "accelerating_bullish"  # Getting more bullish
            else:
                direction = "recovering"  # Was bearish, becoming less so
        elif velocity < -VELOCITY_THRESHOLD:
            if current_rate < 0:
                direction = "accelerating_bearish"  # Getting more bearish
            else:
                direction = "weakening"  # Was bullish, becoming less so
        else:
            direction = "stable"

        if flip_warning:
            direction = "flipping"

        return FundingVelocity(
            coin=pair,
            current_rate=current_rate,
            velocity=velocity,
            direction=direction,
            flip_warning=flip_warning,
            annualized_apr=annualized_apr,
        )

    # =========================================================================
    # VOLUME SURGE DETECTION
    # =========================================================================

    def detect_volume_surge(self, pair: str) -> Optional[VolumeSurge]:
        """
        Detect volume surges that often precede big moves.

        A volume surge with price moving up = bullish confirmation
        A volume surge with price moving down = bearish confirmation
        A volume surge with price flat = breakout imminent (direction unknown)

        Args:
            pair: Trading pair to analyze

        Returns:
            VolumeSurge or None if insufficient data
        """
        history = self._oi_history.get(pair)
        if history is None or len(history) < 3:
            return None

        newest = history[-1]
        current_volume = newest.day_volume

        # Calculate average volume over history
        volumes = [h.day_volume for h in history]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        if avg_volume == 0:
            return None

        surge_ratio = current_volume / avg_volume

        # Threshold for surge detection
        SURGE_THRESHOLD = 1.5  # 50% above average

        is_surging = surge_ratio > SURGE_THRESHOLD

        # Determine surge direction based on price action
        oldest = history[0]
        if oldest.mark_price == 0:
            return None

        price_change_pct = ((newest.mark_price - oldest.mark_price)
                           / oldest.mark_price * 100)

        if is_surging:
            if price_change_pct > 0.5:
                surge_direction = "bullish"
            elif price_change_pct < -0.5:
                surge_direction = "bearish"
            else:
                surge_direction = "breakout_imminent"
        else:
            surge_direction = "normal"

        return VolumeSurge(
            coin=pair,
            current_volume=current_volume,
            avg_volume=avg_volume,
            surge_ratio=surge_ratio,
            is_surging=is_surging,
            surge_direction=surge_direction,
        )

    # =========================================================================
    # HOLY GRAIL - Combined Directional Signal
    # =========================================================================

    def get_holy_grail_signal(self, pair: str) -> Optional[DirectionalSignal]:
        """
        THE HOLY GRAIL: Combined directional signal from ALL leading indicators.

        Combines:
        1. OI Momentum (-100 to +100)
        2. Premium pressure (-100 to +100)
        3. Funding velocity (-100 to +100)
        4. Volume surge confirmation (-100 to +100)

        Weights:
        - OI Momentum: 35% (strongest predictor)
        - Premium: 25% (real-time pressure)
        - Funding Velocity: 25% (momentum shift)
        - Volume: 15% (confirmation)

        Args:
            pair: Trading pair to analyze

        Returns:
            DirectionalSignal with combined recommendation
        """
        warnings = []
        reasoning_parts = []

        # =====================================================================
        # 1. OI MOMENTUM SCORE
        # =====================================================================
        oi_score = 0.0
        oi_analysis = self.analyze_oi_momentum(pair)
        if oi_analysis:
            if oi_analysis.momentum == OIMomentum.BULLISH_CONFIRM:
                oi_score = oi_analysis.conviction_score  # 0 to +100
                reasoning_parts.append(f"OI: Bullish confirm (+{oi_score:.0f})")
            elif oi_analysis.momentum == OIMomentum.BEARISH_CONFIRM:
                oi_score = -oi_analysis.conviction_score  # 0 to -100
                reasoning_parts.append(f"OI: Bearish confirm ({oi_score:.0f})")
            elif oi_analysis.momentum == OIMomentum.BULLISH_EXHAUSTION:
                oi_score = -30  # Penalize - squeeze ending
                warnings.append("Short squeeze ending - avoid longs")
                reasoning_parts.append(f"OI: Bullish exhaustion ({oi_score:.0f})")
            elif oi_analysis.momentum == OIMomentum.BEARISH_EXHAUSTION:
                oi_score = 40  # Favor longs - bounce setup
                reasoning_parts.append(f"OI: Bearish exhaustion/bounce (+{oi_score:.0f})")
            else:
                reasoning_parts.append("OI: Neutral (0)")

        # =====================================================================
        # 2. PREMIUM SCORE
        # =====================================================================
        premium_score = 0.0
        premium_analysis = self.analyze_premium(pair)
        if premium_analysis:
            if premium_analysis.direction_bias == "bullish":
                premium_score = premium_analysis.pressure_score  # 0 to +100
                if premium_analysis.is_extreme:
                    warnings.append(f"Extreme premium ({premium_analysis.premium_pct:.3f}%) - crowded long")
                    premium_score *= 0.5  # Reduce score if too crowded
                reasoning_parts.append(f"Premium: Bullish (+{premium_score:.0f})")
            elif premium_analysis.direction_bias == "bearish":
                premium_score = -premium_analysis.pressure_score  # 0 to -100
                if premium_analysis.is_extreme:
                    warnings.append(f"Extreme discount ({premium_analysis.premium_pct:.3f}%) - crowded short")
                    premium_score *= 0.5  # Reduce score if too crowded
                reasoning_parts.append(f"Premium: Bearish ({premium_score:.0f})")
            else:
                reasoning_parts.append("Premium: Neutral (0)")

        # =====================================================================
        # 3. FUNDING VELOCITY SCORE
        # =====================================================================
        funding_velocity_score = 0.0
        funding_vel = self.analyze_funding_velocity(pair)
        if funding_vel:
            if funding_vel.direction == "accelerating_bullish":
                funding_velocity_score = min(80, abs(funding_vel.velocity) * 100000)
                reasoning_parts.append(f"Funding: Accelerating bullish (+{funding_velocity_score:.0f})")
            elif funding_vel.direction == "accelerating_bearish":
                funding_velocity_score = -min(80, abs(funding_vel.velocity) * 100000)
                reasoning_parts.append(f"Funding: Accelerating bearish ({funding_velocity_score:.0f})")
            elif funding_vel.direction == "recovering":
                funding_velocity_score = 30  # Recovering from bearish
                reasoning_parts.append(f"Funding: Recovering (+{funding_velocity_score:.0f})")
            elif funding_vel.direction == "weakening":
                funding_velocity_score = -30  # Weakening from bullish
                reasoning_parts.append(f"Funding: Weakening ({funding_velocity_score:.0f})")
            elif funding_vel.direction == "flipping":
                warnings.append("Funding rate about to flip - trend reversal possible")
                # Direction depends on which way it's flipping
                if funding_vel.current_rate > 0 and funding_vel.velocity < 0:
                    funding_velocity_score = -50  # Was bullish, flipping bearish
                else:
                    funding_velocity_score = 50  # Was bearish, flipping bullish
                reasoning_parts.append(f"Funding: FLIPPING ({funding_velocity_score:.0f})")
            else:
                reasoning_parts.append("Funding: Stable (0)")

        # =====================================================================
        # 4. VOLUME SCORE
        # =====================================================================
        volume_score = 0.0
        volume_surge = self.detect_volume_surge(pair)
        if volume_surge and volume_surge.is_surging:
            if volume_surge.surge_direction == "bullish":
                volume_score = min(60, (volume_surge.surge_ratio - 1) * 40)
                reasoning_parts.append(f"Volume: Bullish surge (+{volume_score:.0f})")
            elif volume_surge.surge_direction == "bearish":
                volume_score = -min(60, (volume_surge.surge_ratio - 1) * 40)
                reasoning_parts.append(f"Volume: Bearish surge ({volume_score:.0f})")
            elif volume_surge.surge_direction == "breakout_imminent":
                warnings.append(f"Volume surge {volume_surge.surge_ratio:.1f}x - breakout imminent")
                reasoning_parts.append("Volume: Breakout imminent (0)")
        else:
            reasoning_parts.append("Volume: Normal (0)")

        # =====================================================================
        # COMBINE SCORES WITH WEIGHTS
        # =====================================================================
        OI_WEIGHT = 0.35
        PREMIUM_WEIGHT = 0.25
        FUNDING_WEIGHT = 0.25
        VOLUME_WEIGHT = 0.15

        combined_score = (
            oi_score * OI_WEIGHT +
            premium_score * PREMIUM_WEIGHT +
            funding_velocity_score * FUNDING_WEIGHT +
            volume_score * VOLUME_WEIGHT
        )

        # Calculate confidence based on agreement between indicators
        scores = [oi_score, premium_score, funding_velocity_score, volume_score]
        positive_count = sum(1 for s in scores if s > 10)
        negative_count = sum(1 for s in scores if s < -10)

        # Higher confidence when indicators agree
        if positive_count >= 3 or negative_count >= 3:
            confidence = min(95, abs(combined_score) + 20)
        elif positive_count >= 2 or negative_count >= 2:
            confidence = min(80, abs(combined_score) + 10)
        else:
            confidence = min(60, abs(combined_score))

        # =====================================================================
        # DETERMINE DIRECTION
        # =====================================================================
        if combined_score > 40:
            direction = "strong_long"
        elif combined_score > 15:
            direction = "long"
        elif combined_score < -40:
            direction = "strong_short"
        elif combined_score < -15:
            direction = "short"
        else:
            direction = "neutral"

        reasoning = " | ".join(reasoning_parts)

        return DirectionalSignal(
            coin=pair,
            direction=direction,
            confidence=confidence,
            oi_score=oi_score,
            premium_score=premium_score,
            funding_velocity_score=funding_velocity_score,
            volume_score=volume_score,
            warnings=warnings,
            reasoning=reasoning,
        )

    def get_direction_recommendation(self, pair: str) -> Tuple[str, float, List[str]]:
        """
        Simple interface to get direction recommendation.

        Args:
            pair: Trading pair

        Returns:
            Tuple of (direction, confidence, warnings)
            direction: "long", "short", or "neutral"
            confidence: 0-100
            warnings: List of warning strings
        """
        signal = self.get_holy_grail_signal(pair)
        if signal is None:
            return "neutral", 0, []

        # Simplify direction
        if signal.direction in ("strong_long", "long"):
            direction = "long"
        elif signal.direction in ("strong_short", "short"):
            direction = "short"
        else:
            direction = "neutral"

        return direction, signal.confidence, signal.warnings

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

    # =========================================================================
    # PHASE 1 GOD MODE: SPIKE DETECTION
    # =========================================================================

    def detect_oi_spike(self, pair: str) -> Optional[OISpikeResult]:
        """
        Detect instant violent OI changes in a single snapshot.

        Your 12-period lookback smooths out spikes - this catches them instantly.

        Signals:
        - OI spike UP (>10% in 30s) = Big player entering, expect momentum
        - OI spike DOWN (>10% in 30s) = Liquidation cascade starting

        Args:
            pair: Trading pair to check

        Returns:
            OISpikeResult or None if insufficient data
        """
        history = self._oi_history.get(pair)
        if history is None or len(history) < 2:
            return None

        # Compare last two snapshots (instant detection)
        prev = history[-2]
        curr = history[-1]

        if prev.open_interest == 0:
            return None

        change_pct = ((curr.open_interest - prev.open_interest)
                      / prev.open_interest * 100)

        # Spike thresholds
        SPIKE_UP_THRESHOLD = 10.0    # 10% increase in single snapshot
        SPIKE_DOWN_THRESHOLD = -10.0  # 10% decrease in single snapshot

        # Also detect smaller but still significant spikes
        MINOR_SPIKE_UP = 5.0
        MINOR_SPIKE_DOWN = -5.0

        detected = False
        spike_type = "none"
        interpretation = ""

        if change_pct >= SPIKE_UP_THRESHOLD:
            detected = True
            spike_type = "spike_up"
            interpretation = (
                f"MAJOR OI SPIKE UP: {change_pct:.1f}% in ~30s. "
                "Big player entering - expect momentum continuation. "
                "Consider following the direction if price confirms."
            )
        elif change_pct >= MINOR_SPIKE_UP:
            detected = True
            spike_type = "spike_up"
            interpretation = (
                f"OI spike up: {change_pct:.1f}% in ~30s. "
                "Significant new positions opening - momentum building."
            )
        elif change_pct <= SPIKE_DOWN_THRESHOLD:
            detected = True
            spike_type = "spike_down"
            interpretation = (
                f"MAJOR OI SPIKE DOWN: {change_pct:.1f}% in ~30s. "
                "Liquidation cascade or mass exit. "
                "Expect volatility spike then potential reversal."
            )
        elif change_pct <= MINOR_SPIKE_DOWN:
            detected = True
            spike_type = "spike_down"
            interpretation = (
                f"OI spike down: {change_pct:.1f}% in ~30s. "
                "Positions closing rapidly - watch for reversal setup."
            )

        return OISpikeResult(
            coin=pair,
            detected=detected,
            spike_type=spike_type,
            change_pct=change_pct,
            interpretation=interpretation,
            timestamp=curr.timestamp,
        )

    def detect_premium_spike(self, pair: str) -> Optional[PremiumSpikeResult]:
        """
        Detect sudden premium changes that often precede violent moves.

        A premium spike >50% change indicates rapid sentiment shift.

        Args:
            pair: Trading pair to check

        Returns:
            PremiumSpikeResult or None if insufficient data
        """
        history = self._oi_history.get(pair)
        if history is None or len(history) < 2:
            return None

        prev = history[-2]
        curr = history[-1]

        # Handle near-zero premiums carefully
        if abs(prev.premium) < 0.00001:
            # If previous was near zero, use absolute change
            if abs(curr.premium) > 0.0005:  # Significant new premium
                detected = True
                change_pct = 100.0  # Treat as 100% change
            else:
                return PremiumSpikeResult(
                    coin=pair,
                    detected=False,
                    spike_type="none",
                    change_pct=0.0,
                    previous_premium=prev.premium,
                    current_premium=curr.premium,
                    interpretation="Premium stable near zero",
                    timestamp=curr.timestamp,
                )
        else:
            change_pct = ((curr.premium - prev.premium) / abs(prev.premium) * 100)

        # Spike thresholds
        SPIKE_THRESHOLD = 50.0  # 50% change in premium

        detected = abs(change_pct) >= SPIKE_THRESHOLD
        spike_type = "none"
        interpretation = ""

        if detected:
            if curr.premium > prev.premium:
                spike_type = "bullish_spike"
                interpretation = (
                    f"PREMIUM SPIKE UP: {change_pct:+.1f}% in ~30s. "
                    f"Premium went from {prev.premium*100:.4f}% to {curr.premium*100:.4f}%. "
                    "Aggressive long pressure building - potential pump incoming."
                )
            else:
                spike_type = "bearish_spike"
                interpretation = (
                    f"PREMIUM SPIKE DOWN: {change_pct:+.1f}% in ~30s. "
                    f"Premium went from {prev.premium*100:.4f}% to {curr.premium*100:.4f}%. "
                    "Aggressive short pressure building - potential dump incoming."
                )

        return PremiumSpikeResult(
            coin=pair,
            detected=detected,
            spike_type=spike_type,
            change_pct=change_pct,
            previous_premium=prev.premium,
            current_premium=curr.premium,
            interpretation=interpretation,
            timestamp=curr.timestamp,
        )

    # =========================================================================
    # PHASE 1 GOD MODE: MARKET REGIME DETECTION
    # =========================================================================

    def get_market_regime(self) -> Optional[MarketRegime]:
        """
        Detect market-wide regime across all monitored coins.

        When 8+ coins have same-direction funding, the market is crowded
        and a squeeze in the opposite direction becomes likely.

        Returns:
            MarketRegime with cross-market analysis
        """
        pairs_tracked = list(self._oi_history.keys())
        if len(pairs_tracked) < 3:
            return None

        # Thresholds for classifying funding direction
        BULLISH_THRESHOLD = 0.0001   # 0.01% hourly = ~88% APR
        BEARISH_THRESHOLD = -0.0001

        bullish_coins = 0
        bearish_coins = 0
        neutral_coins = 0
        funding_rates = []
        warnings = []

        for pair in pairs_tracked:
            history = self._oi_history.get(pair)
            if history is None or len(history) < 1:
                continue

            funding = history[-1].funding_rate
            funding_rates.append(funding)

            if funding > BULLISH_THRESHOLD:
                bullish_coins += 1
            elif funding < BEARISH_THRESHOLD:
                bearish_coins += 1
            else:
                neutral_coins += 1

        total_coins = bullish_coins + bearish_coins + neutral_coins
        if total_coins == 0:
            return None

        avg_funding = sum(funding_rates) / len(funding_rates) if funding_rates else 0

        # Determine regime
        bullish_pct = bullish_coins / total_coins * 100
        bearish_pct = bearish_coins / total_coins * 100

        # Crowded thresholds
        CROWDED_THRESHOLD = 70  # 70% of coins same direction = crowded

        if bullish_pct >= CROWDED_THRESHOLD:
            regime = "crowded_long"
            squeeze_risk = "high"
            dominant_direction = "long"
            warnings.append(
                f"MARKET CROWDED LONG: {bullish_coins}/{total_coins} coins have positive funding. "
                "Long squeeze risk is LOW, but short squeeze risk is HIGH if sentiment shifts."
            )
        elif bearish_pct >= CROWDED_THRESHOLD:
            regime = "crowded_short"
            squeeze_risk = "high"
            dominant_direction = "short"
            warnings.append(
                f"MARKET CROWDED SHORT: {bearish_coins}/{total_coins} coins have negative funding. "
                "Short squeeze imminent if any catalyst appears."
            )
        elif bullish_pct >= 50 or bearish_pct >= 50:
            regime = "leaning"
            squeeze_risk = "medium"
            dominant_direction = "long" if bullish_pct > bearish_pct else "short"
        else:
            regime = "mixed"
            squeeze_risk = "low"
            dominant_direction = "none"

        # Additional warning for extreme average funding
        if abs(avg_funding) > 0.0005:  # 0.05% hourly = ~438% APR average
            direction = "LONG" if avg_funding > 0 else "SHORT"
            warnings.append(
                f"EXTREME AVG FUNDING: {avg_funding*100:.4f}% hourly across market. "
                f"Market-wide {direction} crowding detected."
            )

        # Calculate confidence
        max_pct = max(bullish_pct, bearish_pct)
        confidence = min(95, max_pct + (10 if squeeze_risk == "high" else 0))

        return MarketRegime(
            regime=regime,
            confidence=confidence,
            bullish_coins=bullish_coins,
            bearish_coins=bearish_coins,
            neutral_coins=neutral_coins,
            total_coins=total_coins,
            avg_funding_rate=avg_funding,
            squeeze_risk=squeeze_risk,
            dominant_direction=dominant_direction,
            warnings=warnings,
        )

    def get_all_spikes(self) -> Dict[str, Dict]:
        """
        Check all tracked pairs for OI and premium spikes.

        Returns:
            Dict of pair -> spike info
        """
        results = {}
        for pair in self._oi_history.keys():
            oi_spike = self.detect_oi_spike(pair)
            premium_spike = self.detect_premium_spike(pair)

            if (oi_spike and oi_spike.detected) or (premium_spike and premium_spike.detected):
                results[pair] = {
                    "oi_spike": {
                        "detected": oi_spike.detected if oi_spike else False,
                        "type": oi_spike.spike_type if oi_spike else "none",
                        "change": f"{oi_spike.change_pct:+.1f}%" if oi_spike else "N/A",
                    },
                    "premium_spike": {
                        "detected": premium_spike.detected if premium_spike else False,
                        "type": premium_spike.spike_type if premium_spike else "none",
                        "change": f"{premium_spike.change_pct:+.1f}%" if premium_spike else "N/A",
                    },
                }
        return results

    def should_block_entry_godmode(
        self,
        pair: str,
        direction: str,
        check_market_regime: bool = True
    ) -> Tuple[bool, str]:
        """
        Enhanced entry blocking with GOD MODE checks.

        Combines original OI blocking with:
        - OI spike detection
        - Premium spike detection
        - Market regime awareness

        Args:
            pair: Trading pair
            direction: "long" or "short"
            check_market_regime: Whether to factor in market-wide regime

        Returns:
            Tuple of (should_block, reason)
        """
        reasons = []

        # Original OI momentum check
        blocked, reason = self.should_block_entry(pair, direction)
        if blocked:
            reasons.append(reason)

        # OI Spike check
        oi_spike = self.detect_oi_spike(pair)
        if oi_spike and oi_spike.detected:
            if direction == "long" and oi_spike.spike_type == "spike_down":
                reasons.append(
                    f"OI SPIKE DOWN ({oi_spike.change_pct:.1f}%): "
                    "Liquidation cascade - wait for dust to settle"
                )
            elif direction == "short" and oi_spike.spike_type == "spike_up":
                reasons.append(
                    f"OI SPIKE UP ({oi_spike.change_pct:.1f}%): "
                    "Big player entering long - don't short into momentum"
                )

        # Premium spike check
        premium_spike = self.detect_premium_spike(pair)
        if premium_spike and premium_spike.detected:
            if direction == "long" and premium_spike.spike_type == "bearish_spike":
                reasons.append(
                    f"PREMIUM SPIKE DOWN ({premium_spike.change_pct:.1f}%): "
                    "Aggressive shorting - wait for stabilization"
                )
            elif direction == "short" and premium_spike.spike_type == "bullish_spike":
                reasons.append(
                    f"PREMIUM SPIKE UP ({premium_spike.change_pct:.1f}%): "
                    "Aggressive longing - don't short into FOMO"
                )

        # Market regime check
        if check_market_regime:
            regime = self.get_market_regime()
            if regime and regime.squeeze_risk == "high":
                if direction == "long" and regime.regime == "crowded_long":
                    # Don't block longs in crowded long - they might still work
                    # But warn
                    pass
                elif direction == "short" and regime.regime == "crowded_short":
                    # Shorting into crowded short is DANGEROUS
                    reasons.append(
                        f"MARKET CROWDED SHORT ({regime.bearish_coins}/{regime.total_coins}): "
                        "Short squeeze imminent - extremely dangerous to short"
                    )
                elif direction == "long" and regime.regime == "crowded_short":
                    # This is actually good for longs
                    pass
                elif direction == "short" and regime.regime == "crowded_long":
                    # Careful shorting crowded longs - they can squeeze higher
                    if regime.confidence > 80:
                        reasons.append(
                            f"MARKET CROWDED LONG ({regime.bullish_coins}/{regime.total_coins}): "
                            "Be cautious shorting - longs can squeeze higher before reversing"
                        )

        if reasons:
            return True, " | ".join(reasons)
        return False, ""
