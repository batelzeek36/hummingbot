"""
Early Warning System - Proactive threat detection.

Monitors market conditions for early signs of trouble:
1. Order book imbalance - detects whale orders waiting in the book
2. Funding rate spikes - detects overleveraged/crowded markets
3. Spread widening - detects liquidity deterioration

These are LEADING indicators that can warn before a violent move happens.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple


class WarningLevel(Enum):
    """Warning severity levels."""
    CLEAR = "clear"           # No warnings, safe conditions
    INFO = "info"             # Minor anomaly, worth noting
    WARNING = "warning"       # Elevated risk, consider reducing exposure
    DANGER = "danger"         # High risk, avoid new entries
    CRITICAL = "critical"     # Immediate action recommended


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state."""
    timestamp: float
    pair: str
    bid_depth: float          # Total bid volume in top N levels
    ask_depth: float          # Total ask volume in top N levels
    imbalance_ratio: float    # bid_depth / ask_depth
    spread_pct: float         # Spread as percentage of mid price
    best_bid: float
    best_ask: float


@dataclass
class OrderBookVelocity:
    """
    Order book velocity analysis - tracks how fast imbalance is changing.

    Rapid changes indicate whale activity:
    - Fast bid accumulation = whale buying
    - Fast ask accumulation = whale selling
    - Sudden disappearance = iceberg order filled, move incoming
    """
    pair: str
    current_ratio: float      # Current bid/ask ratio
    velocity: float           # Rate of change per snapshot
    acceleration: float       # Rate of velocity change (2nd derivative)
    signal: str               # "whale_buying", "whale_selling", "iceberg_detected", "normal"
    confidence: float         # 0-100 confidence in signal
    interpretation: str       # Human-readable explanation


@dataclass
class FundingSnapshot:
    """Snapshot of funding rate state."""
    timestamp: float
    pair: str
    rate: float               # Current funding rate
    apr: float                # Annualized rate
    direction: str            # "long_pays" or "short_pays"
    crowding_score: float     # How crowded the market is (0-100)


@dataclass
class Warning:
    """An active warning."""
    timestamp: float
    pair: str
    level: WarningLevel
    category: str             # "order_book", "funding", "spread", etc.
    message: str
    data: Dict


class EarlyWarningSystem:
    """
    Early Warning System for proactive risk detection.

    Features:
    - Order book imbalance monitoring
    - Funding rate spike detection
    - Spread monitoring
    - Aggregated warning level per pair
    - Historical warning tracking
    """

    def __init__(
        self,
        # Order book thresholds
        ob_imbalance_warning: float = 2.0,       # 2:1 imbalance triggers warning
        ob_imbalance_danger: float = 3.5,        # 3.5:1 imbalance triggers danger
        ob_depth_levels: int = 5,                # How many levels to analyze
        # Funding thresholds (as hourly rate, e.g., 0.01 = 1%)
        funding_warning_rate: float = 0.0005,    # 0.05% hourly (~438% APR)
        funding_danger_rate: float = 0.001,      # 0.1% hourly (~876% APR)
        funding_critical_rate: float = 0.002,    # 0.2% hourly (~1752% APR)
        # Spread thresholds
        spread_warning_pct: float = 0.002,       # 0.2% spread triggers warning
        spread_danger_pct: float = 0.005,        # 0.5% spread triggers danger
        # History
        max_history: int = 100,
        # Callbacks
        on_warning: Optional[Callable[[Warning], None]] = None,
        # Logging
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
    ):
        """
        Initialize the early warning system.

        Args:
            ob_imbalance_*: Order book imbalance thresholds
            ob_depth_levels: Number of order book levels to analyze
            funding_*_rate: Funding rate thresholds (as hourly decimal)
            spread_*_pct: Spread thresholds as percentage
            max_history: Maximum warnings to keep in history
            on_warning: Callback when new warning is raised
            logger: Logger instance
            enabled: Whether early warning is active
        """
        self.enabled = enabled

        # Order book thresholds
        self.ob_imbalance_warning = ob_imbalance_warning
        self.ob_imbalance_danger = ob_imbalance_danger
        self.ob_depth_levels = ob_depth_levels

        # Funding thresholds
        self.funding_warning_rate = funding_warning_rate
        self.funding_danger_rate = funding_danger_rate
        self.funding_critical_rate = funding_critical_rate

        # Spread thresholds
        self.spread_warning_pct = spread_warning_pct
        self.spread_danger_pct = spread_danger_pct

        self.max_history = max_history
        self.on_warning = on_warning
        self.logger = logger or logging.getLogger(__name__)

        # State
        self._order_book_history: Dict[str, Deque[OrderBookSnapshot]] = {}
        self._funding_history: Dict[str, Deque[FundingSnapshot]] = {}
        self._active_warnings: Dict[str, Dict[str, Warning]] = {}  # pair -> category -> warning
        self._warning_history: Deque[Warning] = deque(maxlen=max_history)

        # Aggregate warning level per pair
        self._pair_warning_levels: Dict[str, WarningLevel] = {}

        # Stats
        self._total_warnings = 0
        self._warnings_by_category: Dict[str, int] = {}

    def update_order_book(
        self,
        pair: str,
        bids: List[Tuple[float, float]],  # [(price, size), ...]
        asks: List[Tuple[float, float]],
        timestamp: Optional[float] = None
    ) -> WarningLevel:
        """
        Update order book data and check for imbalances.

        Args:
            pair: Trading pair
            bids: List of (price, size) tuples, best bid first
            asks: List of (price, size) tuples, best ask first
            timestamp: Unix timestamp

        Returns:
            Current warning level for this pair
        """
        if not self.enabled:
            return WarningLevel.CLEAR

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        if pair not in self._order_book_history:
            self._order_book_history[pair] = deque(maxlen=100)

        # Calculate metrics
        bid_depth = sum(size for _, size in bids[:self.ob_depth_levels])
        ask_depth = sum(size for _, size in asks[:self.ob_depth_levels])

        if ask_depth > 0:
            imbalance_ratio = bid_depth / ask_depth
        else:
            imbalance_ratio = float('inf') if bid_depth > 0 else 1.0

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        if mid_price > 0:
            spread_pct = (best_ask - best_bid) / mid_price
        else:
            spread_pct = 0

        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            pair=pair,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance_ratio=imbalance_ratio,
            spread_pct=spread_pct,
            best_bid=best_bid,
            best_ask=best_ask
        )
        self._order_book_history[pair].append(snapshot)

        # Check for warnings
        self._check_order_book_warnings(pair, snapshot)
        self._check_spread_warnings(pair, snapshot)

        return self._update_pair_level(pair)

    def update_funding(
        self,
        pair: str,
        rate: float,
        timestamp: Optional[float] = None
    ) -> WarningLevel:
        """
        Update funding rate and check for extremes.

        Args:
            pair: Trading pair
            rate: Current funding rate (hourly, as decimal)
            timestamp: Unix timestamp

        Returns:
            Current warning level for this pair
        """
        if not self.enabled:
            return WarningLevel.CLEAR

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        if pair not in self._funding_history:
            self._funding_history[pair] = deque(maxlen=100)

        # Calculate metrics
        apr = abs(rate) * 8760 * 100  # Annualized percentage
        direction = "long_pays" if rate > 0 else "short_pays"

        # Crowding score (0-100 based on how extreme funding is)
        # 0.1% hourly = 50 crowding, 0.2% = 100 crowding
        crowding_score = min(100, abs(rate) / 0.002 * 100)

        snapshot = FundingSnapshot(
            timestamp=timestamp,
            pair=pair,
            rate=rate,
            apr=apr,
            direction=direction,
            crowding_score=crowding_score
        )
        self._funding_history[pair].append(snapshot)

        # Check for warnings
        self._check_funding_warnings(pair, snapshot)

        return self._update_pair_level(pair)

    def _check_order_book_warnings(self, pair: str, snapshot: OrderBookSnapshot):
        """Check order book for warning conditions."""
        category = "order_book"
        ratio = snapshot.imbalance_ratio
        inverse_ratio = 1 / ratio if ratio > 0 else float('inf')

        # Check both directions (heavy bids = potential pump, heavy asks = potential dump)
        max_imbalance = max(ratio, inverse_ratio)
        direction = "bid_heavy" if ratio > inverse_ratio else "ask_heavy"

        if max_imbalance >= self.ob_imbalance_danger:
            self._raise_warning(
                pair=pair,
                level=WarningLevel.DANGER,
                category=category,
                message=f"Order book severely imbalanced ({direction}): {max_imbalance:.1f}:1",
                data={"imbalance_ratio": max_imbalance, "direction": direction}
            )
        elif max_imbalance >= self.ob_imbalance_warning:
            self._raise_warning(
                pair=pair,
                level=WarningLevel.WARNING,
                category=category,
                message=f"Order book imbalanced ({direction}): {max_imbalance:.1f}:1",
                data={"imbalance_ratio": max_imbalance, "direction": direction}
            )
        else:
            self._clear_warning(pair, category)

    def _check_spread_warnings(self, pair: str, snapshot: OrderBookSnapshot):
        """Check spread for warning conditions."""
        category = "spread"
        spread = snapshot.spread_pct

        if spread >= self.spread_danger_pct:
            self._raise_warning(
                pair=pair,
                level=WarningLevel.DANGER,
                category=category,
                message=f"Spread dangerously wide: {spread*100:.3f}%",
                data={"spread_pct": spread}
            )
        elif spread >= self.spread_warning_pct:
            self._raise_warning(
                pair=pair,
                level=WarningLevel.WARNING,
                category=category,
                message=f"Spread elevated: {spread*100:.3f}%",
                data={"spread_pct": spread}
            )
        else:
            self._clear_warning(pair, category)

    def _check_funding_warnings(self, pair: str, snapshot: FundingSnapshot):
        """Check funding rate for warning conditions."""
        category = "funding"
        abs_rate = abs(snapshot.rate)

        if abs_rate >= self.funding_critical_rate:
            self._raise_warning(
                pair=pair,
                level=WarningLevel.CRITICAL,
                category=category,
                message=f"EXTREME funding rate: {snapshot.apr:.0f}% APR ({snapshot.direction})",
                data={
                    "rate": snapshot.rate,
                    "apr": snapshot.apr,
                    "direction": snapshot.direction,
                    "crowding_score": snapshot.crowding_score
                }
            )
        elif abs_rate >= self.funding_danger_rate:
            self._raise_warning(
                pair=pair,
                level=WarningLevel.DANGER,
                category=category,
                message=f"High funding rate: {snapshot.apr:.0f}% APR ({snapshot.direction})",
                data={
                    "rate": snapshot.rate,
                    "apr": snapshot.apr,
                    "direction": snapshot.direction,
                    "crowding_score": snapshot.crowding_score
                }
            )
        elif abs_rate >= self.funding_warning_rate:
            self._raise_warning(
                pair=pair,
                level=WarningLevel.WARNING,
                category=category,
                message=f"Elevated funding rate: {snapshot.apr:.0f}% APR ({snapshot.direction})",
                data={
                    "rate": snapshot.rate,
                    "apr": snapshot.apr,
                    "direction": snapshot.direction,
                    "crowding_score": snapshot.crowding_score
                }
            )
        else:
            self._clear_warning(pair, category)

    def _raise_warning(
        self,
        pair: str,
        level: WarningLevel,
        category: str,
        message: str,
        data: Dict
    ):
        """Raise or update a warning."""
        timestamp = datetime.now().timestamp()

        warning = Warning(
            timestamp=timestamp,
            pair=pair,
            level=level,
            category=category,
            message=message,
            data=data
        )

        # Initialize pair dict if needed
        if pair not in self._active_warnings:
            self._active_warnings[pair] = {}

        # Check if this is a new or upgraded warning
        existing = self._active_warnings[pair].get(category)
        is_new = existing is None
        is_upgrade = existing is not None and level.value > existing.level.value

        if is_new or is_upgrade:
            self._active_warnings[pair][category] = warning
            self._warning_history.append(warning)
            self._total_warnings += 1
            self._warnings_by_category[category] = self._warnings_by_category.get(category, 0) + 1

            log_msg = f"EARLY WARNING [{level.value.upper()}]: {pair} - {message}"
            if level in (WarningLevel.DANGER, WarningLevel.CRITICAL):
                self.logger.warning(log_msg)
            else:
                self.logger.info(log_msg)

            if self.on_warning:
                try:
                    self.on_warning(warning)
                except Exception as e:
                    self.logger.error(f"Error in warning callback: {e}")

    def _clear_warning(self, pair: str, category: str):
        """Clear a warning for a pair/category."""
        if pair in self._active_warnings and category in self._active_warnings[pair]:
            old_warning = self._active_warnings[pair][category]
            del self._active_warnings[pair][category]
            self.logger.debug(f"Warning cleared: {pair} {category} (was {old_warning.level.value})")

    def _update_pair_level(self, pair: str) -> WarningLevel:
        """Update aggregate warning level for a pair."""
        if pair not in self._active_warnings or not self._active_warnings[pair]:
            self._pair_warning_levels[pair] = WarningLevel.CLEAR
            return WarningLevel.CLEAR

        # Take highest warning level
        max_level = WarningLevel.CLEAR
        for warning in self._active_warnings[pair].values():
            if self._level_value(warning.level) > self._level_value(max_level):
                max_level = warning.level

        self._pair_warning_levels[pair] = max_level
        return max_level

    def _level_value(self, level: WarningLevel) -> int:
        """Get numeric value for warning level comparison."""
        values = {
            WarningLevel.CLEAR: 0,
            WarningLevel.INFO: 1,
            WarningLevel.WARNING: 2,
            WarningLevel.DANGER: 3,
            WarningLevel.CRITICAL: 4,
        }
        return values.get(level, 0)

    def get_warning_level(self, pair: str) -> WarningLevel:
        """Get current aggregate warning level for a pair."""
        return self._pair_warning_levels.get(pair, WarningLevel.CLEAR)

    def get_active_warnings(self, pair: Optional[str] = None) -> List[Warning]:
        """Get all active warnings, optionally filtered by pair."""
        warnings = []
        for p, categories in self._active_warnings.items():
            if pair is None or p == pair:
                warnings.extend(categories.values())
        return warnings

    def should_block_entry(self, pair: str) -> Tuple[bool, str]:
        """
        Check if new entries should be blocked for a pair.

        Args:
            pair: Trading pair

        Returns:
            Tuple of (should_block, reason)
        """
        level = self.get_warning_level(pair)

        if level == WarningLevel.CRITICAL:
            warnings = self.get_active_warnings(pair)
            reasons = [w.message for w in warnings if w.level == WarningLevel.CRITICAL]
            return True, f"CRITICAL: {'; '.join(reasons)}"

        if level == WarningLevel.DANGER:
            warnings = self.get_active_warnings(pair)
            reasons = [w.message for w in warnings if w.level == WarningLevel.DANGER]
            return True, f"DANGER: {'; '.join(reasons)}"

        return False, ""

    def get_entry_recommendation(self, pair: str) -> Tuple[str, float]:
        """
        Get entry recommendation based on warnings.

        Args:
            pair: Trading pair

        Returns:
            Tuple of (recommendation, position_size_multiplier)
            recommendation: "proceed", "caution", "avoid"
            multiplier: suggested position size multiplier (1.0 = normal, 0.5 = half, etc.)
        """
        level = self.get_warning_level(pair)

        if level == WarningLevel.CRITICAL:
            return "avoid", 0.0
        elif level == WarningLevel.DANGER:
            return "avoid", 0.0
        elif level == WarningLevel.WARNING:
            return "caution", 0.5
        elif level == WarningLevel.INFO:
            return "caution", 0.75
        else:
            return "proceed", 1.0

    def get_funding_crowding(self, pair: str) -> Optional[float]:
        """Get current funding crowding score for a pair (0-100)."""
        if pair in self._funding_history and self._funding_history[pair]:
            return self._funding_history[pair][-1].crowding_score
        return None

    def get_order_book_imbalance(self, pair: str) -> Optional[Tuple[float, str]]:
        """
        Get current order book imbalance.

        Returns:
            Tuple of (imbalance_ratio, direction) or None
            direction is "bid_heavy" or "ask_heavy"
        """
        if pair in self._order_book_history and self._order_book_history[pair]:
            snapshot = self._order_book_history[pair][-1]
            ratio = snapshot.imbalance_ratio
            direction = "bid_heavy" if ratio > 1 else "ask_heavy"
            return max(ratio, 1/ratio if ratio > 0 else 1), direction
        return None

    def get_status(self) -> Dict:
        """Get overall early warning system status."""
        pair_status = {}
        for pair in set(list(self._pair_warning_levels.keys()) +
                       list(self._active_warnings.keys())):
            level = self._pair_warning_levels.get(pair, WarningLevel.CLEAR)
            warnings = self._active_warnings.get(pair, {})
            pair_status[pair] = {
                "level": level.value,
                "active_warnings": len(warnings),
                "categories": list(warnings.keys()),
            }

        return {
            "enabled": self.enabled,
            "pairs_monitored": len(pair_status),
            "pair_status": pair_status,
            "stats": {
                "total_warnings": self._total_warnings,
                "by_category": self._warnings_by_category,
            },
            "thresholds": {
                "ob_imbalance_warning": self.ob_imbalance_warning,
                "ob_imbalance_danger": self.ob_imbalance_danger,
                "funding_warning_apr": self.funding_warning_rate * 8760 * 100,
                "funding_danger_apr": self.funding_danger_rate * 8760 * 100,
                "spread_warning_pct": self.spread_warning_pct * 100,
            }
        }

    # =========================================================================
    # PHASE 1 GOD MODE: ORDER BOOK VELOCITY
    # =========================================================================

    def get_order_book_velocity(self, pair: str) -> Optional[OrderBookVelocity]:
        """
        Calculate order book velocity - how fast the imbalance is changing.

        This detects whale activity:
        - Rapid bid accumulation = whale buying (velocity > 0.3, ratio increasing)
        - Rapid ask accumulation = whale selling (velocity < -0.3, ratio decreasing)
        - Sudden disappearance = iceberg order filled (|velocity| > 0.5 sudden change)

        Args:
            pair: Trading pair to analyze

        Returns:
            OrderBookVelocity or None if insufficient data
        """
        if pair not in self._order_book_history:
            return None

        history = self._order_book_history[pair]
        if len(history) < 3:
            return None

        # Get recent ratios
        ratios = [s.imbalance_ratio for s in history]
        current_ratio = ratios[-1]

        # Calculate velocity (rate of change)
        # Use last 3 snapshots for smoothing
        recent_ratios = ratios[-3:]
        if len(recent_ratios) < 2:
            return None

        velocity = recent_ratios[-1] - recent_ratios[-2]

        # Calculate acceleration (change in velocity)
        if len(recent_ratios) >= 3:
            prev_velocity = recent_ratios[-2] - recent_ratios[-3]
            acceleration = velocity - prev_velocity
        else:
            acceleration = 0.0

        # Thresholds
        WHALE_VELOCITY_THRESHOLD = 0.3       # Significant velocity
        ICEBERG_VELOCITY_THRESHOLD = 0.5     # Sudden large change
        HIGH_CONFIDENCE_VELOCITY = 0.5       # Very confident signal

        # Determine signal
        signal = "normal"
        confidence = 0.0
        interpretation = ""

        # Check for iceberg detection first (sudden large change in either direction)
        if abs(velocity) > ICEBERG_VELOCITY_THRESHOLD:
            signal = "iceberg_detected"
            confidence = min(95, abs(velocity) * 100)
            if velocity > 0:
                interpretation = (
                    f"ICEBERG DETECTED: Bid/ask ratio jumped {velocity:+.2f} suddenly. "
                    "Large hidden sell order likely filled - expect upward move."
                )
            else:
                interpretation = (
                    f"ICEBERG DETECTED: Bid/ask ratio dropped {velocity:+.2f} suddenly. "
                    "Large hidden buy order likely filled - expect downward move."
                )

            # Raise warning for iceberg
            self._raise_warning(
                pair=pair,
                level=WarningLevel.WARNING,
                category="order_book_velocity",
                message=f"Iceberg order detected: velocity {velocity:+.2f}",
                data={"velocity": velocity, "signal": signal}
            )

        elif velocity > WHALE_VELOCITY_THRESHOLD and current_ratio > 1.5:
            signal = "whale_buying"
            confidence = min(90, velocity * 150)
            interpretation = (
                f"WHALE BUYING: Bid depth accumulating rapidly (velocity: {velocity:+.2f}). "
                f"Current ratio {current_ratio:.2f}:1 bid-heavy. "
                "Large buyer loading up - potential pump incoming."
            )

            if velocity > HIGH_CONFIDENCE_VELOCITY:
                self._raise_warning(
                    pair=pair,
                    level=WarningLevel.INFO,
                    category="order_book_velocity",
                    message=f"Whale buying detected: velocity {velocity:+.2f}, ratio {current_ratio:.1f}:1",
                    data={"velocity": velocity, "ratio": current_ratio, "signal": signal}
                )

        elif velocity < -WHALE_VELOCITY_THRESHOLD and current_ratio < 0.67:
            signal = "whale_selling"
            confidence = min(90, abs(velocity) * 150)
            interpretation = (
                f"WHALE SELLING: Ask depth accumulating rapidly (velocity: {velocity:+.2f}). "
                f"Current ratio {current_ratio:.2f}:1 ask-heavy. "
                "Large seller loading up - potential dump incoming."
            )

            if velocity < -HIGH_CONFIDENCE_VELOCITY:
                self._raise_warning(
                    pair=pair,
                    level=WarningLevel.INFO,
                    category="order_book_velocity",
                    message=f"Whale selling detected: velocity {velocity:+.2f}, ratio {current_ratio:.1f}:1",
                    data={"velocity": velocity, "ratio": current_ratio, "signal": signal}
                )

        else:
            signal = "normal"
            confidence = 100 - min(100, abs(velocity) * 200)  # Higher confidence when stable
            interpretation = (
                f"Order book stable. Velocity: {velocity:+.2f}, Ratio: {current_ratio:.2f}:1. "
                "No significant whale activity detected."
            )
            # Clear any existing velocity warning
            self._clear_warning(pair, "order_book_velocity")

        return OrderBookVelocity(
            pair=pair,
            current_ratio=current_ratio,
            velocity=velocity,
            acceleration=acceleration,
            signal=signal,
            confidence=confidence,
            interpretation=interpretation,
        )

    def get_all_velocities(self) -> Dict[str, OrderBookVelocity]:
        """
        Get order book velocity for all tracked pairs.

        Returns:
            Dict of pair -> OrderBookVelocity
        """
        results = {}
        for pair in self._order_book_history.keys():
            velocity = self.get_order_book_velocity(pair)
            if velocity:
                results[pair] = velocity
        return results

    def should_block_entry_velocity(self, pair: str, direction: str) -> Tuple[bool, str]:
        """
        Check if entry should be blocked based on order book velocity.

        Args:
            pair: Trading pair
            direction: "long" or "short"

        Returns:
            Tuple of (should_block, reason)
        """
        velocity = self.get_order_book_velocity(pair)
        if velocity is None:
            return False, ""

        # Block based on whale activity
        if direction == "long":
            if velocity.signal == "whale_selling" and velocity.confidence > 70:
                return True, (
                    f"WHALE SELLING DETECTED: Order book velocity {velocity.velocity:+.2f}. "
                    "Don't buy into whale distribution."
                )
            if velocity.signal == "iceberg_detected" and velocity.velocity < 0:
                return True, (
                    f"ICEBERG SELL DETECTED: Large hidden buy filled. "
                    "Wait for dust to settle before entering long."
                )

        elif direction == "short":
            if velocity.signal == "whale_buying" and velocity.confidence > 70:
                return True, (
                    f"WHALE BUYING DETECTED: Order book velocity {velocity.velocity:+.2f}. "
                    "Don't short into whale accumulation."
                )
            if velocity.signal == "iceberg_detected" and velocity.velocity > 0:
                return True, (
                    f"ICEBERG BUY DETECTED: Large hidden sell filled. "
                    "Wait for dust to settle before entering short."
                )

        return False, ""
