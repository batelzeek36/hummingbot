"""
Circuit Breaker - Rapid price movement detection and emergency response.

Monitors price changes over short time windows and triggers protective actions
when movements exceed configured thresholds.

Actions:
- PAUSE: Cancel all orders, hold existing positions, stop new entries
- FLATTEN: Cancel all orders AND close all positions immediately

Conservative defaults are set to avoid false triggers during normal volatility.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    NORMAL = "normal"           # All systems go
    ALERT = "alert"             # Warning level - increased monitoring
    PAUSED = "paused"           # Orders cancelled, positions held
    FLATTENED = "flattened"     # All positions closed
    COOLDOWN = "cooldown"       # Recovering from trigger, waiting to resume


@dataclass
class PricePoint:
    """A single price observation."""
    timestamp: float  # Unix timestamp
    price: float
    pair: str


@dataclass
class CircuitBreakerTrigger:
    """Record of a circuit breaker trigger event."""
    timestamp: float
    pair: str
    price_change_pct: float
    window_seconds: int
    action: str
    reason: str


class CircuitBreaker:
    """
    Circuit Breaker for whale movement protection.

    Tracks price movements across multiple time windows and triggers
    protective actions when thresholds are exceeded.

    Features:
    - Multi-window monitoring (30s, 60s, 120s)
    - Per-pair tracking
    - Configurable pause vs flatten thresholds
    - Cooldown period before resuming
    - Extensive logging for debugging
    """

    def __init__(
        self,
        # Thresholds (as decimals, e.g., 0.03 = 3%)
        pause_threshold_30s: float = 0.025,     # 2.5% in 30s triggers pause
        pause_threshold_60s: float = 0.035,     # 3.5% in 60s triggers pause
        pause_threshold_120s: float = 0.05,     # 5% in 2min triggers pause
        flatten_threshold_30s: float = 0.05,    # 5% in 30s triggers flatten
        flatten_threshold_60s: float = 0.07,    # 7% in 60s triggers flatten
        flatten_threshold_120s: float = 0.10,   # 10% in 2min triggers flatten
        # Timing
        cooldown_seconds: int = 300,            # 5 min cooldown after trigger
        max_history_seconds: int = 180,         # Keep 3 min of price history
        # Callbacks
        on_pause: Optional[Callable[[], None]] = None,
        on_flatten: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
        # Logging
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
    ):
        """
        Initialize the circuit breaker.

        Args:
            pause_threshold_*: Price change % that triggers PAUSE for each window
            flatten_threshold_*: Price change % that triggers FLATTEN for each window
            cooldown_seconds: Time to wait before resuming after trigger
            max_history_seconds: How much price history to keep
            on_pause: Callback when pause is triggered
            on_flatten: Callback when flatten is triggered
            on_resume: Callback when resuming from cooldown
            logger: Logger instance
            enabled: Whether circuit breaker is active
        """
        self.enabled = enabled

        # Thresholds by window (seconds -> threshold)
        self.pause_thresholds = {
            30: pause_threshold_30s,
            60: pause_threshold_60s,
            120: pause_threshold_120s,
        }
        self.flatten_thresholds = {
            30: flatten_threshold_30s,
            60: flatten_threshold_60s,
            120: flatten_threshold_120s,
        }

        self.cooldown_seconds = cooldown_seconds
        self.max_history_seconds = max_history_seconds

        # Callbacks
        self.on_pause = on_pause
        self.on_flatten = on_flatten
        self.on_resume = on_resume

        self.logger = logger or logging.getLogger(__name__)

        # State
        self.state = CircuitBreakerState.NORMAL
        self._price_history: Dict[str, Deque[PricePoint]] = {}  # pair -> deque of prices
        self._trigger_history: List[CircuitBreakerTrigger] = []
        self._last_trigger_time: float = 0
        self._cooldown_end_time: float = 0

        # Position tracking - only trigger for ADVERSE moves
        # Key: pair, Value: "long" or "short"
        self._positions: Dict[str, str] = {}

        # Extreme chaos threshold - trigger regardless of position direction
        # If market moves this much, something is very wrong
        self._extreme_threshold = 0.08  # 8% move = chaos, trigger regardless

        # Stats
        self._total_triggers = 0
        self._pause_triggers = 0
        self._flatten_triggers = 0

    def register_position(self, pair: str, direction: str):
        """
        Register that we have a position in a pair.
        Circuit breaker will only trigger for ADVERSE moves against this position.

        Args:
            pair: Trading pair
            direction: "long" or "short"
        """
        self._positions[pair] = direction.lower()
        self.logger.debug(f"Circuit breaker registered {direction} position on {pair}")

    def clear_position(self, pair: str):
        """
        Clear position registration for a pair.
        Without a registered position, any large move can trigger.
        """
        if pair in self._positions:
            del self._positions[pair]
            self.logger.debug(f"Circuit breaker cleared position on {pair}")

    def has_any_position(self) -> bool:
        """Check if any positions are registered."""
        return len(self._positions) > 0

    def _is_adverse_move(self, pair: str, price_change_pct: float) -> bool:
        """
        Check if a price move is adverse to our position.

        Args:
            pair: Trading pair
            price_change_pct: Price change as decimal (negative = down, positive = up)

        Returns:
            True if move is against our position OR we have no position
        """
        if pair not in self._positions:
            # No position registered - be conservative, treat as adverse
            return True

        direction = self._positions[pair]

        if direction == "long":
            # Long position - DOWN moves are adverse
            return price_change_pct < 0
        elif direction == "short":
            # Short position - UP moves are adverse
            return price_change_pct > 0

        return True  # Unknown direction, be conservative

    def update_price(self, pair: str, price: float, timestamp: float) -> CircuitBreakerState:
        """
        Update price for a trading pair and check for triggers.

        Args:
            pair: Trading pair (e.g., "BTC-USD")
            price: Current price
            timestamp: Unix timestamp

        Returns:
            Current circuit breaker state
        """
        if not self.enabled:
            return CircuitBreakerState.NORMAL

        # Initialize history for new pairs
        if pair not in self._price_history:
            self._price_history[pair] = deque(maxlen=1000)

        # Add price point
        self._price_history[pair].append(PricePoint(
            timestamp=timestamp,
            price=price,
            pair=pair
        ))

        # Prune old history
        self._prune_history(pair, timestamp)

        # Check cooldown
        if self.state == CircuitBreakerState.COOLDOWN:
            if timestamp >= self._cooldown_end_time:
                self._resume()
            return self.state

        # Skip checks if already triggered
        if self.state in (CircuitBreakerState.PAUSED, CircuitBreakerState.FLATTENED):
            return self.state

        # Check for triggers across all windows
        trigger_action, trigger_reason, trigger_pct, trigger_window = self._check_triggers(pair, timestamp)

        if trigger_action == "flatten":
            self._trigger_flatten(pair, trigger_pct, trigger_window, trigger_reason)
        elif trigger_action == "pause":
            self._trigger_pause(pair, trigger_pct, trigger_window, trigger_reason)
        elif trigger_action == "alert":
            if self.state != CircuitBreakerState.ALERT:
                self.state = CircuitBreakerState.ALERT
                self.logger.warning(f"CIRCUIT BREAKER ALERT: {pair} {trigger_reason}")

        return self.state

    def _check_triggers(
        self,
        pair: str,
        current_time: float
    ) -> Tuple[Optional[str], str, float, int]:
        """
        Check all windows for trigger conditions.

        Returns:
            Tuple of (action, reason, change_pct, window_seconds)
            action is "flatten", "pause", "alert", or None
        """
        if pair not in self._price_history or len(self._price_history[pair]) < 2:
            return None, "", 0.0, 0

        history = self._price_history[pair]
        current_price = history[-1].price

        max_action = None
        max_reason = ""
        max_pct = 0.0
        max_window = 0

        # Check each time window
        for window_seconds in [30, 60, 120]:
            window_start = current_time - window_seconds

            # Find oldest price in window
            oldest_price = None
            for point in history:
                if point.timestamp >= window_start:
                    oldest_price = point.price
                    break

            if oldest_price is None or oldest_price == 0:
                continue

            # Calculate price change
            price_change = (current_price - oldest_price) / oldest_price
            abs_change = abs(price_change)

            # Determine if this move is adverse to our position
            is_adverse = self._is_adverse_move(pair, price_change)
            is_extreme = abs_change >= self._extreme_threshold  # 8%+ = chaos

            # Only trigger for:
            # 1. ADVERSE moves (against our position), OR
            # 2. EXTREME moves (>8%, something is very wrong)
            if not is_adverse and not is_extreme:
                # Move is favorable to our position and not extreme - let it ride!
                continue

            # Check flatten threshold first (more severe)
            flatten_threshold = self.flatten_thresholds.get(window_seconds, 0.10)
            if abs_change >= flatten_threshold:
                direction = "DUMP" if price_change < 0 else "PUMP"
                adverse_str = " (ADVERSE)" if is_adverse else " (EXTREME CHAOS)"
                reason = f"{abs_change*100:.2f}% {direction} in {window_seconds}s{adverse_str} (flatten threshold: {flatten_threshold*100:.1f}%)"
                if max_action != "flatten" or abs_change > max_pct:
                    max_action = "flatten"
                    max_reason = reason
                    max_pct = abs_change
                    max_window = window_seconds
                continue

            # Check pause threshold
            pause_threshold = self.pause_thresholds.get(window_seconds, 0.05)
            if abs_change >= pause_threshold:
                direction = "DUMP" if price_change < 0 else "PUMP"
                adverse_str = " (ADVERSE)" if is_adverse else " (EXTREME CHAOS)"
                reason = f"{abs_change*100:.2f}% {direction} in {window_seconds}s{adverse_str} (pause threshold: {pause_threshold*100:.1f}%)"
                if max_action is None or (max_action != "flatten" and abs_change > max_pct):
                    max_action = "pause"
                    max_reason = reason
                    max_pct = abs_change
                    max_window = window_seconds
                continue

            # Check alert threshold (80% of pause)
            alert_threshold = pause_threshold * 0.8
            if abs_change >= alert_threshold:
                direction = "down" if price_change < 0 else "up"
                reason = f"{abs_change*100:.2f}% {direction} in {window_seconds}s (approaching pause threshold)"
                if max_action is None:
                    max_action = "alert"
                    max_reason = reason
                    max_pct = abs_change
                    max_window = window_seconds

        return max_action, max_reason, max_pct, max_window

    def _trigger_pause(self, pair: str, change_pct: float, window: int, reason: str):
        """Trigger PAUSE state."""
        self.state = CircuitBreakerState.PAUSED
        self._last_trigger_time = datetime.now().timestamp()
        self._cooldown_end_time = self._last_trigger_time + self.cooldown_seconds
        self._total_triggers += 1
        self._pause_triggers += 1

        trigger = CircuitBreakerTrigger(
            timestamp=self._last_trigger_time,
            pair=pair,
            price_change_pct=change_pct,
            window_seconds=window,
            action="pause",
            reason=reason
        )
        self._trigger_history.append(trigger)

        self.logger.error(
            f"CIRCUIT BREAKER TRIGGERED - PAUSE\n"
            f"  Pair: {pair}\n"
            f"  Reason: {reason}\n"
            f"  Action: Cancelling orders, holding positions\n"
            f"  Cooldown: {self.cooldown_seconds}s"
        )

        if self.on_pause:
            try:
                self.on_pause()
            except Exception as e:
                self.logger.error(f"Error in on_pause callback: {e}")

    def _trigger_flatten(self, pair: str, change_pct: float, window: int, reason: str):
        """Trigger FLATTEN state."""
        self.state = CircuitBreakerState.FLATTENED
        self._last_trigger_time = datetime.now().timestamp()
        self._cooldown_end_time = self._last_trigger_time + self.cooldown_seconds
        self._total_triggers += 1
        self._flatten_triggers += 1

        trigger = CircuitBreakerTrigger(
            timestamp=self._last_trigger_time,
            pair=pair,
            price_change_pct=change_pct,
            window_seconds=window,
            action="flatten",
            reason=reason
        )
        self._trigger_history.append(trigger)

        self.logger.error(
            f"CIRCUIT BREAKER TRIGGERED - FLATTEN\n"
            f"  Pair: {pair}\n"
            f"  Reason: {reason}\n"
            f"  Action: Cancelling ALL orders AND closing ALL positions\n"
            f"  Cooldown: {self.cooldown_seconds}s"
        )

        if self.on_flatten:
            try:
                self.on_flatten()
            except Exception as e:
                self.logger.error(f"Error in on_flatten callback: {e}")

    def _resume(self):
        """Resume normal operation after cooldown."""
        previous_state = self.state
        self.state = CircuitBreakerState.NORMAL

        self.logger.info(
            f"CIRCUIT BREAKER RESUMED - Cooldown complete\n"
            f"  Previous state: {previous_state.value}\n"
            f"  Total triggers this session: {self._total_triggers}"
        )

        if self.on_resume:
            try:
                self.on_resume()
            except Exception as e:
                self.logger.error(f"Error in on_resume callback: {e}")

    def _prune_history(self, pair: str, current_time: float):
        """Remove old price points from history."""
        if pair not in self._price_history:
            return

        cutoff = current_time - self.max_history_seconds
        history = self._price_history[pair]

        while history and history[0].timestamp < cutoff:
            history.popleft()

    def start_cooldown(self):
        """Manually start cooldown (for external triggers)."""
        self.state = CircuitBreakerState.COOLDOWN
        self._cooldown_end_time = datetime.now().timestamp() + self.cooldown_seconds
        self.logger.info(f"Circuit breaker entering cooldown for {self.cooldown_seconds}s")

    def force_resume(self):
        """Force immediate resume (manual override)."""
        self._resume()

    def is_triggered(self) -> bool:
        """Check if circuit breaker is in a triggered state."""
        return self.state in (
            CircuitBreakerState.PAUSED,
            CircuitBreakerState.FLATTENED,
            CircuitBreakerState.COOLDOWN
        )

    def should_block_new_orders(self) -> bool:
        """Check if new orders should be blocked."""
        return self.state in (
            CircuitBreakerState.PAUSED,
            CircuitBreakerState.FLATTENED,
            CircuitBreakerState.COOLDOWN
        )

    def should_flatten_positions(self) -> bool:
        """Check if positions should be flattened."""
        return self.state == CircuitBreakerState.FLATTENED

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            "state": self.state.value,
            "enabled": self.enabled,
            "total_triggers": self._total_triggers,
            "pause_triggers": self._pause_triggers,
            "flatten_triggers": self._flatten_triggers,
            "cooldown_remaining": max(0, self._cooldown_end_time - datetime.now().timestamp())
            if self.state == CircuitBreakerState.COOLDOWN else 0,
            "pairs_monitored": len(self._price_history),
            "last_trigger": self._trigger_history[-1].__dict__ if self._trigger_history else None,
        }

    def get_price_change(self, pair: str, window_seconds: int) -> Optional[float]:
        """
        Get price change for a pair over a specific window.

        Args:
            pair: Trading pair
            window_seconds: Time window in seconds

        Returns:
            Price change as decimal (e.g., -0.02 = -2%), or None if insufficient data
        """
        if pair not in self._price_history or len(self._price_history[pair]) < 2:
            return None

        history = self._price_history[pair]
        current_time = history[-1].timestamp
        current_price = history[-1].price
        window_start = current_time - window_seconds

        oldest_price = None
        for point in history:
            if point.timestamp >= window_start:
                oldest_price = point.price
                break

        if oldest_price is None or oldest_price == 0:
            return None

        return (current_price - oldest_price) / oldest_price
