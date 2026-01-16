"""
Grid Protection - Fill rate monitoring for grid trading strategy.

Detects when grid orders are being filled too rapidly on one side,
which indicates a strong directional move (whale dump/pump) that
could accumulate a dangerous position.

Example scenario:
- Price dumps rapidly
- All buy grid orders get filled in sequence
- You end up max long at the bottom
- Protection: Detect rapid one-sided fills and pause grid
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional


class GridProtectionState(Enum):
    """Grid protection states."""
    NORMAL = "normal"           # Grid operating normally
    WARNING = "warning"         # Elevated fill rate, monitoring closely
    PAUSED = "paused"           # Grid paused due to one-sided fills
    COOLDOWN = "cooldown"       # Recovering, waiting to resume


@dataclass
class GridFill:
    """Record of a grid order fill."""
    timestamp: float
    pair: str
    side: str  # "buy" or "sell"
    price: float
    amount: float
    order_id: str


@dataclass
class GridProtectionTrigger:
    """Record of a grid protection trigger."""
    timestamp: float
    pair: str
    buy_fills: int
    sell_fills: int
    window_seconds: int
    reason: str


class GridProtection:
    """
    Grid Protection System.

    Monitors grid order fills and detects dangerous one-sided accumulation.
    When too many fills happen on one side without balancing fills on the other,
    it indicates the market is moving strongly against the grid.

    Features:
    - Track fills per side over rolling windows
    - Configurable imbalance thresholds
    - Automatic pause and cooldown
    - Extensive logging
    """

    def __init__(
        self,
        # Thresholds
        max_one_sided_fills: int = 4,           # Max fills on one side without opposite
        imbalance_ratio_threshold: float = 3.0,  # Ratio of buy:sell (or sell:buy) to trigger
        window_seconds: int = 120,               # Time window for fill counting
        # Timing
        cooldown_seconds: int = 180,             # 3 min cooldown after trigger
        # Callbacks
        on_pause: Optional[Callable[[], None]] = None,
        on_resume: Optional[Callable[[], None]] = None,
        # Logging
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
    ):
        """
        Initialize grid protection.

        Args:
            max_one_sided_fills: Max consecutive fills on one side before pause
            imbalance_ratio_threshold: Fill ratio that triggers pause (e.g., 3.0 = 3:1)
            window_seconds: Time window for counting fills
            cooldown_seconds: Cooldown period after trigger
            on_pause: Callback when grid is paused
            on_resume: Callback when grid resumes
            logger: Logger instance
            enabled: Whether protection is active
        """
        self.enabled = enabled
        self.max_one_sided_fills = max_one_sided_fills
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds

        self.on_pause = on_pause
        self.on_resume = on_resume
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.state = GridProtectionState.NORMAL
        self._fill_history: Dict[str, Deque[GridFill]] = {}  # pair -> deque of fills
        self._trigger_history: List[GridProtectionTrigger] = []
        self._cooldown_end_time: float = 0

        # Stats
        self._total_triggers = 0
        self._total_fills_tracked = 0

    def record_fill(
        self,
        pair: str,
        side: str,
        price: float,
        amount: float,
        order_id: str,
        timestamp: Optional[float] = None
    ) -> GridProtectionState:
        """
        Record a grid order fill and check for protection triggers.

        Args:
            pair: Trading pair
            side: "buy" or "sell"
            price: Fill price
            amount: Fill amount
            order_id: Order ID
            timestamp: Unix timestamp (defaults to now)

        Returns:
            Current protection state
        """
        if not self.enabled:
            return GridProtectionState.NORMAL

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Initialize history for new pairs
        if pair not in self._fill_history:
            self._fill_history[pair] = deque(maxlen=100)

        # Record the fill
        fill = GridFill(
            timestamp=timestamp,
            pair=pair,
            side=side.lower(),
            price=price,
            amount=amount,
            order_id=order_id
        )
        self._fill_history[pair].append(fill)
        self._total_fills_tracked += 1

        # Prune old fills
        self._prune_history(pair, timestamp)

        # Check cooldown
        if self.state == GridProtectionState.COOLDOWN:
            if timestamp >= self._cooldown_end_time:
                self._resume()
            return self.state

        # Skip checks if already paused
        if self.state == GridProtectionState.PAUSED:
            return self.state

        # Check for triggers
        should_pause, reason = self._check_triggers(pair, timestamp)

        if should_pause:
            self._trigger_pause(pair, reason, timestamp)

        return self.state

    def _check_triggers(self, pair: str, current_time: float) -> tuple:
        """
        Check if grid should be paused based on fill patterns.

        Returns:
            Tuple of (should_pause: bool, reason: str)
        """
        if pair not in self._fill_history:
            return False, ""

        fills = self._fill_history[pair]
        if len(fills) < 2:
            return False, ""

        # Count fills by side in window
        window_start = current_time - self.window_seconds
        buy_fills = 0
        sell_fills = 0
        recent_fills = []

        for fill in fills:
            if fill.timestamp >= window_start:
                recent_fills.append(fill)
                if fill.side == "buy":
                    buy_fills += 1
                else:
                    sell_fills += 1

        if len(recent_fills) < 2:
            return False, ""

        # Check 1: Max one-sided fills (consecutive)
        # IMPORTANT: Only trigger on one-sided BUYS (accumulating into weakness)
        # One-sided SELLS are GOOD - we're taking profit into strength
        consecutive_same_side = 1
        last_side = recent_fills[-1].side
        for fill in reversed(list(recent_fills)[:-1]):
            if fill.side == last_side:
                consecutive_same_side += 1
            else:
                break

        if consecutive_same_side >= self.max_one_sided_fills:
            if last_side == "buy":
                # One-sided BUYS = accumulating into weakness = DANGEROUS
                reason = (
                    f"{consecutive_same_side} consecutive BUY fills - "
                    f"accumulating into price dump (dangerous)"
                )
                return True, reason
            else:
                # One-sided SELLS = taking profit into strength = GOOD
                # Log it but don't pause
                self.logger.info(
                    f"GRID PROTECTION: {consecutive_same_side} consecutive SELL fills - "
                    f"taking profit into price pump (this is GOOD, not pausing)"
                )

        # Check 2: Imbalance ratio
        # IMPORTANT: Only trigger on BUY-heavy imbalance (accumulating into weakness)
        # SELL-heavy imbalance is GOOD - we're taking profit into strength
        total_fills = buy_fills + sell_fills
        if total_fills >= 4:  # Need enough fills to judge ratio
            if sell_fills > 0 and buy_fills / sell_fills >= self.imbalance_ratio_threshold:
                # BUY-heavy = accumulating into price dump = DANGEROUS
                reason = (
                    f"Fill imbalance: {buy_fills} buys vs {sell_fills} sells "
                    f"(ratio {buy_fills/sell_fills:.1f}:1) in {self.window_seconds}s - "
                    f"accumulating into price dump (dangerous)"
                )
                return True, reason

            if buy_fills > 0 and sell_fills / buy_fills >= self.imbalance_ratio_threshold:
                # SELL-heavy = taking profit into price pump = GOOD
                # Log it but don't pause
                self.logger.info(
                    f"GRID PROTECTION: Sell-heavy fills ({sell_fills}S vs {buy_fills}B) - "
                    f"taking profit into price pump (this is GOOD, not pausing)"
                )

        # Check for warning state (approaching threshold)
        # Only warn on BUY-side patterns (accumulating is the danger)
        if consecutive_same_side >= self.max_one_sided_fills - 1 and last_side == "buy":
            if self.state != GridProtectionState.WARNING:
                self.state = GridProtectionState.WARNING
                self.logger.warning(
                    f"GRID PROTECTION WARNING: {consecutive_same_side} consecutive "
                    f"BUY fills on {pair} - approaching accumulation threshold"
                )

        return False, ""

    def _trigger_pause(self, pair: str, reason: str, timestamp: float):
        """Trigger grid pause."""
        self.state = GridProtectionState.PAUSED
        self._cooldown_end_time = timestamp + self.cooldown_seconds
        self._total_triggers += 1

        # Get fill counts for logging
        fills = self._fill_history.get(pair, deque())
        window_start = timestamp - self.window_seconds
        buy_fills = sum(1 for f in fills if f.timestamp >= window_start and f.side == "buy")
        sell_fills = sum(1 for f in fills if f.timestamp >= window_start and f.side == "sell")

        trigger = GridProtectionTrigger(
            timestamp=timestamp,
            pair=pair,
            buy_fills=buy_fills,
            sell_fills=sell_fills,
            window_seconds=self.window_seconds,
            reason=reason
        )
        self._trigger_history.append(trigger)

        self.logger.error(
            f"GRID PROTECTION TRIGGERED - PAUSING GRID\n"
            f"  Pair: {pair}\n"
            f"  Reason: {reason}\n"
            f"  Fills in window: {buy_fills} buys, {sell_fills} sells\n"
            f"  Action: Cancelling grid orders, no new grid orders\n"
            f"  Cooldown: {self.cooldown_seconds}s"
        )

        if self.on_pause:
            try:
                self.on_pause()
            except Exception as e:
                self.logger.error(f"Error in grid protection on_pause callback: {e}")

    def _resume(self):
        """Resume normal grid operation."""
        previous_state = self.state
        self.state = GridProtectionState.NORMAL

        self.logger.info(
            f"GRID PROTECTION RESUMED - Cooldown complete\n"
            f"  Previous state: {previous_state.value}\n"
            f"  Total triggers this session: {self._total_triggers}"
        )

        if self.on_resume:
            try:
                self.on_resume()
            except Exception as e:
                self.logger.error(f"Error in grid protection on_resume callback: {e}")

    def _prune_history(self, pair: str, current_time: float):
        """Remove old fills from history."""
        if pair not in self._fill_history:
            return

        cutoff = current_time - (self.window_seconds * 2)  # Keep 2x window for analysis
        history = self._fill_history[pair]

        while history and history[0].timestamp < cutoff:
            history.popleft()

    def check_state(self, timestamp: Optional[float] = None) -> GridProtectionState:
        """
        Check current state and handle cooldown expiry.

        Args:
            timestamp: Current timestamp (defaults to now)

        Returns:
            Current protection state
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        if self.state == GridProtectionState.COOLDOWN:
            if timestamp >= self._cooldown_end_time:
                self._resume()

        return self.state

    def force_resume(self):
        """Force immediate resume (manual override)."""
        self._resume()

    def is_paused(self) -> bool:
        """Check if grid is paused."""
        return self.state in (GridProtectionState.PAUSED, GridProtectionState.COOLDOWN)

    def should_block_new_orders(self) -> bool:
        """Check if new grid orders should be blocked."""
        return self.state in (GridProtectionState.PAUSED, GridProtectionState.COOLDOWN)

    def get_fill_stats(self, pair: str, window_seconds: Optional[int] = None) -> Dict:
        """
        Get fill statistics for a pair.

        Args:
            pair: Trading pair
            window_seconds: Time window (defaults to configured window)

        Returns:
            Dict with buy_fills, sell_fills, total_fills, imbalance_ratio
        """
        if window_seconds is None:
            window_seconds = self.window_seconds

        if pair not in self._fill_history:
            return {
                "buy_fills": 0,
                "sell_fills": 0,
                "total_fills": 0,
                "imbalance_ratio": 1.0,
                "dominant_side": "neutral"
            }

        current_time = datetime.now().timestamp()
        window_start = current_time - window_seconds
        fills = self._fill_history[pair]

        buy_fills = sum(1 for f in fills if f.timestamp >= window_start and f.side == "buy")
        sell_fills = sum(1 for f in fills if f.timestamp >= window_start and f.side == "sell")
        total_fills = buy_fills + sell_fills

        if sell_fills > 0 and buy_fills > sell_fills:
            ratio = buy_fills / sell_fills
            dominant = "buy"
        elif buy_fills > 0 and sell_fills > buy_fills:
            ratio = sell_fills / buy_fills
            dominant = "sell"
        else:
            ratio = 1.0
            dominant = "neutral"

        return {
            "buy_fills": buy_fills,
            "sell_fills": sell_fills,
            "total_fills": total_fills,
            "imbalance_ratio": ratio,
            "dominant_side": dominant
        }

    def get_status(self) -> Dict:
        """Get current grid protection status."""
        return {
            "state": self.state.value,
            "enabled": self.enabled,
            "total_triggers": self._total_triggers,
            "total_fills_tracked": self._total_fills_tracked,
            "cooldown_remaining": max(0, self._cooldown_end_time - datetime.now().timestamp())
            if self.state == GridProtectionState.COOLDOWN else 0,
            "pairs_monitored": len(self._fill_history),
            "last_trigger": self._trigger_history[-1].__dict__ if self._trigger_history else None,
            "config": {
                "max_one_sided_fills": self.max_one_sided_fills,
                "imbalance_ratio_threshold": self.imbalance_ratio_threshold,
                "window_seconds": self.window_seconds,
                "cooldown_seconds": self.cooldown_seconds,
            }
        }
