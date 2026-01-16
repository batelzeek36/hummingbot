"""
Trailing Stop Manager - Dynamic stop loss that follows profitable moves.

Instead of a static stop loss, the trailing stop moves in the direction
of profit, locking in gains while still allowing the position to run.

Example for a LONG position:
- Entry: $100, Initial SL: $98.50 (1.5%)
- Price hits $105 -> SL moves to $103 (trailing 2% below peak)
- Price hits $110 -> SL moves to $107.80
- Price drops to $107.80 -> EXIT (locked in $7.80 profit instead of $1.50 loss)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Dict, Optional, Tuple


class TrailingStopMode(Enum):
    """Trailing stop behavior modes."""
    DISABLED = "disabled"           # No trailing, use static SL
    PERCENTAGE = "percentage"       # Trail by fixed percentage
    ATR_BASED = "atr_based"         # Trail by ATR multiple (volatility-aware)
    BREAKEVEN_THEN_TRAIL = "breakeven_then_trail"  # Move to breakeven first, then trail


@dataclass
class TrailingStopState:
    """State of a trailing stop for a position."""
    pair: str
    direction: str  # "long" or "short"
    entry_price: Decimal
    initial_stop: Decimal
    current_stop: Decimal
    peak_price: Decimal  # Highest (long) or lowest (short) price since entry
    stop_distance_pct: Decimal  # Current trailing distance as percentage
    is_breakeven: bool  # Has stop moved to breakeven?
    is_trailing: bool   # Is stop actively trailing?
    last_update: float


class TrailingStopManager:
    """
    Manages trailing stops for momentum positions.

    Features:
    - Multiple trailing modes
    - Breakeven-first option (move to breakeven, then trail)
    - Configurable activation threshold
    - Per-position tracking
    - Volatility-aware trailing distance
    """

    def __init__(
        self,
        # Core settings
        mode: TrailingStopMode = TrailingStopMode.BREAKEVEN_THEN_TRAIL,
        trail_distance_pct: Decimal = Decimal("0.015"),      # 1.5% trail distance
        activation_profit_pct: Decimal = Decimal("0.01"),    # Activate after 1% profit
        breakeven_activation_pct: Decimal = Decimal("0.008"), # Move to breakeven at 0.8% profit
        # Advanced
        min_trail_distance_pct: Decimal = Decimal("0.008"),  # 0.8% minimum trail
        max_trail_distance_pct: Decimal = Decimal("0.03"),   # 3% maximum trail
        tighten_at_profit_pct: Optional[Decimal] = Decimal("0.025"),  # Tighten trail at 2.5% profit
        tightened_trail_pct: Decimal = Decimal("0.01"),      # Tighten to 1% trail
        # Logging
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
    ):
        """
        Initialize the trailing stop manager.

        Args:
            mode: Trailing stop behavior mode
            trail_distance_pct: Percentage distance for trailing stop
            activation_profit_pct: Minimum profit before trailing activates
            breakeven_activation_pct: Profit level to move stop to breakeven
            min_trail_distance_pct: Minimum trailing distance
            max_trail_distance_pct: Maximum trailing distance
            tighten_at_profit_pct: Profit level to tighten the trail (None to disable)
            tightened_trail_pct: Tightened trail distance
            logger: Logger instance
            enabled: Whether trailing stops are active
        """
        self.enabled = enabled
        self.mode = mode
        self.trail_distance_pct = trail_distance_pct
        self.activation_profit_pct = activation_profit_pct
        self.breakeven_activation_pct = breakeven_activation_pct
        self.min_trail_distance_pct = min_trail_distance_pct
        self.max_trail_distance_pct = max_trail_distance_pct
        self.tighten_at_profit_pct = tighten_at_profit_pct
        self.tightened_trail_pct = tightened_trail_pct

        self.logger = logger or logging.getLogger(__name__)

        # Active trailing stops by pair
        self._stops: Dict[str, TrailingStopState] = {}

        # Stats
        self._total_stops_created = 0
        self._breakeven_moves = 0
        self._trail_updates = 0

    def create_stop(
        self,
        pair: str,
        direction: str,
        entry_price: Decimal,
        initial_stop_pct: Decimal,
    ) -> TrailingStopState:
        """
        Create a new trailing stop for a position.

        Args:
            pair: Trading pair
            direction: "long" or "short"
            entry_price: Position entry price
            initial_stop_pct: Initial stop loss percentage (e.g., 0.015 = 1.5%)

        Returns:
            TrailingStopState for the new stop
        """
        if direction == "long":
            initial_stop = entry_price * (Decimal("1") - initial_stop_pct)
        else:
            initial_stop = entry_price * (Decimal("1") + initial_stop_pct)

        state = TrailingStopState(
            pair=pair,
            direction=direction,
            entry_price=entry_price,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            peak_price=entry_price,
            stop_distance_pct=initial_stop_pct,
            is_breakeven=False,
            is_trailing=False,
            last_update=datetime.now().timestamp()
        )

        self._stops[pair] = state
        self._total_stops_created += 1

        self.logger.info(
            f"TRAILING STOP: Created for {pair} {direction.upper()}\n"
            f"  Entry: {entry_price:.4f}\n"
            f"  Initial Stop: {initial_stop:.4f} ({float(initial_stop_pct)*100:.2f}%)\n"
            f"  Mode: {self.mode.value}"
        )

        return state

    def update_price(
        self,
        pair: str,
        current_price: Decimal,
    ) -> Tuple[Optional[Decimal], bool]:
        """
        Update price and adjust trailing stop if needed.

        Args:
            pair: Trading pair
            current_price: Current market price

        Returns:
            Tuple of (current_stop_price, should_exit)
        """
        if not self.enabled or pair not in self._stops:
            return None, False

        state = self._stops[pair]
        should_exit = False

        # Check if stop is hit
        if state.direction == "long":
            if current_price <= state.current_stop:
                should_exit = True
                self.logger.info(
                    f"TRAILING STOP HIT: {pair} LONG @ {current_price:.4f}\n"
                    f"  Stop was: {state.current_stop:.4f}\n"
                    f"  Entry was: {state.entry_price:.4f}\n"
                    f"  Peak was: {state.peak_price:.4f}"
                )
                return state.current_stop, True
        else:  # short
            if current_price >= state.current_stop:
                should_exit = True
                self.logger.info(
                    f"TRAILING STOP HIT: {pair} SHORT @ {current_price:.4f}\n"
                    f"  Stop was: {state.current_stop:.4f}\n"
                    f"  Entry was: {state.entry_price:.4f}\n"
                    f"  Trough was: {state.peak_price:.4f}"
                )
                return state.current_stop, True

        # Update peak price
        if state.direction == "long":
            if current_price > state.peak_price:
                state.peak_price = current_price
        else:  # short
            if current_price < state.peak_price:
                state.peak_price = current_price

        # Calculate current profit
        if state.direction == "long":
            profit_pct = (current_price - state.entry_price) / state.entry_price
            peak_profit_pct = (state.peak_price - state.entry_price) / state.entry_price
        else:
            profit_pct = (state.entry_price - current_price) / state.entry_price
            peak_profit_pct = (state.entry_price - state.peak_price) / state.entry_price

        # Apply trailing logic based on mode
        if self.mode == TrailingStopMode.DISABLED:
            pass  # Keep initial stop

        elif self.mode == TrailingStopMode.PERCENTAGE:
            self._apply_percentage_trail(state, current_price, peak_profit_pct)

        elif self.mode == TrailingStopMode.BREAKEVEN_THEN_TRAIL:
            self._apply_breakeven_then_trail(state, current_price, profit_pct, peak_profit_pct)

        state.last_update = datetime.now().timestamp()
        return state.current_stop, False

    def _apply_percentage_trail(
        self,
        state: TrailingStopState,
        current_price: Decimal,
        peak_profit_pct: Decimal
    ):
        """Apply simple percentage trailing."""
        # Only trail if above activation threshold
        if peak_profit_pct < self.activation_profit_pct:
            return

        # Determine trail distance (tighten if in significant profit)
        trail_pct = self.trail_distance_pct
        if self.tighten_at_profit_pct and peak_profit_pct >= self.tighten_at_profit_pct:
            trail_pct = self.tightened_trail_pct

        # Calculate new stop from peak
        if state.direction == "long":
            new_stop = state.peak_price * (Decimal("1") - trail_pct)
            if new_stop > state.current_stop:
                old_stop = state.current_stop
                state.current_stop = new_stop
                state.stop_distance_pct = trail_pct
                state.is_trailing = True
                self._trail_updates += 1
                self.logger.debug(
                    f"TRAIL UPDATE: {state.pair} stop raised {old_stop:.4f} -> {new_stop:.4f}"
                )
        else:  # short
            new_stop = state.peak_price * (Decimal("1") + trail_pct)
            if new_stop < state.current_stop:
                old_stop = state.current_stop
                state.current_stop = new_stop
                state.stop_distance_pct = trail_pct
                state.is_trailing = True
                self._trail_updates += 1
                self.logger.debug(
                    f"TRAIL UPDATE: {state.pair} stop lowered {old_stop:.4f} -> {new_stop:.4f}"
                )

    def _apply_breakeven_then_trail(
        self,
        state: TrailingStopState,
        current_price: Decimal,
        profit_pct: Decimal,
        peak_profit_pct: Decimal
    ):
        """Apply breakeven-first, then trail logic."""
        # Step 1: Move to breakeven once we have enough profit
        if not state.is_breakeven and profit_pct >= self.breakeven_activation_pct:
            # Add small buffer above entry (0.1% to cover fees)
            if state.direction == "long":
                breakeven_stop = state.entry_price * Decimal("1.001")
                if breakeven_stop > state.current_stop:
                    state.current_stop = breakeven_stop
                    state.is_breakeven = True
                    self._breakeven_moves += 1
                    self.logger.info(
                        f"TRAILING STOP: {state.pair} moved to BREAKEVEN @ {breakeven_stop:.4f}"
                    )
            else:  # short
                breakeven_stop = state.entry_price * Decimal("0.999")
                if breakeven_stop < state.current_stop:
                    state.current_stop = breakeven_stop
                    state.is_breakeven = True
                    self._breakeven_moves += 1
                    self.logger.info(
                        f"TRAILING STOP: {state.pair} moved to BREAKEVEN @ {breakeven_stop:.4f}"
                    )

        # Step 2: Start trailing once we have more profit
        if state.is_breakeven and peak_profit_pct >= self.activation_profit_pct:
            self._apply_percentage_trail(state, current_price, peak_profit_pct)

    def remove_stop(self, pair: str):
        """Remove a trailing stop (position closed)."""
        if pair in self._stops:
            state = self._stops[pair]
            self.logger.info(
                f"TRAILING STOP: Removed for {pair}\n"
                f"  Was breakeven: {state.is_breakeven}\n"
                f"  Was trailing: {state.is_trailing}\n"
                f"  Final stop: {state.current_stop:.4f}"
            )
            del self._stops[pair]

    def get_stop_price(self, pair: str) -> Optional[Decimal]:
        """Get current stop price for a pair."""
        if pair in self._stops:
            return self._stops[pair].current_stop
        return None

    def get_state(self, pair: str) -> Optional[TrailingStopState]:
        """Get full state for a pair."""
        return self._stops.get(pair)

    def has_stop(self, pair: str) -> bool:
        """Check if a trailing stop exists for a pair."""
        return pair in self._stops

    def get_status(self) -> Dict:
        """Get overall trailing stop manager status."""
        active_stops = []
        for pair, state in self._stops.items():
            active_stops.append({
                "pair": pair,
                "direction": state.direction,
                "entry": float(state.entry_price),
                "current_stop": float(state.current_stop),
                "peak": float(state.peak_price),
                "is_breakeven": state.is_breakeven,
                "is_trailing": state.is_trailing,
            })

        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "active_stops": len(self._stops),
            "stops": active_stops,
            "stats": {
                "total_created": self._total_stops_created,
                "breakeven_moves": self._breakeven_moves,
                "trail_updates": self._trail_updates,
            },
            "config": {
                "trail_distance_pct": float(self.trail_distance_pct),
                "activation_profit_pct": float(self.activation_profit_pct),
                "breakeven_activation_pct": float(self.breakeven_activation_pct),
            }
        }

    def adjust_for_volatility(self, pair: str, volatility_multiplier: float):
        """
        Adjust trail distance based on current volatility.

        Args:
            pair: Trading pair
            volatility_multiplier: Multiplier for trail distance (e.g., 1.5 = 50% wider)
        """
        if pair not in self._stops:
            return

        state = self._stops[pair]
        adjusted_trail = self.trail_distance_pct * Decimal(str(volatility_multiplier))

        # Clamp to min/max
        adjusted_trail = max(self.min_trail_distance_pct, min(self.max_trail_distance_pct, adjusted_trail))

        if adjusted_trail != state.stop_distance_pct:
            self.logger.debug(
                f"TRAIL ADJUSTED: {pair} trail distance "
                f"{float(state.stop_distance_pct)*100:.2f}% -> {float(adjusted_trail)*100:.2f}% "
                f"(volatility multiplier: {volatility_multiplier:.2f})"
            )
            state.stop_distance_pct = adjusted_trail
