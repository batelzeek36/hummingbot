"""
Dynamic Risk Manager - Volatility-scaled risk management.

Adjusts risk parameters based on current market conditions:
- Wider stops during high volatility (avoid getting stopped out on noise)
- Tighter stops during low volatility (protect profits)
- Emergency market orders when critical thresholds exceeded

This prevents:
1. Getting stopped out by normal volatility spikes
2. Losing too much when volatility is calm (should have tighter stops)
3. Limit order exits not filling during violent moves
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple


class VolatilityRegime(Enum):
    """Current volatility classification."""
    VERY_LOW = "very_low"     # Unusually calm
    LOW = "low"               # Below average
    NORMAL = "normal"         # Average volatility
    HIGH = "high"             # Above average
    EXTREME = "extreme"       # Dangerous volatility


class EmergencyAction(Enum):
    """Emergency action types."""
    NONE = "none"
    MARKET_CLOSE = "market_close"       # Close position with market order
    REDUCE_POSITION = "reduce_position"  # Reduce position size
    HEDGE = "hedge"                      # Open offsetting position


@dataclass
class VolatilityMetrics:
    """Current volatility measurements."""
    pair: str
    current_volatility: float        # Current volatility (e.g., std dev of returns)
    avg_volatility: float            # Average volatility over lookback
    volatility_ratio: float          # current / average
    regime: VolatilityRegime
    recommended_sl_multiplier: float # Multiplier for stop loss distance
    timestamp: float


@dataclass
class EmergencyEvent:
    """Record of an emergency action."""
    timestamp: float
    pair: str
    action: EmergencyAction
    reason: str
    loss_pct: float
    price: float


class DynamicRiskManager:
    """
    Dynamic Risk Management System.

    Features:
    - Real-time volatility calculation
    - Volatility regime detection
    - Dynamic stop loss adjustment
    - Emergency market order triggers
    - Position sizing recommendations
    """

    def __init__(
        self,
        # Volatility calculation
        volatility_lookback: int = 30,           # Periods for volatility calc
        volatility_update_interval: int = 10,    # Seconds between updates
        # Regime thresholds (ratio of current/avg volatility)
        very_low_threshold: float = 0.5,         # < 0.5x avg = very low
        low_threshold: float = 0.8,              # < 0.8x avg = low
        high_threshold: float = 1.5,             # > 1.5x avg = high
        extreme_threshold: float = 2.5,          # > 2.5x avg = extreme
        # Stop loss adjustment
        very_low_sl_multiplier: float = 0.7,     # Tighten SL by 30%
        low_sl_multiplier: float = 0.85,         # Tighten SL by 15%
        normal_sl_multiplier: float = 1.0,       # Normal SL
        high_sl_multiplier: float = 1.3,         # Widen SL by 30%
        extreme_sl_multiplier: float = 1.8,      # Widen SL by 80%
        # Emergency thresholds (as % loss)
        emergency_loss_threshold: float = 0.05,  # 5% unrealized loss triggers emergency
        critical_loss_threshold: float = 0.08,   # 8% triggers immediate market close
        # Callbacks
        on_emergency: Optional[Callable[[str, EmergencyAction, str], None]] = None,
        # Logging
        logger: Optional[logging.Logger] = None,
        enabled: bool = True,
    ):
        """
        Initialize the dynamic risk manager.

        Args:
            volatility_lookback: Periods for volatility calculation
            volatility_update_interval: Seconds between volatility updates
            *_threshold: Volatility ratio thresholds for regime classification
            *_sl_multiplier: Stop loss multipliers for each regime
            emergency_loss_threshold: Loss % that triggers emergency action
            critical_loss_threshold: Loss % that triggers immediate market close
            on_emergency: Callback when emergency action is needed
            logger: Logger instance
            enabled: Whether dynamic risk is active
        """
        self.enabled = enabled

        # Volatility settings
        self.volatility_lookback = volatility_lookback
        self.volatility_update_interval = volatility_update_interval

        # Regime thresholds
        self.regime_thresholds = {
            "very_low": very_low_threshold,
            "low": low_threshold,
            "high": high_threshold,
            "extreme": extreme_threshold,
        }

        # SL multipliers by regime
        self.sl_multipliers = {
            VolatilityRegime.VERY_LOW: very_low_sl_multiplier,
            VolatilityRegime.LOW: low_sl_multiplier,
            VolatilityRegime.NORMAL: normal_sl_multiplier,
            VolatilityRegime.HIGH: high_sl_multiplier,
            VolatilityRegime.EXTREME: extreme_sl_multiplier,
        }

        # Emergency settings
        self.emergency_loss_threshold = emergency_loss_threshold
        self.critical_loss_threshold = critical_loss_threshold

        self.on_emergency = on_emergency
        self.logger = logger or logging.getLogger(__name__)

        # State
        self._price_history: Dict[str, Deque[Tuple[float, float]]] = {}  # pair -> (timestamp, price)
        self._volatility_cache: Dict[str, VolatilityMetrics] = {}
        self._last_volatility_update: Dict[str, float] = {}
        self._emergency_history: List[EmergencyEvent] = []

        # Stats
        self._total_emergency_triggers = 0
        self._market_close_count = 0

    def update_price(self, pair: str, price: float, timestamp: Optional[float] = None) -> VolatilityMetrics:
        """
        Update price data and recalculate volatility if needed.

        Args:
            pair: Trading pair
            price: Current price
            timestamp: Unix timestamp (defaults to now)

        Returns:
            Current volatility metrics for the pair
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Initialize history for new pairs
        if pair not in self._price_history:
            self._price_history[pair] = deque(maxlen=500)
            self._last_volatility_update[pair] = 0

        # Add price point
        self._price_history[pair].append((timestamp, price))

        # Check if we need to update volatility
        time_since_update = timestamp - self._last_volatility_update.get(pair, 0)
        if time_since_update >= self.volatility_update_interval:
            self._calculate_volatility(pair, timestamp)
            self._last_volatility_update[pair] = timestamp

        return self._volatility_cache.get(pair, self._default_metrics(pair, timestamp))

    def _calculate_volatility(self, pair: str, timestamp: float):
        """Calculate current volatility for a pair."""
        if pair not in self._price_history:
            return

        history = list(self._price_history[pair])
        if len(history) < self.volatility_lookback:
            # Not enough data, return default
            self._volatility_cache[pair] = self._default_metrics(pair, timestamp)
            return

        # Calculate returns
        recent_prices = [p for _, p in history[-self.volatility_lookback:]]
        returns = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                returns.append(ret)

        if len(returns) < 5:
            self._volatility_cache[pair] = self._default_metrics(pair, timestamp)
            return

        # Calculate current volatility (standard deviation of returns)
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        current_vol = variance ** 0.5

        # Calculate longer-term average volatility
        all_prices = [p for _, p in history]
        all_returns = []
        for i in range(1, len(all_prices)):
            if all_prices[i-1] > 0:
                ret = (all_prices[i] - all_prices[i-1]) / all_prices[i-1]
                all_returns.append(ret)

        if len(all_returns) > self.volatility_lookback:
            avg_mean = sum(all_returns) / len(all_returns)
            avg_variance = sum((r - avg_mean) ** 2 for r in all_returns) / len(all_returns)
            avg_vol = avg_variance ** 0.5
        else:
            avg_vol = current_vol

        # Calculate ratio and determine regime
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
        else:
            vol_ratio = 1.0

        regime = self._classify_regime(vol_ratio)
        sl_multiplier = self.sl_multipliers[regime]

        metrics = VolatilityMetrics(
            pair=pair,
            current_volatility=current_vol,
            avg_volatility=avg_vol,
            volatility_ratio=vol_ratio,
            regime=regime,
            recommended_sl_multiplier=sl_multiplier,
            timestamp=timestamp
        )

        self._volatility_cache[pair] = metrics

        # Log regime changes
        if pair in self._volatility_cache:
            old_regime = self._volatility_cache[pair].regime
            if regime != old_regime:
                self.logger.info(
                    f"VOLATILITY REGIME CHANGE: {pair} {old_regime.value} -> {regime.value} "
                    f"(ratio: {vol_ratio:.2f}, SL multiplier: {sl_multiplier:.2f})"
                )

    def _classify_regime(self, vol_ratio: float) -> VolatilityRegime:
        """Classify volatility regime based on ratio."""
        if vol_ratio < self.regime_thresholds["very_low"]:
            return VolatilityRegime.VERY_LOW
        elif vol_ratio < self.regime_thresholds["low"]:
            return VolatilityRegime.LOW
        elif vol_ratio > self.regime_thresholds["extreme"]:
            return VolatilityRegime.EXTREME
        elif vol_ratio > self.regime_thresholds["high"]:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.NORMAL

    def _default_metrics(self, pair: str, timestamp: float) -> VolatilityMetrics:
        """Return default metrics when insufficient data."""
        return VolatilityMetrics(
            pair=pair,
            current_volatility=0.0,
            avg_volatility=0.0,
            volatility_ratio=1.0,
            regime=VolatilityRegime.NORMAL,
            recommended_sl_multiplier=1.0,
            timestamp=timestamp
        )

    def get_adjusted_stop_loss(
        self,
        pair: str,
        base_stop_loss_pct: Decimal
    ) -> Tuple[Decimal, VolatilityRegime]:
        """
        Get volatility-adjusted stop loss percentage.

        Args:
            pair: Trading pair
            base_stop_loss_pct: Base stop loss percentage

        Returns:
            Tuple of (adjusted_stop_loss_pct, current_regime)
        """
        if not self.enabled or pair not in self._volatility_cache:
            return base_stop_loss_pct, VolatilityRegime.NORMAL

        metrics = self._volatility_cache[pair]
        adjusted = base_stop_loss_pct * Decimal(str(metrics.recommended_sl_multiplier))

        return adjusted, metrics.regime

    def check_emergency(
        self,
        pair: str,
        direction: str,
        entry_price: Decimal,
        current_price: Decimal,
        position_value: Decimal,
    ) -> Tuple[EmergencyAction, str]:
        """
        Check if emergency action is needed based on unrealized loss.

        Args:
            pair: Trading pair
            direction: "long" or "short"
            entry_price: Position entry price
            current_price: Current market price
            position_value: Current position value

        Returns:
            Tuple of (action, reason)
        """
        if not self.enabled:
            return EmergencyAction.NONE, ""

        # Calculate unrealized loss
        if direction == "long":
            pnl_pct = float((current_price - entry_price) / entry_price)
        else:
            pnl_pct = float((entry_price - current_price) / entry_price)

        # Check critical threshold first
        if pnl_pct <= -self.critical_loss_threshold:
            reason = (
                f"CRITICAL LOSS: {pnl_pct*100:.2f}% unrealized loss "
                f"(threshold: {self.critical_loss_threshold*100:.1f}%)"
            )
            self._trigger_emergency(pair, EmergencyAction.MARKET_CLOSE, reason, pnl_pct, float(current_price))
            return EmergencyAction.MARKET_CLOSE, reason

        # Check emergency threshold
        if pnl_pct <= -self.emergency_loss_threshold:
            # Scale action based on volatility regime
            metrics = self._volatility_cache.get(pair)
            if metrics and metrics.regime == VolatilityRegime.EXTREME:
                # In extreme volatility, give more room
                adjusted_threshold = self.emergency_loss_threshold * 1.5
                if pnl_pct > -adjusted_threshold:
                    return EmergencyAction.NONE, ""

            reason = (
                f"EMERGENCY: {pnl_pct*100:.2f}% unrealized loss "
                f"(threshold: {self.emergency_loss_threshold*100:.1f}%)"
            )
            self._trigger_emergency(pair, EmergencyAction.MARKET_CLOSE, reason, pnl_pct, float(current_price))
            return EmergencyAction.MARKET_CLOSE, reason

        return EmergencyAction.NONE, ""

    def _trigger_emergency(
        self,
        pair: str,
        action: EmergencyAction,
        reason: str,
        loss_pct: float,
        price: float
    ):
        """Record and trigger emergency action."""
        timestamp = datetime.now().timestamp()

        event = EmergencyEvent(
            timestamp=timestamp,
            pair=pair,
            action=action,
            reason=reason,
            loss_pct=loss_pct,
            price=price
        )
        self._emergency_history.append(event)
        self._total_emergency_triggers += 1

        if action == EmergencyAction.MARKET_CLOSE:
            self._market_close_count += 1

        self.logger.error(
            f"EMERGENCY ACTION TRIGGERED\n"
            f"  Pair: {pair}\n"
            f"  Action: {action.value}\n"
            f"  Reason: {reason}\n"
            f"  Current Price: {price:.4f}"
        )

        if self.on_emergency:
            try:
                self.on_emergency(pair, action, reason)
            except Exception as e:
                self.logger.error(f"Error in emergency callback: {e}")

    def get_position_size_multiplier(self, pair: str) -> float:
        """
        Get recommended position size multiplier based on volatility.

        In high volatility, reduce position size to manage risk.

        Args:
            pair: Trading pair

        Returns:
            Multiplier for position size (0.5 = halve, 1.0 = normal, etc.)
        """
        if not self.enabled or pair not in self._volatility_cache:
            return 1.0

        metrics = self._volatility_cache[pair]

        # Inverse of SL multiplier (wider SL = smaller position)
        if metrics.recommended_sl_multiplier > 0:
            return 1.0 / metrics.recommended_sl_multiplier
        return 1.0

    def get_volatility_metrics(self, pair: str) -> Optional[VolatilityMetrics]:
        """Get current volatility metrics for a pair."""
        return self._volatility_cache.get(pair)

    def get_current_regime(self, pair: str) -> VolatilityRegime:
        """Get current volatility regime for a pair."""
        if pair in self._volatility_cache:
            return self._volatility_cache[pair].regime
        return VolatilityRegime.NORMAL

    def get_status(self) -> Dict:
        """Get overall dynamic risk manager status."""
        pair_status = {}
        for pair, metrics in self._volatility_cache.items():
            pair_status[pair] = {
                "regime": metrics.regime.value,
                "volatility_ratio": round(metrics.volatility_ratio, 2),
                "sl_multiplier": metrics.recommended_sl_multiplier,
            }

        return {
            "enabled": self.enabled,
            "pairs_monitored": len(self._volatility_cache),
            "pair_status": pair_status,
            "stats": {
                "total_emergencies": self._total_emergency_triggers,
                "market_closes": self._market_close_count,
            },
            "thresholds": {
                "emergency_loss_pct": self.emergency_loss_threshold * 100,
                "critical_loss_pct": self.critical_loss_threshold * 100,
            },
            "last_emergency": self._emergency_history[-1].__dict__ if self._emergency_history else None,
        }

    def should_use_market_order(self, pair: str, loss_pct: float) -> bool:
        """
        Check if market order should be used for exit.

        Args:
            pair: Trading pair
            loss_pct: Current loss percentage (negative)

        Returns:
            True if market order recommended
        """
        if not self.enabled:
            return False

        # Use market order if:
        # 1. Loss exceeds emergency threshold
        # 2. In extreme volatility regime
        if loss_pct <= -self.emergency_loss_threshold:
            return True

        metrics = self._volatility_cache.get(pair)
        if metrics and metrics.regime == VolatilityRegime.EXTREME:
            # Lower threshold in extreme volatility
            if loss_pct <= -self.emergency_loss_threshold * 0.7:
                return True

        return False
