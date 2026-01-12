"""
Momentum/RSI Trading Strategy for Hyperliquid Monster Bot v2.

RSI-based directional trading with take profit and stop loss.
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import PositionAction, PriceType, TradeType

from ..config import HyperliquidMonsterV2Config
from ..models import StrategyMetrics, StrategyMode


class MomentumStrategy:
    """
    Momentum Trading Strategy using RSI.

    Opens long positions on oversold conditions,
    short positions on overbought conditions.
    """

    def __init__(
        self,
        config: HyperliquidMonsterV2Config,
        connector: ConnectorBase,
        metrics: StrategyMetrics,
        place_order_fn: Callable,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the momentum strategy.

        Args:
            config: Bot configuration
            connector: Exchange connector
            metrics: Strategy metrics tracker
            place_order_fn: Function to place orders
            logger: Logger instance
        """
        self.config = config
        self.connector = connector
        self.metrics = metrics
        self.place_order = place_order_fn
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.position: Optional[str] = None  # "long", "short", or None
        self.entry_price: Optional[Decimal] = None

    @property
    def pair(self) -> str:
        """Get the momentum trading pair."""
        return self.config.momentum_pair

    def run(self, global_pnl_updater: Callable[[Decimal], None]):
        """
        Run the momentum strategy.

        Args:
            global_pnl_updater: Callback to update global P&L when closing positions
        """
        price = self.connector.get_price_by_type(self.pair, PriceType.MidPrice)
        if price is None:
            return

        now = datetime.now()
        if self.pair not in self.price_history:
            self.price_history[self.pair] = []

        self.price_history[self.pair].append((now, price))

        # Keep only last hour of data
        cutoff = now - timedelta(hours=1)
        self.price_history[self.pair] = [(t, p) for t, p in self.price_history[self.pair] if t > cutoff]

        # Need enough data for RSI calculation
        if len(self.price_history[self.pair]) < self.config.momentum_lookback:
            return

        # Activate strategy once warmed up
        if self.metrics.status == StrategyMode.WARMING_UP:
            self.metrics.status = StrategyMode.ACTIVE
            self.logger.info("MOMENTUM: Strategy activated")

        if self.metrics.status != StrategyMode.ACTIVE:
            return

        rsi = self._calculate_rsi()
        if rsi is None:
            return

        # Check exit conditions if in position
        if self.position and self.entry_price:
            should_exit, reason = self._check_exit(price)
            if should_exit:
                self._close_position(price, reason, global_pnl_updater)
                return

        # Check entry conditions if not in position
        if self.position is None:
            if rsi < float(self.config.rsi_oversold):
                self._open_position("long", price, rsi)
            elif rsi > float(self.config.rsi_overbought):
                self._open_position("short", price, rsi)

    def _calculate_rsi(self) -> Optional[float]:
        """Calculate RSI from price history."""
        prices = [p for _, p in self.price_history[self.pair]]
        if len(prices) < self.config.momentum_lookback:
            return None

        changes = []
        for i in range(1, len(prices)):
            changes.append(float(prices[i] - prices[i-1]))

        if not changes:
            return None

        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]

        avg_gain = sum(gains) / len(changes) if gains else 0
        avg_loss = sum(losses) / len(changes) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _open_position(self, direction: str, price: Decimal, rsi: float):
        """Open a momentum position."""
        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        amount = position_size / price
        trade_type = TradeType.BUY if direction == "long" else TradeType.SELL

        order_id = self.place_order(
            pair=self.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.OPEN,
            strategy="momentum"
        )

        if order_id:
            self.position = direction
            self.entry_price = price
            self.logger.info(
                f"MOMENTUM: Opened {direction.upper()} @ {price:.2f} "
                f"(RSI: {rsi:.1f}, ${position_size} x {leverage}x)"
            )

    def _check_exit(self, current_price: Decimal) -> Tuple[bool, str]:
        """Check if momentum position should exit."""
        if not self.entry_price:
            return False, ""

        if self.position == "long":
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price

        if pnl_pct >= self.config.momentum_take_profit:
            return True, "take_profit"

        if pnl_pct <= -self.config.momentum_stop_loss:
            return True, "stop_loss"

        return False, ""

    def _close_position(self, price: Decimal, reason: str, global_pnl_updater: Callable[[Decimal], None]):
        """Close momentum position."""
        if not self.position:
            return

        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        amount = position_size / price
        trade_type = TradeType.SELL if self.position == "long" else TradeType.BUY

        order_id = self.place_order(
            pair=self.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.CLOSE,
            strategy="momentum"
        )

        if order_id and self.entry_price:
            if self.position == "long":
                pnl = (price - self.entry_price) * amount * leverage
            else:
                pnl = (self.entry_price - price) * amount * leverage

            self.metrics.realized_pnl += pnl
            global_pnl_updater(pnl)

            pnl_pct = pnl / position_size * 100
            self.logger.info(
                f"MOMENTUM: Closed {self.position.upper()} @ {price:.2f} "
                f"({reason}) P&L: ${pnl:.4f} ({pnl_pct:.2f}%)"
            )

        self.position = None
        self.entry_price = None

    def close_position_for_shutdown(self, global_pnl_updater: Callable[[Decimal], None]):
        """Close position for shutdown."""
        if self.position:
            price = self.connector.get_price_by_type(self.pair, PriceType.MidPrice)
            if price:
                self._close_position(price, "shutdown", global_pnl_updater)

    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.position is not None

    def get_position_info(self) -> Optional[Tuple[str, Decimal]]:
        """Get current position info (direction, entry_price)."""
        if self.position and self.entry_price:
            return (self.position, self.entry_price)
        return None
