"""
Grid Trading Strategy for Hyperliquid Monster Bot v2.

Leveraged grid trading with automatic rebalancing.
"""

import logging
from decimal import Decimal
from typing import Callable, Dict, List, Optional

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import PositionAction, PriceType, TradeType
from hummingbot.core.event.events import OrderFilledEvent

from ..config import HyperliquidMonsterV2Config
from ..models import StrategyMetrics, StrategyMode


class GridStrategy:
    """
    Leveraged Grid Trading Strategy.

    Places buy orders below current price and sell orders above,
    capturing profits from price oscillations.
    """

    def __init__(
        self,
        config: HyperliquidMonsterV2Config,
        connector: ConnectorBase,
        metrics: StrategyMetrics,
        place_order_fn: Callable,
        cancel_order_fn: Callable,
        get_active_orders_fn: Callable,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the grid strategy.

        Args:
            config: Bot configuration
            connector: Exchange connector
            metrics: Strategy metrics tracker
            place_order_fn: Function to place orders
            cancel_order_fn: Function to cancel orders
            get_active_orders_fn: Function to get active orders
            logger: Logger instance
        """
        self.config = config
        self.connector = connector
        self.metrics = metrics
        self.place_order = place_order_fn
        self.cancel_order = cancel_order_fn
        self.get_active_orders = get_active_orders_fn
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.initialized = False
        self.base_price: Optional[Decimal] = None
        self.orders: Dict[str, dict] = {}  # order_id -> grid info

    @property
    def pair(self) -> str:
        """Get the grid trading pair."""
        return self.config.grid_pair

    def run(self):
        """Run the grid strategy."""
        if self.metrics.status != StrategyMode.ACTIVE:
            return

        price = self.connector.get_price_by_type(self.pair, PriceType.MidPrice)
        if price is None:
            return

        if not self.initialized:
            self._initialize_grid(price)
            return

        # Check for rebalance
        if self.base_price:
            deviation = abs(price - self.base_price) / self.base_price
            if deviation > self.config.grid_rebalance_pct:
                self.logger.info(f"GRID: Rebalancing (price moved {deviation*100:.2f}%)")
                self.cancel_all_orders()
                self._initialize_grid(price)

    def _initialize_grid(self, base_price: Decimal):
        """Set up grid levels."""
        # Cancel any existing grid orders on exchange first (handles bot restarts)
        existing_orders = [o for o in self.get_active_orders()
                          if o.trading_pair == self.pair]
        if existing_orders:
            self.logger.info(f"GRID: Cancelling {len(existing_orders)} existing orders before reinitializing")
            for order in existing_orders:
                self.cancel_order(self.pair, order.client_order_id)

        self.base_price = base_price
        self.orders.clear()

        spacing = self.config.grid_spacing_pct
        levels = self.config.grid_levels
        order_size = self.config.grid_order_size
        leverage = self.config.grid_leverage

        self.logger.info(f"GRID: Initializing {levels} levels each side around {base_price:.2f}")

        amount_per_order = order_size / base_price

        # Buy orders below
        for i in range(1, levels + 1):
            level_price = base_price * (Decimal("1") - spacing * i)

            order_id = self.place_order(
                pair=self.pair,
                side=TradeType.BUY,
                amount=amount_per_order,
                price=level_price,
                leverage=leverage,
                position_action=PositionAction.OPEN,
                strategy="grid"
            )

            if order_id:
                self.orders[order_id] = {
                    "level": i,
                    "side": "buy",
                    "price": level_price
                }

        # Sell orders above
        for i in range(1, levels + 1):
            level_price = base_price * (Decimal("1") + spacing * i)

            order_id = self.place_order(
                pair=self.pair,
                side=TradeType.SELL,
                amount=amount_per_order,
                price=level_price,
                leverage=leverage,
                position_action=PositionAction.OPEN,
                strategy="grid"
            )

            if order_id:
                self.orders[order_id] = {
                    "level": i,
                    "side": "sell",
                    "price": level_price
                }

        self.initialized = True
        self.logger.info(f"GRID: {len(self.orders)} orders placed ({leverage}x leverage)")

    def handle_fill(self, event: OrderFilledEvent) -> bool:
        """
        Handle a grid order fill and place replacement order.

        Args:
            event: The order filled event

        Returns:
            True if this was a grid order, False otherwise
        """
        order_id = event.order_id
        if order_id not in self.orders:
            return False

        self.metrics.total_trades += 1

        grid_info = self.orders[order_id]
        del self.orders[order_id]

        spacing = self.config.grid_spacing_pct
        leverage = self.config.grid_leverage

        # Place opposite order
        if grid_info["side"] == "buy":
            new_price = event.price * (Decimal("1") + spacing)
            new_side = TradeType.SELL
            new_side_str = "sell"
            position_action = PositionAction.CLOSE
        else:
            new_price = event.price * (Decimal("1") - spacing)
            new_side = TradeType.BUY
            new_side_str = "buy"
            position_action = PositionAction.OPEN

        new_order_id = self.place_order(
            pair=self.pair,
            side=new_side,
            amount=event.amount,
            price=new_price,
            leverage=leverage,
            position_action=position_action,
            strategy="grid"
        )

        if new_order_id:
            self.orders[new_order_id] = {
                "level": grid_info["level"],
                "side": new_side_str,
                "price": new_price
            }

        return True

    def cancel_all_orders(self):
        """Cancel all grid orders."""
        for order in self.get_active_orders():
            if order.trading_pair == self.pair and order.client_order_id in self.orders:
                self.cancel_order(self.pair, order.client_order_id)

        self.orders.clear()
        self.initialized = False

    def get_order_count(self) -> int:
        """Get number of active grid orders."""
        return len(self.orders)

    def is_grid_order(self, order_id: str) -> bool:
        """Check if an order ID belongs to the grid strategy."""
        return order_id in self.orders
