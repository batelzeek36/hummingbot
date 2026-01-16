"""
Enhanced Grid Trading Strategy for Hyperliquid Monster Bot v2.

Leveraged grid trading with:
- Trend pause (stop during strong trends)
- Volatility-adaptive spacing
- Inventory skew management
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import PositionAction, PriceType, TradeType
from hummingbot.core.event.events import OrderFilledEvent

from ..config import HyperliquidMonsterV2Config
from ..indicators import TechnicalIndicators, TrendInfo
from ..models import StrategyMetrics, StrategyMode
from ..volatility import get_grid_trend_threshold, get_volatility_class


class GridStrategy:
    """
    Enhanced Leveraged Grid Trading Strategy.

    Places buy orders below current price and sell orders above,
    capturing profits from price oscillations.

    Enhancements:
    1. Trend pause - stops placing new orders during strong trends
    2. Volatility-adaptive spacing - wider grids in high vol, tighter in low vol
    3. Inventory management - reduces exposure when position gets skewed
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
        Initialize the enhanced grid strategy.

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

        # Grid state
        self.initialized = False
        self.base_price: Optional[Decimal] = None
        self.orders: Dict[str, dict] = {}  # order_id -> grid info
        self.current_spacing: Decimal = config.grid_spacing_pct

        # Price history for indicators
        self.price_history: List[Tuple[datetime, float]] = []

        # Inventory tracking (positive = net long, negative = net short)
        self.inventory_skew: int = 0  # Count of buy fills - sell fills

        # Status tracking
        self._trend_paused: bool = False
        self._last_trend: Optional[TrendInfo] = None
        self._current_volatility: Optional[float] = None

    @property
    def pair(self) -> str:
        """Get the grid trading pair."""
        return self.config.grid_pair

    def run(self):
        """Run the enhanced grid strategy."""
        if self.metrics.status != StrategyMode.ACTIVE:
            self.logger.debug(f"GRID: Not active, status={self.metrics.status}")
            return

        price = self.connector.get_price_by_type(self.pair, PriceType.MidPrice)
        if price is None:
            self.logger.warning(f"GRID: No price available for {self.pair}")
            return

        # Update price history
        now = datetime.now()
        self.price_history.append((now, float(price)))

        # Keep 1 hour of data
        cutoff = now - timedelta(hours=1)
        self.price_history = [(t, p) for t, p in self.price_history if t > cutoff]

        # Check trend for pause decision
        if self.config.grid_trend_pause:
            self._check_trend_pause()

        # Calculate adaptive spacing if enabled
        if self.config.grid_adaptive_spacing:
            self._calculate_adaptive_spacing()

        # Handle initialization
        if not self.initialized:
            if self._trend_paused:
                self.logger.info("GRID: Waiting for trend to settle before initializing...")
                return
            self._initialize_grid(price)
            return

        # If trend paused, cancel orders and wait
        if self._trend_paused:
            if self.orders:
                self.logger.info("GRID: Strong trend detected - cancelling orders and pausing")
                self.cancel_all_orders()
            return

        # Check for rebalance
        if self.base_price:
            deviation = abs(price - self.base_price) / self.base_price
            if deviation > self.config.grid_rebalance_pct:
                self.logger.info(f"GRID: Rebalancing (price moved {deviation*100:.2f}%)")
                self.cancel_all_orders()
                self._initialize_grid(price)

    def _check_trend_pause(self):
        """Check if grid should be paused due to strong trend (volatility-scaled)."""
        prices = [p for _, p in self.price_history]

        if len(prices) < self.config.grid_trend_ema_long:
            # Not enough data yet
            self._trend_paused = False
            return

        trend = TechnicalIndicators.analyze_trend(
            prices,
            self.config.grid_trend_ema_short,
            self.config.grid_trend_ema_long
        )

        if trend is None:
            self._trend_paused = False
            return

        self._last_trend = trend

        # Use volatility-scaled threshold based on the trading pair
        # SAFE: 0.5%, MEDIUM: 1.0%, HIGH: 1.5%, EXTREME: 2.5%
        threshold = get_grid_trend_threshold(self.pair)
        vol_class = get_volatility_class(self.pair).value.upper()

        # Pause if trend strength exceeds threshold
        was_paused = self._trend_paused
        self._trend_paused = trend.strength >= threshold

        if self._trend_paused and not was_paused:
            self.logger.info(
                f"GRID: Trend pause ACTIVATED - {trend.direction} "
                f"(strength: {trend.strength:.2f}% >= {threshold}% [{vol_class}])"
            )
        elif not self._trend_paused and was_paused:
            self.logger.info(
                f"GRID: Trend pause DEACTIVATED - market ranging "
                f"(strength: {trend.strength:.2f}% < {threshold}% [{vol_class}])"
            )

    def _calculate_adaptive_spacing(self):
        """Calculate grid spacing based on recent volatility."""
        prices = [p for _, p in self.price_history]

        if len(prices) < self.config.grid_volatility_lookback:
            # Use default spacing until we have data
            self.current_spacing = self.config.grid_spacing_pct
            return

        # Calculate volatility as standard deviation of returns
        recent_prices = prices[-self.config.grid_volatility_lookback:]
        returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            returns.append(abs(ret))

        if not returns:
            return

        avg_volatility = sum(returns) / len(returns)
        self._current_volatility = avg_volatility

        # Map volatility to spacing range
        # Low vol (~0.1%) -> min spacing, High vol (~0.5%) -> max spacing
        min_spacing = float(self.config.grid_min_spacing_pct)
        max_spacing = float(self.config.grid_max_spacing_pct)

        # Normalize volatility (0.001 = 0.1% is low, 0.005 = 0.5% is high)
        vol_normalized = min(max((avg_volatility - 0.001) / 0.004, 0), 1)

        # Linear interpolation between min and max spacing
        adaptive_spacing = min_spacing + vol_normalized * (max_spacing - min_spacing)
        self.current_spacing = Decimal(str(round(adaptive_spacing, 4)))

    def _get_order_size(self, side: str) -> Decimal:
        """
        Get order size, adjusted for inventory skew.

        Args:
            side: "buy" or "sell"

        Returns:
            Adjusted order size
        """
        base_size = self.config.grid_order_size

        if not self.config.grid_inventory_management:
            return base_size

        max_skew = self.config.grid_max_inventory_skew
        reduction = self.config.grid_skew_reduction_factor

        # Reduce size if adding to an already skewed position
        if side == "buy" and self.inventory_skew >= max_skew:
            # Already too long, reduce buy size
            self.logger.debug(f"GRID: Inventory skewed long ({self.inventory_skew}), reducing buy size")
            return base_size * reduction
        elif side == "sell" and self.inventory_skew <= -max_skew:
            # Already too short, reduce sell size
            self.logger.debug(f"GRID: Inventory skewed short ({self.inventory_skew}), reducing sell size")
            return base_size * reduction

        return base_size

    def _initialize_grid(self, base_price: Decimal):
        """Set up grid levels with enhanced features."""
        # Cancel any existing grid orders on exchange first
        existing_orders = [o for o in self.get_active_orders()
                          if o.trading_pair == self.pair]
        if existing_orders:
            self.logger.info(f"GRID: Cancelling {len(existing_orders)} existing orders before reinitializing")
            for order in existing_orders:
                self.cancel_order(self.pair, order.client_order_id)

        self.base_price = base_price
        self.orders.clear()

        spacing = self.current_spacing
        levels = self.config.grid_levels
        leverage = self.config.grid_leverage

        self.logger.info(
            f"GRID: Initializing {levels} levels each side around {base_price:.2f} "
            f"(spacing: {float(spacing)*100:.2f}%)"
        )

        # Buy orders below
        for i in range(1, levels + 1):
            level_price = base_price * (Decimal("1") - spacing * i)
            order_size = self._get_order_size("buy")

            # Notional = margin * leverage, then convert to asset quantity
            notional_per_order = order_size * leverage
            amount_per_order = notional_per_order / level_price

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
                    "price": level_price,
                    "amount": amount_per_order
                }

        # Sell orders above
        for i in range(1, levels + 1):
            level_price = base_price * (Decimal("1") + spacing * i)
            order_size = self._get_order_size("sell")

            # Notional = margin * leverage, then convert to asset quantity
            notional_per_order = order_size * leverage
            amount_per_order = notional_per_order / level_price

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
                    "price": level_price,
                    "amount": amount_per_order
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

        # Update inventory tracking
        if grid_info["side"] == "buy":
            self.inventory_skew += 1
        else:
            self.inventory_skew -= 1

        # Don't place new orders if trend paused
        if self._trend_paused:
            self.logger.info(f"GRID: Fill on {grid_info['side']} - not replacing (trend paused)")
            return True

        leverage = self.config.grid_leverage
        spacing = self.current_spacing

        # Place opposite order
        if grid_info["side"] == "buy":
            new_price = event.price * (Decimal("1") + spacing)
            new_side = TradeType.SELL
            new_side_str = "sell"
            position_action = PositionAction.CLOSE
            order_size = self._get_order_size("sell")
        else:
            new_price = event.price * (Decimal("1") - spacing)
            new_side = TradeType.BUY
            new_side_str = "buy"
            position_action = PositionAction.OPEN
            order_size = self._get_order_size("buy")

        # Calculate amount based on potentially adjusted size
        notional = order_size * leverage
        new_amount = notional / new_price

        new_order_id = self.place_order(
            pair=self.pair,
            side=new_side,
            amount=new_amount,
            price=new_price,
            leverage=leverage,
            position_action=position_action,
            strategy="grid"
        )

        if new_order_id:
            self.orders[new_order_id] = {
                "level": grid_info["level"],
                "side": new_side_str,
                "price": new_price,
                "amount": new_amount
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

    def get_status_info(self) -> Dict[str, str]:
        """Get current grid status for display."""
        status = {}

        status["Orders"] = str(len(self.orders))
        status["Spacing"] = f"{float(self.current_spacing)*100:.2f}%"

        # Show trend status with volatility-scaled threshold
        vol_class = get_volatility_class(self.pair).value.upper()
        threshold = get_grid_trend_threshold(self.pair)

        if self._trend_paused:
            status["Trend"] = f"PAUSED (threshold: {threshold}% [{vol_class}])"
        elif self._last_trend:
            status["Trend"] = f"{self._last_trend.direction} ({self._last_trend.strength:.2f}%/{threshold}%)"
        else:
            status["Trend"] = f"warming up (threshold: {threshold}% [{vol_class}])"

        if self._current_volatility is not None:
            status["Volatility"] = f"{self._current_volatility*100:.2f}%"

        if self.config.grid_inventory_management:
            skew_str = "neutral"
            if self.inventory_skew > 0:
                skew_str = f"long {self.inventory_skew}"
            elif self.inventory_skew < 0:
                skew_str = f"short {abs(self.inventory_skew)}"
            status["Inventory"] = skew_str

        return status

    def get_net_direction(self) -> Optional[str]:
        """
        Get net position direction based on inventory skew.
        Used by circuit breaker to determine adverse vs favorable moves.

        Returns:
            "long" if net long, "short" if net short, None if neutral
        """
        if self.inventory_skew > 0:
            return "long"
        elif self.inventory_skew < 0:
            return "short"
        return None  # Neutral - no position bias
