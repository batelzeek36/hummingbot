"""
Dollar-A-Day Market Making Strategy

Goal: Consistent small profits ($1/day) through conservative market making.

Safety Features:
- Inventory-based spread adjustment (widen spread when heavy on one side)
- Max inventory limits (stop buying/selling when too imbalanced)
- P&L tracking and status display
- Conservative default settings

Usage:
1. Paper trade first: Set exchange to "kucoin_paper_trade" or "binance_paper_trade"
2. Run for 1 week, observe fills and P&L
3. If profitable, switch to real exchange with small capital ($100-200)

Author: Generated for DEX Arbitrage Project
"""

import logging
from decimal import Decimal
from typing import Dict, List

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class DollarADayPMMConfig(BaseClientModel):
    """
    Configuration for Dollar-A-Day PMM Strategy.

    Start conservative, adjust after observing paper trade results.
    """
    script_file_name: str = Field(default="dollar_a_day_pmm.py")

    # === EXCHANGE SETTINGS ===
    # Use paper trade first: "kraken_paper_trade" (US legal)
    # Then real: "kraken" (requires KYC)
    exchange: str = Field(
        default="kraken_paper_trade",
        description="Exchange to trade on. Use paper_trade suffix for testing."
    )

    # === PAIR SELECTION ===
    # Good candidates: Mid-cap with $1-10M daily volume
    # Kraken: ATOM-USD, SOL-USD, DOT-USD (uses USD not USDT)
    trading_pair: str = Field(
        default="ATOM-USD",
        description="Trading pair. Pick mid-cap with decent volume."
    )

    # === ORDER SETTINGS ===
    order_amount: Decimal = Field(
        default=Decimal("5"),
        description="Order size in base asset (e.g., 5 ATOM)"
    )

    # === SPREAD SETTINGS ===
    # Conservative: 0.003 (0.3%) - wider spread, fewer fills, safer
    # Aggressive: 0.001 (0.1%) - tighter spread, more fills, riskier
    base_bid_spread: Decimal = Field(
        default=Decimal("0.003"),
        description="Base spread below mid-price (0.003 = 0.3%)"
    )
    base_ask_spread: Decimal = Field(
        default=Decimal("0.003"),
        description="Base spread above mid-price (0.003 = 0.3%)"
    )

    # === TIMING ===
    order_refresh_time: int = Field(
        default=30,
        description="Seconds between order refresh cycles"
    )

    # === INVENTORY MANAGEMENT ===
    # This is the key safety feature!
    # When inventory is imbalanced, we widen the spread on the heavy side
    # to discourage more accumulation and encourage rebalancing
    inventory_skew_enabled: bool = Field(
        default=True,
        description="Adjust spreads based on inventory imbalance"
    )

    # Target ratio of base asset value to total portfolio value
    # 0.5 = 50% base, 50% quote (balanced)
    target_base_ratio: Decimal = Field(
        default=Decimal("0.5"),
        description="Target inventory ratio (0.5 = balanced)"
    )

    # How aggressively to skew spreads based on inventory
    # Higher = more aggressive skew
    inventory_skew_intensity: Decimal = Field(
        default=Decimal("1.0"),
        description="Multiplier for inventory-based spread adjustment"
    )

    # Maximum allowed inventory imbalance before stopping orders on one side
    # 0.8 = stop buying when 80%+ of portfolio is base asset
    max_inventory_ratio: Decimal = Field(
        default=Decimal("0.8"),
        description="Max ratio before stopping orders on heavy side"
    )

    # === PRICE SOURCE ===
    price_type: str = Field(
        default="mid",
        description="Price source: 'mid' or 'last'"
    )


class DollarADayPMM(ScriptStrategyBase):
    """
    Conservative market making strategy designed for consistent small profits.

    Key features:
    1. Places bid/ask orders around mid-price with configurable spreads
    2. Adjusts spreads based on inventory to prevent imbalance
    3. Stops placing orders on one side if inventory gets too heavy
    4. Tracks and displays P&L metrics
    """

    # Timestamp tracking
    create_timestamp = 0

    # P&L tracking
    total_buy_volume: Decimal = Decimal("0")
    total_sell_volume: Decimal = Decimal("0")
    total_buy_quote_volume: Decimal = Decimal("0")  # USDT spent
    total_sell_quote_volume: Decimal = Decimal("0")  # USDT received
    total_trades: int = 0

    # Default markets (overridden by init_markets if config is provided)
    markets = {"kraken_paper_trade": {"ATOM-USD"}}

    @classmethod
    def init_markets(cls, config: DollarADayPMMConfig):
        cls.markets = {config.exchange: {config.trading_pair}}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: DollarADayPMMConfig = None):
        super().__init__(connectors)
        # Use provided config or create default
        self.config = config if config is not None else DollarADayPMMConfig()
        self.price_source = PriceType.LastTrade if self.config.price_type == "last" else PriceType.MidPrice

    def on_tick(self):
        """Main loop - runs every tick."""
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()

            # Calculate inventory-adjusted spreads
            bid_spread, ask_spread, inventory_ratio = self.get_adjusted_spreads()

            # Create and place orders
            proposal = self.create_proposal(bid_spread, ask_spread, inventory_ratio)
            proposal_adjusted = self.adjust_proposal_to_budget(proposal)
            self.place_orders(proposal_adjusted)

            self.create_timestamp = self.config.order_refresh_time + self.current_timestamp

    def get_inventory_ratio(self) -> Decimal:
        """
        Calculate current inventory ratio.
        Returns ratio of base asset value to total portfolio value.
        0.0 = all quote (USDT), 1.0 = all base (e.g., ATOM)
        """
        connector = self.connectors[self.config.exchange]
        base, quote = self.config.trading_pair.split("-")

        base_balance = connector.get_available_balance(base)
        quote_balance = connector.get_available_balance(quote)

        # Get current price to value base in quote terms
        mid_price = connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice)

        if mid_price is None or mid_price == 0:
            return Decimal("0.5")  # Default to balanced if no price

        base_value_in_quote = base_balance * mid_price
        total_value = base_value_in_quote + quote_balance

        if total_value == 0:
            return Decimal("0.5")

        return base_value_in_quote / total_value

    def get_adjusted_spreads(self) -> tuple:
        """
        Calculate spreads adjusted for inventory skew.

        Returns: (bid_spread, ask_spread, inventory_ratio)
        """
        inventory_ratio = self.get_inventory_ratio()

        if not self.config.inventory_skew_enabled:
            return self.config.base_bid_spread, self.config.base_ask_spread, inventory_ratio

        # Calculate deviation from target
        # Positive = too much base asset, Negative = too little base asset
        deviation = inventory_ratio - self.config.target_base_ratio

        # Adjust spreads based on deviation
        # If we have too much base (deviation > 0):
        #   - Widen ask spread (discourage more buying from us / encourage sells to us? No wait...)
        #   - Actually: Widen BID spread (discourage buying more base)
        #   - Tighten ASK spread (encourage selling our base)
        # If we have too little base (deviation < 0):
        #   - Tighten BID spread (encourage buying base)
        #   - Widen ASK spread (discourage selling our base)

        skew_adjustment = deviation * self.config.inventory_skew_intensity

        # Cap the adjustment to ±0.5% max (prevents extreme spreads)
        max_adjustment = Decimal("0.005")  # 0.5% max
        skew_adjustment = max(min(skew_adjustment, max_adjustment), -max_adjustment)

        # Apply adjustment (capped to prevent negative spreads)
        bid_spread = max(
            Decimal("0.0005"),  # Minimum 0.05% spread
            self.config.base_bid_spread + skew_adjustment
        )
        ask_spread = max(
            Decimal("0.0005"),
            min(self.config.base_ask_spread - skew_adjustment, Decimal("0.01"))  # Max 1% ask spread
        )

        return bid_spread, ask_spread, inventory_ratio

    def create_proposal(
        self,
        bid_spread: Decimal,
        ask_spread: Decimal,
        inventory_ratio: Decimal
    ) -> List[OrderCandidate]:
        """Create buy and sell order candidates with inventory limits."""

        connector = self.connectors[self.config.exchange]
        ref_price = connector.get_price_by_type(self.config.trading_pair, self.price_source)

        if ref_price is None:
            return []

        orders = []

        # Check inventory limits before creating orders
        can_buy = inventory_ratio < self.config.max_inventory_ratio
        can_sell = inventory_ratio > (Decimal("1") - self.config.max_inventory_ratio)

        # Create buy order (if not too heavy on base)
        if can_buy:
            buy_price = ref_price * (Decimal("1") - bid_spread)
            buy_order = OrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=self.config.order_amount,
                price=buy_price
            )
            orders.append(buy_order)

        # Create sell order (if not too heavy on quote)
        if can_sell:
            sell_price = ref_price * (Decimal("1") + ask_spread)
            sell_order = OrderCandidate(
                trading_pair=self.config.trading_pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=self.config.order_amount,
                price=sell_price
            )
            orders.append(sell_order)

        return orders

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        """Adjust orders to available budget."""
        if not proposal:
            return []
        return self.connectors[self.config.exchange].budget_checker.adjust_candidates(
            proposal, all_or_none=True
        )

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        """Place the orders."""
        for order in proposal:
            if order.amount > 0:
                self.place_order(connector_name=self.config.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        """Place a single order."""
        if order.order_side == TradeType.SELL:
            self.sell(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        elif order.order_side == TradeType.BUY:
            self.buy(
                connector_name=connector_name,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )

    def cancel_all_orders(self):
        """Cancel all active orders."""
        for order in self.get_active_orders(connector_name=self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        """Track fills for P&L calculation."""
        self.total_trades += 1
        quote_volume = event.amount * event.price

        if event.trade_type == TradeType.BUY:
            self.total_buy_volume += event.amount
            self.total_buy_quote_volume += quote_volume
            msg = f"BUY {event.amount:.4f} {self.config.trading_pair} @ {event.price:.4f}"
        else:
            self.total_sell_volume += event.amount
            self.total_sell_quote_volume += quote_volume
            msg = f"SELL {event.amount:.4f} {self.config.trading_pair} @ {event.price:.4f}"

        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)

    def format_status(self) -> str:
        """Display strategy status with P&L metrics."""
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []

        # Get current state
        connector = self.connectors[self.config.exchange]
        base, quote = self.config.trading_pair.split("-")
        mid_price = connector.get_price_by_type(self.config.trading_pair, PriceType.MidPrice)

        # Balances
        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        # Active orders
        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.extend(["", "  No active orders."])

        lines.append("\n" + "=" * 60)

        # Inventory status
        inventory_ratio = self.get_inventory_ratio()
        bid_spread, ask_spread, _ = self.get_adjusted_spreads()

        lines.append("\n  INVENTORY STATUS")
        lines.append(f"    Current ratio: {inventory_ratio:.1%} {base} / {(1-inventory_ratio):.1%} {quote}")
        lines.append(f"    Target ratio:  {self.config.target_base_ratio:.1%} {base}")

        if inventory_ratio > self.config.max_inventory_ratio:
            lines.append(f"    WARNING: Heavy on {base} - BUY orders disabled")
        elif inventory_ratio < (Decimal("1") - self.config.max_inventory_ratio):
            lines.append(f"    WARNING: Heavy on {quote} - SELL orders disabled")
        else:
            lines.append(f"    Status: Balanced")

        lines.append("\n" + "=" * 60)

        # Spread info
        lines.append("\n  SPREAD STATUS")
        lines.append(f"    Base spreads:     Bid {self.config.base_bid_spread:.2%} | Ask {self.config.base_ask_spread:.2%}")
        lines.append(f"    Adjusted spreads: Bid {bid_spread:.2%} | Ask {ask_spread:.2%}")
        if mid_price:
            lines.append(f"    Mid price: {mid_price:.4f} {quote}")
            lines.append(f"    Bid price: {mid_price * (1 - bid_spread):.4f} | Ask price: {mid_price * (1 + ask_spread):.4f}")

        lines.append("\n" + "=" * 60)

        # P&L tracking
        lines.append("\n  TRADING METRICS")
        lines.append(f"    Total trades: {self.total_trades}")
        lines.append(f"    Buy volume:  {self.total_buy_volume:.4f} {base} ({self.total_buy_quote_volume:.2f} {quote})")
        lines.append(f"    Sell volume: {self.total_sell_volume:.4f} {base} ({self.total_sell_quote_volume:.2f} {quote})")

        # Realized P&L (simplified - assumes matched trades)
        if self.total_buy_volume > 0 and self.total_sell_volume > 0:
            matched_volume = min(self.total_buy_volume, self.total_sell_volume)
            if matched_volume > 0:
                avg_buy = self.total_buy_quote_volume / self.total_buy_volume
                avg_sell = self.total_sell_quote_volume / self.total_sell_volume
                spread_captured = (avg_sell - avg_buy) / avg_buy
                estimated_pnl = matched_volume * (avg_sell - avg_buy)
                lines.append(f"\n    Avg buy price:  {avg_buy:.4f} {quote}")
                lines.append(f"    Avg sell price: {avg_sell:.4f} {quote}")
                lines.append(f"    Spread captured: {spread_captured:.2%}")
                lines.append(f"    Estimated P&L: {estimated_pnl:.2f} {quote}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
