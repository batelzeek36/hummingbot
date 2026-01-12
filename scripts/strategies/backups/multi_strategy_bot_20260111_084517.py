"""
Multi-Strategy Trading Bot

Combines multiple trading strategies in a single script with:
- Market Making (spread capture)
- Grid Trading (volatility profits)
- RSI Directional (momentum trades)
- Per-strategy and global kill switches
- Unified P&L tracking and dashboard

Usage:
    start --script multi_strategy_bot.py

Author: Dollar-A-Day Project
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory


class StrategyStatus(Enum):
    """Status of a sub-strategy."""
    ACTIVE = "Active"
    PAUSED = "Paused"
    KILLED = "Killed"
    WARMING_UP = "Warming Up"


@dataclass
class StrategyMetrics:
    """Tracks metrics for a single sub-strategy."""
    name: str
    status: StrategyStatus = StrategyStatus.WARMING_UP
    total_pnl: Decimal = Decimal("0")
    daily_pnl: Decimal = Decimal("0")
    peak_pnl: Decimal = Decimal("0")
    drawdown: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    buy_volume: Decimal = Decimal("0")
    sell_volume: Decimal = Decimal("0")
    last_trade_time: Optional[datetime] = None

    def update_pnl(self, pnl_change: Decimal):
        """Update P&L and calculate drawdown."""
        self.total_pnl += pnl_change
        self.daily_pnl += pnl_change

        if self.total_pnl > self.peak_pnl:
            self.peak_pnl = self.total_pnl

        if self.peak_pnl > 0:
            self.drawdown = (self.peak_pnl - self.total_pnl) / self.peak_pnl * 100
        else:
            self.drawdown = Decimal("0")

    def reset_daily(self):
        """Reset daily metrics."""
        self.daily_pnl = Decimal("0")


# =============================================================================
# CONFIGURATION
# =============================================================================

class MultiStrategyConfig(BaseClientModel):
    """Configuration for Multi-Strategy Bot."""

    script_file_name: str = Field(default="multi_strategy_bot.py")

    # === EXCHANGE SETTINGS ===
    exchange: str = Field(
        default="kraken",
        description="Exchange to trade on"
    )

    # === TRADING PAIRS ===
    # Can use different pairs for different strategies
    mm_trading_pair: str = Field(
        default="XRP-USD",
        description="Trading pair for Market Making"
    )
    grid_trading_pair: str = Field(
        default="XRP-USD",
        description="Trading pair for Grid Trading"
    )
    rsi_trading_pair: str = Field(
        default="XRP-USD",
        description="Trading pair for RSI Directional"
    )

    # === CAPITAL ALLOCATION ===
    # CURRENT CONFIG: $100 starting capital
    # To upgrade to $250: mm_order=1, grid_order=0.5, grid_levels=5, rsi_order=0.5
    # How much of your balance each strategy can use (percentages)
    mm_capital_pct: Decimal = Field(
        default=Decimal("30"),
        description="% of capital for Market Making"
    )
    grid_capital_pct: Decimal = Field(
        default=Decimal("40"),
        description="% of capital for Grid Trading"
    )
    rsi_capital_pct: Decimal = Field(
        default=Decimal("30"),
        description="% of capital for RSI Directional"
    )

    # === MARKET MAKING SETTINGS ===
    mm_enabled: bool = Field(default=True)
    mm_order_amount: Decimal = Field(default=Decimal("2"))  # 2 XRP ~$4.20 per order (Kraken min 1.65)
    mm_bid_spread: Decimal = Field(default=Decimal("0.003"))
    mm_ask_spread: Decimal = Field(default=Decimal("0.003"))
    mm_order_refresh_time: int = Field(default=30)
    mm_inventory_skew: bool = Field(default=True)
    mm_target_ratio: Decimal = Field(default=Decimal("0.5"))

    # === GRID TRADING SETTINGS ===
    grid_enabled: bool = Field(default=True)
    grid_levels: int = Field(default=2)  # Reduced for $100 capital
    grid_spacing_pct: Decimal = Field(default=Decimal("0.005"))  # 0.5% between levels (tighter for more activity)
    grid_order_amount: Decimal = Field(default=Decimal("2"))  # 2 XRP ~$4.20 per order (Kraken min 1.65)
    grid_rebalance_threshold: Decimal = Field(default=Decimal("0.02"))  # 2%

    # === RSI DIRECTIONAL SETTINGS ===
    rsi_enabled: bool = Field(default=True)
    rsi_period: int = Field(default=14)
    rsi_oversold: int = Field(default=30)
    rsi_overbought: int = Field(default=70)
    rsi_order_amount: Decimal = Field(default=Decimal("2"))  # 2 XRP ~$4.20 per order (Kraken min 1.65)
    rsi_take_profit_pct: Decimal = Field(default=Decimal("0.02"))  # 2%
    rsi_stop_loss_pct: Decimal = Field(default=Decimal("0.01"))  # 1%
    rsi_cooldown_minutes: int = Field(default=60)

    # === KILL SWITCH SETTINGS ===
    # Per-strategy limits
    mm_max_loss: Decimal = Field(
        default=Decimal("20"),
        description="Max loss ($) before killing MM strategy"
    )
    grid_max_loss: Decimal = Field(
        default=Decimal("30"),
        description="Max loss ($) before killing Grid strategy"
    )
    rsi_max_loss: Decimal = Field(
        default=Decimal("25"),
        description="Max loss ($) before killing RSI strategy"
    )

    # Global limits
    global_max_loss: Decimal = Field(
        default=Decimal("50"),
        description="Max total loss ($) before killing ALL strategies"
    )
    global_max_drawdown_pct: Decimal = Field(
        default=Decimal("10"),
        description="Max drawdown (%) before killing ALL strategies"
    )
    daily_loss_limit: Decimal = Field(
        default=Decimal("25"),
        description="Max daily loss ($) before pausing until next day"
    )

    # === TIMING ===
    main_tick_interval: int = Field(
        default=10,
        description="Seconds between main loop ticks"
    )


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class MultiStrategyBot(ScriptStrategyBase):
    """
    Multi-strategy trading bot combining:
    1. Market Making - Spread capture around mid-price
    2. Grid Trading - Fixed levels for volatility profits
    3. RSI Directional - Momentum-based entries

    Each strategy has independent P&L tracking and kill switches.
    """

    # Class-level market definition
    markets = {"kraken": {"XRP-USD"}}

    @classmethod
    def init_markets(cls, config: MultiStrategyConfig):
        pairs = {config.mm_trading_pair, config.grid_trading_pair, config.rsi_trading_pair}
        cls.markets = {config.exchange: pairs}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: MultiStrategyConfig = None):
        super().__init__(connectors)
        self.config = config if config is not None else MultiStrategyConfig()

        # Initialize metrics for each strategy
        self.metrics: Dict[str, StrategyMetrics] = {
            "mm": StrategyMetrics(name="Market Making"),
            "grid": StrategyMetrics(name="Grid Trading"),
            "rsi": StrategyMetrics(name="RSI Directional"),
        }

        # Global metrics
        self.global_pnl = Decimal("0")
        self.global_peak_pnl = Decimal("0")
        self.global_drawdown = Decimal("0")
        self.bot_start_time = datetime.now()
        self.last_daily_reset = datetime.now().date()

        # Timing
        self.last_tick = 0
        self.mm_last_refresh = 0

        # Grid state
        self.grid_orders: Dict[str, Dict] = {}  # order_id -> {level, side, price}
        self.grid_initialized = False
        self.grid_base_price: Optional[Decimal] = None

        # RSI state
        self.rsi_position: Optional[str] = None  # "long", "short", or None
        self.rsi_entry_price: Optional[Decimal] = None
        self.rsi_last_signal_time: Optional[datetime] = None
        self.rsi_values: List[Decimal] = []

        # Candles for RSI calculation
        self.candles_ready = False

        # Order tracking by strategy
        self.strategy_orders: Dict[str, set] = {
            "mm": set(),
            "grid": set(),
            "rsi": set(),
        }

        # Kill switch state
        self.global_killed = False
        self.daily_paused = False

        # Track if strategies have been activated
        self._strategies_activated = False

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def on_tick(self):
        """Main loop - runs all strategies."""

        # Activate strategies on first tick (fallback if on_start didn't run)
        if not self._strategies_activated:
            self.logger().info("Activating strategies...")
            if self.config.mm_enabled:
                self.metrics["mm"].status = StrategyStatus.ACTIVE
                self.logger().info("Market Making: ACTIVATED")
            if self.config.grid_enabled:
                self.metrics["grid"].status = StrategyStatus.ACTIVE
                self.logger().info("Grid Trading: ACTIVATED")
            if self.config.rsi_enabled:
                self.metrics["rsi"].status = StrategyStatus.WARMING_UP
                self.logger().info("RSI Directional: WARMING UP")
            self._strategies_activated = True

        # Check if it's a new day
        self._check_daily_reset()

        # Check kill switches first
        if self.global_killed:
            return

        if self.daily_paused:
            return

        self._check_kill_switches()

        if self.global_killed or self.daily_paused:
            return

        # Throttle main tick
        if self.last_tick > self.current_timestamp - self.config.main_tick_interval:
            return

        self.last_tick = self.current_timestamp

        # Run each enabled strategy
        if self.config.mm_enabled and self.metrics["mm"].status == StrategyStatus.ACTIVE:
            self._run_market_making()

        if self.config.grid_enabled and self.metrics["grid"].status == StrategyStatus.ACTIVE:
            self._run_grid_trading()

        if self.config.rsi_enabled and self.metrics["rsi"].status == StrategyStatus.ACTIVE:
            self._run_rsi_directional()

    def on_start(self):
        """Called when strategy starts."""
        self.logger().info("Multi-Strategy Bot starting...")

        # Activate enabled strategies
        if self.config.mm_enabled:
            self.metrics["mm"].status = StrategyStatus.ACTIVE
        if self.config.grid_enabled:
            self.metrics["grid"].status = StrategyStatus.ACTIVE
        if self.config.rsi_enabled:
            self.metrics["rsi"].status = StrategyStatus.WARMING_UP  # Needs candles

    def on_stop(self):
        """Called when strategy stops."""
        self.logger().info("Multi-Strategy Bot stopping...")
        self._cancel_all_orders()

    # =========================================================================
    # MARKET MAKING STRATEGY
    # =========================================================================

    def _run_market_making(self):
        """Market making: place bid/ask orders around mid-price."""

        # Check refresh timing
        if self.mm_last_refresh > self.current_timestamp - self.config.mm_order_refresh_time:
            return

        self.mm_last_refresh = self.current_timestamp

        # Cancel existing MM orders
        self._cancel_strategy_orders("mm")

        connector = self.connectors[self.config.exchange]
        pair = self.config.mm_trading_pair

        mid_price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if mid_price is None:
            return

        # Calculate inventory ratio
        base, quote = pair.split("-")
        base_balance = connector.get_available_balance(base)
        quote_balance = connector.get_available_balance(quote)

        base_value = base_balance * mid_price
        total_value = base_value + quote_balance

        if total_value == 0:
            return

        inventory_ratio = base_value / total_value

        # Adjust spreads based on inventory
        bid_spread = self.config.mm_bid_spread
        ask_spread = self.config.mm_ask_spread

        if self.config.mm_inventory_skew:
            deviation = inventory_ratio - self.config.mm_target_ratio
            adjustment = min(max(deviation * Decimal("1"), Decimal("-0.005")), Decimal("0.005"))
            bid_spread = max(Decimal("0.0005"), bid_spread + adjustment)
            ask_spread = max(Decimal("0.0005"), min(ask_spread - adjustment, Decimal("0.01")))

        # Create orders
        orders = []

        # Buy order (if not too heavy on base)
        if inventory_ratio < Decimal("0.8"):
            buy_price = mid_price * (Decimal("1") - bid_spread)
            orders.append(OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=self.config.mm_order_amount,
                price=buy_price
            ))

        # Sell order (if not too heavy on quote)
        if inventory_ratio > Decimal("0.2"):
            sell_price = mid_price * (Decimal("1") + ask_spread)
            orders.append(OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=self.config.mm_order_amount,
                price=sell_price
            ))

        # Adjust to budget and place
        adjusted = connector.budget_checker.adjust_candidates(orders, all_or_none=False)
        for order in adjusted:
            if order.amount > 0:
                order_id = self._place_order(order, "mm")
                if order_id:
                    self.strategy_orders["mm"].add(order_id)

    # =========================================================================
    # GRID TRADING STRATEGY
    # =========================================================================

    def _run_grid_trading(self):
        """Grid trading: place orders at fixed price levels."""

        connector = self.connectors[self.config.exchange]
        pair = self.config.grid_trading_pair

        mid_price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if mid_price is None:
            return

        # Initialize grid if needed
        if not self.grid_initialized:
            self._initialize_grid(mid_price)
            return

        # Check if price has moved too far from grid base (need to rebalance)
        price_deviation = abs(mid_price - self.grid_base_price) / self.grid_base_price
        if price_deviation > self.config.grid_rebalance_threshold:
            self.logger().info(f"Grid rebalancing: price moved {price_deviation:.2%}")
            self._cancel_strategy_orders("grid")
            self._initialize_grid(mid_price)
            return

        # Check for missing grid levels and replace
        self._maintain_grid_orders(mid_price)

    def _initialize_grid(self, base_price: Decimal):
        """Set up initial grid levels."""

        self.grid_base_price = base_price
        self._cancel_strategy_orders("grid")
        self.grid_orders.clear()

        connector = self.connectors[self.config.exchange]
        pair = self.config.grid_trading_pair
        spacing = self.config.grid_spacing_pct
        levels = self.config.grid_levels

        orders = []

        # Create buy orders below current price
        for i in range(1, levels + 1):
            price = base_price * (Decimal("1") - spacing * i)
            orders.append(OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=self.config.grid_order_amount,
                price=price
            ))

        # Create sell orders above current price
        for i in range(1, levels + 1):
            price = base_price * (Decimal("1") + spacing * i)
            orders.append(OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=self.config.grid_order_amount,
                price=price
            ))

        # Adjust to budget and place
        adjusted = connector.budget_checker.adjust_candidates(orders, all_or_none=False)
        for i, order in enumerate(adjusted):
            if order.amount > 0:
                order_id = self._place_order(order, "grid")
                if order_id:
                    self.strategy_orders["grid"].add(order_id)
                    level = (i % self.config.grid_levels) + 1
                    side = "buy" if order.order_side == TradeType.BUY else "sell"
                    self.grid_orders[order_id] = {
                        "level": level,
                        "side": side,
                        "price": order.price
                    }

        self.grid_initialized = True
        self.logger().info(f"Grid initialized with {len(self.grid_orders)} orders around {base_price:.4f}")

    def _maintain_grid_orders(self, current_price: Decimal):
        """Replace any filled grid orders with opposite side."""
        # Grid maintenance happens via did_fill_order callback
        pass

    # =========================================================================
    # RSI DIRECTIONAL STRATEGY
    # =========================================================================

    def _run_rsi_directional(self):
        """RSI-based directional trading."""

        connector = self.connectors[self.config.exchange]
        pair = self.config.rsi_trading_pair

        # Get current price
        current_price = connector.get_price_by_type(pair, PriceType.LastTrade)
        if current_price is None:
            return

        # Simple RSI approximation using price changes
        # (In production, use candles feed for proper RSI)
        rsi = self._calculate_simple_rsi(current_price)

        if rsi is None:
            if self.metrics["rsi"].status == StrategyStatus.WARMING_UP:
                pass  # Still warming up
            return

        # Mark as active once we have RSI
        if self.metrics["rsi"].status == StrategyStatus.WARMING_UP:
            self.metrics["rsi"].status = StrategyStatus.ACTIVE

        # Check cooldown
        if self.rsi_last_signal_time:
            cooldown = timedelta(minutes=self.config.rsi_cooldown_minutes)
            if datetime.now() - self.rsi_last_signal_time < cooldown:
                return

        # Check for exit conditions first if in position
        if self.rsi_position == "long" and self.rsi_entry_price:
            pnl_pct = (current_price - self.rsi_entry_price) / self.rsi_entry_price

            # Take profit
            if pnl_pct >= self.config.rsi_take_profit_pct:
                self._close_rsi_position("take_profit", current_price)
                return

            # Stop loss
            if pnl_pct <= -self.config.rsi_stop_loss_pct:
                self._close_rsi_position("stop_loss", current_price)
                return

            # RSI exit signal (overbought)
            if rsi >= self.config.rsi_overbought:
                self._close_rsi_position("rsi_signal", current_price)
                return

        # Check for entry signals
        if self.rsi_position is None:
            # Oversold - potential buy
            if rsi <= self.config.rsi_oversold:
                self._open_rsi_position("long", current_price)

    def _calculate_simple_rsi(self, current_price: Decimal) -> Optional[Decimal]:
        """Simple RSI calculation using recent prices."""

        self.rsi_values.append(current_price)

        # Need enough data points
        if len(self.rsi_values) < self.config.rsi_period + 1:
            return None

        # Keep only what we need
        self.rsi_values = self.rsi_values[-(self.config.rsi_period + 10):]

        # Calculate gains and losses
        gains = []
        losses = []

        for i in range(1, min(len(self.rsi_values), self.config.rsi_period + 1)):
            change = self.rsi_values[-i] - self.rsi_values[-i-1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(abs(change))

        avg_gain = sum(gains) / len(gains) if gains else Decimal("0")
        avg_loss = sum(losses) / len(losses) if losses else Decimal("0")

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    def _open_rsi_position(self, direction: str, price: Decimal):
        """Open a directional position."""

        connector = self.connectors[self.config.exchange]
        pair = self.config.rsi_trading_pair

        if direction == "long":
            order = OrderCandidate(
                trading_pair=pair,
                is_maker=False,  # Market order for immediate fill
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=self.config.rsi_order_amount,
                price=price * Decimal("1.001")  # Slightly above to ensure fill
            )

            adjusted = connector.budget_checker.adjust_candidates([order], all_or_none=True)
            if adjusted and adjusted[0].amount > 0:
                order_id = self._place_order(adjusted[0], "rsi")
                if order_id:
                    self.strategy_orders["rsi"].add(order_id)
                    self.rsi_position = "long"
                    self.rsi_entry_price = price
                    self.rsi_last_signal_time = datetime.now()
                    self.logger().info(f"RSI: Opened LONG at {price:.4f}")

    def _close_rsi_position(self, reason: str, price: Decimal):
        """Close the directional position."""

        connector = self.connectors[self.config.exchange]
        pair = self.config.rsi_trading_pair
        base, _ = pair.split("-")

        # Get available balance to sell
        balance = connector.get_available_balance(base)
        sell_amount = min(balance, self.config.rsi_order_amount)

        if sell_amount <= 0:
            self.rsi_position = None
            self.rsi_entry_price = None
            return

        order = OrderCandidate(
            trading_pair=pair,
            is_maker=False,
            order_type=OrderType.LIMIT,
            order_side=TradeType.SELL,
            amount=sell_amount,
            price=price * Decimal("0.999")  # Slightly below to ensure fill
        )

        order_id = self._place_order(order, "rsi")
        if order_id:
            self.strategy_orders["rsi"].add(order_id)
            pnl = (price - self.rsi_entry_price) * sell_amount if self.rsi_entry_price else Decimal("0")
            self.logger().info(f"RSI: Closed LONG ({reason}) at {price:.4f}, P&L: {pnl:.4f}")

        self.rsi_position = None
        self.rsi_entry_price = None
        self.rsi_last_signal_time = datetime.now()

    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================

    def _place_order(self, order: OrderCandidate, strategy: str) -> Optional[str]:
        """Place an order and track it."""

        if order.order_side == TradeType.BUY:
            return self.buy(
                connector_name=self.config.exchange,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )
        else:
            return self.sell(
                connector_name=self.config.exchange,
                trading_pair=order.trading_pair,
                amount=order.amount,
                order_type=order.order_type,
                price=order.price
            )

    def _cancel_strategy_orders(self, strategy: str):
        """Cancel all orders for a specific strategy."""

        for order in self.get_active_orders(connector_name=self.config.exchange):
            if order.client_order_id in self.strategy_orders[strategy]:
                self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

        self.strategy_orders[strategy].clear()

        if strategy == "grid":
            self.grid_orders.clear()

    def _cancel_all_orders(self):
        """Cancel all orders from all strategies."""
        for strategy in self.strategy_orders:
            self._cancel_strategy_orders(strategy)

    # =========================================================================
    # FILL HANDLING
    # =========================================================================

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fills and update metrics."""

        order_id = event.order_id
        quote_volume = event.amount * event.price

        # Determine which strategy this fill belongs to
        strategy = None
        for strat, orders in self.strategy_orders.items():
            if order_id in orders:
                strategy = strat
                break

        if strategy is None:
            return

        # Update metrics
        metrics = self.metrics[strategy]
        metrics.total_trades += 1
        metrics.last_trade_time = datetime.now()

        if event.trade_type == TradeType.BUY:
            metrics.buy_volume += event.amount
        else:
            metrics.sell_volume += event.amount

        # Estimate P&L for this fill
        # (Simplified - real P&L needs proper tracking of entry prices)
        if event.trade_type == TradeType.SELL and metrics.buy_volume > 0:
            # Assume some profit from spread
            estimated_profit = quote_volume * Decimal("0.002")  # ~0.2% assumption
            metrics.update_pnl(estimated_profit)
            self.global_pnl += estimated_profit

            if self.global_pnl > self.global_peak_pnl:
                self.global_peak_pnl = self.global_pnl

            if self.global_peak_pnl > 0:
                self.global_drawdown = (self.global_peak_pnl - self.global_pnl) / self.global_peak_pnl * 100

        # Log the fill
        side = "BUY" if event.trade_type == TradeType.BUY else "SELL"
        msg = f"[{strategy.upper()}] {side} {event.amount:.4f} @ {event.price:.4f}"
        self.log_with_clock(logging.INFO, msg)

        # Handle grid replacement
        if strategy == "grid" and order_id in self.grid_orders:
            self._handle_grid_fill(event, order_id)

    def _handle_grid_fill(self, event: OrderFilledEvent, order_id: str):
        """When a grid order fills, place opposite order."""

        grid_info = self.grid_orders.get(order_id)
        if not grid_info:
            return

        connector = self.connectors[self.config.exchange]
        pair = self.config.grid_trading_pair
        spacing = self.config.grid_spacing_pct

        # Place opposite order
        if grid_info["side"] == "buy":
            # Buy filled, place sell above
            new_price = event.price * (Decimal("1") + spacing)
            new_order = OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=event.amount,
                price=new_price
            )
        else:
            # Sell filled, place buy below
            new_price = event.price * (Decimal("1") - spacing)
            new_order = OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=event.amount,
                price=new_price
            )

        # Place the replacement order
        new_order_id = self._place_order(new_order, "grid")
        if new_order_id:
            self.strategy_orders["grid"].add(new_order_id)
            self.grid_orders[new_order_id] = {
                "level": grid_info["level"],
                "side": "sell" if grid_info["side"] == "buy" else "buy",
                "price": new_price
            }

        # Clean up old order
        del self.grid_orders[order_id]
        self.strategy_orders["grid"].discard(order_id)

    # =========================================================================
    # KILL SWITCHES
    # =========================================================================

    def _check_kill_switches(self):
        """Check all kill switch conditions."""

        # Per-strategy kill switches
        if self.metrics["mm"].total_pnl <= -self.config.mm_max_loss:
            self._kill_strategy("mm", "max loss reached")

        if self.metrics["grid"].total_pnl <= -self.config.grid_max_loss:
            self._kill_strategy("grid", "max loss reached")

        if self.metrics["rsi"].total_pnl <= -self.config.rsi_max_loss:
            self._kill_strategy("rsi", "max loss reached")

        # Global kill switches
        if self.global_pnl <= -self.config.global_max_loss:
            self._kill_all("global max loss reached")
            return

        if self.global_drawdown >= self.config.global_max_drawdown_pct:
            self._kill_all(f"max drawdown {self.global_drawdown:.1f}% reached")
            return

        # Daily loss limit
        daily_pnl = sum(m.daily_pnl for m in self.metrics.values())
        if daily_pnl <= -self.config.daily_loss_limit:
            self._pause_daily("daily loss limit reached")

    def _kill_strategy(self, strategy: str, reason: str):
        """Kill a specific strategy."""

        if self.metrics[strategy].status == StrategyStatus.KILLED:
            return

        self.metrics[strategy].status = StrategyStatus.KILLED
        self._cancel_strategy_orders(strategy)

        msg = f"KILL SWITCH: {strategy.upper()} killed - {reason}"
        self.logger().warning(msg)
        self.notify_hb_app_with_timestamp(msg)

    def _kill_all(self, reason: str):
        """Kill all strategies."""

        self.global_killed = True
        self._cancel_all_orders()

        for strat in self.metrics:
            self.metrics[strat].status = StrategyStatus.KILLED

        msg = f"GLOBAL KILL SWITCH: All strategies killed - {reason}"
        self.logger().error(msg)
        self.notify_hb_app_with_timestamp(msg)

    def _pause_daily(self, reason: str):
        """Pause all trading until next day."""

        self.daily_paused = True
        self._cancel_all_orders()

        for strat in self.metrics:
            if self.metrics[strat].status == StrategyStatus.ACTIVE:
                self.metrics[strat].status = StrategyStatus.PAUSED

        msg = f"DAILY PAUSE: Trading paused - {reason}"
        self.logger().warning(msg)
        self.notify_hb_app_with_timestamp(msg)

    def _check_daily_reset(self):
        """Reset daily metrics at midnight."""

        today = datetime.now().date()
        if today > self.last_daily_reset:
            self.last_daily_reset = today

            for metrics in self.metrics.values():
                metrics.reset_daily()

            # Unpause if was daily paused
            if self.daily_paused:
                self.daily_paused = False
                for strat in self.metrics:
                    if self.metrics[strat].status == StrategyStatus.PAUSED:
                        self.metrics[strat].status = StrategyStatus.ACTIVE
                self.logger().info("New day - trading resumed")

    # =========================================================================
    # STATUS DISPLAY
    # =========================================================================

    def format_status(self) -> str:
        """Display unified status dashboard."""

        lines = []

        # Header
        lines.append("")
        lines.append("=" * 70)
        lines.append("  MULTI-STRATEGY BOT STATUS")
        lines.append("=" * 70)

        # Runtime
        runtime = datetime.now() - self.bot_start_time
        hours = runtime.seconds // 3600
        minutes = (runtime.seconds % 3600) // 60
        lines.append(f"  Runtime: {runtime.days}d {hours}h {minutes}m")

        # Global status
        lines.append("")
        lines.append("-" * 70)
        lines.append("  GLOBAL METRICS")
        lines.append("-" * 70)
        lines.append(f"  Total P&L:      ${self.global_pnl:.4f}")
        lines.append(f"  Peak P&L:       ${self.global_peak_pnl:.4f}")
        lines.append(f"  Drawdown:       {self.global_drawdown:.2f}%")

        daily_pnl = sum(m.daily_pnl for m in self.metrics.values())
        lines.append(f"  Daily P&L:      ${daily_pnl:.4f}")

        if self.global_killed:
            lines.append(f"  Status:         KILLED")
        elif self.daily_paused:
            lines.append(f"  Status:         PAUSED (daily limit)")
        else:
            lines.append(f"  Status:         RUNNING")

        # Strategy table
        lines.append("")
        lines.append("-" * 70)
        lines.append("  STRATEGY PERFORMANCE")
        lines.append("-" * 70)
        lines.append(f"  {'Strategy':<18} {'Status':<10} {'P&L':<12} {'DD%':<8} {'Trades':<8}")
        lines.append(f"  {'-'*18} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")

        for key, m in self.metrics.items():
            status_icon = {
                StrategyStatus.ACTIVE: "[ON]",
                StrategyStatus.PAUSED: "[||]",
                StrategyStatus.KILLED: "[X]",
                StrategyStatus.WARMING_UP: "[..]",
            }.get(m.status, "[?]")

            pnl_str = f"${m.total_pnl:.2f}"
            if m.total_pnl >= 0:
                pnl_str = f"+{pnl_str}"

            lines.append(
                f"  {m.name:<18} {status_icon:<10} {pnl_str:<12} "
                f"{m.drawdown:.1f}%{'':<4} {m.total_trades:<8}"
            )

        # Balances
        lines.append("")
        lines.append("-" * 70)
        lines.append("  BALANCES")
        lines.append("-" * 70)

        balance_df = self.get_balance_df()
        for line in balance_df.to_string(index=False).split("\n"):
            lines.append(f"  {line}")

        # Active orders summary
        lines.append("")
        lines.append("-" * 70)
        lines.append("  ACTIVE ORDERS")
        lines.append("-" * 70)

        for strategy in ["mm", "grid", "rsi"]:
            count = len([o for o in self.get_active_orders(connector_name=self.config.exchange)
                        if o.client_order_id in self.strategy_orders[strategy]])
            lines.append(f"  {strategy.upper()}: {count} orders")

        # Kill switch status
        lines.append("")
        lines.append("-" * 70)
        lines.append("  KILL SWITCH STATUS")
        lines.append("-" * 70)
        lines.append(f"  MM Loss Limit:     ${-self.config.mm_max_loss} (current: ${self.metrics['mm'].total_pnl:.2f})")
        lines.append(f"  Grid Loss Limit:   ${-self.config.grid_max_loss} (current: ${self.metrics['grid'].total_pnl:.2f})")
        lines.append(f"  RSI Loss Limit:    ${-self.config.rsi_max_loss} (current: ${self.metrics['rsi'].total_pnl:.2f})")
        lines.append(f"  Global Loss Limit: ${-self.config.global_max_loss} (current: ${self.global_pnl:.2f})")
        lines.append(f"  Max Drawdown:      {self.config.global_max_drawdown_pct}% (current: {self.global_drawdown:.1f}%)")
        lines.append(f"  Daily Loss Limit:  ${-self.config.daily_loss_limit} (current: ${daily_pnl:.2f})")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
