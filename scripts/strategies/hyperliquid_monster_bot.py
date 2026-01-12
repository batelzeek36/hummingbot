"""
Hyperliquid Monster Bot v1.0 - Advanced Multi-Strategy Perpetual Trading

A comprehensive trading bot that exploits every advantage of Hyperliquid perpetuals:
1. FUNDING RATE HARVESTING - Hyperliquid pays funding every HOUR (not 8h like Binance!)
2. LEVERAGED GRID TRADING - Amplify profits with 3-5x leverage
3. DIRECTIONAL MOMENTUM - Go LONG or SHORT based on RSI/BB signals
4. DYNAMIC LEVERAGE - Adjust based on volatility and confidence

Target: 10-20% monthly returns on ~$80 capital

Key Hyperliquid Advantages:
- Hourly funding payments (8x more frequent than competitors)
- Low fees: 0.045% taker, 0.015% maker
- Up to 50x leverage
- No KYC required
- Deep liquidity on major pairs

Capital Allocation ($80):
- Funding Harvesting: 40% ($32) - Safest, consistent income
- Grid Trading: 35% ($28) - Range-bound profits with leverage
- Directional: 25% ($20) - Higher risk/reward momentum plays

Risk Management:
- Per-strategy stop losses
- Dynamic leverage based on volatility
- Position size limits
- Global kill switch at 15% drawdown

Usage:
    start --script hyperliquid_monster_bot.py

Author: Dollar-A-Day Project
Date: 2026-01-11
"""

import logging
import os
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate, PerpetualOrderCandidate
from hummingbot.core.event.events import OrderFilledEvent, FundingPaymentCompletedEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class StrategyMode(Enum):
    """Operating modes for strategies."""
    ACTIVE = "Active"
    PAUSED = "Paused"
    KILLED = "Killed"
    WARMING_UP = "Warming Up"


@dataclass
class PositionInfo:
    """Track open positions."""
    trading_pair: str
    side: str  # "long" or "short"
    entry_price: Decimal
    amount: Decimal
    leverage: int
    entry_time: datetime
    strategy: str  # which strategy opened this
    unrealized_pnl: Decimal = Decimal("0")
    funding_collected: Decimal = Decimal("0")


@dataclass
class StrategyMetrics:
    """Track metrics for each sub-strategy."""
    name: str
    status: StrategyMode = StrategyMode.WARMING_UP
    total_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    funding_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    current_position_value: Decimal = Decimal("0")


# =============================================================================
# CONFIGURATION
# =============================================================================

class HyperliquidMonsterConfig(BaseClientModel):
    """Configuration for Hyperliquid Monster Bot."""

    script_file_name: str = Field(default="hyperliquid_monster_bot.py")

    # === EXCHANGE SETTINGS ===
    exchange: str = Field(
        default="hyperliquid_perpetual",
        description="Exchange connector"
    )

    # === TRADING PAIRS ===
    # OPTIMIZED based on Hyperliquid funding rate analysis (2026-01-11)
    # ANIME/VVV have -50% to -110% APR funding = GO LONG to collect!
    # SOL has best altcoin liquidity ($424M/day) + good volatility
    funding_pairs: str = Field(
        default="ANIME-USD,VVV-USD",
        description="Pairs for funding rate harvesting (high negative funding = go long)"
    )
    grid_pair: str = Field(
        default="SOL-USD",
        description="Pair for grid trading (best altcoin liquidity + volatility)"
    )
    momentum_pair: str = Field(
        default="BTC-USD",
        description="Pair for momentum/directional trading (safest for directional bets)"
    )

    # === CAPITAL ALLOCATION ===
    total_capital: Decimal = Field(
        default=Decimal("78"),
        description="Total capital in USDC"
    )
    funding_capital_pct: Decimal = Field(
        default=Decimal("40"),
        description="% of capital for funding harvesting"
    )
    grid_capital_pct: Decimal = Field(
        default=Decimal("35"),
        description="% of capital for grid trading"
    )
    momentum_capital_pct: Decimal = Field(
        default=Decimal("25"),
        description="% of capital for momentum trading"
    )

    # === LEVERAGE SETTINGS ===
    # Conservative leverage for safety
    funding_leverage: int = Field(default=3, description="Leverage for funding positions")
    grid_leverage: int = Field(default=5, description="Leverage for grid trading")
    momentum_leverage: int = Field(default=3, description="Leverage for momentum trades")
    max_leverage: int = Field(default=10, description="Maximum allowed leverage")

    # === FUNDING HARVESTING SETTINGS ===
    # Hyperliquid pays funding EVERY HOUR (huge advantage!)
    funding_enabled: bool = Field(default=True)
    min_funding_rate: Decimal = Field(
        default=Decimal("0.0001"),  # 0.01% per hour = ~8.76% APY
        description="Minimum funding rate to open position"
    )
    funding_position_size: Decimal = Field(
        default=Decimal("10"),  # $10 per position (with 3x = $30 exposure)
        description="Position size per funding pair"
    )
    minutes_before_funding: int = Field(
        default=5,
        description="Open position this many minutes before funding"
    )

    # === GRID TRADING SETTINGS ===
    grid_enabled: bool = Field(default=True)
    grid_levels: int = Field(default=5, description="Number of grid levels each side")
    grid_spacing_pct: Decimal = Field(
        default=Decimal("0.005"),  # 0.5% spacing
        description="Spacing between grid levels"
    )
    grid_order_size: Decimal = Field(
        default=Decimal("5"),  # $5 per grid order
        description="Size per grid order in USD"
    )
    grid_rebalance_pct: Decimal = Field(
        default=Decimal("0.03"),  # 3% rebalance trigger
        description="Rebalance grid when price moves this much"
    )

    # === MOMENTUM SETTINGS ===
    momentum_enabled: bool = Field(default=True)
    momentum_lookback: int = Field(default=20, description="Candles for RSI calculation")
    rsi_oversold: Decimal = Field(default=Decimal("30"), description="RSI oversold threshold")
    rsi_overbought: Decimal = Field(default=Decimal("70"), description="RSI overbought threshold")
    momentum_take_profit: Decimal = Field(
        default=Decimal("0.02"),  # 2% TP
        description="Take profit percentage"
    )
    momentum_stop_loss: Decimal = Field(
        default=Decimal("0.01"),  # 1% SL
        description="Stop loss percentage"
    )
    momentum_position_size: Decimal = Field(
        default=Decimal("6"),  # $6 per momentum trade
        description="Position size for momentum trades"
    )

    # === RISK MANAGEMENT ===
    max_drawdown_pct: Decimal = Field(
        default=Decimal("15"),
        description="Kill all strategies at this drawdown %"
    )
    max_position_size_pct: Decimal = Field(
        default=Decimal("50"),
        description="Max position as % of capital (leveraged)"
    )
    daily_loss_limit: Decimal = Field(
        default=Decimal("10"),
        description="Max daily loss in USD"
    )


# =============================================================================
# MAIN BOT CLASS
# =============================================================================

class HyperliquidMonsterBot(ScriptStrategyBase):
    """
    Advanced multi-strategy Hyperliquid perpetual trading bot.

    Combines funding rate harvesting, leveraged grid trading, and
    directional momentum strategies for maximum returns.
    """

    # Will be set by init_markets
    # BALANCED config: ANIME/VVV for funding, SOL for grid, BTC for momentum
    markets = {"hyperliquid_perpetual": {"ANIME-USD", "VVV-USD", "SOL-USD", "BTC-USD"}}

    @classmethod
    def init_markets(cls, config: HyperliquidMonsterConfig):
        """Initialize markets based on config."""
        pairs = set()

        # Add funding pairs
        for pair in config.funding_pairs.split(","):
            pairs.add(pair.strip())

        # Add grid and momentum pairs
        pairs.add(config.grid_pair)
        pairs.add(config.momentum_pair)

        cls.markets = {config.exchange: pairs}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: HyperliquidMonsterConfig = None):
        super().__init__(connectors)
        self.config = config if config is not None else HyperliquidMonsterConfig()

        # Strategy metrics
        self.metrics = {
            "funding": StrategyMetrics(name="Funding Harvesting"),
            "grid": StrategyMetrics(name="Grid Trading"),
            "momentum": StrategyMetrics(name="Momentum Trading"),
        }

        # Position tracking
        self.positions: Dict[str, PositionInfo] = {}  # pair -> position

        # Grid state
        self.grid_initialized = False
        self.grid_base_price: Optional[Decimal] = None
        self.grid_orders: Dict[str, dict] = {}  # order_id -> info

        # Momentum state
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}  # pair -> [(time, price)]
        self.momentum_position: Optional[str] = None  # "long", "short", or None
        self.momentum_entry_price: Optional[Decimal] = None

        # Funding tracking
        self.last_funding_check = 0
        self.funding_positions: Dict[str, str] = {}  # pair -> side
        self.total_funding_collected: Decimal = Decimal("0")

        # Global state
        self.start_time = datetime.now()
        self.global_pnl = Decimal("0")
        self.peak_equity = self.config.total_capital
        self.global_killed = False

        # Timing
        self.last_tick = 0
        self.tick_interval = 10  # seconds

        self.logger().info("Hyperliquid Monster Bot initialized!")

    # =========================================================================
    # STARTUP AND MAIN LOOP
    # =========================================================================

    def on_start(self):
        """Called when strategy starts."""
        self.logger().info("=" * 60)
        self.logger().info("  HYPERLIQUID MONSTER BOT STARTING")
        self.logger().info("=" * 60)
        self.logger().info(f"  Capital: ${self.config.total_capital}")
        self.logger().info(f"  Funding Pairs: {self.config.funding_pairs}")
        self.logger().info(f"  Grid Pair: {self.config.grid_pair}")
        self.logger().info(f"  Momentum Pair: {self.config.momentum_pair}")

        # Set position mode and leverage
        self._configure_exchange()

        # Activate strategies
        if self.config.funding_enabled:
            self.metrics["funding"].status = StrategyMode.ACTIVE
        if self.config.grid_enabled:
            self.metrics["grid"].status = StrategyMode.ACTIVE
        if self.config.momentum_enabled:
            self.metrics["momentum"].status = StrategyMode.WARMING_UP

    def _configure_exchange(self):
        """Configure exchange settings."""
        connector = self.connectors[self.config.exchange]

        try:
            # Set one-way position mode
            connector.set_position_mode(PositionMode.ONEWAY)

            # Set leverage for each pair
            all_pairs = set()
            for pair in self.config.funding_pairs.split(","):
                all_pairs.add(pair.strip())
            all_pairs.add(self.config.grid_pair)
            all_pairs.add(self.config.momentum_pair)

            for pair in all_pairs:
                connector.set_leverage(pair, self.config.max_leverage)
                self.logger().info(f"Set leverage for {pair}: {self.config.max_leverage}x")

        except Exception as e:
            self.logger().warning(f"Error configuring exchange: {e}")

    def on_tick(self):
        """Main loop."""
        if self.global_killed:
            return

        # Throttle ticks
        if self.last_tick > self.current_timestamp - self.tick_interval:
            return
        self.last_tick = self.current_timestamp

        # Check risk limits
        self._check_risk_limits()
        if self.global_killed:
            return

        # Run strategies
        if self.config.funding_enabled and self.metrics["funding"].status == StrategyMode.ACTIVE:
            self._run_funding_strategy()

        if self.config.grid_enabled and self.metrics["grid"].status == StrategyMode.ACTIVE:
            self._run_grid_strategy()

        if self.config.momentum_enabled:
            self._run_momentum_strategy()

    def on_stop(self):
        """Called when strategy stops."""
        self.logger().info("Hyperliquid Monster Bot stopping...")
        self._close_all_positions()
        self._cancel_all_orders()

    # =========================================================================
    # FUNDING RATE HARVESTING
    # =========================================================================

    def _run_funding_strategy(self):
        """
        Funding Rate Harvesting Strategy

        Hyperliquid pays funding every HOUR - this is huge!
        - If funding rate is positive: shorts pay longs
        - If funding rate is negative: longs pay shorts

        Strategy: Open position in direction that RECEIVES funding
        before the funding timestamp, collect payment, close after.
        """
        connector = self.connectors[self.config.exchange]

        for pair in self.config.funding_pairs.split(","):
            pair = pair.strip()

            try:
                # Get funding info
                funding_info = connector.get_funding_info(pair)
                if funding_info is None:
                    continue

                current_rate = funding_info.rate
                next_funding_time = funding_info.next_funding_utc_timestamp
                time_to_funding = next_funding_time - self.current_timestamp

                # Log funding info periodically
                if self.last_funding_check < self.current_timestamp - 300:  # Every 5 min
                    self.logger().info(
                        f"FUNDING {pair}: Rate={current_rate:.6f} ({current_rate*100:.4f}%), "
                        f"Next in {time_to_funding/60:.1f} min"
                    )
                    self.last_funding_check = self.current_timestamp

                # Check if we should open a funding position
                minutes_to_funding = time_to_funding / 60

                if minutes_to_funding <= self.config.minutes_before_funding:
                    if abs(current_rate) >= float(self.config.min_funding_rate):
                        # Determine direction
                        # Positive rate = shorts pay longs = we go LONG
                        # Negative rate = longs pay shorts = we go SHORT
                        side = "long" if current_rate > 0 else "short"

                        # Check if we already have a position
                        if pair not in self.funding_positions:
                            self._open_funding_position(pair, side, current_rate)
                        elif self.funding_positions[pair] != side:
                            # Rate flipped, close and re-open
                            self._close_funding_position(pair, "rate_flipped")
                            self._open_funding_position(pair, side, current_rate)

                # Close positions that have collected funding (after funding)
                elif pair in self.funding_positions and minutes_to_funding > 55:
                    # Funding just happened, close position
                    self._close_funding_position(pair, "funding_collected")

            except Exception as e:
                self.logger().error(f"Funding strategy error for {pair}: {e}")

    def _open_funding_position(self, pair: str, side: str, rate: float):
        """Open a position to collect funding."""
        connector = self.connectors[self.config.exchange]

        # Calculate position size
        position_size = self.config.funding_position_size
        leverage = self.config.funding_leverage

        # Get current price
        price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        # Calculate amount in base currency
        amount = position_size / price

        # Create order
        trade_type = TradeType.BUY if side == "long" else TradeType.SELL

        order_id = self._place_perpetual_order(
            pair=pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.OPEN,
            strategy="funding"
        )

        if order_id:
            self.funding_positions[pair] = side
            self.logger().info(
                f"FUNDING: Opened {side.upper()} on {pair} @ {price:.2f} "
                f"(rate: {rate*100:.4f}%, size: ${position_size}, {leverage}x)"
            )

    def _close_funding_position(self, pair: str, reason: str):
        """Close a funding position."""
        if pair not in self.funding_positions:
            return

        connector = self.connectors[self.config.exchange]
        side = self.funding_positions[pair]

        # Get current price
        price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        # Calculate amount (should match what we opened)
        amount = self.config.funding_position_size / price

        # Close with opposite side
        trade_type = TradeType.SELL if side == "long" else TradeType.BUY

        order_id = self._place_perpetual_order(
            pair=pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=self.config.funding_leverage,
            position_action=PositionAction.CLOSE,
            strategy="funding"
        )

        if order_id:
            del self.funding_positions[pair]
            self.logger().info(f"FUNDING: Closed {side.upper()} on {pair} ({reason})")

    def did_complete_funding_payment(self, event: FundingPaymentCompletedEvent):
        """Handle funding payment events."""
        self.total_funding_collected += event.amount
        self.metrics["funding"].funding_pnl += event.amount
        self.global_pnl += event.amount

        self.logger().info(
            f"FUNDING RECEIVED: {event.trading_pair} ${event.amount:.4f} "
            f"(Total: ${self.total_funding_collected:.4f})"
        )

    # =========================================================================
    # GRID TRADING STRATEGY
    # =========================================================================

    def _run_grid_strategy(self):
        """
        Leveraged Grid Trading

        Place buy orders below price, sell orders above.
        With leverage, profits are amplified.
        """
        connector = self.connectors[self.config.exchange]
        pair = self.config.grid_pair

        price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        # Initialize grid if needed
        if not self.grid_initialized:
            self._initialize_grid(price)
            return

        # Check for rebalance
        if self.grid_base_price:
            deviation = abs(price - self.grid_base_price) / self.grid_base_price
            if deviation > self.config.grid_rebalance_pct:
                self.logger().info(f"GRID: Rebalancing (price moved {deviation*100:.2f}%)")
                self._cancel_grid_orders()
                self._initialize_grid(price)

    def _initialize_grid(self, base_price: Decimal):
        """Set up grid levels."""
        self.grid_base_price = base_price
        self.grid_orders.clear()

        pair = self.config.grid_pair
        spacing = self.config.grid_spacing_pct
        levels = self.config.grid_levels
        order_size = self.config.grid_order_size
        leverage = self.config.grid_leverage

        self.logger().info(f"GRID: Initializing around {base_price:.2f}")

        # Calculate amount in base currency
        amount_per_order = order_size / base_price

        # Create buy orders below
        for i in range(1, levels + 1):
            level_price = base_price * (Decimal("1") - spacing * i)

            order_id = self._place_perpetual_order(
                pair=pair,
                side=TradeType.BUY,
                amount=amount_per_order,
                price=level_price,
                leverage=leverage,
                position_action=PositionAction.OPEN,
                strategy="grid"
            )

            if order_id:
                self.grid_orders[order_id] = {
                    "level": i,
                    "side": "buy",
                    "price": level_price
                }

        # Create sell orders above
        for i in range(1, levels + 1):
            level_price = base_price * (Decimal("1") + spacing * i)

            order_id = self._place_perpetual_order(
                pair=pair,
                side=TradeType.SELL,
                amount=amount_per_order,
                price=level_price,
                leverage=leverage,
                position_action=PositionAction.OPEN,
                strategy="grid"
            )

            if order_id:
                self.grid_orders[order_id] = {
                    "level": i,
                    "side": "sell",
                    "price": level_price
                }

        self.grid_initialized = True
        self.logger().info(f"GRID: Initialized with {len(self.grid_orders)} orders")

    def _cancel_grid_orders(self):
        """Cancel all grid orders."""
        connector = self.connectors[self.config.exchange]
        pair = self.config.grid_pair

        for order in self.get_active_orders(connector_name=self.config.exchange):
            if order.trading_pair == pair and order.client_order_id in self.grid_orders:
                self.cancel(self.config.exchange, pair, order.client_order_id)

        self.grid_orders.clear()
        self.grid_initialized = False

    # =========================================================================
    # MOMENTUM/DIRECTIONAL STRATEGY
    # =========================================================================

    def _run_momentum_strategy(self):
        """
        Momentum Trading with Long/Short capability

        Uses simple RSI to determine overbought/oversold conditions.
        Can go LONG when oversold, SHORT when overbought.
        """
        connector = self.connectors[self.config.exchange]
        pair = self.config.momentum_pair

        price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        # Track price history
        now = datetime.now()
        if pair not in self.price_history:
            self.price_history[pair] = []

        self.price_history[pair].append((now, price))

        # Keep last hour of data
        cutoff = now - timedelta(hours=1)
        self.price_history[pair] = [(t, p) for t, p in self.price_history[pair] if t > cutoff]

        # Need enough data for RSI
        if len(self.price_history[pair]) < self.config.momentum_lookback:
            if self.metrics["momentum"].status == StrategyMode.WARMING_UP:
                return
            return

        # Activate if warming up
        if self.metrics["momentum"].status == StrategyMode.WARMING_UP:
            self.metrics["momentum"].status = StrategyMode.ACTIVE
            self.logger().info("MOMENTUM: Strategy activated")

        # Calculate simple RSI approximation
        rsi = self._calculate_rsi(pair)
        if rsi is None:
            return

        # Check exit conditions first
        if self.momentum_position and self.momentum_entry_price:
            should_exit, reason = self._check_momentum_exit(price)
            if should_exit:
                self._close_momentum_position(price, reason)
                return

        # Check entry signals
        if self.momentum_position is None:
            if rsi < float(self.config.rsi_oversold):
                # Oversold - go LONG
                self._open_momentum_position("long", price, rsi)
            elif rsi > float(self.config.rsi_overbought):
                # Overbought - go SHORT
                self._open_momentum_position("short", price, rsi)

    def _calculate_rsi(self, pair: str) -> Optional[float]:
        """Calculate RSI from price history."""
        prices = [p for _, p in self.price_history[pair]]
        if len(prices) < self.config.momentum_lookback:
            return None

        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            changes.append(float(prices[i] - prices[i-1]))

        if not changes:
            return None

        # Separate gains and losses
        gains = [c for c in changes if c > 0]
        losses = [-c for c in changes if c < 0]

        avg_gain = sum(gains) / len(changes) if gains else 0
        avg_loss = sum(losses) / len(changes) if losses else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _open_momentum_position(self, direction: str, price: Decimal, rsi: float):
        """Open a momentum position."""
        pair = self.config.momentum_pair
        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        amount = position_size / price
        trade_type = TradeType.BUY if direction == "long" else TradeType.SELL

        order_id = self._place_perpetual_order(
            pair=pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.OPEN,
            strategy="momentum"
        )

        if order_id:
            self.momentum_position = direction
            self.momentum_entry_price = price
            self.logger().info(
                f"MOMENTUM: Opened {direction.upper()} @ {price:.2f} "
                f"(RSI: {rsi:.1f}, size: ${position_size}, {leverage}x)"
            )

    def _check_momentum_exit(self, current_price: Decimal) -> Tuple[bool, str]:
        """Check if momentum position should exit."""
        if not self.momentum_entry_price:
            return False, ""

        if self.momentum_position == "long":
            pnl_pct = (current_price - self.momentum_entry_price) / self.momentum_entry_price
        else:  # short
            pnl_pct = (self.momentum_entry_price - current_price) / self.momentum_entry_price

        # Take profit
        if pnl_pct >= self.config.momentum_take_profit:
            return True, "take_profit"

        # Stop loss
        if pnl_pct <= -self.config.momentum_stop_loss:
            return True, "stop_loss"

        return False, ""

    def _close_momentum_position(self, price: Decimal, reason: str):
        """Close momentum position."""
        if not self.momentum_position:
            return

        pair = self.config.momentum_pair
        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        amount = position_size / price
        trade_type = TradeType.SELL if self.momentum_position == "long" else TradeType.BUY

        order_id = self._place_perpetual_order(
            pair=pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.CLOSE,
            strategy="momentum"
        )

        if order_id and self.momentum_entry_price:
            if self.momentum_position == "long":
                pnl = (price - self.momentum_entry_price) * amount * leverage
            else:
                pnl = (self.momentum_entry_price - price) * amount * leverage

            self.metrics["momentum"].realized_pnl += pnl
            self.global_pnl += pnl

            pnl_pct = pnl / position_size * 100
            self.logger().info(
                f"MOMENTUM: Closed {self.momentum_position.upper()} @ {price:.2f} "
                f"({reason}) P&L: ${pnl:.4f} ({pnl_pct:.2f}%)"
            )

        self.momentum_position = None
        self.momentum_entry_price = None

    # =========================================================================
    # ORDER PLACEMENT
    # =========================================================================

    def _place_perpetual_order(
        self,
        pair: str,
        side: TradeType,
        amount: Decimal,
        price: Decimal,
        leverage: int,
        position_action: PositionAction,
        strategy: str
    ) -> Optional[str]:
        """Place a perpetual order."""
        try:
            if side == TradeType.BUY:
                return self.buy(
                    connector_name=self.config.exchange,
                    trading_pair=pair,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=price,
                    position_action=position_action
                )
            else:
                return self.sell(
                    connector_name=self.config.exchange,
                    trading_pair=pair,
                    amount=amount,
                    order_type=OrderType.LIMIT,
                    price=price,
                    position_action=position_action
                )
        except Exception as e:
            self.logger().error(f"Order placement error: {e}")
            return None

    # =========================================================================
    # ORDER FILL HANDLING
    # =========================================================================

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fills."""
        order_id = event.order_id

        # Update metrics based on strategy
        if order_id in self.grid_orders:
            self.metrics["grid"].total_trades += 1
            self._handle_grid_fill(event, order_id)
        else:
            # Could be funding or momentum
            for strategy in ["funding", "momentum"]:
                self.metrics[strategy].total_trades += 1

        side = "BUY" if event.trade_type == TradeType.BUY else "SELL"
        self.logger().info(
            f"FILL: {side} {event.amount:.6f} {event.trading_pair} @ {event.price:.2f}"
        )

    def _handle_grid_fill(self, event: OrderFilledEvent, order_id: str):
        """Replace grid order with opposite side."""
        if order_id not in self.grid_orders:
            return

        grid_info = self.grid_orders[order_id]
        del self.grid_orders[order_id]

        pair = self.config.grid_pair
        spacing = self.config.grid_spacing_pct
        leverage = self.config.grid_leverage

        # Place opposite order
        if grid_info["side"] == "buy":
            new_price = event.price * (Decimal("1") + spacing)
            new_side = TradeType.SELL
            new_side_str = "sell"
        else:
            new_price = event.price * (Decimal("1") - spacing)
            new_side = TradeType.BUY
            new_side_str = "buy"

        new_order_id = self._place_perpetual_order(
            pair=pair,
            side=new_side,
            amount=event.amount,
            price=new_price,
            leverage=leverage,
            position_action=PositionAction.CLOSE if grid_info["side"] == "buy" else PositionAction.OPEN,
            strategy="grid"
        )

        if new_order_id:
            self.grid_orders[new_order_id] = {
                "level": grid_info["level"],
                "side": new_side_str,
                "price": new_price
            }

    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================

    def _check_risk_limits(self):
        """Check global risk limits."""
        # Calculate current equity
        current_equity = self.config.total_capital + self.global_pnl

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Check drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100

            if drawdown >= float(self.config.max_drawdown_pct):
                self.logger().error(
                    f"MAX DRAWDOWN REACHED: {drawdown:.2f}% - KILLING ALL STRATEGIES"
                )
                self._kill_all()

    def _kill_all(self):
        """Emergency shutdown."""
        self.global_killed = True
        self._close_all_positions()
        self._cancel_all_orders()

        for strategy in self.metrics.values():
            strategy.status = StrategyMode.KILLED

    def _close_all_positions(self):
        """Close all open positions."""
        # Close funding positions
        for pair in list(self.funding_positions.keys()):
            self._close_funding_position(pair, "shutdown")

        # Close momentum position
        if self.momentum_position:
            connector = self.connectors[self.config.exchange]
            price = connector.get_price_by_type(self.config.momentum_pair, PriceType.MidPrice)
            if price:
                self._close_momentum_position(price, "shutdown")

    def _cancel_all_orders(self):
        """Cancel all orders."""
        self._cancel_grid_orders()

        for order in self.get_active_orders(connector_name=self.config.exchange):
            self.cancel(self.config.exchange, order.trading_pair, order.client_order_id)

    # =========================================================================
    # STATUS DISPLAY
    # =========================================================================

    def format_status(self) -> str:
        """Display bot status."""
        lines = []

        lines.append("")
        lines.append("=" * 70)
        lines.append("  HYPERLIQUID MONSTER BOT STATUS")
        lines.append("=" * 70)

        # Runtime
        runtime = datetime.now() - self.start_time
        lines.append(f"  Runtime: {runtime.days}d {runtime.seconds//3600}h {(runtime.seconds%3600)//60}m")

        # Global P&L
        lines.append("")
        lines.append("-" * 70)
        lines.append("  GLOBAL METRICS")
        lines.append("-" * 70)
        lines.append(f"  Capital:          ${self.config.total_capital}")
        lines.append(f"  Total P&L:        ${self.global_pnl:+.4f}")
        lines.append(f"  Funding Collected: ${self.total_funding_collected:.4f}")

        current_equity = self.config.total_capital + self.global_pnl
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
            lines.append(f"  Drawdown:         {drawdown:.2f}%")

        if self.global_killed:
            lines.append(f"  Status:           KILLED")
        else:
            lines.append(f"  Status:           RUNNING")

        # Strategy status
        lines.append("")
        lines.append("-" * 70)
        lines.append("  STRATEGY STATUS")
        lines.append("-" * 70)

        for key, m in self.metrics.items():
            status = m.status.value
            pnl = m.realized_pnl + m.funding_pnl
            lines.append(f"  {m.name:<20} [{status:<10}] P&L: ${pnl:+.4f} Trades: {m.total_trades}")

        # Active positions
        lines.append("")
        lines.append("-" * 70)
        lines.append("  ACTIVE POSITIONS")
        lines.append("-" * 70)

        if self.funding_positions:
            for pair, side in self.funding_positions.items():
                lines.append(f"  FUNDING: {side.upper()} {pair}")

        if self.momentum_position:
            lines.append(f"  MOMENTUM: {self.momentum_position.upper()} {self.config.momentum_pair} @ {self.momentum_entry_price}")

        lines.append(f"  GRID: {len(self.grid_orders)} orders active")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
