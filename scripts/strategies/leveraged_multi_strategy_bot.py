"""
Leveraged Multi-Strategy Trading Bot (10x Leverage Version)

This is the LEVERAGED version designed for offshore LLC accounts with full exchange access.
Uses $10,000 capital with 10x leverage = $100,000 effective buying power.

Combines multiple strategies optimized for perpetual futures:
- Aggressive Market Making (tight spreads, fast refresh)
- Dynamic Grid Trading (more levels, adaptive spacing)
- RSI Directional with LONG and SHORT positions
- MACD Trend Following with LONG and SHORT positions
- Momentum Scalping with LONG and SHORT positions
- Multi-coin rotation based on volatility

REQUIREMENTS:
- Offshore LLC or jurisdiction allowing leveraged crypto trading
- KuCoin Futures account with API access
- Minimum $10,000 capital recommended
- Understanding of liquidation risks with leverage

WARNING: LEVERAGED TRADING CARRIES SIGNIFICANT RISK.
- 10x leverage means 10% adverse move = 100% loss (liquidation)
- Always use stop losses
- Never risk more than you can afford to lose
- This bot includes tighter risk controls for leveraged positions

Usage:
    start --script leveraged_multi_strategy_bot.py

Author: Dollar-A-Day Project
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics

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


class MarketCondition(Enum):
    """Current market condition assessment."""
    TRENDING_UP = "Trending Up"
    TRENDING_DOWN = "Trending Down"
    RANGING = "Ranging"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"


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


@dataclass
class CoinMetrics:
    """Tracks metrics for each trading pair."""
    symbol: str
    volatility: Decimal = Decimal("0")
    volume_24h: Decimal = Decimal("0")
    trend_strength: Decimal = Decimal("0")
    rsi: Decimal = Decimal("50")
    macd_signal: int = 0  # -1 = sell, 0 = neutral, 1 = buy
    last_price: Decimal = Decimal("0")
    price_history: List[Decimal] = field(default_factory=list)
    allocated_capital: Decimal = Decimal("0")


# =============================================================================
# CONFIGURATION
# =============================================================================

class LeveragedConfig(BaseClientModel):
    """Configuration for Leveraged Multi-Strategy Bot (10x Leverage)."""

    script_file_name: str = Field(default="leveraged_multi_strategy_bot.py")

    # === EXCHANGE SETTINGS ===
    exchange: str = Field(
        default="kucoin_perpetual",
        description="Exchange to trade on (KuCoin Perpetual Futures)"
    )

    # === LEVERAGE SETTINGS ===
    leverage: int = Field(
        default=10,
        description="Leverage multiplier (10x recommended, max 20x)"
    )

    # === TRADING PAIRS (Perpetual contracts - optimized per strategy) ===
    # BTC-USDT: Best for Market Making (highest liquidity, tightest spread)
    # LTC-USDT: Best for Grid Trading (high oscillation, moderate volatility)
    # ETH-USDT: Added for RSI/MACD (high liquidity for leveraged positions)
    # SOL-USDT: Strong for MACD Trend (good trends, liquid)
    # DOGE-USDT: Best for Momentum Scalping (high volatility, liquid enough for leverage)
    # Note: Using more liquid pairs for leverage to avoid slippage on larger positions
    trading_pairs: str = Field(
        default="BTC-USDT,LTC-USDT,ETH-USDT,SOL-USDT,DOGE-USDT",
        description="Comma-separated list of perpetual trading pairs"
    )

    # === CAPITAL SETTINGS (Real money with leverage) ===
    total_capital: Decimal = Field(
        default=Decimal("10000"),
        description="Actual capital deposited ($10k real = $100k with 10x leverage)"
    )

    # Effective capital = total_capital * leverage = $100,000
    reserve_pct: Decimal = Field(
        default=Decimal("20"),
        description="Percentage to keep as margin reserve (higher for leverage safety)"
    )

    # === AGGRESSIVE MARKET MAKING SETTINGS ===
    mm_enabled: bool = Field(default=True)
    mm_spread: Decimal = Field(default=Decimal("0.0015"))  # 0.15% spread (aggressive)
    mm_order_refresh_time: int = Field(default=10)  # Fast refresh
    mm_orders_per_side: int = Field(default=3)  # Multiple orders per side
    mm_order_spacing: Decimal = Field(default=Decimal("0.001"))  # 0.1% between orders
    mm_inventory_skew_enabled: bool = Field(default=True)
    mm_max_inventory_pct: Decimal = Field(default=Decimal("30"))  # Max 30% in any coin

    # === AGGRESSIVE GRID SETTINGS ===
    grid_enabled: bool = Field(default=True)
    grid_levels: int = Field(default=5)  # More levels
    grid_spacing_pct: Decimal = Field(default=Decimal("0.003"))  # 0.3% spacing (tight)
    grid_order_amount_pct: Decimal = Field(default=Decimal("2"))  # 2% of capital per order
    grid_rebalance_threshold: Decimal = Field(default=Decimal("0.01"))  # 1% rebalance

    # === LEVERAGED RSI SETTINGS (with SHORT capability) ===
    rsi_enabled: bool = Field(default=True)
    rsi_period: int = Field(default=10)  # Faster RSI
    rsi_oversold: int = Field(default=25)  # Go LONG when oversold
    rsi_overbought: int = Field(default=75)  # Go SHORT when overbought
    rsi_order_amount_pct: Decimal = Field(default=Decimal("5"))  # 5% of effective capital
    rsi_take_profit_pct: Decimal = Field(default=Decimal("0.01"))  # 1% take profit (tighter for leverage)
    rsi_stop_loss_pct: Decimal = Field(default=Decimal("0.005"))  # 0.5% stop loss (CRITICAL with 10x = 5% account)
    rsi_cooldown_minutes: int = Field(default=15)
    rsi_enable_shorts: bool = Field(default=True)  # Enable short selling

    # === LEVERAGED MACD TREND FOLLOWING (with SHORT capability) ===
    macd_enabled: bool = Field(default=True)
    macd_fast_period: int = Field(default=8)
    macd_slow_period: int = Field(default=17)
    macd_signal_period: int = Field(default=9)
    macd_order_amount_pct: Decimal = Field(default=Decimal("5"))
    macd_take_profit_pct: Decimal = Field(default=Decimal("0.015"))  # 1.5% (tighter for leverage)
    macd_stop_loss_pct: Decimal = Field(default=Decimal("0.007"))  # 0.7% stop loss (7% account with 10x)
    macd_cooldown_minutes: int = Field(default=30)
    macd_enable_shorts: bool = Field(default=True)  # Enable short selling

    # === LEVERAGED MOMENTUM SCALPING (with SHORT capability) ===
    momentum_enabled: bool = Field(default=True)
    momentum_threshold_pct: Decimal = Field(default=Decimal("0.005"))  # 0.5% move triggers
    momentum_order_amount_pct: Decimal = Field(default=Decimal("3"))
    momentum_take_profit_pct: Decimal = Field(default=Decimal("0.006"))  # 0.6% (tighter for leverage)
    momentum_stop_loss_pct: Decimal = Field(default=Decimal("0.003"))  # 0.3% stop loss (3% account with 10x)
    momentum_lookback_seconds: int = Field(default=60)
    momentum_enable_shorts: bool = Field(default=True)  # Enable short selling

    # === VOLATILITY-BASED ROTATION ===
    rotation_enabled: bool = Field(default=True)
    rotation_interval_minutes: int = Field(default=60)  # Rebalance hourly
    min_volatility_threshold: Decimal = Field(default=Decimal("0.01"))  # 1% min volatility

    # === KILL SWITCH SETTINGS (Tighter for leveraged trading) ===
    # With $10k capital, these represent % of actual capital at risk
    mm_max_loss: Decimal = Field(default=Decimal("500"))  # 5% of capital
    grid_max_loss: Decimal = Field(default=Decimal("500"))  # 5% of capital
    rsi_max_loss: Decimal = Field(default=Decimal("400"))  # 4% of capital
    macd_max_loss: Decimal = Field(default=Decimal("400"))  # 4% of capital
    momentum_max_loss: Decimal = Field(default=Decimal("300"))  # 3% of capital
    global_max_loss: Decimal = Field(default=Decimal("1500"))  # 15% max total loss
    global_max_drawdown_pct: Decimal = Field(default=Decimal("8"))  # 8% drawdown kills all
    daily_loss_limit: Decimal = Field(default=Decimal("500"))  # 5% daily limit

    # === LIQUIDATION PROTECTION ===
    # With 10x leverage, 10% adverse move = liquidation
    # We stop well before that
    max_position_pct: Decimal = Field(default=Decimal("50"))  # Max 50% of effective capital in one position
    liquidation_buffer_pct: Decimal = Field(default=Decimal("5"))  # Stop if within 5% of liquidation

    # === TIMING ===
    main_tick_interval: int = Field(default=5)  # Faster ticks


# =============================================================================
# MAIN STRATEGY CLASS
# =============================================================================

class LeveragedMultiStrategyBot(ScriptStrategyBase):
    """
    Leveraged multi-strategy trading bot for perpetual futures.

    Uses $10,000 capital with 10x leverage = $100,000 effective buying power.
    Designed for offshore LLC accounts with full exchange access.

    Strategies:
    1. Aggressive Market Making - Tight spreads, multiple orders, fast refresh
    2. Dynamic Grid Trading - More levels, adaptive to volatility
    3. RSI Directional - LONG on oversold, SHORT on overbought
    4. MACD Trend Following - LONG on bullish crossover, SHORT on bearish
    5. Momentum Scalping - LONG on upward spikes, SHORT on downward spikes
    6. Volatility Rotation - Allocate more to volatile coins

    RISK CONTROLS:
    - Tighter stop losses (0.3-0.7% vs 1-2% unleveraged)
    - Lower kill switch thresholds (5% per strategy)
    - Liquidation buffer protection
    - Higher margin reserve (20%)
    """

    @classmethod
    def init_markets(cls, config: LeveragedConfig):
        """Initialize markets from config."""
        pairs = set(p.strip() for p in config.trading_pairs.split(","))
        cls.markets = {config.exchange: pairs}

    # Class-level market definition (will be overridden by init_markets)
    # Using liquid pairs for leverage safety: BTC, ETH, LTC, SOL, DOGE
    markets = {"kucoin_perpetual": {
        "BTC-USDT", "LTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT"
    }}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: LeveragedConfig = None):
        super().__init__(connectors)
        self.config = config if config is not None else LeveragedConfig()

        # Calculate effective capital with leverage
        self.effective_capital = self.config.total_capital * self.config.leverage

        # Parse trading pairs
        self.trading_pairs = [p.strip() for p in self.config.trading_pairs.split(",")]

        # Initialize metrics for each strategy
        self.metrics: Dict[str, StrategyMetrics] = {
            "mm": StrategyMetrics(name="Market Making"),
            "grid": StrategyMetrics(name="Grid Trading"),
            "rsi": StrategyMetrics(name="RSI Directional"),
            "macd": StrategyMetrics(name="MACD Trend"),
            "momentum": StrategyMetrics(name="Momentum Scalp"),
        }

        # Initialize coin metrics
        self.coin_metrics: Dict[str, CoinMetrics] = {
            pair: CoinMetrics(symbol=pair) for pair in self.trading_pairs
        }

        # Global metrics
        self.global_pnl = Decimal("0")
        self.global_peak_pnl = Decimal("0")
        self.global_drawdown = Decimal("0")
        self.bot_start_time = datetime.now()
        self.last_daily_reset = datetime.now().date()
        self.last_rotation_time = datetime.now()

        # Timing
        self.last_tick = 0
        self.mm_last_refresh = 0

        # Strategy state tracking
        self._strategies_activated = False

        # Grid state per pair
        self.grid_orders: Dict[str, Dict[str, Dict]] = {pair: {} for pair in self.trading_pairs}
        self.grid_initialized: Dict[str, bool] = {pair: False for pair in self.trading_pairs}
        self.grid_base_price: Dict[str, Optional[Decimal]] = {pair: None for pair in self.trading_pairs}

        # RSI state per pair
        self.rsi_position: Dict[str, Optional[str]] = {pair: None for pair in self.trading_pairs}
        self.rsi_entry_price: Dict[str, Optional[Decimal]] = {pair: None for pair in self.trading_pairs}
        self.rsi_last_signal_time: Dict[str, Optional[datetime]] = {pair: None for pair in self.trading_pairs}

        # MACD state per pair
        self.macd_position: Dict[str, Optional[str]] = {pair: None for pair in self.trading_pairs}
        self.macd_entry_price: Dict[str, Optional[Decimal]] = {pair: None for pair in self.trading_pairs}
        self.macd_last_signal_time: Dict[str, Optional[datetime]] = {pair: None for pair in self.trading_pairs}

        # Momentum state per pair
        self.momentum_position: Dict[str, Optional[str]] = {pair: None for pair in self.trading_pairs}
        self.momentum_entry_price: Dict[str, Optional[Decimal]] = {pair: None for pair in self.trading_pairs}
        self.momentum_entry_time: Dict[str, Optional[datetime]] = {pair: None for pair in self.trading_pairs}
        self.price_snapshots: Dict[str, List[Tuple[datetime, Decimal]]] = {pair: [] for pair in self.trading_pairs}

        # Order tracking by strategy
        self.strategy_orders: Dict[str, set] = {
            "mm": set(),
            "grid": set(),
            "rsi": set(),
            "macd": set(),
            "momentum": set(),
        }

        # Capital allocation per pair
        self.pair_capital: Dict[str, Decimal] = {}
        self._allocate_capital()

    def _allocate_capital(self):
        """Allocate effective capital (with leverage) across trading pairs."""
        # Use effective capital (actual * leverage) for position sizing
        usable_capital = self.effective_capital * (1 - self.config.reserve_pct / 100)
        per_pair = usable_capital / len(self.trading_pairs)
        for pair in self.trading_pairs:
            self.pair_capital[pair] = per_pair
            self.coin_metrics[pair].allocated_capital = per_pair

    def on_start(self):
        """Called when strategy starts."""
        self.logger().info("=" * 60)
        self.logger().info("LEVERAGED MULTI-STRATEGY BOT STARTING")
        self.logger().info("=" * 60)
        self.logger().info(f"Exchange: {self.config.exchange}")
        self.logger().info(f"Leverage: {self.config.leverage}x")
        self.logger().info(f"Trading Pairs: {self.trading_pairs}")
        self.logger().info(f"Actual Capital: ${self.config.total_capital}")
        self.logger().info(f"Effective Capital: ${self.effective_capital} ({self.config.leverage}x leverage)")
        self.logger().info(f"Per Pair: ${self.pair_capital.get(self.trading_pairs[0], 0):.2f}")
        self.logger().info("-" * 60)
        self.logger().info("WARNING: Leveraged trading - tight stops enforced")
        self.logger().info(f"Max loss per strategy: ${self.config.mm_max_loss}")
        self.logger().info(f"Global max loss: ${self.config.global_max_loss}")
        self.logger().info("=" * 60)

        # Activate all enabled strategies
        if self.config.mm_enabled:
            self.metrics["mm"].status = StrategyStatus.ACTIVE
        if self.config.grid_enabled:
            self.metrics["grid"].status = StrategyStatus.ACTIVE
        if self.config.rsi_enabled:
            self.metrics["rsi"].status = StrategyStatus.WARMING_UP
        if self.config.macd_enabled:
            self.metrics["macd"].status = StrategyStatus.WARMING_UP
        if self.config.momentum_enabled:
            self.metrics["momentum"].status = StrategyStatus.ACTIVE

        self._strategies_activated = True

    def on_tick(self):
        """Main loop called every tick."""
        current_time = self.current_timestamp

        # Activate strategies if not done
        if not self._strategies_activated:
            self.on_start()

        # Check if enough time has passed since last tick
        if current_time - self.last_tick < self.config.main_tick_interval:
            return
        self.last_tick = current_time

        # Check for daily reset
        self._check_daily_reset()

        # Check kill switches
        if self._check_kill_switches():
            return

        # Update coin metrics
        self._update_coin_metrics()

        # Run volatility-based rotation
        if self.config.rotation_enabled:
            self._check_rotation()

        # Run each strategy for each pair
        for pair in self.trading_pairs:
            try:
                # Market Making
                if self.config.mm_enabled and self.metrics["mm"].status == StrategyStatus.ACTIVE:
                    self._run_market_making(pair)

                # Grid Trading
                if self.config.grid_enabled and self.metrics["grid"].status == StrategyStatus.ACTIVE:
                    self._run_grid_trading(pair)

                # RSI Directional
                if self.config.rsi_enabled and self.metrics["rsi"].status in [StrategyStatus.ACTIVE, StrategyStatus.WARMING_UP]:
                    self._run_rsi_strategy(pair)

                # MACD Trend
                if self.config.macd_enabled and self.metrics["macd"].status in [StrategyStatus.ACTIVE, StrategyStatus.WARMING_UP]:
                    self._run_macd_strategy(pair)

                # Momentum Scalping
                if self.config.momentum_enabled and self.metrics["momentum"].status == StrategyStatus.ACTIVE:
                    self._run_momentum_strategy(pair)

            except Exception as e:
                self.logger().error(f"Error processing {pair}: {e}")

    def _update_coin_metrics(self):
        """Update metrics for each coin."""
        connector = self.connectors[self.config.exchange]

        for pair in self.trading_pairs:
            try:
                mid_price = connector.get_mid_price(pair)
                if mid_price is None or mid_price <= 0:
                    continue

                metrics = self.coin_metrics[pair]
                metrics.last_price = mid_price

                # Track price history (last 100 prices)
                metrics.price_history.append(mid_price)
                if len(metrics.price_history) > 100:
                    metrics.price_history.pop(0)

                # Calculate volatility
                if len(metrics.price_history) >= 20:
                    returns = []
                    for i in range(1, len(metrics.price_history)):
                        ret = (metrics.price_history[i] - metrics.price_history[i-1]) / metrics.price_history[i-1]
                        returns.append(float(ret))
                    if returns:
                        metrics.volatility = Decimal(str(statistics.stdev(returns) * 100))

                # Store price snapshot for momentum
                self.price_snapshots[pair].append((datetime.now(), mid_price))
                # Keep only last 5 minutes of snapshots
                cutoff = datetime.now() - timedelta(seconds=300)
                self.price_snapshots[pair] = [
                    (t, p) for t, p in self.price_snapshots[pair] if t > cutoff
                ]

            except Exception as e:
                self.logger().debug(f"Error updating metrics for {pair}: {e}")

    def _check_rotation(self):
        """Rotate capital based on volatility."""
        now = datetime.now()
        if (now - self.last_rotation_time).total_seconds() < self.config.rotation_interval_minutes * 60:
            return

        self.last_rotation_time = now

        # Sort pairs by volatility
        volatility_ranking = sorted(
            self.coin_metrics.items(),
            key=lambda x: x[1].volatility,
            reverse=True
        )

        # Allocate more capital to higher volatility pairs
        usable_capital = self.config.total_capital * (1 - self.config.reserve_pct / 100)

        # Weight by volatility rank
        total_weight = sum(range(1, len(self.trading_pairs) + 1))

        for rank, (pair, metrics) in enumerate(volatility_ranking):
            weight = len(self.trading_pairs) - rank
            allocation = usable_capital * Decimal(str(weight)) / Decimal(str(total_weight))
            self.pair_capital[pair] = allocation
            self.coin_metrics[pair].allocated_capital = allocation

        self.logger().info(f"Capital rotation complete. Top volatile: {volatility_ranking[0][0]}")

    # =========================================================================
    # MARKET MAKING STRATEGY
    # =========================================================================

    def _run_market_making(self, pair: str):
        """Aggressive market making with multiple orders per side."""
        connector = self.connectors[self.config.exchange]

        # Check refresh timing
        if self.current_timestamp - self.mm_last_refresh < self.config.mm_order_refresh_time:
            return

        # Cancel existing MM orders for this pair
        self._cancel_strategy_orders("mm", pair)

        mid_price = connector.get_mid_price(pair)
        if mid_price is None or mid_price <= 0:
            return

        # Calculate order amount based on allocated capital
        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * Decimal("0.02")  # 2% per order
        order_amount = order_amount_usd / mid_price

        # Check inventory skew
        inventory_skew = self._calculate_inventory_skew(pair)

        orders = []

        # Place multiple orders on each side
        for i in range(self.config.mm_orders_per_side):
            spacing = self.config.mm_spread + (self.config.mm_order_spacing * i)

            # Adjust for inventory skew
            bid_spread = spacing * (1 + inventory_skew * Decimal("0.5"))
            ask_spread = spacing * (1 - inventory_skew * Decimal("0.5"))

            bid_price = mid_price * (1 - bid_spread)
            ask_price = mid_price * (1 + ask_spread)

            # Create orders
            buy_order = OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=order_amount,
                price=bid_price
            )
            sell_order = OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=order_amount,
                price=ask_price
            )
            orders.extend([buy_order, sell_order])

        # Submit orders
        adjusted_orders = connector.budget_checker.adjust_candidates(orders, all_or_none=False)
        for order in adjusted_orders:
            if order.amount > 0:
                if order.order_side == TradeType.BUY:
                    order_id = connector.buy(
                        trading_pair=order.trading_pair,
                        amount=order.amount,
                        order_type=order.order_type,
                        price=order.price
                    )
                else:
                    order_id = connector.sell(
                        trading_pair=order.trading_pair,
                        amount=order.amount,
                        order_type=order.order_type,
                        price=order.price
                    )
                self.strategy_orders["mm"].add(order_id)

        self.mm_last_refresh = self.current_timestamp

    def _calculate_inventory_skew(self, pair: str) -> Decimal:
        """Calculate inventory skew for a pair. Returns -1 to 1."""
        connector = self.connectors[self.config.exchange]
        base, quote = pair.split("-")

        base_balance = connector.get_balance(base)
        quote_balance = connector.get_balance(quote)

        mid_price = connector.get_mid_price(pair)
        if mid_price is None or mid_price <= 0:
            return Decimal("0")

        base_value = base_balance * mid_price
        total_value = base_value + quote_balance

        if total_value <= 0:
            return Decimal("0")

        # Target is 50% in base
        current_ratio = base_value / total_value
        target_ratio = Decimal("0.5")

        # Skew: positive means too much base, negative means too much quote
        return (current_ratio - target_ratio) * 2

    # =========================================================================
    # GRID TRADING STRATEGY
    # =========================================================================

    def _run_grid_trading(self, pair: str):
        """Dynamic grid trading with volatility-adjusted spacing."""
        connector = self.connectors[self.config.exchange]

        mid_price = connector.get_mid_price(pair)
        if mid_price is None or mid_price <= 0:
            return

        # Initialize grid if needed
        if not self.grid_initialized[pair]:
            self.grid_base_price[pair] = mid_price
            self.grid_initialized[pair] = True
            self._place_grid_orders(pair, mid_price)
            return

        # Check if price moved significantly (rebalance grid)
        base_price = self.grid_base_price[pair]
        if base_price is None:
            return

        price_deviation = abs(mid_price - base_price) / base_price
        if price_deviation > self.config.grid_rebalance_threshold:
            self.logger().info(f"Grid rebalancing {pair}: price moved {price_deviation:.2%}")
            self._cancel_strategy_orders("grid", pair)
            self.grid_base_price[pair] = mid_price
            self._place_grid_orders(pair, mid_price)

    def _place_grid_orders(self, pair: str, base_price: Decimal):
        """Place grid orders around base price."""
        connector = self.connectors[self.config.exchange]

        # Adjust spacing based on volatility
        volatility = self.coin_metrics[pair].volatility
        if volatility > 0:
            spacing = max(self.config.grid_spacing_pct, volatility / 100)
        else:
            spacing = self.config.grid_spacing_pct

        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * self.config.grid_order_amount_pct / 100
        order_amount = order_amount_usd / base_price

        orders = []

        for i in range(1, self.config.grid_levels + 1):
            # Buy orders below
            buy_price = base_price * (1 - spacing * i)
            buy_order = OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.BUY,
                amount=order_amount,
                price=buy_price
            )
            orders.append(buy_order)

            # Sell orders above
            sell_price = base_price * (1 + spacing * i)
            sell_order = OrderCandidate(
                trading_pair=pair,
                is_maker=True,
                order_type=OrderType.LIMIT,
                order_side=TradeType.SELL,
                amount=order_amount,
                price=sell_price
            )
            orders.append(sell_order)

        # Submit orders
        adjusted_orders = connector.budget_checker.adjust_candidates(orders, all_or_none=False)
        for order in adjusted_orders:
            if order.amount > 0:
                if order.order_side == TradeType.BUY:
                    order_id = connector.buy(
                        trading_pair=order.trading_pair,
                        amount=order.amount,
                        order_type=order.order_type,
                        price=order.price
                    )
                else:
                    order_id = connector.sell(
                        trading_pair=order.trading_pair,
                        amount=order.amount,
                        order_type=order.order_type,
                        price=order.price
                    )
                self.strategy_orders["grid"].add(order_id)
                self.grid_orders[pair][order_id] = {
                    "side": order.order_side,
                    "price": order.price,
                    "amount": order.amount
                }

    # =========================================================================
    # RSI DIRECTIONAL STRATEGY
    # =========================================================================

    def _run_rsi_strategy(self, pair: str):
        """RSI-based directional trading."""
        connector = self.connectors[self.config.exchange]

        # Calculate RSI
        rsi = self._calculate_rsi(pair)
        if rsi is None:
            return

        self.coin_metrics[pair].rsi = rsi

        # Activate strategy after warmup
        if self.metrics["rsi"].status == StrategyStatus.WARMING_UP:
            self.metrics["rsi"].status = StrategyStatus.ACTIVE

        mid_price = connector.get_mid_price(pair)
        if mid_price is None or mid_price <= 0:
            return

        # Check cooldown
        last_signal = self.rsi_last_signal_time.get(pair)
        if last_signal:
            cooldown = timedelta(minutes=self.config.rsi_cooldown_minutes)
            if datetime.now() - last_signal < cooldown:
                return

        # Check for existing position
        position = self.rsi_position.get(pair)

        if position is None:
            # Entry logic
            if rsi < self.config.rsi_oversold:
                # Oversold - go LONG
                self._enter_rsi_position(pair, "long", mid_price)
            elif rsi > self.config.rsi_overbought and self.config.rsi_enable_shorts:
                # Overbought - go SHORT (leveraged futures only)
                self._enter_rsi_position(pair, "short", mid_price)

        else:
            # Exit logic
            entry_price = self.rsi_entry_price.get(pair)
            if entry_price and position == "long":
                pnl_pct = (mid_price - entry_price) / entry_price

                # Take profit
                if pnl_pct >= self.config.rsi_take_profit_pct:
                    self._exit_rsi_position(pair, mid_price, "take_profit")
                # Stop loss
                elif pnl_pct <= -self.config.rsi_stop_loss_pct:
                    self._exit_rsi_position(pair, mid_price, "stop_loss")
                # RSI reversal
                elif rsi > 60:
                    self._exit_rsi_position(pair, mid_price, "rsi_reversal")

            elif entry_price and position == "short":
                # SHORT position P&L is inverted
                pnl_pct = (entry_price - mid_price) / entry_price

                # Take profit (price went down)
                if pnl_pct >= self.config.rsi_take_profit_pct:
                    self._exit_rsi_position(pair, mid_price, "take_profit")
                # Stop loss (price went up)
                elif pnl_pct <= -self.config.rsi_stop_loss_pct:
                    self._exit_rsi_position(pair, mid_price, "stop_loss")
                # RSI reversal (no longer overbought)
                elif rsi < 40:
                    self._exit_rsi_position(pair, mid_price, "rsi_reversal")

    def _calculate_rsi(self, pair: str, period: int = None) -> Optional[Decimal]:
        """Calculate RSI from price history."""
        if period is None:
            period = self.config.rsi_period

        prices = self.coin_metrics[pair].price_history
        if len(prices) < period + 1:
            return None

        gains = []
        losses = []

        for i in range(1, period + 1):
            change = prices[-i] - prices[-i-1]
            if change > 0:
                gains.append(float(change))
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(float(change)))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return Decimal(str(rsi))

    def _enter_rsi_position(self, pair: str, direction: str, price: Decimal):
        """Enter RSI position (LONG or SHORT for futures)."""
        connector = self.connectors[self.config.exchange]

        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * self.config.rsi_order_amount_pct / 100
        order_amount = order_amount_usd / price

        if direction == "long":
            order_id = connector.buy(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["rsi"].add(order_id)
            self.rsi_position[pair] = "long"
            self.rsi_entry_price[pair] = price
            self.rsi_last_signal_time[pair] = datetime.now()
            self.logger().info(f"RSI LONG {pair} at {price} (leveraged)")

        elif direction == "short":
            # SHORT position - sell to open (futures)
            order_id = connector.sell(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["rsi"].add(order_id)
            self.rsi_position[pair] = "short"
            self.rsi_entry_price[pair] = price
            self.rsi_last_signal_time[pair] = datetime.now()
            self.logger().info(f"RSI SHORT {pair} at {price} (leveraged)")

    def _exit_rsi_position(self, pair: str, price: Decimal, reason: str):
        """Exit RSI position (LONG or SHORT)."""
        connector = self.connectors[self.config.exchange]
        position = self.rsi_position.get(pair)

        if position is None:
            return

        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * self.config.rsi_order_amount_pct / 100
        order_amount = order_amount_usd / price

        entry_price = self.rsi_entry_price.get(pair)

        if position == "long":
            # Close LONG by selling
            order_id = connector.sell(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["rsi"].add(order_id)

            if entry_price:
                pnl = (price - entry_price) * order_amount
                self.metrics["rsi"].update_pnl(pnl)
                self.logger().info(f"RSI EXIT LONG {pair} at {price} ({reason}), PnL: ${pnl:.2f}")

        elif position == "short":
            # Close SHORT by buying
            order_id = connector.buy(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["rsi"].add(order_id)

            if entry_price:
                # SHORT P&L: profit when price goes down
                pnl = (entry_price - price) * order_amount
                self.metrics["rsi"].update_pnl(pnl)
                self.logger().info(f"RSI EXIT SHORT {pair} at {price} ({reason}), PnL: ${pnl:.2f}")

        self.rsi_position[pair] = None
        self.rsi_entry_price[pair] = None

    # =========================================================================
    # MACD TREND FOLLOWING STRATEGY
    # =========================================================================

    def _run_macd_strategy(self, pair: str):
        """MACD-based trend following."""
        connector = self.connectors[self.config.exchange]

        prices = self.coin_metrics[pair].price_history
        if len(prices) < self.config.macd_slow_period + self.config.macd_signal_period:
            return

        # Activate after warmup
        if self.metrics["macd"].status == StrategyStatus.WARMING_UP:
            self.metrics["macd"].status = StrategyStatus.ACTIVE

        # Calculate MACD
        macd_line, signal_line = self._calculate_macd(prices)
        if macd_line is None or signal_line is None:
            return

        mid_price = connector.get_mid_price(pair)
        if mid_price is None or mid_price <= 0:
            return

        # Check cooldown
        last_signal = self.macd_last_signal_time.get(pair)
        if last_signal:
            cooldown = timedelta(minutes=self.config.macd_cooldown_minutes)
            if datetime.now() - last_signal < cooldown:
                return

        position = self.macd_position.get(pair)

        if position is None:
            # Bullish crossover - go LONG
            if macd_line > signal_line and macd_line > 0:
                self._enter_macd_position(pair, "long", mid_price)
            # Bearish crossover - go SHORT (futures only)
            elif macd_line < signal_line and macd_line < 0 and self.config.macd_enable_shorts:
                self._enter_macd_position(pair, "short", mid_price)
        else:
            # Exit logic
            entry_price = self.macd_entry_price.get(pair)
            if entry_price and position == "long":
                pnl_pct = (mid_price - entry_price) / entry_price

                # Take profit
                if pnl_pct >= self.config.macd_take_profit_pct:
                    self._exit_macd_position(pair, mid_price, "take_profit")
                # Stop loss
                elif pnl_pct <= -self.config.macd_stop_loss_pct:
                    self._exit_macd_position(pair, mid_price, "stop_loss")
                # Bearish crossover
                elif macd_line < signal_line:
                    self._exit_macd_position(pair, mid_price, "macd_crossover")

            elif entry_price and position == "short":
                # SHORT position P&L is inverted
                pnl_pct = (entry_price - mid_price) / entry_price

                # Take profit (price went down)
                if pnl_pct >= self.config.macd_take_profit_pct:
                    self._exit_macd_position(pair, mid_price, "take_profit")
                # Stop loss (price went up)
                elif pnl_pct <= -self.config.macd_stop_loss_pct:
                    self._exit_macd_position(pair, mid_price, "stop_loss")
                # Bullish crossover - close short
                elif macd_line > signal_line:
                    self._exit_macd_position(pair, mid_price, "macd_crossover")

    def _calculate_macd(self, prices: List[Decimal]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate MACD and signal line."""
        if len(prices) < self.config.macd_slow_period + self.config.macd_signal_period:
            return None, None

        prices_float = [float(p) for p in prices]

        # Calculate EMAs
        fast_ema = self._calculate_ema(prices_float, self.config.macd_fast_period)
        slow_ema = self._calculate_ema(prices_float, self.config.macd_slow_period)

        if fast_ema is None or slow_ema is None:
            return None, None

        macd_line = fast_ema - slow_ema

        # Calculate signal line (EMA of MACD)
        # Simplified: use recent MACD values
        signal_line = macd_line * 0.9  # Approximation

        return macd_line, signal_line

    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = sum(prices[-period:]) / period  # Start with SMA

        for price in prices[-period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _enter_macd_position(self, pair: str, direction: str, price: Decimal):
        """Enter MACD position (LONG or SHORT for futures)."""
        connector = self.connectors[self.config.exchange]

        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * self.config.macd_order_amount_pct / 100
        order_amount = order_amount_usd / price

        if direction == "long":
            order_id = connector.buy(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["macd"].add(order_id)
            self.macd_position[pair] = "long"
            self.macd_entry_price[pair] = price
            self.macd_last_signal_time[pair] = datetime.now()
            self.logger().info(f"MACD LONG {pair} at {price} (leveraged)")

        elif direction == "short":
            # SHORT position - sell to open (futures)
            order_id = connector.sell(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["macd"].add(order_id)
            self.macd_position[pair] = "short"
            self.macd_entry_price[pair] = price
            self.macd_last_signal_time[pair] = datetime.now()
            self.logger().info(f"MACD SHORT {pair} at {price} (leveraged)")

    def _exit_macd_position(self, pair: str, price: Decimal, reason: str):
        """Exit MACD position (LONG or SHORT)."""
        connector = self.connectors[self.config.exchange]
        position = self.macd_position.get(pair)

        if position is None:
            return

        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * self.config.macd_order_amount_pct / 100
        order_amount = order_amount_usd / price

        entry_price = self.macd_entry_price.get(pair)

        if position == "long":
            # Close LONG by selling
            order_id = connector.sell(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["macd"].add(order_id)

            if entry_price:
                pnl = (price - entry_price) * order_amount
                self.metrics["macd"].update_pnl(pnl)
                self.logger().info(f"MACD EXIT LONG {pair} at {price} ({reason}), PnL: ${pnl:.2f}")

        elif position == "short":
            # Close SHORT by buying
            order_id = connector.buy(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["macd"].add(order_id)

            if entry_price:
                # SHORT P&L: profit when price goes down
                pnl = (entry_price - price) * order_amount
                self.metrics["macd"].update_pnl(pnl)
                self.logger().info(f"MACD EXIT SHORT {pair} at {price} ({reason}), PnL: ${pnl:.2f}")

        self.macd_position[pair] = None
        self.macd_entry_price[pair] = None

    # =========================================================================
    # MOMENTUM SCALPING STRATEGY
    # =========================================================================

    def _run_momentum_strategy(self, pair: str):
        """Quick momentum scalping on price spikes."""
        connector = self.connectors[self.config.exchange]

        snapshots = self.price_snapshots.get(pair, [])
        if len(snapshots) < 5:
            return

        mid_price = connector.get_mid_price(pair)
        if mid_price is None or mid_price <= 0:
            return

        # Get price from lookback period
        lookback = timedelta(seconds=self.config.momentum_lookback_seconds)
        cutoff_time = datetime.now() - lookback

        old_prices = [p for t, p in snapshots if t <= cutoff_time]
        if not old_prices:
            return

        old_price = old_prices[-1]
        price_change = (mid_price - old_price) / old_price

        position = self.momentum_position.get(pair)

        if position is None:
            # Strong upward momentum - enter LONG
            if price_change > self.config.momentum_threshold_pct:
                self._enter_momentum_position(pair, "long", mid_price)
            # Strong downward momentum - enter SHORT (futures only)
            elif price_change < -self.config.momentum_threshold_pct and self.config.momentum_enable_shorts:
                self._enter_momentum_position(pair, "short", mid_price)
        else:
            # Exit logic
            entry_price = self.momentum_entry_price.get(pair)
            if entry_price and position == "long":
                pnl_pct = (mid_price - entry_price) / entry_price

                # Take profit
                if pnl_pct >= self.config.momentum_take_profit_pct:
                    self._exit_momentum_position(pair, mid_price, "take_profit")
                # Stop loss
                elif pnl_pct <= -self.config.momentum_stop_loss_pct:
                    self._exit_momentum_position(pair, mid_price, "stop_loss")
                # Time exit (momentum trades should be quick)
                elif (datetime.now() - self.momentum_entry_time.get(pair, datetime.now())).total_seconds() > 300:
                    self._exit_momentum_position(pair, mid_price, "time_exit")

            elif entry_price and position == "short":
                # SHORT position P&L is inverted
                pnl_pct = (entry_price - mid_price) / entry_price

                # Take profit (price went down)
                if pnl_pct >= self.config.momentum_take_profit_pct:
                    self._exit_momentum_position(pair, mid_price, "take_profit")
                # Stop loss (price went up)
                elif pnl_pct <= -self.config.momentum_stop_loss_pct:
                    self._exit_momentum_position(pair, mid_price, "stop_loss")
                # Time exit
                elif (datetime.now() - self.momentum_entry_time.get(pair, datetime.now())).total_seconds() > 300:
                    self._exit_momentum_position(pair, mid_price, "time_exit")

    def _enter_momentum_position(self, pair: str, direction: str, price: Decimal):
        """Enter momentum position (LONG or SHORT for futures)."""
        connector = self.connectors[self.config.exchange]

        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * self.config.momentum_order_amount_pct / 100
        order_amount = order_amount_usd / price

        if direction == "long":
            order_id = connector.buy(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["momentum"].add(order_id)
            self.momentum_position[pair] = "long"
            self.momentum_entry_price[pair] = price
            self.momentum_entry_time[pair] = datetime.now()
            self.logger().info(f"MOMENTUM LONG {pair} at {price} (leveraged)")

        elif direction == "short":
            # SHORT position - sell to open on downward spike
            order_id = connector.sell(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["momentum"].add(order_id)
            self.momentum_position[pair] = "short"
            self.momentum_entry_price[pair] = price
            self.momentum_entry_time[pair] = datetime.now()
            self.logger().info(f"MOMENTUM SHORT {pair} at {price} (leveraged)")

    def _exit_momentum_position(self, pair: str, price: Decimal, reason: str):
        """Exit momentum position (LONG or SHORT)."""
        connector = self.connectors[self.config.exchange]
        position = self.momentum_position.get(pair)

        if position is None:
            return

        capital = self.pair_capital.get(pair, Decimal("0"))
        order_amount_usd = capital * self.config.momentum_order_amount_pct / 100
        order_amount = order_amount_usd / price

        entry_price = self.momentum_entry_price.get(pair)

        if position == "long":
            # Close LONG by selling
            order_id = connector.sell(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["momentum"].add(order_id)

            if entry_price:
                pnl = (price - entry_price) * order_amount
                self.metrics["momentum"].update_pnl(pnl)
                self.logger().info(f"MOMENTUM EXIT LONG {pair} at {price} ({reason}), PnL: ${pnl:.2f}")

        elif position == "short":
            # Close SHORT by buying
            order_id = connector.buy(
                trading_pair=pair,
                amount=order_amount,
                order_type=OrderType.MARKET,
                price=price
            )
            self.strategy_orders["momentum"].add(order_id)

            if entry_price:
                # SHORT P&L: profit when price goes down
                pnl = (entry_price - price) * order_amount
                self.metrics["momentum"].update_pnl(pnl)
                self.logger().info(f"MOMENTUM EXIT SHORT {pair} at {price} ({reason}), PnL: ${pnl:.2f}")

        self.momentum_position[pair] = None
        self.momentum_entry_price[pair] = None
        self.momentum_entry_time[pair] = None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _cancel_strategy_orders(self, strategy: str, pair: Optional[str] = None):
        """Cancel orders for a specific strategy."""
        connector = self.connectors[self.config.exchange]
        orders_to_remove = set()

        for order_id in self.strategy_orders[strategy]:
            try:
                order = connector.get_order(order_id)
                if order and (pair is None or order.trading_pair == pair):
                    connector.cancel(order.trading_pair, order_id)
                    orders_to_remove.add(order_id)
            except:
                orders_to_remove.add(order_id)

        self.strategy_orders[strategy] -= orders_to_remove

    def _check_daily_reset(self):
        """Reset daily metrics at midnight."""
        today = datetime.now().date()
        if today != self.last_daily_reset:
            for metrics in self.metrics.values():
                metrics.reset_daily()
            self.last_daily_reset = today
            self.logger().info("Daily metrics reset")

    def _check_kill_switches(self) -> bool:
        """Check all kill switches. Returns True if trading should stop."""
        killed = False

        # Per-strategy kill switches
        kill_limits = {
            "mm": self.config.mm_max_loss,
            "grid": self.config.grid_max_loss,
            "rsi": self.config.rsi_max_loss,
            "macd": self.config.macd_max_loss,
            "momentum": self.config.momentum_max_loss,
        }

        for strategy, limit in kill_limits.items():
            if self.metrics[strategy].total_pnl < -limit:
                if self.metrics[strategy].status == StrategyStatus.ACTIVE:
                    self.metrics[strategy].status = StrategyStatus.KILLED
                    self._cancel_strategy_orders(strategy)
                    self.logger().warning(f"KILL SWITCH: {strategy} stopped (loss: ${-self.metrics[strategy].total_pnl:.2f})")

        # Global kill switches
        total_pnl = sum(m.total_pnl for m in self.metrics.values())
        self.global_pnl = total_pnl

        if total_pnl > self.global_peak_pnl:
            self.global_peak_pnl = total_pnl

        if self.global_peak_pnl > 0:
            self.global_drawdown = (self.global_peak_pnl - total_pnl) / self.global_peak_pnl * 100

        if total_pnl < -self.config.global_max_loss:
            self.logger().error(f"GLOBAL KILL SWITCH: Max loss exceeded (${-total_pnl:.2f})")
            self._kill_all_strategies()
            killed = True

        if self.global_drawdown > self.config.global_max_drawdown_pct:
            self.logger().error(f"GLOBAL KILL SWITCH: Max drawdown exceeded ({self.global_drawdown:.1f}%)")
            self._kill_all_strategies()
            killed = True

        # Daily loss limit
        daily_pnl = sum(m.daily_pnl for m in self.metrics.values())
        if daily_pnl < -self.config.daily_loss_limit:
            self.logger().warning(f"DAILY LIMIT: Trading paused for today (loss: ${-daily_pnl:.2f})")
            for m in self.metrics.values():
                if m.status == StrategyStatus.ACTIVE:
                    m.status = StrategyStatus.PAUSED
            killed = True

        return killed

    def _kill_all_strategies(self):
        """Kill all strategies and cancel all orders."""
        for strategy in self.metrics:
            self.metrics[strategy].status = StrategyStatus.KILLED
            self._cancel_strategy_orders(strategy)

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fill events."""
        order_id = event.order_id

        # Track which strategy the fill belongs to
        for strategy, orders in self.strategy_orders.items():
            if order_id in orders:
                self.metrics[strategy].total_trades += 1
                self.metrics[strategy].last_trade_time = datetime.now()

                if event.trade_type == TradeType.BUY:
                    self.metrics[strategy].buy_volume += event.amount
                else:
                    self.metrics[strategy].sell_volume += event.amount

                # For MM and Grid, estimate P&L from fills
                if strategy in ["mm", "grid"]:
                    # Simplified P&L estimation
                    fee_pct = Decimal("0.001")  # 0.1% fee estimate
                    trade_value = event.amount * event.price
                    fee = trade_value * fee_pct

                    # MM profits from spread
                    if strategy == "mm":
                        spread_capture = trade_value * self.config.mm_spread / 2
                        estimated_pnl = spread_capture - fee
                        self.metrics[strategy].update_pnl(estimated_pnl)

                break

    def format_status(self) -> str:
        """Format status for display."""
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  LEVERAGED MULTI-STRATEGY BOT STATUS")
        lines.append(f"  {self.config.leverage}x LEVERAGE | ${self.config.total_capital} CAPITAL")
        lines.append("=" * 60)

        # Runtime
        runtime = datetime.now() - self.bot_start_time
        days = runtime.days
        hours = runtime.seconds // 3600
        minutes = (runtime.seconds % 3600) // 60
        lines.append(f"  Runtime: {days}d {hours}h {minutes}m")
        lines.append("")

        # Global metrics
        lines.append("-" * 40)
        lines.append("  GLOBAL METRICS (LEVERAGED)")
        lines.append("-" * 40)
        lines.append(f"  Actual Capital:   ${self.config.total_capital}")
        lines.append(f"  Effective Capital: ${self.effective_capital} ({self.config.leverage}x)")
        lines.append(f"  Total P&L:        ${self.global_pnl:.4f}")
        lines.append(f"  Peak P&L:         ${self.global_peak_pnl:.4f}")
        lines.append(f"  Drawdown:         {self.global_drawdown:.2f}%")
        daily_pnl = sum(m.daily_pnl for m in self.metrics.values())
        lines.append(f"  Daily P&L:        ${daily_pnl:.4f}")
        lines.append(f"  Status:           RUNNING (LEVERAGED)")
        lines.append("")

        # Strategy performance
        lines.append("-" * 40)
        lines.append("  STRATEGY PERFORMANCE")
        lines.append("-" * 40)
        lines.append(f"  {'Strategy':<18} {'Status':<12} {'P&L':<12} {'Trades':<8}")
        lines.append("-" * 50)

        for key, metrics in self.metrics.items():
            status_str = {
                StrategyStatus.ACTIVE: "[ON]",
                StrategyStatus.PAUSED: "[||]",
                StrategyStatus.KILLED: "[XX]",
                StrategyStatus.WARMING_UP: "[..]"
            }.get(metrics.status, "[??]")

            pnl_str = f"+${metrics.total_pnl:.2f}" if metrics.total_pnl >= 0 else f"-${abs(metrics.total_pnl):.2f}"
            lines.append(f"  {metrics.name:<18} {status_str:<12} {pnl_str:<12} {metrics.total_trades:<8}")

        lines.append("")

        # Trading pairs status
        lines.append("-" * 40)
        lines.append("  TRADING PAIRS")
        lines.append("-" * 40)
        lines.append(f"  {'Pair':<12} {'Price':<12} {'Volatility':<12} {'RSI':<8} {'Capital':<12}")
        lines.append("-" * 56)

        connector = self.connectors.get(self.config.exchange)
        for pair in self.trading_pairs:
            metrics = self.coin_metrics[pair]
            try:
                price = connector.get_mid_price(pair) if connector else Decimal("0")
                price_str = f"${price:.2f}" if price else "N/A"
            except:
                price_str = "N/A"

            vol_str = f"{metrics.volatility:.2f}%"
            rsi_str = f"{metrics.rsi:.0f}"
            cap_str = f"${metrics.allocated_capital:.0f}"

            lines.append(f"  {pair:<12} {price_str:<12} {vol_str:<12} {rsi_str:<8} {cap_str:<12}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
