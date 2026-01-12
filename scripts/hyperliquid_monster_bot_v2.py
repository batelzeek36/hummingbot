"""
Hyperliquid Monster Bot v2.3 - MODULAR Multi-Strategy Perpetual Trading

This is the main orchestrator that coordinates the modular strategy components.
All strategy logic has been extracted into separate modules in the hyperliquid_monster/ package.

LEVERAGE TIERS (based on volatility analysis 2026-01-11):
- SAFE (max daily <10%):     BTC → 8x leverage
- MEDIUM (max daily <15%):   ETH, SOL, TAO → 5x leverage
- HIGH (max daily <25%):     DOGE, HYPE, PURR, AVNT → 3x leverage
- EXTREME (max daily 25%+):  kPEPE, kBONK, WIF, VVV, HYPER, IP → 2x leverage

Target: 12-16% monthly returns on ~$80 capital

Capital Allocation ($78):
- Funding Harvesting: 45% ($35) - Smart leverage by coin (scans 13 pairs)
- Grid Trading: 35% ($27) - 5x on SOL (best liquidity)
- Directional: 20% ($16) - 8x on BTC (safest)

Author: Dollar-A-Day Project
Date: 2026-01-12
Version: 2.3 (Modularized)
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.event.events import OrderFilledEvent, FundingPaymentCompletedEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase

# Import from modular package
from hyperliquid_monster import (
    HyperliquidMonsterV2Config,
    StrategyMetrics,
    StrategyMode,
    PerformanceTracker,
    get_safe_leverage,
    COIN_VOLATILITY,
    VOLATILITY_LEVERAGE,
    CoinVolatility,
)
from hyperliquid_monster.strategies import (
    FundingHunterStrategy,
    GridStrategy,
    MomentumStrategy,
)


class HyperliquidMonsterBotV2(ScriptStrategyBase):
    """
    Modular multi-strategy Hyperliquid perpetual trading bot V2.

    This class serves as the orchestrator, delegating strategy logic to
    modular components in the hyperliquid_monster package.
    """

    # All validated Hyperliquid pairs
    markets = {"hyperliquid_perpetual": {
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "TAO-USD",
        "HYPE-USD", "AVNT-USD",
        "kPEPE-USD", "kBONK-USD", "WIF-USD", "VVV-USD", "HYPER-USD", "IP-USD"
    }}

    @classmethod
    def init_markets(cls, config: HyperliquidMonsterV2Config):
        """Initialize markets based on config."""
        pairs = set()
        for pair in config.funding_scan_pairs.split(","):
            pairs.add(pair.strip())
        pairs.add(config.grid_pair)
        pairs.add(config.momentum_pair)
        cls.markets = {config.exchange: pairs}

    def __init__(self, connectors: Dict[str, ConnectorBase], config: HyperliquidMonsterV2Config = None):
        super().__init__(connectors)
        self.config = config if config is not None else HyperliquidMonsterV2Config()

        # Strategy metrics
        self.metrics = {
            "funding": StrategyMetrics(name="Funding Hunting"),
            "grid": StrategyMetrics(name="Grid Trading"),
            "momentum": StrategyMetrics(name="Momentum Trading"),
        }

        # Performance tracker
        self.performance_tracker = PerformanceTracker(logger=self.logger())
        self.performance_tracker.load_data()

        # Global state
        self.start_time = datetime.now()
        self.global_pnl = Decimal("0")
        self.total_funding_collected = Decimal("0")
        self.peak_equity = self.config.total_capital
        self.global_killed = False
        self._is_stopping = False
        self._first_run = True

        # Timing
        self.last_tick = 0
        self.tick_interval = 10

        # Initialize strategies (will be done after connector is ready)
        self._funding_strategy: Optional[FundingHunterStrategy] = None
        self._grid_strategy: Optional[GridStrategy] = None
        self._momentum_strategy: Optional[MomentumStrategy] = None

        self.logger().info("Hyperliquid Monster Bot V2 (MODULAR) initialized!")

    def _init_strategies(self):
        """Initialize strategy instances."""
        connector = self.connectors[self.config.exchange]

        self._funding_strategy = FundingHunterStrategy(
            config=self.config,
            connector=connector,
            metrics=self.metrics["funding"],
            place_order_fn=self._place_perpetual_order,
            logger=self.logger(),
        )

        self._grid_strategy = GridStrategy(
            config=self.config,
            connector=connector,
            metrics=self.metrics["grid"],
            place_order_fn=self._place_perpetual_order,
            cancel_order_fn=lambda pair, order_id: self.cancel(self.config.exchange, pair, order_id),
            get_active_orders_fn=lambda: self.get_active_orders(connector_name=self.config.exchange),
            logger=self.logger(),
        )

        self._momentum_strategy = MomentumStrategy(
            config=self.config,
            connector=connector,
            metrics=self.metrics["momentum"],
            place_order_fn=self._place_perpetual_order,
            logger=self.logger(),
        )

    # =========================================================================
    # STARTUP AND MAIN LOOP
    # =========================================================================

    def on_start(self):
        """Called when strategy starts."""
        self.logger().info("=" * 70)
        self.logger().info("  HYPERLIQUID MONSTER BOT V2.3 - MODULAR SMART LEVERAGE")
        self.logger().info("=" * 70)
        self.logger().info(f"  Capital: ${self.config.total_capital}")
        self.logger().info(f"  Smart Leverage: ENABLED (volatility-based)")
        self.logger().info(f"  Grid Leverage: {self.config.grid_leverage}x (SOL)")
        self.logger().info(f"  Momentum Leverage: {self.config.momentum_leverage}x (BTC)")
        self.logger().info(f"  Min Funding APR: {self.config.min_funding_apr}%")
        self.logger().info(f"  Scanning: {self.config.funding_scan_pairs}")

        self._configure_exchange()
        self._init_strategies()
        self._is_stopping = False

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
            connector.set_position_mode(PositionMode.ONEWAY)

            all_pairs = set()
            for pair in self.config.funding_scan_pairs.split(","):
                all_pairs.add(pair.strip())
            all_pairs.add(self.config.grid_pair)
            all_pairs.add(self.config.momentum_pair)

            for pair in all_pairs:
                safe_lev = get_safe_leverage(pair, self.config.max_leverage)
                connector.set_leverage(pair, safe_lev)

            self.logger().info(f"Exchange configured: {len(all_pairs)} pairs, smart leverage per coin")

        except Exception as e:
            self.logger().warning(f"Error configuring exchange: {e}")

    def on_tick(self):
        """Main loop."""
        if self.global_killed or self._is_stopping:
            return

        # First run activation
        if self._first_run:
            self._first_run = False
            if not self._funding_strategy:
                self._init_strategies()
            self.logger().info("Strategies activated on first tick")

        if self.last_tick > self.current_timestamp - self.tick_interval:
            return
        self.last_tick = self.current_timestamp

        # Risk check
        self._check_risk_limits()
        if self.global_killed:
            return

        # Run strategies
        if self.config.funding_enabled and self._funding_strategy:
            self._funding_strategy.run(self.current_timestamp)

        if self.config.grid_enabled and self._grid_strategy:
            self._grid_strategy.run()

        if self.config.momentum_enabled and self._momentum_strategy:
            self._momentum_strategy.run(self._update_global_pnl)

        # Periodic performance save
        self.performance_tracker.check_save_interval(self.start_time, self.global_pnl)

    def on_stop(self):
        """Called when strategy stops."""
        self.logger().info("Monster Bot V2 stopping...")
        self._is_stopping = True

        # Cancel all orders first
        self._cancel_all_orders()

        # Close all positions
        self._close_all_positions()

        # Save final performance data
        self.performance_tracker.save_data(self.start_time, self.global_pnl)
        self.logger().info("Shutdown complete.")

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
    # EVENT HANDLERS
    # =========================================================================

    def did_fill_order(self, event: OrderFilledEvent):
        """Handle order fills."""
        # Check if it's a grid order
        if self._grid_strategy and self._grid_strategy.handle_fill(event):
            pass  # Grid strategy handled it
        else:
            # Other strategies
            for strategy in ["funding", "momentum"]:
                self.metrics[strategy].total_trades += 1

        side = "BUY" if event.trade_type == TradeType.BUY else "SELL"
        self.logger().info(
            f"FILL: {side} {event.amount:.6f} {event.trading_pair} @ {event.price:.4f}"
        )

    def did_complete_funding_payment(self, event: FundingPaymentCompletedEvent):
        """Handle funding payment events."""
        self.total_funding_collected += event.amount
        self.metrics["funding"].funding_pnl += event.amount
        self.global_pnl += event.amount

        # Track per-coin performance
        coin_perf = self.performance_tracker.get_coin_performance(event.trading_pair)
        apr = 0.0
        if self._funding_strategy and event.trading_pair in self._funding_strategy.positions:
            apr = self._funding_strategy.positions[event.trading_pair].apr
        coin_perf.add_payment(float(event.amount), apr)

        self.logger().info(
            f"FUNDING RECEIVED: {event.trading_pair} ${event.amount:.4f} "
            f"(Session Total: ${self.total_funding_collected:.4f})"
        )

        # Save immediately for real-time tracking
        self.performance_tracker.save_data(self.start_time, self.global_pnl)

    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================

    def _check_risk_limits(self):
        """Check global risk limits."""
        current_equity = self.config.total_capital + self.global_pnl

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100

            if drawdown >= float(self.config.max_drawdown_pct):
                self.logger().error(
                    f"MAX DRAWDOWN REACHED: {drawdown:.2f}% - EMERGENCY SHUTDOWN"
                )
                self._kill_all()

    def _kill_all(self):
        """Emergency shutdown."""
        self.global_killed = True
        self._is_stopping = True
        self._cancel_all_orders()
        self._close_all_positions()

        for strategy in self.metrics.values():
            strategy.status = StrategyMode.KILLED

    def _update_global_pnl(self, pnl: Decimal):
        """Update global P&L (callback for strategies)."""
        self.global_pnl += pnl

    def _close_all_positions(self):
        """Close all open positions."""
        if self._funding_strategy:
            self._funding_strategy.close_all_positions(self.current_timestamp)

        if self._momentum_strategy:
            self._momentum_strategy.close_position_for_shutdown(self._update_global_pnl)

    def _cancel_all_orders(self):
        """Cancel all orders."""
        if self._grid_strategy:
            self._grid_strategy.cancel_all_orders()

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
        lines.append("  HYPERLIQUID MONSTER BOT V2.3 - MODULAR")
        lines.append("=" * 70)

        runtime = datetime.now() - self.start_time
        lines.append(f"  Runtime: {runtime.days}d {runtime.seconds//3600}h {(runtime.seconds%3600)//60}m")

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
            lines.append(f"  Drawdown:         {drawdown:.2f}% (max: {self.config.max_drawdown_pct}%)")

        status = "KILLED" if self.global_killed else "RUNNING"
        lines.append(f"  Status:           {status}")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  SMART LEVERAGE (by volatility)")
        lines.append("-" * 70)
        lines.append(f"  Grid (SOL): {self.config.grid_leverage}x | Momentum (BTC): {self.config.momentum_leverage}x")
        lines.append(f"  Funding: Dynamic per coin (2x-8x based on volatility)")

        # Funding opportunities
        if self._funding_strategy and self._funding_strategy.opportunities:
            lines.append("")
            lines.append("-" * 70)
            lines.append("  TOP FUNDING OPPORTUNITIES")
            lines.append("-" * 70)
            for opp in self._funding_strategy.opportunities[:5]:
                minutes = (opp.next_funding_time - self.current_timestamp) / 60
                in_position = "***" if opp.pair in self._funding_strategy.positions else "   "
                lines.append(f"  {in_position} {opp.pair:<12} {opp.apr:>6.1f}% APR ({opp.direction:<5}) {minutes:>4.0f}min")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  STRATEGY STATUS")
        lines.append("-" * 70)

        for m in self.metrics.values():
            status = m.status.value
            pnl = m.realized_pnl + m.funding_pnl
            lines.append(f"  {m.name:<20} [{status:<10}] P&L: ${pnl:+.4f} Trades: {m.total_trades}")

        # Coin performance leaderboard
        lines.append("")
        lines.append("-" * 70)
        lines.append("  COIN PERFORMANCE LEADERBOARD")
        lines.append("-" * 70)
        if self.performance_tracker.coin_performance:
            sorted_coins = sorted(
                self.performance_tracker.coin_performance.items(),
                key=lambda x: x[1].total_funding_received,
                reverse=True
            )[:5]
            lines.append(f"  {'Rank':<4} {'Coin':<12} {'Funding':<12} {'Payments':<10} {'Avg APR':<10}")
            lines.append("  " + "-" * 48)
            for rank, (symbol, perf) in enumerate(sorted_coins, 1):
                lines.append(
                    f"  {rank:<4} {symbol:<12} ${perf.total_funding_received:<10.4f} "
                    f"{perf.funding_payments_count:<10} {perf.avg_apr_captured:<10.0f}%"
                )
        else:
            lines.append("  No funding payments recorded yet")

        # Active positions
        lines.append("")
        lines.append("-" * 70)
        lines.append("  ACTIVE POSITIONS")
        lines.append("-" * 70)

        if self._funding_strategy and self._funding_strategy.positions:
            for pair, opp in self._funding_strategy.positions.items():
                lines.append(f"  FUNDING: {opp.direction.upper():<5} {pair} ({opp.apr:.1f}% APR)")

        if self._momentum_strategy:
            pos_info = self._momentum_strategy.get_position_info()
            if pos_info:
                direction, entry = pos_info
                lines.append(f"  MOMENTUM: {direction.upper()} {self.config.momentum_pair} @ {entry}")

        if self._grid_strategy:
            lines.append(f"  GRID: {self._grid_strategy.get_order_count()} orders active on {self.config.grid_pair}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
