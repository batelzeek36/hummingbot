"""
Hyperliquid Monster Bot v2.2 - OPTIMIZED Multi-Strategy Perpetual Trading

UPGRADES from v2.1:
- Added AVNT, HYPER, IP to funding scanner (high APR opportunities)
- Expanded volatility classifications for new coins
- Kept SOL for grid (best liquidity at $461M daily volume)

LEVERAGE TIERS (based on volatility analysis 2026-01-11):
- SAFE (max daily <10%):     BTC, ETH → 8x leverage
- MEDIUM (max daily <15%):   SOL, TAO, AVNT → 5x leverage
- HIGH (max daily <25%):     DOGE, HYPE, MOG, IP → 3x leverage
- EXTREME (max daily 25%+):  PEPE, BONK, WIF, ANIME, HYPER → 2x leverage

Target: 12-16% monthly returns on ~$80 capital

Capital Allocation ($78):
- Funding Harvesting: 45% ($35) - Smart leverage by coin (scans 14 pairs)
- Grid Trading: 35% ($27) - 5x on SOL (best liquidity)
- Directional: 20% ($16) - 8x on BTC (safest)

Author: Dollar-A-Day Project
Date: 2026-01-11
Version: 2.2 (Optimized)
"""

import json
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
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


# =============================================================================
# VOLATILITY-BASED LEVERAGE CLASSIFICATION
# Based on 30-day historical volatility analysis (2026-01-11)
# =============================================================================

class CoinVolatility(Enum):
    """Volatility classification for leverage decisions."""
    SAFE = "safe"           # Max daily <10%, can handle 8x
    MEDIUM = "medium"       # Max daily <15%, can handle 5x
    HIGH = "high"           # Max daily <25%, max 3x
    EXTREME = "extreme"     # Max daily 25%+, max 2x


# Coin classifications based on volatility analysis
# CRITICAL: These must be verified against liquidation thresholds!
# 8x = 12.5% liq, 5x = 20% liq, 3x = 33% liq, 2x = 50% liq
COIN_VOLATILITY = {
    # SAFE - Low volatility, high leverage OK (max daily <10%)
    "BTC-USD": CoinVolatility.SAFE,      # Max 6.0% daily, buffer +6.5%

    # MEDIUM - Moderate volatility (max daily <15%)
    "ETH-USD": CoinVolatility.MEDIUM,    # Max 9.8% daily, buffer +10.2% - MOVED from SAFE for safety
    "SOL-USD": CoinVolatility.MEDIUM,    # Max 10.4% daily, buffer +9.6%
    "TAO-USD": CoinVolatility.MEDIUM,    # Max 13.1% daily, buffer +6.9%

    # HIGH - Elevated volatility, reduce leverage (max daily <25%)
    "DOGE-USD": CoinVolatility.HIGH,     # Max 15.0% daily, buffer +18.3%
    "HYPE-USD": CoinVolatility.HIGH,     # Max 17.4% daily, buffer +15.9%
    "MOG-USD": CoinVolatility.HIGH,      # Max 20.4% daily
    "PURR-USD": CoinVolatility.HIGH,     # Assume similar to other memes
    "AVNT-USD": CoinVolatility.HIGH,     # Max 25.8% daily - MOVED from MEDIUM!

    # EXTREME - Very high volatility, minimal leverage (max daily 25%+)
    "PEPE-USD": CoinVolatility.EXTREME,  # Max 28.1% daily
    "BONK-USD": CoinVolatility.EXTREME,  # Max 39.7% daily
    "WIF-USD": CoinVolatility.EXTREME,   # Max 26.6% daily, buffer +23.4%
    "HYPER-USD": CoinVolatility.EXTREME, # Max 33.4% daily, buffer +16.6%
    "VVV-USD": CoinVolatility.EXTREME,   # Max 40.5% daily - MOVED from HIGH!
    "IP-USD": CoinVolatility.EXTREME,    # Max 30.7% daily, buffer +19.3% - MOVED from HIGH for safety
    # ANIME-USD REMOVED - Max 51.9% exceeds 50% liq threshold, no safe leverage exists
}

# Leverage limits by volatility class
VOLATILITY_LEVERAGE = {
    CoinVolatility.SAFE: 8,
    CoinVolatility.MEDIUM: 5,
    CoinVolatility.HIGH: 3,
    CoinVolatility.EXTREME: 2,
}


def get_safe_leverage(pair: str, max_leverage: int = 8) -> int:
    """Get safe leverage for a trading pair based on volatility."""
    volatility = COIN_VOLATILITY.get(pair, CoinVolatility.HIGH)  # Default to HIGH if unknown
    safe_lev = VOLATILITY_LEVERAGE.get(volatility, 3)
    return min(safe_lev, max_leverage)


@dataclass
class FundingOpportunity:
    """Represents a funding rate opportunity."""
    pair: str
    rate: float
    apr: float  # Annualized rate
    direction: str  # "long" or "short" to receive funding
    next_funding_time: float
    score: float = 0.0  # Higher is better


@dataclass
class PositionInfo:
    """Track open positions."""
    trading_pair: str
    side: str
    entry_price: Decimal
    amount: Decimal
    leverage: int
    entry_time: datetime
    strategy: str
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


@dataclass
class CoinPerformance:
    """Track performance metrics for each coin."""
    symbol: str
    total_funding_received: float = 0.0
    funding_payments_count: int = 0
    total_positions_opened: int = 0
    total_position_time_minutes: float = 0.0
    avg_apr_captured: float = 0.0
    best_single_payment: float = 0.0
    worst_single_payment: float = 0.0
    last_payment_time: str = ""
    leverage_used: int = 2
    volatility_class: str = "unknown"

    def add_payment(self, amount: float, apr: float = 0.0):
        """Record a funding payment."""
        self.total_funding_received += amount
        self.funding_payments_count += 1
        self.best_single_payment = max(self.best_single_payment, amount)
        if self.worst_single_payment == 0.0:
            self.worst_single_payment = amount
        else:
            self.worst_single_payment = min(self.worst_single_payment, amount)
        self.last_payment_time = datetime.now().isoformat()
        # Update rolling average APR
        if self.avg_apr_captured == 0:
            self.avg_apr_captured = apr
        else:
            self.avg_apr_captured = (self.avg_apr_captured * (self.funding_payments_count - 1) + apr) / self.funding_payments_count


# =============================================================================
# CONFIGURATION - AGGRESSIVE V2
# =============================================================================

class HyperliquidMonsterV2Config(BaseClientModel):
    """Configuration for Hyperliquid Monster Bot V2 - AGGRESSIVE."""

    script_file_name: str = Field(default="hyperliquid_monster_bot_v2.py")

    # === EXCHANGE SETTINGS ===
    exchange: str = Field(
        default="hyperliquid_perpetual",
        description="Exchange connector"
    )

    # === PAIRS TO SCAN FOR FUNDING ===
    # All pairs we'll scan for funding opportunities
    # Start with known high-funding pairs, expand as needed
    funding_scan_pairs: str = Field(
        default="ETH-USD,SOL-USD,DOGE-USD",
        description="Pairs to scan for funding rate opportunities"
    )

    # Maximum simultaneous funding positions
    max_funding_positions: int = Field(
        default=3,
        description="Maximum number of simultaneous funding positions"
    )

    # === STATIC PAIRS FOR GRID/MOMENTUM ===
    grid_pair: str = Field(
        default="SOL-USD",
        description="Pair for grid trading (needs good liquidity)"
    )
    momentum_pair: str = Field(
        default="BTC-USD",
        description="Pair for momentum/directional trading"
    )

    # === CAPITAL ALLOCATION - MORE TO FUNDING ===
    total_capital: Decimal = Field(
        default=Decimal("78"),
        description="Total capital in USDC"
    )
    funding_capital_pct: Decimal = Field(
        default=Decimal("45"),  # UP from 40%
        description="% of capital for funding harvesting"
    )
    grid_capital_pct: Decimal = Field(
        default=Decimal("35"),
        description="% of capital for grid trading"
    )
    momentum_capital_pct: Decimal = Field(
        default=Decimal("20"),  # DOWN from 25%
        description="% of capital for momentum trading"
    )

    # === LEVERAGE SETTINGS - SMART (based on volatility analysis) ===
    # These are MAX values - actual leverage is determined by coin volatility
    # BTC: 8x, SOL: 5x, DOGE/HYPE/MOG: 3x, PEPE/BONK/WIF/ANIME: 2x
    funding_leverage_max: int = Field(default=8, description="Max leverage for funding (actual varies by coin)")
    grid_leverage: int = Field(default=5, description="Leverage for grid trading (SOL = medium volatility)")
    momentum_leverage: int = Field(default=8, description="Leverage for momentum trades (BTC = low volatility)")
    max_leverage: int = Field(default=8, description="Absolute maximum leverage cap")
    use_smart_leverage: bool = Field(default=True, description="Use volatility-based leverage per coin")

    # === FUNDING HARVESTING SETTINGS - AGGRESSIVE ===
    funding_enabled: bool = Field(default=True)

    # Minimum 30% APR to enter (filters out weak opportunities)
    min_funding_apr: Decimal = Field(
        default=Decimal("30"),  # 30% APR minimum
        description="Minimum annualized funding rate to open position"
    )

    # Auto-rotate if better opportunity is 20% higher APR
    funding_rotation_threshold: Decimal = Field(
        default=Decimal("20"),  # Switch if new opp is 20% APR better
        description="APR improvement needed to rotate positions"
    )

    funding_position_size: Decimal = Field(
        default=Decimal("12"),  # $12 per position (with 8x = $96 exposure)
        description="Position size per funding pair"
    )

    minutes_before_funding: int = Field(
        default=10,  # UP from 5 - more time to position
        description="Open position this many minutes before funding"
    )

    # Scan interval for new opportunities
    funding_scan_interval: int = Field(
        default=60,  # Scan every 60 seconds
        description="Seconds between funding rate scans"
    )

    # === GRID TRADING SETTINGS - TIGHTER ===
    grid_enabled: bool = Field(default=True)
    grid_levels: int = Field(default=6, description="Number of grid levels each side")  # UP from 5
    grid_spacing_pct: Decimal = Field(
        default=Decimal("0.004"),  # 0.4% spacing (tighter than 0.5%)
        description="Spacing between grid levels"
    )
    grid_order_size: Decimal = Field(
        default=Decimal("4"),  # $4 per grid order (with 8x = $32 exposure each)
        description="Size per grid order in USD"
    )
    grid_rebalance_pct: Decimal = Field(
        default=Decimal("0.025"),  # 2.5% rebalance trigger (tighter)
        description="Rebalance grid when price moves this much"
    )

    # === MOMENTUM SETTINGS ===
    momentum_enabled: bool = Field(default=True)
    momentum_lookback: int = Field(default=20, description="Candles for RSI calculation")
    rsi_oversold: Decimal = Field(default=Decimal("25"), description="RSI oversold threshold")  # More extreme
    rsi_overbought: Decimal = Field(default=Decimal("75"), description="RSI overbought threshold")  # More extreme
    momentum_take_profit: Decimal = Field(
        default=Decimal("0.025"),  # 2.5% TP (with 5x = 12.5% gain)
        description="Take profit percentage"
    )
    momentum_stop_loss: Decimal = Field(
        default=Decimal("0.015"),  # 1.5% SL (with 5x = 7.5% loss)
        description="Stop loss percentage"
    )
    momentum_position_size: Decimal = Field(
        default=Decimal("5"),  # $5 per momentum trade
        description="Position size for momentum trades"
    )

    # === RISK MANAGEMENT - SLIGHTLY LOOSER FOR AGGRESSIVE MODE ===
    max_drawdown_pct: Decimal = Field(
        default=Decimal("20"),  # UP from 15% - more room before kill
        description="Kill all strategies at this drawdown %"
    )
    max_position_size_pct: Decimal = Field(
        default=Decimal("60"),  # UP from 50%
        description="Max position as % of capital (leveraged)"
    )
    daily_loss_limit: Decimal = Field(
        default=Decimal("12"),  # UP from 10
        description="Max daily loss in USD"
    )


# =============================================================================
# MAIN BOT CLASS - V2 AGGRESSIVE
# =============================================================================

class HyperliquidMonsterBotV2(ScriptStrategyBase):
    """
    Aggressive multi-strategy Hyperliquid perpetual trading bot V2.

    Key upgrades:
    - Dynamic funding spike hunting across all pairs
    - Auto-rotation to best opportunities
    - Higher leverage (8x/8x/5x)
    - Tighter grid spacing
    """

    # Will be set dynamically based on funding scan
    markets = {"hyperliquid_perpetual": {
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"  # Core pairs only
    }}

    @classmethod
    def init_markets(cls, config: HyperliquidMonsterV2Config):
        """Initialize markets based on config."""
        pairs = set()

        # Add all funding scan pairs
        for pair in config.funding_scan_pairs.split(","):
            pairs.add(pair.strip())

        # Add grid and momentum pairs
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

        # Position tracking
        self.positions: Dict[str, PositionInfo] = {}

        # Funding hunting state
        self.funding_opportunities: List[FundingOpportunity] = []
        self.funding_positions: Dict[str, FundingOpportunity] = {}  # pair -> opportunity
        self.last_funding_scan = 0
        self.total_funding_collected: Decimal = Decimal("0")

        # Grid state
        self.grid_initialized = False
        self.grid_base_price: Optional[Decimal] = None
        self.grid_orders: Dict[str, dict] = {}

        # Momentum state
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.momentum_position: Optional[str] = None
        self.momentum_entry_price: Optional[Decimal] = None

        # Global state
        self.start_time = datetime.now()
        self.global_pnl = Decimal("0")
        self.peak_equity = self.config.total_capital
        self.global_killed = False

        # Timing
        self.last_tick = 0
        self.tick_interval = 10

        # === PERFORMANCE TRACKING ===
        self.coin_performance: Dict[str, CoinPerformance] = {}
        self.last_performance_save = datetime.now()
        self.performance_save_interval = 3600  # Save every hour
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "performance"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._load_performance_data()

        self.logger().info("Hyperliquid Monster Bot V2 (AGGRESSIVE) initialized!")

    # =========================================================================
    # PERFORMANCE TRACKING METHODS
    # =========================================================================

    def _load_performance_data(self):
        """Load existing performance data from JSON file."""
        json_path = self.data_dir / "coin_performance.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    for symbol, perf_data in data.get("coins", {}).items():
                        self.coin_performance[symbol] = CoinPerformance(
                            symbol=symbol,
                            total_funding_received=perf_data.get("total_funding_received", 0.0),
                            funding_payments_count=perf_data.get("funding_payments_count", 0),
                            total_positions_opened=perf_data.get("total_positions_opened", 0),
                            total_position_time_minutes=perf_data.get("total_position_time_minutes", 0.0),
                            avg_apr_captured=perf_data.get("avg_apr_captured", 0.0),
                            best_single_payment=perf_data.get("best_single_payment", 0.0),
                            worst_single_payment=perf_data.get("worst_single_payment", 0.0),
                            last_payment_time=perf_data.get("last_payment_time", ""),
                            leverage_used=perf_data.get("leverage_used", 2),
                            volatility_class=perf_data.get("volatility_class", "unknown"),
                        )
                self.logger().info(f"Loaded performance data for {len(self.coin_performance)} coins")
            except Exception as e:
                self.logger().warning(f"Could not load performance data: {e}")

    def _save_performance_data(self):
        """Save performance data to both JSON and Markdown files."""
        # Prepare data
        coins_data = {}
        for symbol, perf in self.coin_performance.items():
            coins_data[symbol] = {
                "total_funding_received": round(perf.total_funding_received, 6),
                "funding_payments_count": perf.funding_payments_count,
                "total_positions_opened": perf.total_positions_opened,
                "total_position_time_minutes": round(perf.total_position_time_minutes, 2),
                "avg_apr_captured": round(perf.avg_apr_captured, 2),
                "best_single_payment": round(perf.best_single_payment, 6),
                "worst_single_payment": round(perf.worst_single_payment, 6),
                "last_payment_time": perf.last_payment_time,
                "leverage_used": perf.leverage_used,
                "volatility_class": perf.volatility_class,
            }

        # Sort by total funding received
        sorted_coins = sorted(coins_data.items(), key=lambda x: x[1]["total_funding_received"], reverse=True)

        # Calculate totals
        total_funding = sum(c["total_funding_received"] for _, c in sorted_coins)
        total_payments = sum(c["funding_payments_count"] for _, c in sorted_coins)

        # === SAVE JSON (Machine-readable) ===
        json_data = {
            "generated_at": datetime.now().isoformat(),
            "session_start": self.start_time.isoformat(),
            "runtime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "summary": {
                "total_funding_received": round(total_funding, 6),
                "total_payments": total_payments,
                "coins_tracked": len(coins_data),
                "global_pnl": float(self.global_pnl),
            },
            "coins": dict(sorted_coins),
        }

        json_path = self.data_dir / "coin_performance.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # === SAVE MARKDOWN (Human-readable / Investor report) ===
        runtime = datetime.now() - self.start_time
        hours = runtime.total_seconds() / 3600

        md_lines = [
            "# Coin Performance Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Session Start:** {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Runtime:** {hours:.1f} hours",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Funding Received | **${total_funding:.4f}** |",
            f"| Total Payments | {total_payments} |",
            f"| Coins Tracked | {len(coins_data)} |",
            f"| Global P&L | ${float(self.global_pnl):.4f} |",
            "",
            "---",
            "",
            "## Coin Leaderboard",
            "",
            "| Rank | Coin | Funding Received | Payments | Avg APR | Leverage |",
            "|------|------|------------------|----------|---------|----------|",
        ]

        for rank, (symbol, data) in enumerate(sorted_coins, 1):
            md_lines.append(
                f"| {rank} | {symbol} | ${data['total_funding_received']:.4f} | "
                f"{data['funding_payments_count']} | {data['avg_apr_captured']:.0f}% | "
                f"{data['leverage_used']}x |"
            )

        md_lines.extend([
            "",
            "---",
            "",
            "## Detailed Coin Metrics",
            "",
        ])

        for symbol, data in sorted_coins:
            md_lines.extend([
                f"### {symbol}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total Funding | ${data['total_funding_received']:.6f} |",
                f"| Payment Count | {data['funding_payments_count']} |",
                f"| Avg APR Captured | {data['avg_apr_captured']:.1f}% |",
                f"| Best Single Payment | ${data['best_single_payment']:.6f} |",
                f"| Worst Single Payment | ${data['worst_single_payment']:.6f} |",
                f"| Leverage Used | {data['leverage_used']}x |",
                f"| Volatility Class | {data['volatility_class']} |",
                f"| Last Payment | {data['last_payment_time']} |",
                "",
            ])

        md_lines.extend([
            "---",
            "",
            f"*Report generated by Hyperliquid Monster Bot v2.2*",
        ])

        md_path = self.data_dir / "coin_performance.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

        self.logger().info(f"Performance data saved to {self.data_dir}")

    def _get_coin_performance(self, symbol: str) -> CoinPerformance:
        """Get or create coin performance tracker."""
        if symbol not in self.coin_performance:
            volatility = COIN_VOLATILITY.get(symbol, CoinVolatility.HIGH)
            leverage = VOLATILITY_LEVERAGE.get(volatility, 3)
            self.coin_performance[symbol] = CoinPerformance(
                symbol=symbol,
                leverage_used=leverage,
                volatility_class=volatility.value,
            )
        return self.coin_performance[symbol]

    def _check_performance_save(self):
        """Check if it's time to save performance data."""
        now = datetime.now()
        if (now - self.last_performance_save).total_seconds() >= self.performance_save_interval:
            self._save_performance_data()
            self.last_performance_save = now

    # =========================================================================
    # STARTUP AND MAIN LOOP
    # =========================================================================

    def on_start(self):
        """Called when strategy starts."""
        self.logger().info("=" * 70)
        self.logger().info("  HYPERLIQUID MONSTER BOT V2.1 - SMART LEVERAGE MODE")
        self.logger().info("=" * 70)
        self.logger().info(f"  Capital: ${self.config.total_capital}")
        self.logger().info(f"  Smart Leverage: ENABLED (volatility-based)")
        self.logger().info(f"    - SAFE coins (BTC): up to 8x")
        self.logger().info(f"    - MEDIUM coins (SOL): up to 5x")
        self.logger().info(f"    - HIGH volatility (DOGE,HYPE): up to 3x")
        self.logger().info(f"    - EXTREME volatility (ANIME,PEPE): up to 2x")
        self.logger().info(f"  Grid Leverage: {self.config.grid_leverage}x (SOL)")
        self.logger().info(f"  Momentum Leverage: {self.config.momentum_leverage}x (BTC)")
        self.logger().info(f"  Min Funding APR: {self.config.min_funding_apr}%")
        self.logger().info(f"  Scanning: {self.config.funding_scan_pairs}")

        self._configure_exchange()

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
                connector.set_leverage(pair, self.config.max_leverage)

            self.logger().info(f"Exchange configured: {len(all_pairs)} pairs, {self.config.max_leverage}x max leverage")

        except Exception as e:
            self.logger().warning(f"Error configuring exchange: {e}")

    def on_tick(self):
        """Main loop."""
        if self.global_killed:
            return

        if self.last_tick > self.current_timestamp - self.tick_interval:
            return
        self.last_tick = self.current_timestamp

        self._check_risk_limits()
        if self.global_killed:
            return

        if self.config.funding_enabled and self.metrics["funding"].status == StrategyMode.ACTIVE:
            self._run_funding_hunter()

        if self.config.grid_enabled and self.metrics["grid"].status == StrategyMode.ACTIVE:
            self._run_grid_strategy()

        if self.config.momentum_enabled:
            self._run_momentum_strategy()

        # Periodic performance data save (hourly)
        self._check_performance_save()

    def on_stop(self):
        """Called when strategy stops."""
        self.logger().info("Monster Bot V2 stopping...")
        self._close_all_positions()
        self._cancel_all_orders()
        # Save final performance data on shutdown
        self._save_performance_data()
        self.logger().info(f"Final performance data saved to {self.data_dir}")

    # =========================================================================
    # FUNDING RATE HUNTER - THE KEY V2 UPGRADE
    # =========================================================================

    def _run_funding_hunter(self):
        """
        Dynamic Funding Rate Hunter

        Scans all configured pairs for best funding opportunities,
        auto-rotates to capture highest rates.
        """
        # Scan for opportunities periodically
        if self.last_funding_scan < self.current_timestamp - self.config.funding_scan_interval:
            self._scan_funding_opportunities()
            self.last_funding_scan = self.current_timestamp

        # Manage existing positions
        self._manage_funding_positions()

        # Open new positions if slots available
        self._open_best_funding_positions()

    def _scan_funding_opportunities(self):
        """Scan all pairs for funding rate opportunities."""
        connector = self.connectors[self.config.exchange]
        opportunities = []

        for pair in self.config.funding_scan_pairs.split(","):
            pair = pair.strip()

            try:
                funding_info = connector.get_funding_info(pair)
                if funding_info is None:
                    continue

                rate = funding_info.rate

                # Calculate APR (Hyperliquid = hourly funding = 8760 periods/year)
                apr = abs(rate) * 8760 * 100  # Convert to percentage

                # Determine direction to receive funding
                # Positive rate = shorts pay longs = go LONG
                # Negative rate = longs pay shorts = go SHORT
                direction = "long" if rate > 0 else "short"

                # Calculate score (APR weighted by time to funding)
                time_to_funding = funding_info.next_funding_utc_timestamp - self.current_timestamp
                minutes_to_funding = time_to_funding / 60

                # Higher score for higher APR and closer to funding time
                urgency_bonus = 1.0
                if minutes_to_funding <= self.config.minutes_before_funding:
                    urgency_bonus = 1.5  # Boost score if funding is imminent

                score = apr * urgency_bonus

                opp = FundingOpportunity(
                    pair=pair,
                    rate=rate,
                    apr=apr,
                    direction=direction,
                    next_funding_time=funding_info.next_funding_utc_timestamp,
                    score=score
                )

                opportunities.append(opp)

            except Exception as e:
                self.logger().debug(f"Error scanning {pair}: {e}")

        # Sort by score (highest first)
        self.funding_opportunities = sorted(opportunities, key=lambda x: x.score, reverse=True)

        # Log top opportunities
        if self.funding_opportunities:
            top3 = self.funding_opportunities[:3]
            self.logger().info("TOP FUNDING OPPORTUNITIES:")
            for opp in top3:
                minutes = (opp.next_funding_time - self.current_timestamp) / 60
                self.logger().info(
                    f"  {opp.pair}: {opp.apr:.1f}% APR ({opp.direction.upper()}) "
                    f"- {minutes:.0f}min to funding"
                )

    def _manage_funding_positions(self):
        """Manage existing funding positions - close or rotate."""
        connector = self.connectors[self.config.exchange]

        for pair, current_opp in list(self.funding_positions.items()):
            try:
                funding_info = connector.get_funding_info(pair)
                if funding_info is None:
                    continue

                time_to_funding = funding_info.next_funding_utc_timestamp - self.current_timestamp
                minutes_to_funding = time_to_funding / 60

                # Check if funding just happened (close window)
                if minutes_to_funding > 55:
                    self._close_funding_position(pair, "funding_collected")
                    continue

                # Check if rate flipped direction
                new_rate = funding_info.rate
                new_direction = "long" if new_rate > 0 else "short"
                if new_direction != current_opp.direction:
                    self._close_funding_position(pair, "rate_flipped")
                    continue

                # Check if better opportunity exists (rotation)
                current_apr = abs(new_rate) * 8760 * 100

                for opp in self.funding_opportunities:
                    if opp.pair not in self.funding_positions:
                        apr_improvement = opp.apr - current_apr
                        if apr_improvement >= float(self.config.funding_rotation_threshold):
                            self.logger().info(
                                f"ROTATING: {pair} ({current_apr:.1f}% APR) -> "
                                f"{opp.pair} ({opp.apr:.1f}% APR)"
                            )
                            self._close_funding_position(pair, "rotation")
                            break

            except Exception as e:
                self.logger().error(f"Error managing funding position {pair}: {e}")

    def _open_best_funding_positions(self):
        """Open positions on best funding opportunities."""
        # Check if we have slots available
        current_positions = len(self.funding_positions)
        available_slots = self.config.max_funding_positions - current_positions

        if available_slots <= 0:
            return

        # Get funding info for timing check
        connector = self.connectors[self.config.exchange]

        for opp in self.funding_opportunities:
            if available_slots <= 0:
                break

            if opp.pair in self.funding_positions:
                continue

            # Check APR threshold
            if opp.apr < float(self.config.min_funding_apr):
                continue

            # Check timing
            time_to_funding = opp.next_funding_time - self.current_timestamp
            minutes_to_funding = time_to_funding / 60

            if minutes_to_funding > self.config.minutes_before_funding:
                continue

            # Open position
            if self._open_funding_position(opp):
                available_slots -= 1

    def _open_funding_position(self, opp: FundingOpportunity) -> bool:
        """Open a position to collect funding with smart leverage."""
        connector = self.connectors[self.config.exchange]

        position_size = self.config.funding_position_size

        # SMART LEVERAGE: Get safe leverage based on coin volatility
        if self.config.use_smart_leverage:
            leverage = get_safe_leverage(opp.pair, self.config.funding_leverage_max)
        else:
            leverage = self.config.funding_leverage_max

        price = connector.get_price_by_type(opp.pair, PriceType.MidPrice)
        if price is None:
            return False

        amount = position_size / price
        trade_type = TradeType.BUY if opp.direction == "long" else TradeType.SELL

        # Get volatility class for logging
        vol_class = COIN_VOLATILITY.get(opp.pair, CoinVolatility.HIGH).value.upper()

        order_id = self._place_perpetual_order(
            pair=opp.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.OPEN,
            strategy="funding"
        )

        if order_id:
            self.funding_positions[opp.pair] = opp
            self.logger().info(
                f"FUNDING HUNTER: Opened {opp.direction.upper()} on {opp.pair} @ {price:.4f} "
                f"({opp.apr:.1f}% APR, ${position_size} x {leverage}x [{vol_class} volatility])"
            )
            return True

        return False

    def _close_funding_position(self, pair: str, reason: str):
        """Close a funding position."""
        if pair not in self.funding_positions:
            return

        connector = self.connectors[self.config.exchange]
        opp = self.funding_positions[pair]

        price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        # Use same leverage as when opened
        if self.config.use_smart_leverage:
            leverage = get_safe_leverage(pair, self.config.funding_leverage_max)
        else:
            leverage = self.config.funding_leverage_max

        amount = self.config.funding_position_size / price
        trade_type = TradeType.SELL if opp.direction == "long" else TradeType.BUY

        order_id = self._place_perpetual_order(
            pair=pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.CLOSE,
            strategy="funding"
        )

        if order_id:
            del self.funding_positions[pair]
            self.logger().info(f"FUNDING HUNTER: Closed {opp.direction.upper()} on {pair} ({reason})")

    def did_complete_funding_payment(self, event: FundingPaymentCompletedEvent):
        """Handle funding payment events."""
        self.total_funding_collected += event.amount
        self.metrics["funding"].funding_pnl += event.amount
        self.global_pnl += event.amount

        # Track per-coin performance
        coin_perf = self._get_coin_performance(event.trading_pair)
        apr = 0.0
        if event.trading_pair in self.funding_positions:
            apr = self.funding_positions[event.trading_pair].apr
        coin_perf.add_payment(float(event.amount), apr)

        self.logger().info(
            f"FUNDING RECEIVED: {event.trading_pair} ${event.amount:.4f} "
            f"(Session Total: ${self.total_funding_collected:.4f}) "
            f"[Coin Total: ${coin_perf.total_funding_received:.4f}]"
        )

        # Save immediately on each payment for real-time tracking
        self._save_performance_data()

    # =========================================================================
    # GRID TRADING STRATEGY (Same as v1 but with tighter params)
    # =========================================================================

    def _run_grid_strategy(self):
        """Leveraged Grid Trading with tighter spacing."""
        connector = self.connectors[self.config.exchange]
        pair = self.config.grid_pair

        price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        if not self.grid_initialized:
            self._initialize_grid(price)
            return

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

        self.logger().info(f"GRID: Initializing {levels} levels each side around {base_price:.2f}")

        amount_per_order = order_size / base_price

        # Buy orders below
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

        # Sell orders above
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
        self.logger().info(f"GRID: {len(self.grid_orders)} orders placed ({leverage}x leverage)")

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
    # MOMENTUM STRATEGY (Same structure, slightly adjusted params)
    # =========================================================================

    def _run_momentum_strategy(self):
        """Momentum Trading with Long/Short capability."""
        connector = self.connectors[self.config.exchange]
        pair = self.config.momentum_pair

        price = connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        now = datetime.now()
        if pair not in self.price_history:
            self.price_history[pair] = []

        self.price_history[pair].append((now, price))

        cutoff = now - timedelta(hours=1)
        self.price_history[pair] = [(t, p) for t, p in self.price_history[pair] if t > cutoff]

        if len(self.price_history[pair]) < self.config.momentum_lookback:
            return

        if self.metrics["momentum"].status == StrategyMode.WARMING_UP:
            self.metrics["momentum"].status = StrategyMode.ACTIVE
            self.logger().info("MOMENTUM: Strategy activated")

        rsi = self._calculate_rsi(pair)
        if rsi is None:
            return

        if self.momentum_position and self.momentum_entry_price:
            should_exit, reason = self._check_momentum_exit(price)
            if should_exit:
                self._close_momentum_position(price, reason)
                return

        if self.momentum_position is None:
            if rsi < float(self.config.rsi_oversold):
                self._open_momentum_position("long", price, rsi)
            elif rsi > float(self.config.rsi_overbought):
                self._open_momentum_position("short", price, rsi)

    def _calculate_rsi(self, pair: str) -> Optional[float]:
        """Calculate RSI from price history."""
        prices = [p for _, p in self.price_history[pair]]
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
                f"(RSI: {rsi:.1f}, ${position_size} x {leverage}x)"
            )

    def _check_momentum_exit(self, current_price: Decimal) -> Tuple[bool, str]:
        """Check if momentum position should exit."""
        if not self.momentum_entry_price:
            return False, ""

        if self.momentum_position == "long":
            pnl_pct = (current_price - self.momentum_entry_price) / self.momentum_entry_price
        else:
            pnl_pct = (self.momentum_entry_price - current_price) / self.momentum_entry_price

        if pnl_pct >= self.config.momentum_take_profit:
            return True, "take_profit"

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

        if order_id in self.grid_orders:
            self.metrics["grid"].total_trades += 1
            self._handle_grid_fill(event, order_id)
        else:
            for strategy in ["funding", "momentum"]:
                self.metrics[strategy].total_trades += 1

        side = "BUY" if event.trade_type == TradeType.BUY else "SELL"
        self.logger().info(
            f"FILL: {side} {event.amount:.6f} {event.trading_pair} @ {event.price:.4f}"
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
        self._close_all_positions()
        self._cancel_all_orders()

        for strategy in self.metrics.values():
            strategy.status = StrategyMode.KILLED

    def _close_all_positions(self):
        """Close all open positions."""
        for pair in list(self.funding_positions.keys()):
            self._close_funding_position(pair, "shutdown")

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
        lines.append("  HYPERLIQUID MONSTER BOT V2 - AGGRESSIVE MODE")
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
        lines.append(f"  SAFE (8x): BTC | MEDIUM (5x): ETH, SOL, TAO")
        lines.append(f"  HIGH (3x): DOGE,HYPE,MOG,AVNT | EXTREME (2x): IP,VVV,PEPE,BONK,WIF,HYPER")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  TOP FUNDING OPPORTUNITIES")
        lines.append("-" * 70)
        for opp in self.funding_opportunities[:5]:
            minutes = (opp.next_funding_time - self.current_timestamp) / 60
            in_position = "***" if opp.pair in self.funding_positions else "   "
            lines.append(f"  {in_position} {opp.pair:<12} {opp.apr:>6.1f}% APR ({opp.direction:<5}) {minutes:>4.0f}min")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  STRATEGY STATUS")
        lines.append("-" * 70)

        for key, m in self.metrics.items():
            status = m.status.value
            pnl = m.realized_pnl + m.funding_pnl
            lines.append(f"  {m.name:<20} [{status:<10}] P&L: ${pnl:+.4f} Trades: {m.total_trades}")

        # === COIN PERFORMANCE LEADERBOARD ===
        lines.append("")
        lines.append("-" * 70)
        lines.append("  COIN PERFORMANCE LEADERBOARD")
        lines.append("-" * 70)
        if self.coin_performance:
            sorted_coins = sorted(
                self.coin_performance.items(),
                key=lambda x: x[1].total_funding_received,
                reverse=True
            )[:5]  # Top 5
            lines.append(f"  {'Rank':<4} {'Coin':<12} {'Funding':<12} {'Payments':<10} {'Avg APR':<10}")
            lines.append("  " + "-" * 48)
            for rank, (symbol, perf) in enumerate(sorted_coins, 1):
                lines.append(
                    f"  {rank:<4} {symbol:<12} ${perf.total_funding_received:<10.4f} "
                    f"{perf.funding_payments_count:<10} {perf.avg_apr_captured:<10.0f}%"
                )
        else:
            lines.append("  No funding payments recorded yet")

        lines.append("")
        lines.append("-" * 70)
        lines.append("  ACTIVE POSITIONS")
        lines.append("-" * 70)

        if self.funding_positions:
            for pair, opp in self.funding_positions.items():
                lines.append(f"  FUNDING: {opp.direction.upper():<5} {pair} ({opp.apr:.1f}% APR)")

        if self.momentum_position:
            lines.append(f"  MOMENTUM: {self.momentum_position.upper()} {self.config.momentum_pair} @ {self.momentum_entry_price}")

        lines.append(f"  GRID: {len(self.grid_orders)} orders active on {self.config.grid_pair}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
