"""
Status Formatter - Formats bot status display.

Extracts the complex status formatting logic into a dedicated module.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional, Any


class StatusFormatter:
    """Formats bot status for display."""
    
    def __init__(self, config, performance_tracker):
        """
        Initialize StatusFormatter.
        
        Args:
            config: Bot configuration
            performance_tracker: PerformanceTracker instance
        """
        self.config = config
        self.performance_tracker = performance_tracker
    
    def format_status(
        self,
        start_time: datetime,
        global_pnl: Decimal,
        total_funding_collected: Decimal,
        peak_equity: Decimal,
        global_killed: bool,
        metrics: Dict[str, Any],
        funding_strategy=None,
        momentum_strategy=None,
        grid_strategy=None,
        whale_orchestrator=None,
        pause_manager=None,
        current_timestamp: float = 0,
    ) -> str:
        """
        Format complete bot status.
        
        Args:
            start_time: Bot start time
            global_pnl: Global P&L
            total_funding_collected: Total funding collected
            peak_equity: Peak equity value
            global_killed: Whether bot is killed
            metrics: Strategy metrics dict
            funding_strategy: FundingHunterStrategy instance
            momentum_strategy: MomentumStrategy instance
            grid_strategy: GridStrategy instance
            whale_orchestrator: WhaleProtectionOrchestrator instance
            pause_manager: PauseManager instance
            current_timestamp: Current timestamp
            
        Returns:
            Formatted status string
        """
        lines = []
        
        lines.append("")
        lines.append("=" * 70)
        lines.append("  HYPERLIQUID MONSTER BOT V2.7 - HOLY GRAIL SIGNALS")
        lines.append("=" * 70)
        
        runtime = datetime.now() - start_time
        lines.append(f"  Runtime: {runtime.days}d {runtime.seconds//3600}h {(runtime.seconds%3600)//60}m")
        
        # Global Metrics
        lines.extend(self._format_global_metrics(global_pnl, total_funding_collected, peak_equity, global_killed))
        
        # Smart Leverage
        lines.extend(self._format_leverage_info())
        
        # Funding Opportunities
        if funding_strategy and funding_strategy.opportunities:
            lines.extend(self._format_funding_opportunities(funding_strategy, current_timestamp))
        
        # Strategy Status
        lines.extend(self._format_strategy_status(metrics))
        
        # Coin Performance Leaderboard
        lines.extend(self._format_coin_leaderboard())
        
        # Active Positions
        lines.extend(self._format_active_positions(funding_strategy, momentum_strategy, grid_strategy))
        
        # Funding Status
        if funding_strategy:
            lines.extend(self._format_funding_status(funding_strategy))
        
        # Grid Status
        if grid_strategy:
            lines.extend(self._format_grid_status(grid_strategy))
        
        # Momentum Indicators
        if momentum_strategy:
            lines.extend(self._format_momentum_indicators(momentum_strategy))
        
        # Whale Protection Status
        if self.config.whale_protection_enabled and whale_orchestrator:
            lines.extend(self._format_whale_protection_status(
                whale_orchestrator, pause_manager, current_timestamp
            ))
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _format_global_metrics(
        self,
        global_pnl: Decimal,
        total_funding_collected: Decimal,
        peak_equity: Decimal,
        global_killed: bool,
    ) -> list:
        """Format global metrics section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  GLOBAL METRICS")
        lines.append("-" * 70)
        lines.append(f"  Capital:          ${self.config.total_capital}")
        lines.append(f"  Total P&L:        ${global_pnl:+.4f}")
        lines.append(f"  Funding Collected: ${total_funding_collected:.4f}")
        
        current_equity = self.config.total_capital + global_pnl
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity * 100
            lines.append(f"  Drawdown:         {drawdown:.2f}% (max: {self.config.max_drawdown_pct}%)")
        
        status = "KILLED" if global_killed else "RUNNING"
        lines.append(f"  Status:           {status}")
        return lines
    
    def _format_leverage_info(self) -> list:
        """Format smart leverage section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  SMART LEVERAGE (by volatility)")
        lines.append("-" * 70)
        lines.append(f"  Grid (SOL): {self.config.grid_leverage}x | Momentum (BTC): {self.config.momentum_leverage}x")
        lines.append(f"  Funding: Dynamic per coin (2x-8x based on volatility)")
        return lines
    
    def _format_funding_opportunities(self, funding_strategy, current_timestamp: float) -> list:
        """Format funding opportunities section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  TOP FUNDING OPPORTUNITIES")
        lines.append("-" * 70)
        for opp in funding_strategy.opportunities[:5]:
            minutes = (opp.next_funding_time - current_timestamp) / 60
            in_position = "***" if opp.pair in funding_strategy.positions else "   "
            oi_status = getattr(opp, '_oi_status', '')
            lines.append(f"  {in_position} {opp.pair:<12} {opp.apr:>6.1f}% APR ({opp.direction:<5}) {minutes:>4.0f}min{oi_status}")
        return lines
    
    def _format_strategy_status(self, metrics: Dict[str, Any]) -> list:
        """Format strategy status section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  STRATEGY STATUS")
        lines.append("-" * 70)
        
        for m in metrics.values():
            status = m.status.value
            pnl = m.realized_pnl + m.funding_pnl
            lines.append(f"  {m.name:<20} [{status:<10}] P&L: ${pnl:+.4f} Trades: {m.total_trades}")
        return lines
    
    def _format_coin_leaderboard(self) -> list:
        """Format coin performance leaderboard section."""
        lines = []
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
        return lines
    
    def _format_active_positions(self, funding_strategy, momentum_strategy, grid_strategy) -> list:
        """Format active positions section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  ACTIVE POSITIONS")
        lines.append("-" * 70)
        
        if funding_strategy and funding_strategy.positions:
            for pair, opp in funding_strategy.positions.items():
                lines.append(f"  FUNDING: {opp.direction.upper():<5} {pair} ({opp.apr:.1f}% APR)")
        
        if momentum_strategy:
            pos_info = momentum_strategy.get_position_info()
            if pos_info:
                direction, entry = pos_info
                lines.append(f"  MOMENTUM: {direction.upper()} {self.config.momentum_pair} @ {entry}")
        
        if grid_strategy:
            lines.append(f"  GRID: {grid_strategy.get_order_count()} orders active on {self.config.grid_pair}")
            grid_status = grid_strategy.get_status_info()
            if grid_status:
                status_parts = [f"{k}: {v}" for k, v in grid_status.items() if k != "Orders"]
                if status_parts:
                    lines.append(f"        [{', '.join(status_parts)}]")
        return lines
    
    def _format_funding_status(self, funding_strategy) -> list:
        """Format funding status section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  FUNDING STATUS")
        lines.append("-" * 70)
        funding_status = funding_strategy.get_status_info()
        if funding_status:
            for name, value in funding_status.items():
                lines.append(f"  {name:<20}: {value}")

        # Show OI analysis for active funding positions
        oi_tracker = funding_strategy.get_oi_tracker()
        if oi_tracker and funding_strategy.positions:
            lines.append("")
            lines.append("  OI Analysis (Leading Indicator):")
            for pair in list(funding_strategy.positions.keys())[:3]:
                oi_analysis = oi_tracker.analyze_oi_momentum(pair)
                if oi_analysis:
                    momentum = oi_analysis.momentum.value.replace("_", " ").title()
                    status_icon = "+" if oi_analysis.oi_change_pct > 0 else ""
                    lines.append(
                        f"    {pair}: {momentum} (OI: {status_icon}{oi_analysis.oi_change_pct:.1f}%)"
                    )

        return lines
    
    def _format_grid_status(self, grid_strategy) -> list:
        """Format grid status section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  GRID STATUS")
        lines.append("-" * 70)
        grid_status = grid_strategy.get_status_info()
        if grid_status:
            for name, value in grid_status.items():
                lines.append(f"  {name:<10}: {value}")
        else:
            lines.append("  Warming up - collecting price data...")
        return lines
    
    def _format_momentum_indicators(self, momentum_strategy) -> list:
        """Format momentum indicators section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  MOMENTUM INDICATORS")
        lines.append("-" * 70)
        indicator_status = momentum_strategy.get_indicator_status()
        if indicator_status:
            # Show lagging indicators first
            for name in ["RSI", "Trend", "MACD", "Volume", "Funding"]:
                if name in indicator_status:
                    lines.append(f"  {name:<10}: {indicator_status[name]}")

            # Show OI (leading indicator) with emphasis
            if "OI" in indicator_status:
                lines.append(f"  {'OI':<10}: {indicator_status['OI']}  <-- LEADING")

            # Show HOLY GRAIL combined signal (THE ULTIMATE INDICATOR)
            if "HolyGrail" in indicator_status:
                lines.append("")
                lines.append(f"  {'HOLY GRAIL':<10}: {indicator_status['HolyGrail']}  <-- COMBINED")

            signals_list = "[Trend, MACD, Volume, Funding"
            if self.config.use_oi_filter:
                signals_list += ", OI"
            signals_list += "]"
            lines.append(f"  Signals Required: RSI + {self.config.min_signals_required} of {signals_list}")
        else:
            lines.append("  Warming up - collecting price data...")

        # OI API Health Stats
        oi_tracker = momentum_strategy.get_oi_tracker()
        if oi_tracker:
            oi_status = oi_tracker.get_status()
            api_health = "OK" if oi_status["error_count"] == 0 else f"ERRORS: {oi_status['error_count']}"
            lines.append("")
            lines.append(f"  OI API Status:    {api_health} (fetches: {oi_status['fetch_count']}, pairs: {oi_status['pairs_tracked']})")

            # Show OI analysis for tracked pairs
            if oi_status.get("oi_analysis"):
                for pair, analysis in list(oi_status["oi_analysis"].items())[:3]:
                    momentum = analysis["momentum"].replace("_", " ").title()
                    lines.append(
                        f"                    {pair}: {momentum} "
                        f"(OI: {analysis['oi_change']}, Price: {analysis['price_change']})"
                    )

        return lines
    
    def _format_whale_protection_status(
        self,
        whale_orchestrator,
        pause_manager,
        current_timestamp: float,
    ) -> list:
        """Format whale protection status section."""
        lines = []
        lines.append("")
        lines.append("-" * 70)
        lines.append("  WHALE PROTECTION STATUS")
        lines.append("-" * 70)
        
        # Circuit Breaker
        if whale_orchestrator.circuit_breaker:
            cb_status = whale_orchestrator.circuit_breaker.get_status()
            state_str = cb_status["state"].upper()
            if cb_status["state"] == "cooldown":
                state_str += f" ({cb_status['cooldown_remaining']:.0f}s left)"
            lines.append(f"  Circuit Breaker:  {state_str} (triggers: {cb_status['total_triggers']})")
        
        # Grid Protection
        if whale_orchestrator.grid_protection:
            gp_status = whale_orchestrator.grid_protection.get_status()
            state_str = gp_status["state"].upper()
            if gp_status["state"] == "cooldown":
                state_str += f" ({gp_status['cooldown_remaining']:.0f}s left)"
            lines.append(f"  Grid Protection:  {state_str} (triggers: {gp_status['total_triggers']})")
            
            # Show fill stats for grid pair
            fill_stats = whale_orchestrator.grid_protection.get_fill_stats(self.config.grid_pair)
            if fill_stats["total_fills"] > 0:
                lines.append(
                    f"                    Fills: {fill_stats['buy_fills']}B/{fill_stats['sell_fills']}S "
                    f"(ratio: {fill_stats['imbalance_ratio']:.1f}:1 {fill_stats['dominant_side']})"
                )
        
        # Trailing Stop
        if whale_orchestrator.trailing_stop:
            ts_status = whale_orchestrator.trailing_stop.get_status()
            if ts_status["active_stops"] > 0:
                for stop in ts_status["stops"]:
                    be_str = "BE" if stop["is_breakeven"] else "--"
                    tr_str = "TRAIL" if stop["is_trailing"] else "-----"
                    lines.append(
                        f"  Trailing Stop:    {stop['pair']} {stop['direction'].upper()} "
                        f"stop@{stop['current_stop']:.2f} [{be_str}|{tr_str}]"
                    )
            else:
                lines.append(f"  Trailing Stop:    No active stops (mode: {ts_status['mode']})")
        
        # Dynamic Risk
        if whale_orchestrator.dynamic_risk:
            dr_status = whale_orchestrator.dynamic_risk.get_status()
            lines.append(f"  Dynamic Risk:     {dr_status['pairs_monitored']} pairs monitored")
            for pair, info in list(dr_status.get("pair_status", {}).items())[:3]:
                lines.append(
                    f"                    {pair}: {info['regime']} (SL mult: {info['sl_multiplier']:.2f}x)"
                )
        
        # Early Warning
        if whale_orchestrator.early_warning:
            ew_status = whale_orchestrator.early_warning.get_status()
            active_warnings = sum(p["active_warnings"] for p in ew_status.get("pair_status", {}).values())
            if active_warnings > 0:
                lines.append(f"  Early Warning:    {active_warnings} ACTIVE WARNINGS")
                for pair, info in ew_status.get("pair_status", {}).items():
                    if info["active_warnings"] > 0:
                        lines.append(
                            f"                    {pair}: {info['level'].upper()} ({', '.join(info['categories'])})"
                        )
            else:
                lines.append(f"  Early Warning:    CLEAR ({ew_status['pairs_monitored']} pairs monitored)")
        
        # Overall status - show if paused and why
        if pause_manager and pause_manager.is_paused:
            lines.append("")
            lines.append("  TRADING PAUSED - Waiting for conditions to normalize")
            for system, reason in pause_manager.pause_reasons.items():
                lines.append(f"     {system}: {reason}")
            pause_duration = current_timestamp - pause_manager.last_pause_time
            lines.append(f"     Paused for: {pause_duration:.0f} seconds")
        
        return lines
