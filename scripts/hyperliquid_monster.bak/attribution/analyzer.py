"""
Attribution Analyzer - Analyzes which signals contribute to wins/losses.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from .models import (
    AttributionReport,
    SignalPerformance,
    SignalType,
    TradeAttribution,
)


class AttributionAnalyzer:
    """
    Analyzes trade attribution data to determine signal effectiveness.

    Usage:
        analyzer = AttributionAnalyzer(tracker.completed_trades)
        report = analyzer.generate_report()
        print(report.get_summary())
    """

    MIN_TRADES_FOR_ANALYSIS = 10

    def __init__(
        self,
        trades: List[TradeAttribution],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize analyzer with trade data.

        Args:
            trades: List of completed trades with attribution
            logger: Logger instance
        """
        self.trades = [t for t in trades if t.is_win is not None]
        self.logger = logger or logging.getLogger(__name__)

    def get_signal_performance(self) -> Dict[SignalType, SignalPerformance]:
        """
        Calculate performance statistics for each signal type.

        Returns:
            Dictionary mapping signal type to performance stats
        """
        signal_stats: Dict[SignalType, Dict] = defaultdict(lambda: {
            "wins": [],
            "losses": [],
            "pnl_list": [],
            "hold_times": [],
        })

        for trade in self.trades:
            if not trade.entry_signals or trade.pnl_pct is None:
                continue

            for signal in trade.entry_signals.active_signals:
                stats = signal_stats[signal]
                stats["pnl_list"].append(trade.pnl_pct)
                if trade.hold_time_seconds:
                    stats["hold_times"].append(trade.hold_time_seconds)

                if trade.is_win:
                    stats["wins"].append(trade.pnl_pct)
                else:
                    stats["losses"].append(trade.pnl_pct)

        # Convert to SignalPerformance objects
        performances: Dict[SignalType, SignalPerformance] = {}

        for signal_type, stats in signal_stats.items():
            total_trades = len(stats["pnl_list"])
            if total_trades == 0:
                continue

            wins = stats["wins"]
            losses = stats["losses"]

            # Calculate metrics
            total_pnl = sum(stats["pnl_list"])
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

            max_win = max(wins) if wins else 0.0
            max_loss = min(losses) if losses else 0.0

            win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0.0

            # Profit factor
            gross_profit = sum(wins) if wins else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0
                else float('inf') if gross_profit > 0
                else 0.0
            )

            avg_hold = (
                sum(stats["hold_times"]) / len(stats["hold_times"])
                if stats["hold_times"] else 0.0
            )

            performances[signal_type] = SignalPerformance(
                signal_type=signal_type,
                total_trades=total_trades,
                wins=len(wins),
                losses=len(losses),
                total_pnl_pct=total_pnl,
                avg_pnl_pct=avg_pnl,
                max_win_pct=max_win,
                max_loss_pct=max_loss,
                win_rate=win_rate,
                profit_factor=profit_factor if profit_factor != float('inf') else 999.0,
                avg_hold_time_seconds=avg_hold,
            )

        return performances

    def get_signal_correlations(self) -> Dict[str, float]:
        """
        Calculate correlation between signals and win rate.

        Returns:
            Dictionary of signal names to correlation scores
        """
        correlations = {}

        for signal_type in SignalType:
            trades_with_signal = [
                t for t in self.trades
                if t.entry_signals and signal_type in t.entry_signals.active_signals
            ]

            trades_without_signal = [
                t for t in self.trades
                if t.entry_signals and signal_type not in t.entry_signals.active_signals
            ]

            if len(trades_with_signal) < 5 or len(trades_without_signal) < 5:
                continue

            # Win rates
            wr_with = sum(1 for t in trades_with_signal if t.is_win) / len(trades_with_signal)
            wr_without = sum(1 for t in trades_without_signal if t.is_win) / len(trades_without_signal)

            # Correlation as difference in win rates
            # Positive = signal helps, Negative = signal hurts
            correlations[signal_type.value] = (wr_with - wr_without) * 100

        return correlations

    def get_signal_combinations(
        self,
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Find which signal combinations perform best.

        Returns:
            List of signal combo stats, sorted by win rate
        """
        combo_stats: Dict[frozenset, Dict] = defaultdict(lambda: {
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
        })

        for trade in self.trades:
            if not trade.entry_signals or trade.pnl_pct is None:
                continue

            # Create a frozenset of active signals for this trade
            combo = frozenset(trade.entry_signals.active_signals)

            if trade.is_win:
                combo_stats[combo]["wins"] += 1
            else:
                combo_stats[combo]["losses"] += 1
            combo_stats[combo]["total_pnl"] += trade.pnl_pct

        # Convert to list and calculate win rates
        combos = []
        for combo, stats in combo_stats.items():
            total = stats["wins"] + stats["losses"]
            if total < 3:  # Need minimum sample
                continue

            combos.append({
                "signals": [s.value for s in combo],
                "signal_count": len(combo),
                "total_trades": total,
                "win_rate": stats["wins"] / total * 100,
                "avg_pnl": stats["total_pnl"] / total,
            })

        # Sort by win rate (with minimum trades)
        combos.sort(key=lambda x: (-x["win_rate"], -x["total_trades"]))

        return combos[:top_n]

    def generate_recommendations(
        self,
        performances: Dict[SignalType, SignalPerformance],
    ) -> List[str]:
        """
        Generate actionable recommendations based on analysis.

        Args:
            performances: Signal performance dictionary

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if len(self.trades) < self.MIN_TRADES_FOR_ANALYSIS:
            return [
                f"Need more trades for analysis ({len(self.trades)}/{self.MIN_TRADES_FOR_ANALYSIS})"
            ]

        # Find underperforming signals (below 40% win rate with 10+ trades)
        underperforming = [
            (st, sp) for st, sp in performances.items()
            if sp.total_trades >= 10 and sp.win_rate < 40
        ]
        for signal_type, perf in underperforming:
            recommendations.append(
                f"DISABLE {signal_type.value}: {perf.win_rate:.0f}% WR over {perf.total_trades} trades"
            )

        # Find high-performing signals that should be required
        top_performers = [
            (st, sp) for st, sp in performances.items()
            if sp.total_trades >= 10 and sp.win_rate > 65
        ]
        for signal_type, perf in top_performers:
            recommendations.append(
                f"REQUIRE {signal_type.value}: {perf.win_rate:.0f}% WR with {perf.avg_pnl_pct:+.2f}% avg"
            )

        # Check for negative-edge signals (lose money on average)
        negative_edge = [
            (st, sp) for st, sp in performances.items()
            if sp.total_trades >= 10 and sp.avg_pnl_pct < -0.5
        ]
        for signal_type, perf in negative_edge:
            recommendations.append(
                f"WARNING {signal_type.value}: Negative edge ({perf.avg_pnl_pct:+.2f}% avg)"
            )

        # Check profit factors
        low_pf = [
            (st, sp) for st, sp in performances.items()
            if sp.total_trades >= 15 and sp.profit_factor < 1.0
        ]
        for signal_type, perf in low_pf:
            recommendations.append(
                f"REVIEW {signal_type.value}: Profit factor {perf.profit_factor:.2f} < 1.0"
            )

        return recommendations

    def generate_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> AttributionReport:
        """
        Generate complete attribution analysis report.

        Args:
            period_start: Start of analysis period
            period_end: End of analysis period

        Returns:
            Complete AttributionReport
        """
        # Filter by period if specified
        trades = self.trades
        if period_start:
            trades = [t for t in trades if t.entry_time >= period_start]
        if period_end:
            trades = [t for t in trades if t.entry_time <= period_end]

        # Calculate overall stats
        total_trades = len(trades)
        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]

        total_pnl_pct = sum(t.pnl_pct or 0 for t in trades)
        total_pnl_usd = sum(t.pnl_usd or 0 for t in trades if t.pnl_usd)

        # Get signal performance
        performances = self.get_signal_performance()

        # Sort signals by different metrics
        signals_by_win_rate = sorted(
            performances.values(),
            key=lambda x: (x.win_rate, x.total_trades),
            reverse=True
        )

        signals_by_pnl = sorted(
            performances.values(),
            key=lambda x: (x.total_pnl_pct, x.total_trades),
            reverse=True
        )

        # Best and worst signals (min 5 trades)
        best_signals = [
            sp.signal_type for sp in signals_by_win_rate
            if sp.total_trades >= 5
        ][:5]

        worst_signals = [
            sp.signal_type for sp in reversed(signals_by_win_rate)
            if sp.total_trades >= 5
        ][:5]

        most_profitable = [
            sp.signal_type for sp in signals_by_pnl
            if sp.total_trades >= 5
        ][:5]

        # Generate recommendations
        recommendations = self.generate_recommendations(performances)

        return AttributionReport(
            generated_at=datetime.utcnow(),
            period_start=period_start,
            period_end=period_end,
            total_trades=total_trades,
            total_wins=len(wins),
            total_losses=len(losses),
            overall_win_rate=len(wins) / max(1, total_trades) * 100,
            total_pnl_pct=total_pnl_pct,
            total_pnl_usd=total_pnl_usd,
            signal_performance=list(performances.values()),
            best_signals=best_signals,
            worst_signals=worst_signals,
            most_profitable_signals=most_profitable,
            recommendations=recommendations,
        )

    def get_summary_stats(self) -> dict:
        """Get summary statistics for display."""
        performances = self.get_signal_performance()

        # Find best/worst
        by_win_rate = sorted(
            performances.values(),
            key=lambda x: x.win_rate if x.total_trades >= 5 else 0,
            reverse=True
        )

        best = by_win_rate[0] if by_win_rate else None
        worst = by_win_rate[-1] if by_win_rate else None

        return {
            "total_trades": len(self.trades),
            "signals_tracked": len(performances),
            "best_signal": best.signal_type.value if best else None,
            "best_win_rate": best.win_rate if best else 0,
            "worst_signal": worst.signal_type.value if worst else None,
            "worst_win_rate": worst.win_rate if worst else 0,
        }
