"""
Backtest Performance Metrics.

Calculates Sharpe ratio, max drawdown, Sortino ratio, etc.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from .models import BacktestTrade, EquityCurvePoint


@dataclass
class BacktestMetrics:
    """
    Calculator for backtest performance metrics.

    Usage:
        metrics = BacktestMetrics(trades, equity_curve, initial_capital)
        sharpe = metrics.sharpe_ratio()
        max_dd = metrics.max_drawdown()
    """

    trades: List[BacktestTrade]
    equity_curve: List[EquityCurvePoint]
    initial_capital: float
    risk_free_rate: float = 0.05  # 5% annual risk-free rate

    def total_return_pct(self) -> float:
        """Calculate total return percentage."""
        if not self.equity_curve:
            return 0.0

        final_equity = self.equity_curve[-1].equity
        return (final_equity - self.initial_capital) / self.initial_capital * 100

    def total_return_usd(self) -> float:
        """Calculate total return in USD."""
        if not self.equity_curve:
            return 0.0

        return self.equity_curve[-1].equity - self.initial_capital

    def max_drawdown(self) -> tuple:
        """
        Calculate maximum drawdown.

        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_usd)
        """
        if not self.equity_curve:
            return 0.0, 0.0

        peak = self.initial_capital
        max_dd_pct = 0.0
        max_dd_usd = 0.0

        for point in self.equity_curve:
            if point.equity > peak:
                peak = point.equity

            dd_usd = peak - point.equity
            dd_pct = dd_usd / peak * 100 if peak > 0 else 0

            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_usd = dd_usd

        return max_dd_pct, max_dd_usd

    def _get_returns(self) -> List[float]:
        """Get daily/period returns from equity curve."""
        if len(self.equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_eq = self.equity_curve[i - 1].equity
            curr_eq = self.equity_curve[i].equity
            if prev_eq > 0:
                ret = (curr_eq - prev_eq) / prev_eq
                returns.append(ret)

        return returns

    def sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            periods_per_year: Number of periods in a year (252 for daily, 365*24 for hourly)

        Returns:
            Annualized Sharpe ratio
        """
        returns = self._get_returns()
        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)
        std_return = self._std(returns)

        if std_return == 0:
            return 0.0

        # Annualize
        rf_per_period = self.risk_free_rate / periods_per_year
        excess_return = avg_return - rf_per_period

        sharpe = (excess_return / std_return) * math.sqrt(periods_per_year)
        return sharpe

    def sortino_ratio(self, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (penalizes only downside volatility).

        Args:
            periods_per_year: Number of periods in a year

        Returns:
            Annualized Sortino ratio
        """
        returns = self._get_returns()
        if not returns:
            return 0.0

        avg_return = sum(returns) / len(returns)

        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf') if avg_return > 0 else 0.0

        downside_deviation = self._std(negative_returns)

        if downside_deviation == 0:
            return float('inf') if avg_return > 0 else 0.0

        rf_per_period = self.risk_free_rate / periods_per_year
        excess_return = avg_return - rf_per_period

        sortino = (excess_return / downside_deviation) * math.sqrt(periods_per_year)
        return sortino

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Returns:
            Calmar ratio (annualized return / max drawdown)
        """
        max_dd_pct, _ = self.max_drawdown()
        if max_dd_pct == 0:
            return float('inf') if self.total_return_pct() > 0 else 0.0

        # Annualize return (assuming equity curve spans the test period)
        total_return = self.total_return_pct()

        return total_return / max_dd_pct

    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor
        """
        gross_profit = sum(t.pnl_usd for t in self.trades if t.pnl_usd and t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd and t.pnl_usd < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if not self.trades:
            return 0.0

        winners = sum(1 for t in self.trades if t.pnl_usd and t.pnl_usd > 0)
        return winners / len(self.trades) * 100

    def avg_win(self) -> float:
        """Calculate average winning trade percentage."""
        winners = [t.pnl_pct for t in self.trades if t.pnl_pct and t.pnl_pct > 0]
        return sum(winners) / len(winners) if winners else 0.0

    def avg_loss(self) -> float:
        """Calculate average losing trade percentage."""
        losers = [t.pnl_pct for t in self.trades if t.pnl_pct and t.pnl_pct < 0]
        return sum(losers) / len(losers) if losers else 0.0

    def avg_trade(self) -> float:
        """Calculate average trade return percentage."""
        if not self.trades:
            return 0.0

        all_pnl = [t.pnl_pct for t in self.trades if t.pnl_pct is not None]
        return sum(all_pnl) / len(all_pnl) if all_pnl else 0.0

    def largest_win(self) -> float:
        """Get largest winning trade percentage."""
        winners = [t.pnl_pct for t in self.trades if t.pnl_pct and t.pnl_pct > 0]
        return max(winners) if winners else 0.0

    def largest_loss(self) -> float:
        """Get largest losing trade percentage."""
        losers = [t.pnl_pct for t in self.trades if t.pnl_pct and t.pnl_pct < 0]
        return min(losers) if losers else 0.0

    def avg_hold_time_hours(self) -> float:
        """Calculate average trade hold time in hours."""
        hold_times = []
        for t in self.trades:
            if t.entry_time and t.exit_time:
                hours = (t.exit_time - t.entry_time).total_seconds() / 3600
                hold_times.append(hours)

        return sum(hold_times) / len(hold_times) if hold_times else 0.0

    def expectancy(self) -> float:
        """
        Calculate trading expectancy.

        Expectancy = (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
        Positive expectancy means profitable system over time.
        """
        wr = self.win_rate() / 100
        lr = 1 - wr
        avg_w = self.avg_win()
        avg_l = self.avg_loss()  # Already negative

        return (wr * avg_w) + (lr * avg_l)

    def recovery_factor(self) -> float:
        """
        Calculate recovery factor (net profit / max drawdown).

        Higher is better - how much profit relative to worst drawdown.
        """
        max_dd_pct, max_dd_usd = self.max_drawdown()
        if max_dd_usd == 0:
            return float('inf') if self.total_return_usd() > 0 else 0.0

        return self.total_return_usd() / max_dd_usd

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def get_all_metrics(self) -> dict:
        """Get all metrics as a dictionary."""
        max_dd_pct, max_dd_usd = self.max_drawdown()

        return {
            "total_return_pct": self.total_return_pct(),
            "total_return_usd": self.total_return_usd(),
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_usd": max_dd_usd,
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "profit_factor": self.profit_factor(),
            "win_rate": self.win_rate(),
            "avg_win_pct": self.avg_win(),
            "avg_loss_pct": self.avg_loss(),
            "avg_trade_pct": self.avg_trade(),
            "largest_win_pct": self.largest_win(),
            "largest_loss_pct": self.largest_loss(),
            "avg_hold_time_hours": self.avg_hold_time_hours(),
            "expectancy": self.expectancy(),
            "recovery_factor": self.recovery_factor(),
            "total_trades": len(self.trades),
            "winning_trades": sum(1 for t in self.trades if t.pnl_usd and t.pnl_usd > 0),
            "losing_trades": sum(1 for t in self.trades if t.pnl_usd and t.pnl_usd < 0),
        }
