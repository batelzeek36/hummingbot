"""
Performance Tracker for Hummingbot Trading Strategies

Analyzes trade history from Hummingbot's SQLite database and generates
performance reports in both human-readable (Markdown) and machine-readable (JSON) formats.

Usage:
    python scripts/analytics/performance_tracker.py [options]

Options:
    --strategy NAME     Filter by strategy name (default: all)
    --market NAME       Filter by exchange (default: all)
    --pair PAIR         Filter by trading pair (default: all)
    --days N            Analyze last N days (default: 30)
    --output DIR        Output directory (default: data/reports)
    --format FORMAT     Output format: json, markdown, both (default: both)

Example:
    python scripts/analytics/performance_tracker.py --strategy dollar_a_day_pmm --days 7
"""

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class TradeRecord:
    """Single trade record from database."""
    timestamp: int
    market: str
    symbol: str
    base_asset: str
    quote_asset: str
    trade_type: str  # BUY or SELL
    price: Decimal
    amount: Decimal
    trade_fee_in_quote: Optional[Decimal]
    strategy: str


@dataclass
class DailyMetrics:
    """Metrics for a single day."""
    date: str
    pnl: float
    cumulative_pnl: float
    num_trades: int
    buy_volume: float
    sell_volume: float
    fees_paid: float
    inventory_change: float


@dataclass
class PerformanceReport:
    """Complete performance report."""
    # Meta
    generated_at: str
    period_start: str
    period_end: str
    strategy: Optional[str]
    market: Optional[str]
    trading_pair: Optional[str]

    # Summary stats
    total_trades: int
    total_buys: int
    total_sells: int
    total_buy_volume: float
    total_sell_volume: float
    total_fees_paid: float

    # P&L metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    avg_trade_pnl: float

    # Return metrics
    total_return_pct: float
    avg_daily_return_pct: float
    best_day_pnl: float
    worst_day_pnl: float

    # Risk metrics
    sharpe_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    win_rate: float
    profit_factor: float

    # Trade metrics
    avg_buy_price: float
    avg_sell_price: float
    avg_spread_captured: float
    avg_trade_size: float

    # Daily breakdown
    daily_metrics: List[DailyMetrics]


class PerformanceTracker:
    """
    Analyzes Hummingbot trade history and generates performance reports.
    """

    DECIMAL_SCALE = 1_000_000  # SqliteDecimal uses 6 decimal places

    def __init__(self, db_path: str = None):
        """
        Initialize tracker with database path.

        Args:
            db_path: Path to hummingbot_trades.sqlite. If None, uses default location.
        """
        if db_path is None:
            # Default location
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "hummingbot_trades.sqlite"

        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        """Create database connection."""
        return sqlite3.connect(str(self.db_path))

    def _scale_decimal(self, value: int) -> Decimal:
        """Convert stored integer back to Decimal."""
        if value is None:
            return Decimal("0")
        return Decimal(value) / Decimal(self.DECIMAL_SCALE)

    def get_trades(
        self,
        strategy: str = None,
        market: str = None,
        trading_pair: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[TradeRecord]:
        """
        Fetch trades from database with optional filters.

        Args:
            strategy: Filter by strategy name
            market: Filter by exchange name
            trading_pair: Filter by trading pair (e.g., "ATOM-USDT")
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of TradeRecord objects
        """
        conn = self._connect()
        cursor = conn.cursor()

        query = """
            SELECT timestamp, market, symbol, base_asset, quote_asset,
                   trade_type, price, amount, trade_fee_in_quote, strategy
            FROM TradeFill
            WHERE 1=1
        """
        params = []

        if strategy:
            query += " AND strategy LIKE ?"
            params.append(f"%{strategy}%")

        if market:
            query += " AND market = ?"
            params.append(market)

        if trading_pair:
            query += " AND symbol = ?"
            params.append(trading_pair)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp() * 1000))

        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp() * 1000))

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        trades = []
        for row in rows:
            trades.append(TradeRecord(
                timestamp=row[0],
                market=row[1],
                symbol=row[2],
                base_asset=row[3],
                quote_asset=row[4],
                trade_type=row[5],
                price=self._scale_decimal(row[6]),
                amount=self._scale_decimal(row[7]),
                trade_fee_in_quote=self._scale_decimal(row[8]) if row[8] else Decimal("0"),
                strategy=row[9]
            ))

        return trades

    def calculate_metrics(
        self,
        trades: List[TradeRecord],
        initial_capital: float = 1000.0
    ) -> PerformanceReport:
        """
        Calculate all performance metrics from trade list.

        Args:
            trades: List of TradeRecord objects
            initial_capital: Starting capital for return calculations

        Returns:
            PerformanceReport with all metrics
        """
        if not trades:
            return self._empty_report()

        # Basic stats
        total_trades = len(trades)
        buys = [t for t in trades if t.trade_type == "BUY"]
        sells = [t for t in trades if t.trade_type == "SELL"]

        total_buys = len(buys)
        total_sells = len(sells)

        total_buy_volume = float(sum(t.amount for t in buys))
        total_sell_volume = float(sum(t.amount for t in sells))
        total_buy_quote = float(sum(t.amount * t.price for t in buys))
        total_sell_quote = float(sum(t.amount * t.price for t in sells))
        total_fees = float(sum(t.trade_fee_in_quote or Decimal("0") for t in trades))

        # Average prices
        avg_buy_price = total_buy_quote / total_buy_volume if total_buy_volume > 0 else 0
        avg_sell_price = total_sell_quote / total_sell_volume if total_sell_volume > 0 else 0

        # Realized P&L (matched trades)
        matched_volume = min(total_buy_volume, total_sell_volume)
        realized_pnl = 0.0
        if matched_volume > 0 and avg_buy_price > 0:
            realized_pnl = matched_volume * (avg_sell_price - avg_buy_price) - total_fees

        # Unrealized P&L (remaining inventory valued at last price)
        inventory_diff = total_buy_volume - total_sell_volume
        last_price = float(trades[-1].price) if trades else 0
        unrealized_pnl = inventory_diff * last_price if inventory_diff != 0 else 0

        total_pnl = realized_pnl + unrealized_pnl

        # Spread captured
        avg_spread = 0.0
        if avg_buy_price > 0 and avg_sell_price > 0:
            avg_spread = (avg_sell_price - avg_buy_price) / avg_buy_price

        # Daily breakdown
        daily_metrics = self._calculate_daily_metrics(trades, initial_capital)

        # Return metrics
        total_return_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        daily_returns = [d.pnl / initial_capital * 100 for d in daily_metrics] if daily_metrics else []
        avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        best_day = max(d.pnl for d in daily_metrics) if daily_metrics else 0
        worst_day = min(d.pnl for d in daily_metrics) if daily_metrics else 0

        # Risk metrics
        sharpe = self._calculate_sharpe_ratio(daily_returns)
        max_dd, max_dd_duration = self._calculate_max_drawdown(daily_metrics)
        win_rate = self._calculate_win_rate(daily_metrics)
        profit_factor = self._calculate_profit_factor(daily_metrics)

        # Period info
        period_start = datetime.fromtimestamp(trades[0].timestamp / 1000).strftime("%Y-%m-%d")
        period_end = datetime.fromtimestamp(trades[-1].timestamp / 1000).strftime("%Y-%m-%d")

        # Get unique values for filters
        strategies = set(t.strategy for t in trades)
        markets = set(t.market for t in trades)
        pairs = set(t.symbol for t in trades)

        return PerformanceReport(
            generated_at=datetime.now().isoformat(),
            period_start=period_start,
            period_end=period_end,
            strategy=list(strategies)[0] if len(strategies) == 1 else None,
            market=list(markets)[0] if len(markets) == 1 else None,
            trading_pair=list(pairs)[0] if len(pairs) == 1 else None,
            total_trades=total_trades,
            total_buys=total_buys,
            total_sells=total_sells,
            total_buy_volume=round(total_buy_volume, 6),
            total_sell_volume=round(total_sell_volume, 6),
            total_fees_paid=round(total_fees, 4),
            total_pnl=round(total_pnl, 4),
            realized_pnl=round(realized_pnl, 4),
            unrealized_pnl=round(unrealized_pnl, 4),
            avg_trade_pnl=round(realized_pnl / total_trades, 4) if total_trades > 0 else 0,
            total_return_pct=round(total_return_pct, 2),
            avg_daily_return_pct=round(avg_daily_return, 4),
            best_day_pnl=round(best_day, 4),
            worst_day_pnl=round(worst_day, 4),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_dd, 2),
            max_drawdown_duration_days=max_dd_duration,
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            avg_buy_price=round(avg_buy_price, 6),
            avg_sell_price=round(avg_sell_price, 6),
            avg_spread_captured=round(avg_spread * 100, 4),  # As percentage
            avg_trade_size=round((total_buy_volume + total_sell_volume) / total_trades, 6) if total_trades > 0 else 0,
            daily_metrics=daily_metrics
        )

    def _calculate_daily_metrics(
        self,
        trades: List[TradeRecord],
        initial_capital: float
    ) -> List[DailyMetrics]:
        """Group trades by day and calculate daily metrics."""
        if not trades:
            return []

        # Group by date
        daily_trades: Dict[str, List[TradeRecord]] = {}
        for trade in trades:
            date = datetime.fromtimestamp(trade.timestamp / 1000).strftime("%Y-%m-%d")
            if date not in daily_trades:
                daily_trades[date] = []
            daily_trades[date].append(trade)

        # Calculate metrics per day
        daily_metrics = []
        cumulative_pnl = 0.0

        for date in sorted(daily_trades.keys()):
            day_trades = daily_trades[date]

            buys = [t for t in day_trades if t.trade_type == "BUY"]
            sells = [t for t in day_trades if t.trade_type == "SELL"]

            buy_volume = float(sum(t.amount for t in buys))
            sell_volume = float(sum(t.amount for t in sells))
            buy_quote = float(sum(t.amount * t.price for t in buys))
            sell_quote = float(sum(t.amount * t.price for t in sells))
            fees = float(sum(t.trade_fee_in_quote or Decimal("0") for t in day_trades))

            # Simple daily P&L (sell proceeds - buy cost - fees)
            # This is simplified; real P&L would need inventory tracking
            daily_pnl = sell_quote - buy_quote - fees
            cumulative_pnl += daily_pnl

            daily_metrics.append(DailyMetrics(
                date=date,
                pnl=round(daily_pnl, 4),
                cumulative_pnl=round(cumulative_pnl, 4),
                num_trades=len(day_trades),
                buy_volume=round(buy_volume, 6),
                sell_volume=round(sell_volume, 6),
                fees_paid=round(fees, 4),
                inventory_change=round(buy_volume - sell_volume, 6)
            ))

        return daily_metrics

    def _calculate_sharpe_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            daily_returns: List of daily return percentages
            risk_free_rate: Annual risk-free rate (default 0)

        Returns:
            Annualized Sharpe ratio
        """
        if len(daily_returns) < 2:
            return 0.0

        avg_return = sum(daily_returns) / len(daily_returns)

        # Calculate standard deviation
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        if std_dev == 0:
            return 0.0

        # Annualize (assuming 365 trading days for crypto)
        daily_rf = risk_free_rate / 365
        sharpe = (avg_return - daily_rf) / std_dev * math.sqrt(365)

        return sharpe

    def _calculate_max_drawdown(self, daily_metrics: List[DailyMetrics]) -> Tuple[float, int]:
        """
        Calculate maximum drawdown percentage and duration.

        Returns:
            Tuple of (max_drawdown_pct, max_duration_days)
        """
        if not daily_metrics:
            return 0.0, 0

        cumulative = [d.cumulative_pnl for d in daily_metrics]

        peak = cumulative[0]
        max_drawdown = 0.0
        max_duration = 0
        current_duration = 0

        for value in cumulative:
            if value > peak:
                peak = value
                current_duration = 0
            else:
                current_duration += 1
                drawdown = (peak - value) / abs(peak) * 100 if peak != 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                if current_duration > max_duration:
                    max_duration = current_duration

        return max_drawdown, max_duration

    def _calculate_win_rate(self, daily_metrics: List[DailyMetrics]) -> float:
        """Calculate percentage of profitable days."""
        if not daily_metrics:
            return 0.0

        winning_days = sum(1 for d in daily_metrics if d.pnl > 0)
        return winning_days / len(daily_metrics) * 100

    def _calculate_profit_factor(self, daily_metrics: List[DailyMetrics]) -> float:
        """Calculate ratio of gross profit to gross loss."""
        if not daily_metrics:
            return 0.0

        gross_profit = sum(d.pnl for d in daily_metrics if d.pnl > 0)
        gross_loss = abs(sum(d.pnl for d in daily_metrics if d.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def _empty_report(self) -> PerformanceReport:
        """Return empty report when no trades found."""
        return PerformanceReport(
            generated_at=datetime.now().isoformat(),
            period_start="N/A",
            period_end="N/A",
            strategy=None,
            market=None,
            trading_pair=None,
            total_trades=0,
            total_buys=0,
            total_sells=0,
            total_buy_volume=0,
            total_sell_volume=0,
            total_fees_paid=0,
            total_pnl=0,
            realized_pnl=0,
            unrealized_pnl=0,
            avg_trade_pnl=0,
            total_return_pct=0,
            avg_daily_return_pct=0,
            best_day_pnl=0,
            worst_day_pnl=0,
            sharpe_ratio=0,
            max_drawdown_pct=0,
            max_drawdown_duration_days=0,
            win_rate=0,
            profit_factor=0,
            avg_buy_price=0,
            avg_sell_price=0,
            avg_spread_captured=0,
            avg_trade_size=0,
            daily_metrics=[]
        )

    def generate_report(
        self,
        strategy: str = None,
        market: str = None,
        trading_pair: str = None,
        days: int = 30,
        initial_capital: float = 1000.0
    ) -> PerformanceReport:
        """
        Generate complete performance report.

        Args:
            strategy: Filter by strategy name
            market: Filter by exchange
            trading_pair: Filter by trading pair
            days: Number of days to analyze
            initial_capital: Starting capital for calculations

        Returns:
            PerformanceReport object
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        trades = self.get_trades(
            strategy=strategy,
            market=market,
            trading_pair=trading_pair,
            start_time=start_time,
            end_time=end_time
        )

        return self.calculate_metrics(trades, initial_capital)

    def to_json(self, report: PerformanceReport) -> str:
        """Convert report to JSON string."""
        data = asdict(report)
        return json.dumps(data, indent=2, default=str)

    def to_markdown(self, report: PerformanceReport) -> str:
        """Convert report to Markdown string."""
        lines = [
            "# Trading Performance Report",
            "",
            f"**Generated:** {report.generated_at}",
            f"**Period:** {report.period_start} to {report.period_end}",
            "",
        ]

        if report.strategy:
            lines.append(f"**Strategy:** {report.strategy}")
        if report.market:
            lines.append(f"**Exchange:** {report.market}")
        if report.trading_pair:
            lines.append(f"**Trading Pair:** {report.trading_pair}")

        lines.extend([
            "",
            "---",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Trades | {report.total_trades} |",
            f"| Buys / Sells | {report.total_buys} / {report.total_sells} |",
            f"| Total Fees Paid | ${report.total_fees_paid:.4f} |",
            "",
            "---",
            "",
            "## P&L Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| **Total P&L** | **${report.total_pnl:.4f}** |",
            f"| Realized P&L | ${report.realized_pnl:.4f} |",
            f"| Unrealized P&L | ${report.unrealized_pnl:.4f} |",
            f"| Avg Trade P&L | ${report.avg_trade_pnl:.4f} |",
            "",
            "---",
            "",
            "## Return Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Return | {report.total_return_pct:.2f}% |",
            f"| Avg Daily Return | {report.avg_daily_return_pct:.4f}% |",
            f"| Best Day | ${report.best_day_pnl:.4f} |",
            f"| Worst Day | ${report.worst_day_pnl:.4f} |",
            "",
            "---",
            "",
            "## Risk Metrics",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|",
            f"| Sharpe Ratio | {report.sharpe_ratio:.2f} | {'✅ Good' if report.sharpe_ratio > 1.5 else '⚠️ Moderate' if report.sharpe_ratio > 0.5 else '❌ Low'} |",
            f"| Max Drawdown | {report.max_drawdown_pct:.2f}% | {'✅ Low' if report.max_drawdown_pct < 10 else '⚠️ Moderate' if report.max_drawdown_pct < 20 else '❌ High'} |",
            f"| Max DD Duration | {report.max_drawdown_duration_days} days | |",
            f"| Win Rate | {report.win_rate:.2f}% | {'✅ Good' if report.win_rate > 50 else '⚠️ Low'} |",
            f"| Profit Factor | {report.profit_factor:.2f} | {'✅ Good' if report.profit_factor > 1.5 else '⚠️ Moderate' if report.profit_factor > 1 else '❌ Losing'} |",
            "",
            "---",
            "",
            "## Trade Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Buy Volume | {report.total_buy_volume:.6f} |",
            f"| Sell Volume | {report.total_sell_volume:.6f} |",
            f"| Avg Buy Price | ${report.avg_buy_price:.6f} |",
            f"| Avg Sell Price | ${report.avg_sell_price:.6f} |",
            f"| Avg Spread Captured | {report.avg_spread_captured:.4f}% |",
            f"| Avg Trade Size | {report.avg_trade_size:.6f} |",
            "",
        ])

        # Daily breakdown table (last 10 days)
        if report.daily_metrics:
            lines.extend([
                "---",
                "",
                "## Daily Breakdown (Last 10 Days)",
                "",
                "| Date | P&L | Cumulative | Trades | Buy Vol | Sell Vol |",
                "|------|-----|------------|--------|---------|----------|",
            ])

            for dm in report.daily_metrics[-10:]:
                pnl_icon = "🟢" if dm.pnl > 0 else "🔴" if dm.pnl < 0 else "⚪"
                lines.append(
                    f"| {dm.date} | {pnl_icon} ${dm.pnl:.2f} | ${dm.cumulative_pnl:.2f} | "
                    f"{dm.num_trades} | {dm.buy_volume:.4f} | {dm.sell_volume:.4f} |"
                )

            lines.append("")

        # Investor-ready summary
        lines.extend([
            "---",
            "",
            "## Investor Summary",
            "",
            "```",
            f"Period:           {report.period_start} to {report.period_end}",
            f"Total Return:     {report.total_return_pct:.2f}%",
            f"Sharpe Ratio:     {report.sharpe_ratio:.2f}",
            f"Max Drawdown:     {report.max_drawdown_pct:.2f}%",
            f"Win Rate:         {report.win_rate:.2f}%",
            f"Total Trades:     {report.total_trades}",
            "```",
            "",
        ])

        return "\n".join(lines)

    def save_report(
        self,
        report: PerformanceReport,
        output_dir: str = None,
        format: str = "both"
    ) -> List[str]:
        """
        Save report to files.

        Args:
            report: PerformanceReport to save
            output_dir: Directory to save files (default: data/reports)
            format: "json", "markdown", or "both"

        Returns:
            List of saved file paths
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "reports"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        if format in ("json", "both"):
            json_path = output_dir / f"performance_{timestamp}.json"
            with open(json_path, "w") as f:
                f.write(self.to_json(report))
            saved_files.append(str(json_path))

        if format in ("markdown", "both"):
            md_path = output_dir / f"performance_{timestamp}.md"
            with open(md_path, "w") as f:
                f.write(self.to_markdown(report))
            saved_files.append(str(md_path))

        return saved_files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate performance reports from Hummingbot trade history"
    )
    parser.add_argument("--strategy", type=str, help="Filter by strategy name")
    parser.add_argument("--market", type=str, help="Filter by exchange")
    parser.add_argument("--pair", type=str, help="Filter by trading pair")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze (default: 30)")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital (default: 1000)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both")
    parser.add_argument("--db", type=str, help="Path to database file")
    parser.add_argument("--print", action="store_true", help="Print report to console")

    args = parser.parse_args()

    try:
        tracker = PerformanceTracker(db_path=args.db)

        print(f"Analyzing last {args.days} days of trades...")

        report = tracker.generate_report(
            strategy=args.strategy,
            market=args.market,
            trading_pair=args.pair,
            days=args.days,
            initial_capital=args.capital
        )

        if report.total_trades == 0:
            print("No trades found for the specified criteria.")
            return

        print(f"Found {report.total_trades} trades")

        if getattr(args, 'print', False):
            print("\n" + tracker.to_markdown(report))

        saved = tracker.save_report(report, args.output, args.format)

        for path in saved:
            print(f"Saved: {path}")

        # Quick summary
        print(f"\n=== Quick Summary ===")
        print(f"Total P&L: ${report.total_pnl:.4f}")
        print(f"Return: {report.total_return_pct:.2f}%")
        print(f"Sharpe Ratio: {report.sharpe_ratio:.2f}")
        print(f"Win Rate: {report.win_rate:.2f}%")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure Hummingbot has run at least once to create the database.")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
