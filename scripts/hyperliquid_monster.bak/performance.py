"""
Performance tracking for Hyperliquid Monster Bot v2.

Tracks per-coin funding performance and generates reports.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Optional

from .volatility import COIN_VOLATILITY, VOLATILITY_LEVERAGE, CoinVolatility


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


class PerformanceTracker:
    """Manages performance tracking and reporting."""

    def __init__(self, data_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize performance tracker.

        Args:
            data_dir: Directory for storing performance data
            logger: Logger instance for output
        """
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "performance"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.coin_performance: Dict[str, CoinPerformance] = {}
        self.last_save_time = datetime.now()
        self.save_interval_seconds = 3600  # Save every hour

    def load_data(self):
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
                self.logger.info(f"Loaded performance data for {len(self.coin_performance)} coins")
            except Exception as e:
                self.logger.warning(f"Could not load performance data: {e}")

    def save_data(self, start_time: datetime, global_pnl: Decimal):
        """
        Save performance data to both JSON and Markdown files.

        Args:
            start_time: Bot start time for runtime calculation
            global_pnl: Current global P&L
        """
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
            "session_start": start_time.isoformat(),
            "runtime_hours": (datetime.now() - start_time).total_seconds() / 3600,
            "summary": {
                "total_funding_received": round(total_funding, 6),
                "total_payments": total_payments,
                "coins_tracked": len(coins_data),
                "global_pnl": float(global_pnl),
            },
            "coins": dict(sorted_coins),
        }

        json_path = self.data_dir / "coin_performance.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # === SAVE MARKDOWN (Human-readable / Investor report) ===
        self._save_markdown_report(start_time, global_pnl, sorted_coins, total_funding, total_payments)

        self.logger.info(f"Performance data saved to {self.data_dir}")

    def _save_markdown_report(
        self,
        start_time: datetime,
        global_pnl: Decimal,
        sorted_coins: list,
        total_funding: float,
        total_payments: int
    ):
        """Generate and save markdown report."""
        runtime = datetime.now() - start_time
        hours = runtime.total_seconds() / 3600

        md_lines = [
            "# Coin Performance Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Session Start:** {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
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
            f"| Coins Tracked | {len(sorted_coins)} |",
            f"| Global P&L | ${float(global_pnl):.4f} |",
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
            f"*Report generated by Hyperliquid Monster Bot v2.3*",
        ])

        md_path = self.data_dir / "coin_performance.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

    def get_coin_performance(self, symbol: str) -> CoinPerformance:
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

    def check_save_interval(self, start_time: datetime, global_pnl: Decimal):
        """Check if it's time to save performance data."""
        now = datetime.now()
        if (now - self.last_save_time).total_seconds() >= self.save_interval_seconds:
            self.save_data(start_time, global_pnl)
            self.last_save_time = now
