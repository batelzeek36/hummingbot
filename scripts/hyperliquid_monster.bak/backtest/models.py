"""
Data models for backtest module.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Optional extended data
    trades: Optional[int] = None
    vwap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trades": self.trades,
            "vwap": self.vwap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candle":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            trades=data.get("trades"),
            vwap=data.get("vwap"),
        )


@dataclass
class HistoricalDataPoint:
    """
    Single point in time with all available data.

    This combines OHLCV with leading indicator data for backtesting.
    """
    timestamp: datetime
    candle: Candle

    # Leading indicators (from Hyperliquid)
    open_interest: Optional[float] = None
    oi_change_pct: Optional[float] = None
    funding_rate: Optional[float] = None
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    premium_pct: Optional[float] = None

    # Coinglass data (if available)
    cvd: Optional[float] = None
    long_short_ratio: Optional[float] = None
    liquidations_long: Optional[float] = None
    liquidations_short: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "candle": self.candle.to_dict(),
            "open_interest": self.open_interest,
            "oi_change_pct": self.oi_change_pct,
            "funding_rate": self.funding_rate,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "premium_pct": self.premium_pct,
            "cvd": self.cvd,
            "long_short_ratio": self.long_short_ratio,
            "liquidations_long": self.liquidations_long,
            "liquidations_short": self.liquidations_short,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalDataPoint":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            candle=Candle.from_dict(data["candle"]),
            open_interest=data.get("open_interest"),
            oi_change_pct=data.get("oi_change_pct"),
            funding_rate=data.get("funding_rate"),
            mark_price=data.get("mark_price"),
            index_price=data.get("index_price"),
            premium_pct=data.get("premium_pct"),
            cvd=data.get("cvd"),
            long_short_ratio=data.get("long_short_ratio"),
            liquidations_long=data.get("liquidations_long"),
            liquidations_short=data.get("liquidations_short"),
        )


class TradeDirection(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class TradeExitReason(Enum):
    """Reason for trade exit."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    SIGNAL_EXIT = "signal_exit"
    END_OF_DATA = "end_of_data"
    MANUAL = "manual"


@dataclass
class BacktestTrade:
    """Single trade in a backtest."""
    trade_id: int
    symbol: str
    direction: TradeDirection

    # Entry
    entry_time: datetime
    entry_price: float
    entry_reason: str

    # Position sizing
    position_size_usd: float
    leverage: float
    notional_value: float  # position_size * leverage

    # Exit (filled after close)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[TradeExitReason] = None

    # Results
    pnl_pct: Optional[float] = None
    pnl_usd: Optional[float] = None
    fees_usd: float = 0.0

    # High water mark tracking
    max_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # Signals at entry (for attribution)
    entry_signals: List[str] = field(default_factory=list)

    def complete_exit(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: TradeExitReason,
        fee_rate: float = 0.0005,  # 0.05% per trade
    ):
        """Complete the trade with exit info."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate PnL
        if self.direction == TradeDirection.LONG:
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100

        # Calculate USD PnL (leveraged)
        self.pnl_usd = self.notional_value * (self.pnl_pct / 100)

        # Subtract fees (entry + exit)
        self.fees_usd = self.notional_value * fee_rate * 2
        self.pnl_usd -= self.fees_usd

    @property
    def is_winner(self) -> bool:
        return self.pnl_usd is not None and self.pnl_usd > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "entry_reason": self.entry_reason,
            "position_size_usd": self.position_size_usd,
            "leverage": self.leverage,
            "notional_value": self.notional_value,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "pnl_pct": self.pnl_pct,
            "pnl_usd": self.pnl_usd,
            "fees_usd": self.fees_usd,
            "max_profit_pct": self.max_profit_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "entry_signals": self.entry_signals,
        }


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    # Data range
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: str = "1m"  # 1m, 5m, 15m, 1h, etc.

    # Capital
    initial_capital: float = 1000.0
    position_size_pct: float = 10.0  # % of capital per trade
    max_positions: int = 1

    # Risk management
    leverage: float = 5.0
    take_profit_pct: float = 2.5
    stop_loss_pct: float = 1.5
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 1.0

    # Fees
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%
    funding_rate_interval_hours: int = 8

    # Strategy settings (use defaults from main config)
    use_holy_grail: bool = True
    use_mtf: bool = True
    use_cvd: bool = True
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    min_signals: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "timeframe": self.timeframe,
            "initial_capital": self.initial_capital,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "leverage": self.leverage,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_enabled": self.trailing_stop_enabled,
            "trailing_stop_pct": self.trailing_stop_pct,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
        }


@dataclass
class EquityCurvePoint:
    """Single point on equity curve."""
    timestamp: datetime
    equity: float
    unrealized_pnl: float = 0.0
    drawdown_pct: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    config: BacktestConfig

    # Trades
    trades: List[BacktestTrade]
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Returns
    total_return_pct: float
    total_return_usd: float
    final_equity: float

    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_usd: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float

    # Trade metrics
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    avg_trade_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    avg_hold_time_hours: float

    # Equity curve
    equity_curve: List[EquityCurvePoint]

    # Timing
    backtest_duration_seconds: float

    def get_summary(self) -> str:
        """Get human-readable summary."""
        duration_days = (self.config.end_date - self.config.start_date).days

        lines = [
            f"=== BACKTEST RESULTS: {self.config.symbol} ===",
            f"Period: {self.config.start_date.strftime('%Y-%m-%d')} to {self.config.end_date.strftime('%Y-%m-%d')} ({duration_days} days)",
            f"Initial: ${self.config.initial_capital:.2f} | Final: ${self.final_equity:.2f}",
            "",
            "PERFORMANCE:",
            f"  Total Return: {self.total_return_pct:+.2f}% (${self.total_return_usd:+.2f})",
            f"  Max Drawdown: {self.max_drawdown_pct:.2f}% (${self.max_drawdown_usd:.2f})",
            f"  Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio: {self.sortino_ratio:.2f}",
            f"  Calmar Ratio: {self.calmar_ratio:.2f}",
            "",
            "TRADES:",
            f"  Total: {self.total_trades} ({self.winning_trades}W / {self.losing_trades}L)",
            f"  Win Rate: {self.win_rate:.1f}%",
            f"  Profit Factor: {self.profit_factor:.2f}",
            f"  Avg Win: {self.avg_win_pct:+.2f}% | Avg Loss: {self.avg_loss_pct:+.2f}%",
            f"  Best: {self.largest_win_pct:+.2f}% | Worst: {self.largest_loss_pct:+.2f}%",
            f"  Avg Hold Time: {self.avg_hold_time_hours:.1f}h",
        ]

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_return_pct": self.total_return_pct,
            "total_return_usd": self.total_return_usd,
            "final_equity": self.final_equity,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_usd": self.max_drawdown_usd,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "profit_factor": self.profit_factor,
            "win_rate": self.win_rate,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "avg_trade_pct": self.avg_trade_pct,
            "largest_win_pct": self.largest_win_pct,
            "largest_loss_pct": self.largest_loss_pct,
            "avg_hold_time_hours": self.avg_hold_time_hours,
            "backtest_duration_seconds": self.backtest_duration_seconds,
        }
