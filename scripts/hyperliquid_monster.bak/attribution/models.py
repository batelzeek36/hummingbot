"""
Data models for performance attribution module.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalType(Enum):
    """Types of signals that can be tracked."""
    # Core indicators
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    TREND_BULLISH = "trend_bullish"
    TREND_BEARISH = "trend_bearish"
    MACD_BULLISH = "macd_bullish"
    MACD_BEARISH = "macd_bearish"
    VOLUME_SURGE = "volume_surge"

    # Leading indicators
    OI_BULLISH = "oi_bullish"
    OI_BEARISH = "oi_bearish"
    OI_EXHAUSTION = "oi_exhaustion"
    PREMIUM_HIGH = "premium_high"
    PREMIUM_LOW = "premium_low"
    FUNDING_BULLISH = "funding_bullish"
    FUNDING_BEARISH = "funding_bearish"

    # Holy Grail
    HOLY_GRAIL_LONG = "holy_grail_long"
    HOLY_GRAIL_SHORT = "holy_grail_short"
    HOLY_GRAIL_STRONG = "holy_grail_strong"

    # Phase 2: Coinglass
    CVD_BULLISH = "cvd_bullish"
    CVD_BEARISH = "cvd_bearish"
    CVD_DIVERGENCE = "cvd_divergence"
    LIQ_MAGNET_UP = "liq_magnet_up"
    LIQ_MAGNET_DOWN = "liq_magnet_down"
    LS_RATIO_CROWDED_LONG = "ls_ratio_crowded_long"
    LS_RATIO_CROWDED_SHORT = "ls_ratio_crowded_short"

    # Phase 3: MTF
    MTF_CONFLUENCE_BULLISH = "mtf_confluence_bullish"
    MTF_CONFLUENCE_BEARISH = "mtf_confluence_bearish"
    MTF_1H_BULLISH = "mtf_1h_bullish"
    MTF_1H_BEARISH = "mtf_1h_bearish"

    # Phase 4: ML
    ML_HIGH_CONFIDENCE = "ml_high_confidence"
    ML_LOW_CONFIDENCE = "ml_low_confidence"

    # News Events (Phase 5)
    NEWS_EVENT_ACTIVE = "news_event_active"
    NEWS_EVENT_WARNING = "news_event_warning"


@dataclass
class SignalSnapshot:
    """
    Snapshot of all signal states at a point in time.

    This captures which signals were active when a trade was opened.
    """
    timestamp: datetime
    active_signals: List[SignalType]
    signal_values: Dict[str, float] = field(default_factory=dict)

    # Core indicator values
    rsi: Optional[float] = None
    trend_strength: Optional[float] = None
    macd_histogram: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Leading indicator values
    oi_change_pct: Optional[float] = None
    premium_pct: Optional[float] = None
    funding_rate: Optional[float] = None

    # Holy Grail
    holy_grail_confidence: Optional[float] = None
    holy_grail_direction: Optional[str] = None

    # Coinglass
    cvd_direction: Optional[str] = None
    cvd_divergence_strength: Optional[float] = None
    liq_imbalance: Optional[float] = None

    # MTF
    mtf_weighted_score: Optional[float] = None
    mtf_confluence_count: Optional[int] = None

    # ML
    ml_confidence: Optional[float] = None
    ml_win_probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "active_signals": [s.value for s in self.active_signals],
            "signal_values": self.signal_values,
            "rsi": self.rsi,
            "trend_strength": self.trend_strength,
            "macd_histogram": self.macd_histogram,
            "volume_ratio": self.volume_ratio,
            "oi_change_pct": self.oi_change_pct,
            "premium_pct": self.premium_pct,
            "funding_rate": self.funding_rate,
            "holy_grail_confidence": self.holy_grail_confidence,
            "holy_grail_direction": self.holy_grail_direction,
            "cvd_direction": self.cvd_direction,
            "cvd_divergence_strength": self.cvd_divergence_strength,
            "liq_imbalance": self.liq_imbalance,
            "mtf_weighted_score": self.mtf_weighted_score,
            "mtf_confluence_count": self.mtf_confluence_count,
            "ml_confidence": self.ml_confidence,
            "ml_win_probability": self.ml_win_probability,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalSnapshot":
        """Create from dictionary."""
        active_signals = [SignalType(s) for s in data.get("active_signals", [])]
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            active_signals=active_signals,
            signal_values=data.get("signal_values", {}),
            rsi=data.get("rsi"),
            trend_strength=data.get("trend_strength"),
            macd_histogram=data.get("macd_histogram"),
            volume_ratio=data.get("volume_ratio"),
            oi_change_pct=data.get("oi_change_pct"),
            premium_pct=data.get("premium_pct"),
            funding_rate=data.get("funding_rate"),
            holy_grail_confidence=data.get("holy_grail_confidence"),
            holy_grail_direction=data.get("holy_grail_direction"),
            cvd_direction=data.get("cvd_direction"),
            cvd_divergence_strength=data.get("cvd_divergence_strength"),
            liq_imbalance=data.get("liq_imbalance"),
            mtf_weighted_score=data.get("mtf_weighted_score"),
            mtf_confluence_count=data.get("mtf_confluence_count"),
            ml_confidence=data.get("ml_confidence"),
            ml_win_probability=data.get("ml_win_probability"),
        )


@dataclass
class TradeAttribution:
    """
    Complete attribution record for a single trade.
    """
    trade_id: str
    symbol: str
    direction: str  # "long" or "short"

    # Entry info
    entry_time: datetime
    entry_price: float
    entry_signals: SignalSnapshot

    # Exit info (filled after close)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    # Outcome
    pnl_pct: Optional[float] = None
    pnl_usd: Optional[float] = None
    is_win: Optional[bool] = None
    hold_time_seconds: Optional[float] = None

    # Additional context
    leverage: Optional[float] = None
    position_size_usd: Optional[float] = None

    def complete_exit(
        self,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
        pnl_usd: Optional[float] = None,
    ):
        """Fill in exit information."""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate PnL
        if self.direction == "long":
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100

        self.pnl_usd = pnl_usd
        self.is_win = self.pnl_pct > 0
        self.hold_time_seconds = (exit_time - self.entry_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "entry_signals": self.entry_signals.to_dict(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_pct": self.pnl_pct,
            "pnl_usd": self.pnl_usd,
            "is_win": self.is_win,
            "hold_time_seconds": self.hold_time_seconds,
            "leverage": self.leverage,
            "position_size_usd": self.position_size_usd,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeAttribution":
        """Create from dictionary."""
        trade = cls(
            trade_id=data["trade_id"],
            symbol=data["symbol"],
            direction=data["direction"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            entry_price=data["entry_price"],
            entry_signals=SignalSnapshot.from_dict(data["entry_signals"]),
        )
        if data.get("exit_time"):
            trade.exit_time = datetime.fromisoformat(data["exit_time"])
            trade.exit_price = data.get("exit_price")
            trade.exit_reason = data.get("exit_reason")
            trade.pnl_pct = data.get("pnl_pct")
            trade.pnl_usd = data.get("pnl_usd")
            trade.is_win = data.get("is_win")
            trade.hold_time_seconds = data.get("hold_time_seconds")
        trade.leverage = data.get("leverage")
        trade.position_size_usd = data.get("position_size_usd")
        return trade


@dataclass
class SignalPerformance:
    """
    Performance statistics for a single signal type.
    """
    signal_type: SignalType

    # Counts
    total_trades: int = 0
    wins: int = 0
    losses: int = 0

    # PnL
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    max_win_pct: float = 0.0
    max_loss_pct: float = 0.0

    # Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0  # gross profit / gross loss

    # Timing
    avg_hold_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl_pct": self.total_pnl_pct,
            "avg_pnl_pct": self.avg_pnl_pct,
            "max_win_pct": self.max_win_pct,
            "max_loss_pct": self.max_loss_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_hold_time_seconds": self.avg_hold_time_seconds,
        }


@dataclass
class AttributionReport:
    """
    Complete attribution analysis report.
    """
    generated_at: datetime
    period_start: Optional[datetime]
    period_end: Optional[datetime]

    # Overall stats
    total_trades: int
    total_wins: int
    total_losses: int
    overall_win_rate: float
    total_pnl_pct: float
    total_pnl_usd: float

    # Per-signal performance
    signal_performance: List[SignalPerformance]

    # Best/worst signals
    best_signals: List[SignalType]  # By win rate
    worst_signals: List[SignalType]  # By win rate
    most_profitable_signals: List[SignalType]  # By total PnL

    # Recommendations
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "overall_win_rate": self.overall_win_rate,
            "total_pnl_pct": self.total_pnl_pct,
            "total_pnl_usd": self.total_pnl_usd,
            "signal_performance": [sp.to_dict() for sp in self.signal_performance],
            "best_signals": [s.value for s in self.best_signals],
            "worst_signals": [s.value for s in self.worst_signals],
            "most_profitable_signals": [s.value for s in self.most_profitable_signals],
            "recommendations": self.recommendations,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"=== Attribution Report ({self.generated_at.strftime('%Y-%m-%d %H:%M')}) ===",
            f"Total Trades: {self.total_trades} ({self.total_wins}W / {self.total_losses}L)",
            f"Win Rate: {self.overall_win_rate:.1f}%",
            f"Total PnL: {self.total_pnl_pct:+.2f}% (${self.total_pnl_usd:+.2f})",
            "",
            "TOP PERFORMING SIGNALS:",
        ]

        for i, signal in enumerate(self.best_signals[:5], 1):
            perf = next((sp for sp in self.signal_performance if sp.signal_type == signal), None)
            if perf:
                lines.append(f"  {i}. {signal.value}: {perf.win_rate:.1f}% WR, {perf.avg_pnl_pct:+.2f}% avg")

        lines.append("")
        lines.append("WORST PERFORMING SIGNALS:")

        for i, signal in enumerate(self.worst_signals[:5], 1):
            perf = next((sp for sp in self.signal_performance if sp.signal_type == signal), None)
            if perf:
                lines.append(f"  {i}. {signal.value}: {perf.win_rate:.1f}% WR, {perf.avg_pnl_pct:+.2f}% avg")

        if self.recommendations:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)
