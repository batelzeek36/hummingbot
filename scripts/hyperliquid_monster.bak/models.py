"""
Data models for Hyperliquid Monster Bot v2.

Contains dataclasses and enums for tracking positions, metrics, and opportunities.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional


class StrategyMode(Enum):
    """Operating modes for strategies."""
    ACTIVE = "Active"
    PAUSED = "Paused"
    KILLED = "Killed"
    WARMING_UP = "Warming Up"


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
