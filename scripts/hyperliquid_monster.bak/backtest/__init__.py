"""
Backtest Module - GOD MODE Phase 5

Historical backtesting framework for validating strategies
before risking real capital.

Features:
- Historical data fetching from Hyperliquid API
- OHLCV, OI, Funding Rate, Premium data
- Strategy replay engine
- Performance metrics (Sharpe, Max Drawdown, etc.)

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

from .models import (
    Candle,
    HistoricalDataPoint,
    BacktestTrade,
    BacktestResult,
    BacktestConfig,
)
from .data_fetcher import (
    HyperliquidHistoricalData,
    DataCache,
)
from .engine import BacktestEngine
from .metrics import BacktestMetrics

__all__ = [
    # Models
    "Candle",
    "HistoricalDataPoint",
    "BacktestTrade",
    "BacktestResult",
    "BacktestConfig",
    # Data
    "HyperliquidHistoricalData",
    "DataCache",
    # Engine
    "BacktestEngine",
    # Metrics
    "BacktestMetrics",
]
