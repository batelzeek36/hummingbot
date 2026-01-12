"""
Hyperliquid Monster Bot v2 - Modular Package

A multi-strategy perpetual trading bot for Hyperliquid with:
- Dynamic funding rate hunting
- Grid trading
- Momentum/RSI trading
- Smart volatility-based leverage

Author: Dollar-A-Day Project
Version: 2.3
"""

from .config import HyperliquidMonsterV2Config
from .models import FundingOpportunity, PositionInfo, StrategyMetrics, StrategyMode
from .performance import CoinPerformance, PerformanceTracker
from .volatility import (
    COIN_VOLATILITY,
    VOLATILITY_LEVERAGE,
    CoinVolatility,
    get_safe_leverage,
)
from .strategies import FundingHunterStrategy, GridStrategy, MomentumStrategy

__all__ = [
    # Config
    "HyperliquidMonsterV2Config",
    # Models
    "FundingOpportunity",
    "PositionInfo",
    "StrategyMetrics",
    "StrategyMode",
    # Performance
    "CoinPerformance",
    "PerformanceTracker",
    # Volatility
    "CoinVolatility",
    "COIN_VOLATILITY",
    "VOLATILITY_LEVERAGE",
    "get_safe_leverage",
    # Strategies
    "FundingHunterStrategy",
    "GridStrategy",
    "MomentumStrategy",
]

__version__ = "2.3"
