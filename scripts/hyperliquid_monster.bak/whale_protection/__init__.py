"""
Whale Protection System for Hyperliquid Monster Bot v2.

This module provides protection against sudden large price movements ("whale moves")
that can cause significant losses in leveraged trading.

Components:
- CircuitBreaker: Detects rapid price spikes and pauses/flattens positions
- GridProtection: Monitors grid fill rates to detect one-sided accumulation
- TrailingStop: Dynamic stop loss that follows profitable moves
- DynamicRisk: Volatility-scaled stop losses and emergency market orders
- EarlyWarning: Order book imbalance and funding spike detection
- OrderBookVelocity: GOD MODE - Tracks rate of change in order book imbalance (v2.8)
"""

from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .grid_protection import GridProtection, GridProtectionState
from .trailing_stop import TrailingStopManager
from .dynamic_risk import DynamicRiskManager
from .early_warning import EarlyWarningSystem, WarningLevel, OrderBookVelocity
from .orchestrator import WhaleProtectionOrchestrator

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerState",
    "GridProtection",
    "GridProtectionState",
    "TrailingStopManager",
    "DynamicRiskManager",
    "EarlyWarningSystem",
    "WarningLevel",
    "WhaleProtectionOrchestrator",
    # Phase 1 GOD MODE
    "OrderBookVelocity",
]
