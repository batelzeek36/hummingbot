"""
Momentum Strategy Package for Hyperliquid Monster Bot v2.11

This package contains the modularized momentum trading strategy:
- strategy.py: Main MomentumStrategy class
- signals.py: Signal evaluation functions
- exits.py: Exit condition checking
- positions.py: Position management
- godmode_filters.py: GOD MODE entry filters (MTF, LIQ, CVD)
- indicators.py: Indicator calculation helpers

Re-exports MomentumStrategy for backward compatibility.
"""

from .strategy import MomentumStrategy

__all__ = ["MomentumStrategy"]
