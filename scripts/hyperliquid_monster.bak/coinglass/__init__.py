"""
Coinglass API Package

GOD MODE Phase 2: Real aggregated market data from Coinglass API.

Provides:
- CVD (Cumulative Volume Delta) - Aggregated across exchanges
- Spot vs Perp CVD Divergence - The killer reversal signal
- Liquidation Heatmap - Know where liquidation clusters sit
- Long/Short Ratios - Market positioning data

API: https://open-api-v4.coinglass.com
Docs: https://docs.coinglass.com
"""

from .api import CoinglassAPI
from .models import (
    CVDDirection,
    CVDSnapshot,
    LiquidationCluster,
    LiquidationHeatmap,
    LongShortRatio,
    SpotPerpDivergence,
)

__all__ = [
    # Main API class
    "CoinglassAPI",
    # Models
    "CVDDirection",
    "CVDSnapshot",
    "SpotPerpDivergence",
    "LiquidationCluster",
    "LiquidationHeatmap",
    "LongShortRatio",
]
