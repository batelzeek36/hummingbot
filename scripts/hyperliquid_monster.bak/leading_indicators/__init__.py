"""
Leading Indicators Package for Hyperliquid Monster Bot v2.

This package provides TRUE leading indicators that predict price moves,
unlike RSI/MACD which only react to past price.

Modules:
- models: Data classes and enums for all indicator types
- oi_analysis: Open Interest momentum analysis
- premium: Premium/discount directional pressure
- funding_velocity: Funding rate acceleration/deceleration
- volume: Volume surge detection
- holy_grail: Combined directional signal from all indicators
- godmode: Spike detection and market regime
- tracker: Main HyperliquidLeadingIndicators class with API access
"""

# Re-export all models for backward compatibility
from .models import (
    OIMomentum,
    OISnapshot,
    OIAnalysis,
    LiquidationCluster,
    PremiumAnalysis,
    FundingVelocity,
    VolumeSurge,
    DirectionalSignal,
    OISpikeResult,
    PremiumSpikeResult,
    MarketRegime,
)

# Re-export main tracker class
from .tracker import HyperliquidLeadingIndicators

__all__ = [
    # Enums
    "OIMomentum",
    # Data classes
    "OISnapshot",
    "OIAnalysis",
    "LiquidationCluster",
    "PremiumAnalysis",
    "FundingVelocity",
    "VolumeSurge",
    "DirectionalSignal",
    "OISpikeResult",
    "PremiumSpikeResult",
    "MarketRegime",
    # Main class
    "HyperliquidLeadingIndicators",
]
