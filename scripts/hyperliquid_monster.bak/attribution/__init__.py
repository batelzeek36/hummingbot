"""
Performance Attribution Module - GOD MODE Phase 5

Tracks which signals were active during each trade and analyzes
which signals actually contribute to wins vs losses.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

from .models import (
    SignalSnapshot,
    SignalType,
    TradeAttribution,
    SignalPerformance,
    AttributionReport,
)
from .tracker import AttributionTracker
from .analyzer import AttributionAnalyzer

__all__ = [
    # Models
    "SignalSnapshot",
    "SignalType",
    "TradeAttribution",
    "SignalPerformance",
    "AttributionReport",
    # Tracker
    "AttributionTracker",
    # Analyzer
    "AttributionAnalyzer",
]
