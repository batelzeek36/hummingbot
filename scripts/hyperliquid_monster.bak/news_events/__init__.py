"""
News Events Module - GOD MODE Phase 5

Provides economic calendar integration to auto-pause trading
around major market-moving events like FOMC, CPI, NFP, etc.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

from .models import (
    EconomicEvent,
    EventImpact,
    EventType,
    MarketState,
    EventCheckResult,
)
from .calendar import EconomicCalendar
from .detector import NewsEventDetector

__all__ = [
    # Models
    "EconomicEvent",
    "EventImpact",
    "EventType",
    "MarketState",
    "EventCheckResult",
    # Calendar
    "EconomicCalendar",
    # Detector
    "NewsEventDetector",
]
