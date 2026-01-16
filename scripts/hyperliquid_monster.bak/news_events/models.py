"""
Data models for news events module.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class EventImpact(Enum):
    """Impact level of economic event."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # FOMC, NFP, CPI


class EventType(Enum):
    """Type of economic event."""
    # US Events
    FOMC = "fomc"  # Fed interest rate decision
    NFP = "nfp"  # Non-Farm Payrolls
    CPI = "cpi"  # Consumer Price Index
    PPI = "ppi"  # Producer Price Index
    GDP = "gdp"  # Gross Domestic Product
    RETAIL_SALES = "retail_sales"
    UNEMPLOYMENT = "unemployment"
    PMI = "pmi"  # Purchasing Managers Index
    FED_SPEECH = "fed_speech"  # Fed Chair/Governor speeches

    # Crypto-specific
    ETH_UPGRADE = "eth_upgrade"
    BTC_HALVING = "btc_halving"
    MAJOR_UNLOCK = "major_unlock"  # Token unlocks

    # Other
    CUSTOM = "custom"
    OTHER = "other"


class MarketState(Enum):
    """Current market state relative to events."""
    NORMAL = "normal"  # No events nearby
    PRE_EVENT = "pre_event"  # Event coming soon
    DURING_EVENT = "during_event"  # Event happening now
    POST_EVENT = "post_event"  # Just after event


@dataclass
class EconomicEvent:
    """Single economic event."""
    name: str
    event_type: EventType
    impact: EventImpact
    timestamp: datetime
    currency: str = "USD"  # Primary currency affected
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None

    # Pause settings
    pause_before_minutes: int = 30
    pause_after_minutes: int = 15
    block_new_entries: bool = True
    close_existing: bool = False  # Only for CRITICAL events

    # Source tracking
    source: str = ""
    source_url: str = ""

    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """Check if we're in the event's danger zone."""
        current_time = current_time or datetime.utcnow()

        from datetime import timedelta
        start = self.timestamp - timedelta(minutes=self.pause_before_minutes)
        end = self.timestamp + timedelta(minutes=self.pause_after_minutes)

        return start <= current_time <= end

    def time_until(self, current_time: Optional[datetime] = None) -> float:
        """Get seconds until event starts."""
        current_time = current_time or datetime.utcnow()
        delta = self.timestamp - current_time
        return delta.total_seconds()

    def get_state(self, current_time: Optional[datetime] = None) -> MarketState:
        """Get current market state relative to this event."""
        current_time = current_time or datetime.utcnow()

        from datetime import timedelta

        pre_start = self.timestamp - timedelta(minutes=self.pause_before_minutes)
        event_end = self.timestamp + timedelta(minutes=5)  # Event typically lasts ~5min
        post_end = self.timestamp + timedelta(minutes=self.pause_after_minutes)

        if current_time < pre_start:
            return MarketState.NORMAL
        elif current_time < self.timestamp:
            return MarketState.PRE_EVENT
        elif current_time < event_end:
            return MarketState.DURING_EVENT
        elif current_time < post_end:
            return MarketState.POST_EVENT
        else:
            return MarketState.NORMAL


@dataclass
class EventCalendarDay:
    """All events for a single day."""
    date: datetime
    events: List[EconomicEvent] = field(default_factory=list)

    def get_high_impact_events(self) -> List[EconomicEvent]:
        """Get only HIGH and CRITICAL impact events."""
        return [e for e in self.events if e.impact in (EventImpact.HIGH, EventImpact.CRITICAL)]

    def has_critical_event(self) -> bool:
        """Check if day has any CRITICAL events."""
        return any(e.impact == EventImpact.CRITICAL for e in self.events)


@dataclass
class EventCheckResult:
    """Result of checking for active events."""
    should_pause: bool
    active_events: List[EconomicEvent]
    upcoming_events: List[EconomicEvent]  # Next 2 hours
    market_state: MarketState
    reason: str
    pause_until: Optional[datetime] = None

    def __str__(self) -> str:
        if not self.should_pause:
            return "CLEAR: No events affecting trading"

        event_names = [e.name for e in self.active_events]
        return f"PAUSE: {', '.join(event_names)} - {self.reason}"
