"""
Economic Calendar - Fetches upcoming market events.

Data sources:
1. Static known events (FOMC dates, etc.)
2. ForexFactory API (free, no key needed)
3. TradingEconomics (optional, paid)

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

import aiohttp

from .models import (
    EconomicEvent,
    EventCalendarDay,
    EventImpact,
    EventType,
)


# Known FOMC meeting dates for 2024-2026
# These are the most market-moving events
FOMC_DATES_2024 = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"
]

FOMC_DATES_2025 = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17"
]

FOMC_DATES_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16"
]

# CPI release dates (typically mid-month)
# These cause massive volatility in crypto
CPI_DATES_2026 = [
    "2026-01-14", "2026-02-12", "2026-03-11", "2026-04-10",
    "2026-05-13", "2026-06-10", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-13", "2026-11-12", "2026-12-10"
]

# NFP (Non-Farm Payrolls) - First Friday of each month
NFP_DATES_2026 = [
    "2026-01-02", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-02", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04"
]


class EconomicCalendar:
    """
    Fetches and manages economic calendar events.

    Primary source is static known events (FOMC, CPI, NFP).
    Optional integration with ForexFactory for more events.
    """

    FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        cache_duration_hours: int = 6,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.cache_duration = timedelta(hours=cache_duration_hours)

        # Event cache
        self._events: Dict[str, EconomicEvent] = {}
        self._last_fetch: Optional[datetime] = None
        self._calendar_days: Dict[str, EventCalendarDay] = {}

        # Load static events
        self._load_static_events()

    def _load_static_events(self):
        """Load known static events (FOMC, CPI, NFP)."""
        self.logger.info("NEWS CALENDAR: Loading static economic events...")

        # FOMC meetings - CRITICAL impact
        for date_str in FOMC_DATES_2024 + FOMC_DATES_2025 + FOMC_DATES_2026:
            try:
                # FOMC announcements are at 2:00 PM ET (18:00 or 19:00 UTC depending on DST)
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=19, minute=0)
                event = EconomicEvent(
                    name="FOMC Interest Rate Decision",
                    event_type=EventType.FOMC,
                    impact=EventImpact.CRITICAL,
                    timestamp=dt,
                    currency="USD",
                    pause_before_minutes=60,  # 1 hour before
                    pause_after_minutes=30,   # 30 min after (volatility settles)
                    block_new_entries=True,
                    close_existing=False,
                    source="static",
                )
                key = f"fomc_{date_str}"
                self._events[key] = event
            except Exception as e:
                self.logger.warning(f"Failed to parse FOMC date {date_str}: {e}")

        # CPI releases - CRITICAL impact
        for date_str in CPI_DATES_2026:
            try:
                # CPI releases at 8:30 AM ET (12:30 or 13:30 UTC)
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=13, minute=30)
                event = EconomicEvent(
                    name="US CPI (Inflation)",
                    event_type=EventType.CPI,
                    impact=EventImpact.CRITICAL,
                    timestamp=dt,
                    currency="USD",
                    pause_before_minutes=30,
                    pause_after_minutes=20,
                    block_new_entries=True,
                    close_existing=False,
                    source="static",
                )
                key = f"cpi_{date_str}"
                self._events[key] = event
            except Exception as e:
                self.logger.warning(f"Failed to parse CPI date {date_str}: {e}")

        # NFP releases - HIGH impact
        for date_str in NFP_DATES_2026:
            try:
                # NFP releases at 8:30 AM ET
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=13, minute=30)
                event = EconomicEvent(
                    name="US Non-Farm Payrolls",
                    event_type=EventType.NFP,
                    impact=EventImpact.HIGH,
                    timestamp=dt,
                    currency="USD",
                    pause_before_minutes=20,
                    pause_after_minutes=15,
                    block_new_entries=True,
                    close_existing=False,
                    source="static",
                )
                key = f"nfp_{date_str}"
                self._events[key] = event
            except Exception as e:
                self.logger.warning(f"Failed to parse NFP date {date_str}: {e}")

        self.logger.info(f"NEWS CALENDAR: Loaded {len(self._events)} static events")

    async def fetch_forex_factory(self) -> List[EconomicEvent]:
        """
        Fetch events from ForexFactory (free, no API key needed).

        Returns list of events for the current week.
        """
        events = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.FOREX_FACTORY_URL,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"ForexFactory returned status {response.status}")
                        return events

                    data = await response.json()

                    for item in data:
                        try:
                            event = self._parse_forex_factory_event(item)
                            if event and event.currency == "USD":
                                events.append(event)
                        except Exception as e:
                            self.logger.debug(f"Failed to parse FF event: {e}")

                    self.logger.info(f"NEWS CALENDAR: Fetched {len(events)} USD events from ForexFactory")

        except aiohttp.ClientError as e:
            self.logger.warning(f"Failed to fetch ForexFactory calendar: {e}")
        except Exception as e:
            self.logger.warning(f"Unexpected error fetching calendar: {e}")

        return events

    def _parse_forex_factory_event(self, item: dict) -> Optional[EconomicEvent]:
        """Parse a ForexFactory calendar item."""
        # ForexFactory JSON format:
        # {"title": "...", "country": "USD", "date": "2024-01-15", "time": "8:30am", "impact": "High", ...}

        country = item.get("country", "")
        if country != "USD":
            return None

        title = item.get("title", "Unknown Event")
        date_str = item.get("date", "")
        time_str = item.get("time", "")
        impact_str = item.get("impact", "Low")

        # Parse impact
        impact_map = {
            "Low": EventImpact.LOW,
            "Medium": EventImpact.MEDIUM,
            "High": EventImpact.HIGH,
        }
        impact = impact_map.get(impact_str, EventImpact.LOW)

        # Parse datetime
        try:
            # Handle "All Day" events
            if time_str.lower() in ("all day", "tentative", ""):
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=12, minute=0)
            else:
                # Parse time like "8:30am" or "2:00pm"
                time_str = time_str.lower().replace(" ", "")
                dt_str = f"{date_str} {time_str}"
                dt = datetime.strptime(dt_str, "%Y-%m-%d %I:%M%p")
        except Exception:
            return None

        # Determine event type
        event_type = EventType.OTHER
        title_lower = title.lower()
        if "fomc" in title_lower or "fed" in title_lower:
            event_type = EventType.FOMC
            impact = EventImpact.CRITICAL
        elif "cpi" in title_lower or "inflation" in title_lower:
            event_type = EventType.CPI
            impact = EventImpact.CRITICAL
        elif "non-farm" in title_lower or "nfp" in title_lower:
            event_type = EventType.NFP
            impact = EventImpact.HIGH
        elif "ppi" in title_lower:
            event_type = EventType.PPI
        elif "gdp" in title_lower:
            event_type = EventType.GDP
        elif "retail" in title_lower:
            event_type = EventType.RETAIL_SALES
        elif "unemployment" in title_lower or "jobless" in title_lower:
            event_type = EventType.UNEMPLOYMENT
        elif "pmi" in title_lower:
            event_type = EventType.PMI

        # Set pause durations based on impact
        if impact == EventImpact.CRITICAL:
            pause_before = 30
            pause_after = 20
        elif impact == EventImpact.HIGH:
            pause_before = 15
            pause_after = 10
        else:
            pause_before = 5
            pause_after = 5

        return EconomicEvent(
            name=title,
            event_type=event_type,
            impact=impact,
            timestamp=dt,
            currency=country,
            forecast=item.get("forecast"),
            previous=item.get("previous"),
            pause_before_minutes=pause_before,
            pause_after_minutes=pause_after,
            block_new_entries=(impact in (EventImpact.HIGH, EventImpact.CRITICAL)),
            source="forex_factory",
            source_url=self.FOREX_FACTORY_URL,
        )

    def add_custom_event(
        self,
        name: str,
        timestamp: datetime,
        impact: EventImpact = EventImpact.HIGH,
        event_type: EventType = EventType.CUSTOM,
        pause_before: int = 30,
        pause_after: int = 15,
    ) -> EconomicEvent:
        """
        Add a custom event (e.g., known token unlock, ETH upgrade).

        Args:
            name: Event name
            timestamp: When event occurs (UTC)
            impact: Event impact level
            event_type: Type of event
            pause_before: Minutes to pause before event
            pause_after: Minutes to pause after event

        Returns:
            Created event
        """
        event = EconomicEvent(
            name=name,
            event_type=event_type,
            impact=impact,
            timestamp=timestamp,
            pause_before_minutes=pause_before,
            pause_after_minutes=pause_after,
            block_new_entries=True,
            source="custom",
        )

        key = f"custom_{timestamp.isoformat()}_{name}"
        self._events[key] = event
        self.logger.info(f"NEWS CALENDAR: Added custom event '{name}' at {timestamp}")

        return event

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
        min_impact: EventImpact = EventImpact.MEDIUM,
    ) -> List[EconomicEvent]:
        """
        Get all events in a time range.

        Args:
            start: Start of range (UTC)
            end: End of range (UTC)
            min_impact: Minimum impact level to include

        Returns:
            List of events in range, sorted by timestamp
        """
        impact_order = {
            EventImpact.LOW: 0,
            EventImpact.MEDIUM: 1,
            EventImpact.HIGH: 2,
            EventImpact.CRITICAL: 3,
        }
        min_impact_val = impact_order.get(min_impact, 0)

        events = []
        for event in self._events.values():
            if start <= event.timestamp <= end:
                if impact_order.get(event.impact, 0) >= min_impact_val:
                    events.append(event)

        return sorted(events, key=lambda e: e.timestamp)

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        min_impact: EventImpact = EventImpact.MEDIUM,
    ) -> List[EconomicEvent]:
        """Get events in the next N hours."""
        now = datetime.utcnow()
        end = now + timedelta(hours=hours_ahead)
        return self.get_events_in_range(now, end, min_impact)

    def get_active_events(self) -> List[EconomicEvent]:
        """Get events that are currently in their pause window."""
        now = datetime.utcnow()
        return [e for e in self._events.values() if e.is_active(now)]

    def get_next_event(
        self,
        min_impact: EventImpact = EventImpact.HIGH,
    ) -> Optional[EconomicEvent]:
        """Get the next upcoming event of at least the specified impact."""
        upcoming = self.get_upcoming_events(hours_ahead=168, min_impact=min_impact)  # 1 week
        return upcoming[0] if upcoming else None

    def get_today_events(
        self,
        min_impact: EventImpact = EventImpact.MEDIUM,
    ) -> List[EconomicEvent]:
        """Get all events for today."""
        now = datetime.utcnow()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return self.get_events_in_range(start, end, min_impact)

    def get_status(self) -> dict:
        """Get calendar status for display."""
        now = datetime.utcnow()
        active = self.get_active_events()
        upcoming_24h = self.get_upcoming_events(24, EventImpact.HIGH)
        next_event = self.get_next_event(EventImpact.HIGH)

        return {
            "total_events": len(self._events),
            "active_events": len(active),
            "active_event_names": [e.name for e in active],
            "upcoming_24h": len(upcoming_24h),
            "next_event": next_event.name if next_event else None,
            "next_event_time": next_event.timestamp.isoformat() if next_event else None,
            "next_event_hours": (
                next_event.time_until(now) / 3600 if next_event else None
            ),
        }
