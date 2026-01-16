"""
News Event Detector - Determines if trading should be paused.

This is the main integration point for strategies.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .models import (
    EconomicEvent,
    EventCheckResult,
    EventImpact,
    MarketState,
)
from .calendar import EconomicCalendar


class NewsEventDetector:
    """
    Detects news events and recommends trading pauses.

    Usage:
        detector = NewsEventDetector()

        # In strategy tick:
        result = detector.check_trading_allowed()
        if not result.should_pause:
            # Safe to trade
        else:
            logger.warning(f"Trading paused: {result.reason}")
    """

    def __init__(
        self,
        calendar: Optional[EconomicCalendar] = None,
        logger: Optional[logging.Logger] = None,
        # Config options
        pause_on_high_impact: bool = True,
        pause_on_critical_only: bool = False,
        close_positions_on_critical: bool = False,
        warning_minutes_ahead: int = 120,  # 2 hours
    ):
        """
        Initialize News Event Detector.

        Args:
            calendar: Economic calendar instance (created if None)
            logger: Logger instance
            pause_on_high_impact: Pause for HIGH impact events
            pause_on_critical_only: Only pause for CRITICAL events
            close_positions_on_critical: Close positions before CRITICAL events
            warning_minutes_ahead: How far ahead to warn about events
        """
        self.calendar = calendar or EconomicCalendar(logger=logger)
        self.logger = logger or logging.getLogger(__name__)

        self.pause_on_high_impact = pause_on_high_impact
        self.pause_on_critical_only = pause_on_critical_only
        self.close_positions_on_critical = close_positions_on_critical
        self.warning_minutes_ahead = warning_minutes_ahead

        # Track state
        self._last_check: Optional[datetime] = None
        self._last_result: Optional[EventCheckResult] = None
        self._paused_since: Optional[datetime] = None

    def check_trading_allowed(
        self,
        current_time: Optional[datetime] = None,
    ) -> EventCheckResult:
        """
        Check if trading should be allowed right now.

        Args:
            current_time: Current time (UTC), defaults to now

        Returns:
            EventCheckResult with pause recommendation and details
        """
        current_time = current_time or datetime.utcnow()
        self._last_check = current_time

        # Get active events (in their pause window)
        active_events = self.calendar.get_active_events()

        # Filter by impact preference
        if self.pause_on_critical_only:
            active_events = [e for e in active_events if e.impact == EventImpact.CRITICAL]
        elif self.pause_on_high_impact:
            active_events = [e for e in active_events if e.impact in (EventImpact.HIGH, EventImpact.CRITICAL)]

        # Get upcoming events for warning
        upcoming_events = self.calendar.get_upcoming_events(
            hours_ahead=self.warning_minutes_ahead / 60,
            min_impact=EventImpact.HIGH,
        )
        # Filter out currently active events from upcoming
        active_ids = {id(e) for e in active_events}
        upcoming_events = [e for e in upcoming_events if id(e) not in active_ids]

        # Determine if we should pause
        should_pause = len(active_events) > 0

        # Determine market state
        if active_events:
            # Use the highest impact active event's state
            critical_events = [e for e in active_events if e.impact == EventImpact.CRITICAL]
            check_event = critical_events[0] if critical_events else active_events[0]
            market_state = check_event.get_state(current_time)
        else:
            market_state = MarketState.NORMAL

        # Build reason string
        if should_pause:
            event_names = [e.name for e in active_events[:3]]
            reason = f"{', '.join(event_names)}"

            # Calculate when pause ends
            latest_end = max(
                e.timestamp + timedelta(minutes=e.pause_after_minutes)
                for e in active_events
            )
            pause_until = latest_end
        else:
            reason = ""
            pause_until = None

        result = EventCheckResult(
            should_pause=should_pause,
            active_events=active_events,
            upcoming_events=upcoming_events,
            market_state=market_state,
            reason=reason,
            pause_until=pause_until,
        )

        self._last_result = result

        # Log state changes
        if should_pause and self._paused_since is None:
            self._paused_since = current_time
            self.logger.warning(
                f"NEWS EVENT: Pausing trading - {reason} "
                f"(until {pause_until.strftime('%H:%M UTC') if pause_until else 'TBD'})"
            )
        elif not should_pause and self._paused_since is not None:
            pause_duration = (current_time - self._paused_since).total_seconds() / 60
            self.logger.info(f"NEWS EVENT: Resuming trading after {pause_duration:.0f}min pause")
            self._paused_since = None

        return result

    def should_close_positions(
        self,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Check if positions should be closed (CRITICAL events only).

        Args:
            current_time: Current time (UTC)

        Returns:
            Tuple of (should_close, reason)
        """
        if not self.close_positions_on_critical:
            return False, ""

        result = self.check_trading_allowed(current_time)

        critical_events = [
            e for e in result.active_events
            if e.impact == EventImpact.CRITICAL and e.close_existing
        ]

        if critical_events:
            names = [e.name for e in critical_events]
            return True, f"CRITICAL event: {', '.join(names)}"

        return False, ""

    def get_warning_message(self) -> Optional[str]:
        """
        Get warning message for upcoming events.

        Returns:
            Warning message string or None if no warnings
        """
        if not self._last_result:
            self.check_trading_allowed()

        result = self._last_result
        if not result:
            return None

        # Current pause message
        if result.should_pause:
            return f"PAUSED: {result.reason}"

        # Upcoming event warning
        if result.upcoming_events:
            next_event = result.upcoming_events[0]
            minutes_until = next_event.time_until() / 60

            if minutes_until <= 60:
                return (
                    f"WARNING: {next_event.name} in {minutes_until:.0f}min "
                    f"({next_event.impact.value} impact)"
                )

        return None

    def add_crypto_event(
        self,
        name: str,
        timestamp: datetime,
        impact: EventImpact = EventImpact.HIGH,
        pause_before: int = 30,
        pause_after: int = 15,
    ) -> EconomicEvent:
        """
        Add a crypto-specific event (ETH upgrade, token unlock, etc.).

        Args:
            name: Event name
            timestamp: Event time (UTC)
            impact: Impact level
            pause_before: Minutes to pause before
            pause_after: Minutes to pause after

        Returns:
            Created event
        """
        from .models import EventType
        return self.calendar.add_custom_event(
            name=name,
            timestamp=timestamp,
            impact=impact,
            event_type=EventType.CUSTOM,
            pause_before=pause_before,
            pause_after=pause_after,
        )

    def get_status(self) -> dict:
        """Get detector status for display."""
        result = self._last_result
        calendar_status = self.calendar.get_status()

        status = {
            "enabled": True,
            "pause_high_impact": self.pause_on_high_impact,
            "pause_critical_only": self.pause_on_critical_only,
            "currently_paused": result.should_pause if result else False,
            "market_state": result.market_state.value if result else "unknown",
            "paused_since": self._paused_since.isoformat() if self._paused_since else None,
            "warning": self.get_warning_message(),
            **calendar_status,
        }

        return status
