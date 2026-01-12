"""
Analytics module for Hummingbot trading performance tracking.

Usage:
    from scripts.analytics import PerformanceTracker

    tracker = PerformanceTracker()
    report = tracker.generate_report(strategy="dollar_a_day_pmm", days=7)

    # Output formats
    print(tracker.to_markdown(report))
    print(tracker.to_json(report))

    # Save to files
    tracker.save_report(report, format="both")
"""

from .performance_tracker import (
    PerformanceTracker,
    PerformanceReport,
    DailyMetrics,
    TradeRecord
)

__all__ = [
    "PerformanceTracker",
    "PerformanceReport",
    "DailyMetrics",
    "TradeRecord"
]
