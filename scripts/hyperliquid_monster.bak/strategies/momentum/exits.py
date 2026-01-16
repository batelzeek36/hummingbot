"""
Exit logic for Momentum Strategy.

Contains functions to check exit conditions (take profit, stop loss, trend reversal).
"""

from decimal import Decimal
from typing import Optional, Tuple, TYPE_CHECKING

from ...indicators import TrendInfo

if TYPE_CHECKING:
    from ...config import HyperliquidMonsterV2Config


def check_exit(
    position: str,
    entry_price: Decimal,
    current_price: Decimal,
    trend: Optional[TrendInfo],
    config: "HyperliquidMonsterV2Config",
) -> Tuple[bool, str]:
    """
    Check if momentum position should exit.

    Includes standard TP/SL plus trend reversal exit.

    Args:
        position: Current position ("long" or "short")
        entry_price: Position entry price
        current_price: Current market price
        trend: Current trend analysis
        config: Bot configuration

    Returns:
        Tuple of (should_exit, reason)
    """
    if position == "long":
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price

    # Take profit
    if pnl_pct >= config.momentum_take_profit:
        return True, "take_profit"

    # Stop loss
    if pnl_pct <= -config.momentum_stop_loss:
        return True, "stop_loss"

    # Trend reversal exit (optional enhancement)
    if config.use_trend_filter and trend:
        if position == "long" and trend.direction == "downtrend":
            # Exit long if trend turns down
            if pnl_pct > 0:  # Only if in profit
                return True, "trend_reversal"
        elif position == "short" and trend.direction == "uptrend":
            # Exit short if trend turns up
            if pnl_pct > 0:
                return True, "trend_reversal"

    return False, ""


def calculate_pnl_pct(
    position: str,
    entry_price: Decimal,
    current_price: Decimal,
) -> Decimal:
    """
    Calculate the P&L percentage for a position.

    Args:
        position: Current position ("long" or "short")
        entry_price: Position entry price
        current_price: Current market price

    Returns:
        P&L percentage as Decimal
    """
    if position == "long":
        return (current_price - entry_price) / entry_price
    else:
        return (entry_price - current_price) / entry_price
