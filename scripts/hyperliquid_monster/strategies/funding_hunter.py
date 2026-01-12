"""
Funding Rate Hunter Strategy for Hyperliquid Monster Bot v2.

Dynamically scans for best funding rate opportunities across all configured pairs
and automatically rotates positions to maximize funding income.
"""

import logging
from decimal import Decimal
from typing import Callable, Dict, List, Optional

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType

from ..config import HyperliquidMonsterV2Config
from ..models import FundingOpportunity, StrategyMetrics, StrategyMode
from ..volatility import COIN_VOLATILITY, CoinVolatility, get_safe_leverage


class FundingHunterStrategy:
    """
    Dynamic Funding Rate Hunter.

    Scans all configured pairs for best funding opportunities,
    auto-rotates to capture highest rates.
    """

    def __init__(
        self,
        config: HyperliquidMonsterV2Config,
        connector: ConnectorBase,
        metrics: StrategyMetrics,
        place_order_fn: Callable,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the funding hunter strategy.

        Args:
            config: Bot configuration
            connector: Exchange connector
            metrics: Strategy metrics tracker
            place_order_fn: Function to place orders (from main bot)
            logger: Logger instance
        """
        self.config = config
        self.connector = connector
        self.metrics = metrics
        self.place_order = place_order_fn
        self.logger = logger or logging.getLogger(__name__)

        # State
        self.opportunities: List[FundingOpportunity] = []
        self.positions: Dict[str, FundingOpportunity] = {}  # pair -> opportunity
        self.last_scan_timestamp = 0

    def run(self, current_timestamp: float):
        """
        Run the funding hunter strategy.

        Args:
            current_timestamp: Current timestamp in seconds
        """
        if self.metrics.status != StrategyMode.ACTIVE:
            return

        # Scan for opportunities periodically
        if self.last_scan_timestamp < current_timestamp - self.config.funding_scan_interval:
            self._scan_opportunities(current_timestamp)
            self.last_scan_timestamp = current_timestamp

        # Manage existing positions
        self._manage_positions(current_timestamp)

        # Open new positions if slots available
        self._open_best_positions(current_timestamp)

    def _scan_opportunities(self, current_timestamp: float):
        """Scan all pairs for funding rate opportunities."""
        opportunities = []

        for pair in self.config.funding_scan_pairs.split(","):
            pair = pair.strip()

            try:
                funding_info = self.connector.get_funding_info(pair)
                if funding_info is None:
                    continue

                rate = float(funding_info.rate)

                # Calculate APR (Hyperliquid = hourly funding = 8760 periods/year)
                apr = abs(rate) * 8760 * 100  # Convert to percentage

                # Determine direction to receive funding
                # Positive rate = shorts pay longs = go LONG
                # Negative rate = longs pay shorts = go SHORT
                direction = "long" if rate > 0 else "short"

                # Calculate score (APR weighted by time to funding)
                next_funding_ts = float(funding_info.next_funding_utc_timestamp)
                time_to_funding = next_funding_ts - current_timestamp
                minutes_to_funding = time_to_funding / 60

                # Higher score for higher APR and closer to funding time
                urgency_bonus = 1.0
                if minutes_to_funding <= self.config.minutes_before_funding:
                    urgency_bonus = 1.5  # Boost score if funding is imminent

                score = apr * urgency_bonus

                opp = FundingOpportunity(
                    pair=pair,
                    rate=rate,
                    apr=apr,
                    direction=direction,
                    next_funding_time=next_funding_ts,
                    score=score
                )

                opportunities.append(opp)

            except Exception as e:
                self.logger.warning(f"Error scanning {pair}: {e}")

        # Sort by score (highest first)
        self.opportunities = sorted(opportunities, key=lambda x: x.score, reverse=True)

        # Log scan results
        if not self.opportunities:
            self.logger.info(f"FUNDING SCAN: No opportunities found from {len(self.config.funding_scan_pairs.split(','))} pairs")
        else:
            top3 = self.opportunities[:3]
            self.logger.info("TOP FUNDING OPPORTUNITIES:")
            for opp in top3:
                minutes = (opp.next_funding_time - current_timestamp) / 60
                self.logger.info(
                    f"  {opp.pair}: {opp.apr:.1f}% APR ({opp.direction.upper()}) "
                    f"- {minutes:.0f}min to funding"
                )

    def _manage_positions(self, current_timestamp: float):
        """Manage existing funding positions - close or rotate."""
        for pair, current_opp in list(self.positions.items()):
            try:
                funding_info = self.connector.get_funding_info(pair)
                if funding_info is None:
                    continue

                time_to_funding = float(funding_info.next_funding_utc_timestamp) - current_timestamp
                minutes_to_funding = time_to_funding / 60

                # Check if funding just happened (close window)
                if minutes_to_funding > 55:
                    self._close_position(pair, "funding_collected", current_timestamp)
                    continue

                # Check if rate flipped direction
                new_rate = float(funding_info.rate)
                new_direction = "long" if new_rate > 0 else "short"
                if new_direction != current_opp.direction:
                    self._close_position(pair, "rate_flipped", current_timestamp)
                    continue

                # Check if better opportunity exists (rotation)
                current_apr = abs(new_rate) * 8760 * 100

                for opp in self.opportunities:
                    if opp.pair not in self.positions:
                        apr_improvement = opp.apr - current_apr
                        if apr_improvement >= float(self.config.funding_rotation_threshold):
                            self.logger.info(
                                f"ROTATING: {pair} ({current_apr:.1f}% APR) -> "
                                f"{opp.pair} ({opp.apr:.1f}% APR)"
                            )
                            self._close_position(pair, "rotation", current_timestamp)
                            break

            except Exception as e:
                self.logger.error(f"Error managing funding position {pair}: {e}")

    def _open_best_positions(self, current_timestamp: float):
        """Open positions on best funding opportunities."""
        # Check if we have slots available
        current_positions = len(self.positions)
        available_slots = self.config.max_funding_positions - current_positions

        if available_slots <= 0:
            return

        for opp in self.opportunities:
            if available_slots <= 0:
                break

            if opp.pair in self.positions:
                continue

            # Check APR threshold
            if opp.apr < float(self.config.min_funding_apr):
                continue

            # Check timing
            time_to_funding = opp.next_funding_time - current_timestamp
            minutes_to_funding = time_to_funding / 60

            if minutes_to_funding > self.config.minutes_before_funding:
                continue

            # Open position
            if self._open_position(opp):
                available_slots -= 1

    def _open_position(self, opp: FundingOpportunity) -> bool:
        """Open a position to collect funding with smart leverage."""
        position_size = self.config.funding_position_size

        # SMART LEVERAGE: Get safe leverage based on coin volatility
        if self.config.use_smart_leverage:
            leverage = get_safe_leverage(opp.pair, self.config.funding_leverage_max)
        else:
            leverage = self.config.funding_leverage_max

        price = self.connector.get_price_by_type(opp.pair, PriceType.MidPrice)
        if price is None:
            return False

        amount = position_size / price
        trade_type = TradeType.BUY if opp.direction == "long" else TradeType.SELL

        # Get volatility class for logging
        vol_class = COIN_VOLATILITY.get(opp.pair, CoinVolatility.HIGH).value.upper()

        order_id = self.place_order(
            pair=opp.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.OPEN,
            strategy="funding"
        )

        if order_id:
            self.positions[opp.pair] = opp
            self.logger.info(
                f"FUNDING HUNTER: Opened {opp.direction.upper()} on {opp.pair} @ {price:.4f} "
                f"({opp.apr:.1f}% APR, ${position_size} x {leverage}x [{vol_class} volatility])"
            )
            return True

        return False

    def _close_position(self, pair: str, reason: str, current_timestamp: float):
        """Close a funding position."""
        if pair not in self.positions:
            return

        opp = self.positions[pair]

        price = self.connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        # Use same leverage as when opened
        if self.config.use_smart_leverage:
            leverage = get_safe_leverage(pair, self.config.funding_leverage_max)
        else:
            leverage = self.config.funding_leverage_max

        amount = self.config.funding_position_size / price
        trade_type = TradeType.SELL if opp.direction == "long" else TradeType.BUY

        order_id = self.place_order(
            pair=pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.CLOSE,
            strategy="funding"
        )

        if order_id:
            del self.positions[pair]
            self.logger.info(f"FUNDING HUNTER: Closed {opp.direction.upper()} on {pair} ({reason})")

    def close_all_positions(self, current_timestamp: float):
        """Close all funding positions (for shutdown)."""
        for pair in list(self.positions.keys()):
            self._close_position(pair, "shutdown", current_timestamp)
