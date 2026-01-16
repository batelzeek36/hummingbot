"""
Funding Rate Hunter Strategy for Hyperliquid Monster Bot v2.

Dynamically scans for best funding rate opportunities across all configured pairs
and automatically rotates positions to maximize funding income.

Enhanced with:
- Max loss exit protection
- **OI Leading Indicators (v2.6)** - Predicts funding rate sustainability
- **HOLY GRAIL Signal (v2.7)** - Combined leading indicators for better entry filtering

OI Integration:
- Rising OI = More positions opening = Funding likely to stay elevated
- Falling OI = Positions closing = Funding will normalize/drop
- OI-boosted scoring = Prefer pairs where OI confirms sustainability

Holy Grail Integration:
- Blocks entries when combined signal predicts imminent reversal
- Uses all leading indicators (OI + Premium + Funding Velocity + Volume)
- Adds warning tags when signals conflict with intended direction
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType

from ..coinglass import CoinglassAPI, LiquidationHeatmap, LongShortRatio, SpotPerpDivergence
from ..config import HyperliquidMonsterV2Config
from ..leading_indicators import DirectionalSignal, HyperliquidLeadingIndicators
from ..models import FundingOpportunity, StrategyMetrics, StrategyMode
from ..volatility import (
    COIN_VOLATILITY,
    CoinVolatility,
    get_safe_leverage,
    get_min_loss_threshold,
    get_volatility_class,
)


@dataclass
class OpenFundingPosition:
    """Tracks an open funding position with entry details."""
    opportunity: FundingOpportunity
    entry_price: Decimal
    amount: Decimal
    leverage: int
    expected_funding_pct: float  # Expected funding as % of position


class FundingHunterStrategy:
    """
    Dynamic Funding Rate Hunter.

    Scans all configured pairs for best funding opportunities,
    auto-rotates to capture highest rates.

    Enhanced with max loss exit - exits early if price move
    exceeds expected funding payment.
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
        self.positions: Dict[str, OpenFundingPosition] = {}  # pair -> position info
        self.last_scan_timestamp = 0

        # Stats
        self._max_loss_exits = 0
        self._oi_early_exits = 0
        self._oi_boosted_entries = 0

        # OI Leading Indicators
        self._oi_tracker: Optional[HyperliquidLeadingIndicators] = None
        if self.config.funding_use_oi:
            self._oi_tracker = HyperliquidLeadingIndicators(
                oi_lookback_periods=12,
                fetch_interval_seconds=30,
                oi_change_threshold=2.0,
                logger=self.logger,
            )
            self.logger.info("FUNDING: OI Leading Indicators enabled - will predict funding sustainability")

        # Coinglass API Integration (GOD MODE Phase 2)
        self._coinglass: Optional[CoinglassAPI] = None
        self._coinglass_blocks = 0  # Entries blocked by Coinglass
        self._coinglass_exits = 0   # Exits triggered by Coinglass
        if self.config.coinglass_enabled:
            self._coinglass = CoinglassAPI(
                api_key=self.config.coinglass_api_key,
                cache_ttl=self.config.coinglass_cache_ttl,
                request_interval=float(self.config.coinglass_request_interval),
                logger=self.logger,
            )
            self.logger.info("FUNDING: Coinglass API enabled - crowding/divergence/liquidation signals active")

    def run(self, current_timestamp: float):
        """
        Run the funding hunter strategy.

        Args:
            current_timestamp: Current timestamp in seconds
        """
        if self.metrics.status != StrategyMode.ACTIVE:
            self.logger.debug(f"FUNDING: Not active, status={self.metrics.status}")
            return

        # Scan for opportunities periodically
        if self.last_scan_timestamp < current_timestamp - self.config.funding_scan_interval:
            self.logger.info(f"FUNDING: Starting scan (last scan: {self.last_scan_timestamp}, now: {current_timestamp})")
            self._scan_opportunities(current_timestamp)
            self.last_scan_timestamp = current_timestamp

        # Manage existing positions
        self._manage_positions(current_timestamp)

        # Open new positions if slots available
        self._open_best_positions(current_timestamp)

    def _scan_opportunities(self, current_timestamp: float):
        """Scan all pairs for funding rate opportunities with OI analysis."""
        opportunities = []

        # Update OI data for all pairs (single API call, cached)
        all_pairs = [p.strip() for p in self.config.funding_scan_pairs.split(",")]
        if self._oi_tracker:
            self._oi_tracker.update(all_pairs)

        for pair in all_pairs:
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

                # === OI LEADING INDICATOR BOOST ===
                # Rising OI = more positions opening = funding likely sustainable
                oi_status = ""
                if self._oi_tracker and self.config.funding_oi_entry_boost:
                    oi_analysis = self._oi_tracker.analyze_oi_momentum(pair)
                    if oi_analysis:
                        if oi_analysis.oi_change_pct > 2.0:  # OI rising
                            # Boost score - this funding is more sustainable
                            score += float(self.config.funding_oi_rising_bonus)
                            oi_status = f" [OI+{oi_analysis.oi_change_pct:.1f}%]"
                        elif oi_analysis.oi_change_pct < -2.0:  # OI falling
                            # Penalize - funding may drop soon
                            score -= float(self.config.funding_oi_rising_bonus) * 0.5
                            oi_status = f" [OI{oi_analysis.oi_change_pct:.1f}%]"

                opp = FundingOpportunity(
                    pair=pair,
                    rate=rate,
                    apr=apr,
                    direction=direction,
                    next_funding_time=next_funding_ts,
                    score=score
                )
                # Store OI status for logging
                opp._oi_status = oi_status  # type: ignore

                opportunities.append(opp)

            except Exception as e:
                self.logger.warning(f"Error scanning {pair}: {e}")

        # Sort by score (highest first)
        self.opportunities = sorted(opportunities, key=lambda x: x.score, reverse=True)

        # Log scan results
        if not self.opportunities:
            self.logger.info(f"FUNDING SCAN: No opportunities found from {len(all_pairs)} pairs")
        else:
            top3 = self.opportunities[:3]
            self.logger.info("TOP FUNDING OPPORTUNITIES:")
            for opp in top3:
                minutes = (opp.next_funding_time - current_timestamp) / 60
                oi_status = getattr(opp, '_oi_status', '')
                self.logger.info(
                    f"  {opp.pair}: {opp.apr:.1f}% APR ({opp.direction.upper()}) "
                    f"- {minutes:.0f}min to funding{oi_status}"
                )

    def _manage_positions(self, current_timestamp: float):
        """Manage existing funding positions - close or rotate."""
        for pair, position in list(self.positions.items()):
            try:
                current_opp = position.opportunity

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

                # === MAX LOSS EXIT CHECK ===
                if self.config.funding_max_loss_exit:
                    should_exit, reason = self._check_max_loss(pair, position)
                    if should_exit:
                        self._close_position(pair, reason, current_timestamp)
                        self._max_loss_exits += 1
                        continue

                # === OI EARLY EXIT CHECK (LEADING INDICATOR) ===
                # If OI is falling significantly, funding will likely drop - exit early
                if self._oi_tracker and self.config.funding_use_oi:
                    should_exit, reason = self._check_oi_exit(pair)
                    if should_exit:
                        self._close_position(pair, reason, current_timestamp)
                        self._oi_early_exits += 1
                        continue

                # === HOLY GRAIL EARLY EXIT CHECK ===
                # If combined signals flip strongly against our position, exit early
                if self._oi_tracker and self.config.use_holy_grail_signal:
                    should_exit, reason = self._check_holy_grail_exit(pair, current_opp.direction)
                    if should_exit:
                        self._close_position(pair, reason, current_timestamp)
                        continue

                # === COINGLASS LIQUIDATION EXIT CHECK (GOD MODE Phase 2) ===
                # If liquidation clusters are building nearby, exit to avoid squeeze
                if self._coinglass:
                    should_exit, reason = self._check_coinglass_exit(pair, current_opp.direction)
                    if should_exit:
                        self.logger.warning(f"FUNDING COINGLASS EXIT: {pair} - {reason}")
                        self._close_position(pair, f"coinglass_{reason}", current_timestamp)
                        self._coinglass_exits += 1
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

    def _check_max_loss(self, pair: str, position: OpenFundingPosition) -> tuple[bool, str]:
        """
        Check if unrealized loss exceeds expected funding payment.

        Uses volatility-scaled thresholds to avoid false exits on volatile coins.
        Exit triggers only if BOTH conditions are met:
        1. Loss exceeds (expected funding * multiplier)
        2. Loss exceeds minimum threshold for the coin's volatility class

        Args:
            pair: Trading pair
            position: Open position info

        Returns:
            Tuple of (should_exit, reason)
        """
        current_price = self.connector.get_price_by_type(pair, PriceType.MidPrice)
        if current_price is None:
            return False, ""

        entry_price = position.entry_price
        direction = position.opportunity.direction

        # Calculate unrealized P&L as percentage
        if direction == "long":
            pnl_pct = float((current_price - entry_price) / entry_price)
        else:
            pnl_pct = float((entry_price - current_price) / entry_price)

        # Expected funding payment (hourly rate as percentage)
        expected_funding_pct = position.expected_funding_pct

        # Loss threshold based on funding rate
        funding_based_threshold = expected_funding_pct * float(self.config.funding_loss_multiplier)

        # Minimum threshold based on volatility class (prevents false exits on volatile coins)
        # This is in percentage points (e.g., 2.0 = 2%)
        volatility_min_threshold = get_min_loss_threshold(pair) / 100  # Convert to decimal

        # Use the HIGHER of the two thresholds
        # This ensures we don't exit on normal volatility noise
        effective_threshold = max(funding_based_threshold, volatility_min_threshold)

        # Exit if loss exceeds effective threshold
        if pnl_pct < -effective_threshold:
            vol_class = get_volatility_class(pair).value.upper()
            self.logger.warning(
                f"FUNDING MAX LOSS: {pair} [{vol_class}] loss {pnl_pct*100:.3f}% exceeds "
                f"threshold {effective_threshold*100:.3f}% "
                f"(funding-based: {funding_based_threshold*100:.3f}%, vol-min: {volatility_min_threshold*100:.2f}%)"
            )
            return True, "max_loss_exit"

        return False, ""

    def _check_oi_exit(self, pair: str) -> Tuple[bool, str]:
        """
        Check if OI is falling - indicates funding rate will likely drop.

        This is a LEADING indicator - we exit BEFORE the rate actually drops,
        instead of waiting for the rate to flip (reactive).

        Args:
            pair: Trading pair

        Returns:
            Tuple of (should_exit, reason)
        """
        if not self._oi_tracker:
            return False, ""

        oi_analysis = self._oi_tracker.analyze_oi_momentum(pair)
        if oi_analysis is None:
            return False, ""

        threshold = float(self.config.funding_oi_exit_threshold)

        # Check if OI is falling beyond threshold
        if oi_analysis.oi_change_pct < threshold:
            self.logger.warning(
                f"FUNDING OI EXIT: {pair} OI falling {oi_analysis.oi_change_pct:.1f}% "
                f"(threshold: {threshold}%) - funding likely to drop"
            )
            return True, f"oi_falling_{oi_analysis.oi_change_pct:.1f}pct"

        return False, ""

    def _check_holy_grail_exit(self, pair: str, position_direction: str) -> Tuple[bool, str]:
        """
        Check if Holy Grail combined signal flips against our position.

        This provides an even earlier exit signal by combining:
        - OI momentum
        - Premium pressure
        - Funding velocity
        - Volume surge

        If all these indicate strong reversal, exit before funding rate actually flips.

        Args:
            pair: Trading pair
            position_direction: Current position direction ("long" or "short")

        Returns:
            Tuple of (should_exit, reason)
        """
        if not self._oi_tracker:
            return False, ""

        holy_grail = self._oi_tracker.get_holy_grail_signal(pair)
        if holy_grail is None:
            return False, ""

        min_confidence = float(self.config.holy_grail_min_confidence)

        # Only exit on strong signals (high confidence)
        # For funding, we want even higher confidence since we're expecting the payment
        exit_confidence_threshold = min_confidence + 20  # Higher bar for exits

        # Check if signal strongly contradicts position
        if position_direction == "long":
            if holy_grail.direction == "strong_short" and holy_grail.confidence >= exit_confidence_threshold:
                self.logger.warning(
                    f"FUNDING HOLY GRAIL EXIT: {pair} LONG - combined signal flipped to "
                    f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"
                )
                return True, f"holy_grail_flip_{holy_grail.direction}"
        elif position_direction == "short":
            if holy_grail.direction == "strong_long" and holy_grail.confidence >= exit_confidence_threshold:
                self.logger.warning(
                    f"FUNDING HOLY GRAIL EXIT: {pair} SHORT - combined signal flipped to "
                    f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"
                )
                return True, f"holy_grail_flip_{holy_grail.direction}"

        return False, ""

    def _check_coinglass_entry(self, pair: str, direction: str) -> Tuple[bool, str]:
        """
        Check if Coinglass signals should block entry.

        Uses:
        - Long/Short ratio (crowding detection)
        - Spot/Perp divergence (trapped traders)

        Args:
            pair: Trading pair
            direction: Intended position direction ("long" or "short")

        Returns:
            Tuple of (should_block, reason)
        """
        if not self._coinglass:
            return False, ""

        try:
            should_block, reason = self._coinglass.should_block_entry(pair, direction)
            return should_block, reason
        except Exception as e:
            self.logger.debug(f"COINGLASS entry check error for {pair}: {e}")
            return False, ""

    def _check_coinglass_exit(self, pair: str, direction: str) -> Tuple[bool, str]:
        """
        Check if Coinglass liquidation heatmap suggests imminent squeeze.

        Exit early if large liquidation clusters are building nearby.

        Args:
            pair: Trading pair
            direction: Current position direction ("long" or "short")

        Returns:
            Tuple of (should_exit, reason)
        """
        if not self._coinglass:
            return False, ""

        try:
            heatmap = self._coinglass.get_liquidation_heatmap(pair)
            if heatmap is None:
                return False, ""

            # Check if liquidations are clustered dangerously close
            # For longs: worry about long liquidations below price (cascade risk)
            # For shorts: worry about short liquidations above price (squeeze risk)

            if direction == "long" and heatmap.nearest_long_liq:
                # Long liquidations below current price
                distance_pct = ((heatmap.current_price - heatmap.nearest_long_liq)
                               / heatmap.current_price) * 100
                # Exit if large liquidation cluster within 2%
                if distance_pct <= 2.0:
                    for cluster in heatmap.long_clusters:
                        if cluster.risk_level == "high" and cluster.distance_pct <= 2.0:
                            return True, (
                                f"LIQUIDATION CLUSTER: ${cluster.estimated_size_usd/1e6:.1f}M longs "
                                f"within {cluster.distance_pct:.1f}% - cascade risk"
                            )

            elif direction == "short" and heatmap.nearest_short_liq:
                # Short liquidations above current price
                distance_pct = ((heatmap.nearest_short_liq - heatmap.current_price)
                               / heatmap.current_price) * 100
                # Exit if large liquidation cluster within 2%
                if distance_pct <= 2.0:
                    for cluster in heatmap.short_clusters:
                        if cluster.risk_level == "high" and cluster.distance_pct <= 2.0:
                            return True, (
                                f"LIQUIDATION CLUSTER: ${cluster.estimated_size_usd/1e6:.1f}M shorts "
                                f"within {cluster.distance_pct:.1f}% - squeeze risk"
                            )

            return False, ""

        except Exception as e:
            self.logger.debug(f"COINGLASS exit check error for {pair}: {e}")
            return False, ""

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

            # === HOLY GRAIL CHECK ===
            # Block entry if combined leading indicators suggest reversal
            if self._oi_tracker and self.config.use_holy_grail_signal:
                holy_grail = self._oi_tracker.get_holy_grail_signal(opp.pair)
                if holy_grail and self.config.holy_grail_block_contrary:
                    min_confidence = float(self.config.holy_grail_min_confidence)

                    # Check if Holy Grail contradicts funding direction
                    if opp.direction == "long" and holy_grail.direction in ("short", "strong_short"):
                        if holy_grail.confidence >= min_confidence:
                            self.logger.info(
                                f"FUNDING: {opp.pair} LONG blocked by HOLY GRAIL - "
                                f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"
                            )
                            continue
                    elif opp.direction == "short" and holy_grail.direction in ("long", "strong_long"):
                        if holy_grail.confidence >= min_confidence:
                            self.logger.info(
                                f"FUNDING: {opp.pair} SHORT blocked by HOLY GRAIL - "
                                f"{holy_grail.direction} ({holy_grail.confidence:.0f}%)"
                            )
                            continue

            # === COINGLASS ENTRY CHECK (GOD MODE Phase 2) ===
            # Block entry if crowding or divergence signals suggest risk
            if self._coinglass:
                should_block, reason = self._check_coinglass_entry(opp.pair, opp.direction)
                if should_block:
                    self.logger.info(
                        f"FUNDING: {opp.pair} {opp.direction.upper()} blocked by COINGLASS - {reason}"
                    )
                    self._coinglass_blocks += 1
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

        # Notional = margin * leverage, then convert to asset quantity
        notional = position_size * leverage
        amount = notional / price
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
            # Store position with entry details for max loss tracking
            self.positions[opp.pair] = OpenFundingPosition(
                opportunity=opp,
                entry_price=price,
                amount=amount,
                leverage=leverage,
                expected_funding_pct=abs(opp.rate)  # Hourly rate as decimal
            )

            # Check if this was OI-boosted
            oi_status = getattr(opp, '_oi_status', '')
            if oi_status and 'OI+' in oi_status:
                self._oi_boosted_entries += 1

            self.logger.info(
                f"FUNDING HUNTER: Opened {opp.direction.upper()} on {opp.pair} @ {price:.4f} "
                f"({opp.apr:.1f}% APR, ${position_size} x {leverage}x [{vol_class}]){oi_status}"
            )
            return True

        return False

    def _close_position(self, pair: str, reason: str, current_timestamp: float):
        """Close a funding position."""
        if pair not in self.positions:
            return

        position = self.positions[pair]
        opp = position.opportunity

        price = self.connector.get_price_by_type(pair, PriceType.MidPrice)
        if price is None:
            return

        # Use same leverage as when opened
        leverage = position.leverage

        # Notional = margin * leverage, then convert to asset quantity
        notional = self.config.funding_position_size * leverage
        amount = notional / price
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
            # Calculate realized P&L for logging
            entry_price = position.entry_price
            if opp.direction == "long":
                pnl_pct = float((price - entry_price) / entry_price) * 100
            else:
                pnl_pct = float((entry_price - price) / entry_price) * 100

            del self.positions[pair]
            self.logger.info(
                f"FUNDING HUNTER: Closed {opp.direction.upper()} on {pair} "
                f"({reason}) P&L: {pnl_pct:+.2f}%"
            )

    def close_all_positions(self, current_timestamp: float):
        """Close all funding positions (for shutdown)."""
        for pair in list(self.positions.keys()):
            self._close_position(pair, "shutdown", current_timestamp)

    def get_status_info(self) -> Dict[str, str]:
        """Get current funding strategy status for display."""
        status = {}
        status["Positions"] = str(len(self.positions))
        status["Max Loss Exits"] = str(self._max_loss_exits)

        if self.config.funding_max_loss_exit:
            status["Max Loss Protection"] = f"ON (volatility-scaled)"
            # Show thresholds by class
            status["Thresholds"] = "SAFE:0.3% MED:0.5% HIGH:1% EXT:2%"
        else:
            status["Max Loss Protection"] = "OFF"

        # OI Leading Indicators status
        if self.config.funding_use_oi and self._oi_tracker:
            oi_stats = self._oi_tracker.get_status()
            api_health = "OK" if oi_stats["error_count"] == 0 else f"ERR:{oi_stats['error_count']}"
            status["OI Predictor"] = f"ON ({api_health})"
            status["OI Exits/Boosts"] = f"{self._oi_early_exits}/{self._oi_boosted_entries}"
        else:
            status["OI Predictor"] = "OFF"

        # Holy Grail signal status
        if self.config.use_holy_grail_signal and self._oi_tracker:
            status["Holy Grail"] = "ON (entry filter + exit trigger)"
        else:
            status["Holy Grail"] = "OFF"

        # Coinglass GOD MODE Phase 2 status
        if self.config.coinglass_enabled and self._coinglass:
            cg_stats = self._coinglass.get_status() if hasattr(self._coinglass, 'get_status') else {}
            api_health = "OK" if cg_stats.get("error_count", 0) == 0 else f"ERR:{cg_stats.get('error_count', '?')}"
            status["Coinglass"] = f"ON ({api_health})"
            status["CG Blocks/Exits"] = f"{self._coinglass_blocks}/{self._coinglass_exits}"
        else:
            status["Coinglass"] = "OFF"

        return status

    def get_oi_tracker(self) -> Optional[HyperliquidLeadingIndicators]:
        """Get the OI tracker for external access."""
        return self._oi_tracker

    def get_active_positions(self) -> Dict[str, Dict]:
        """
        Get all active funding positions with their directions.
        Used by circuit breaker to determine adverse vs favorable moves.

        Returns:
            Dict of pair -> {"direction": "long"/"short", "entry_price": Decimal}
        """
        result = {}
        for pair, pos in self.positions.items():
            result[pair] = {
                "direction": pos.direction,  # "long" or "short"
                "entry_price": pos.entry_price,
            }
        return result
