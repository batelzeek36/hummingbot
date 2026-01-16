"""
Position management for Momentum Strategy.

Contains functions to open and close positions with proper logging and ML tracking.
"""

import logging
import uuid
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from hummingbot.core.data_type.common import PositionAction, PriceType, TradeType

from ...ml_models import MLFeatures

if TYPE_CHECKING:
    from hummingbot.connector.connector_base import ConnectorBase
    from ...config import HyperliquidMonsterV2Config
    from ...models import StrategyMetrics
    from ...ml_models import SignalConfirmationModel


class PositionManager:
    """
    Manages position state and operations for the Momentum Strategy.

    Handles opening, closing positions, P&L tracking, and ML learning integration.
    """

    def __init__(
        self,
        config: "HyperliquidMonsterV2Config",
        connector: "ConnectorBase",
        metrics: "StrategyMetrics",
        place_order_fn: Callable,
        ml_model: Optional["SignalConfirmationModel"] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the position manager.

        Args:
            config: Bot configuration
            connector: Exchange connector
            metrics: Strategy metrics tracker
            place_order_fn: Function to place orders
            ml_model: Optional ML model for tracking
            logger: Logger instance
        """
        self.config = config
        self.connector = connector
        self.metrics = metrics
        self.place_order = place_order_fn
        self._ml_model = ml_model
        self.logger = logger or logging.getLogger(__name__)

        # Position state
        self.position: Optional[str] = None  # "long", "short", or None
        self.entry_price: Optional[Decimal] = None
        self.entry_signals: List[str] = []  # Signals that triggered entry
        self._current_trade_id: Optional[str] = None  # For ML outcome tracking

    @property
    def pair(self) -> str:
        """Get the momentum trading pair."""
        return self.config.momentum_pair

    def has_position(self) -> bool:
        """Check if there's an open position."""
        return self.position is not None

    def get_position_info(self) -> Optional[Tuple[str, Decimal]]:
        """Get current position info (direction, entry_price)."""
        if self.position and self.entry_price:
            return (self.position, self.entry_price)
        return None

    def open_position(
        self,
        direction: str,
        price: Decimal,
        rsi: float,
        signals: Dict[str, bool],
        extra_info: str = "",
        ml_features: Optional[MLFeatures] = None,
    ) -> bool:
        """
        Open a momentum position with enhanced logging and ML tracking.

        Args:
            direction: "long" or "short"
            price: Entry price
            rsi: RSI value at entry
            signals: Signals that triggered entry
            extra_info: Additional info string for logging
            ml_features: ML features for tracking

        Returns:
            True if position was opened successfully
        """
        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        # Notional = margin * leverage, then convert to asset quantity
        notional = position_size * leverage
        amount = notional / price
        trade_type = TradeType.BUY if direction == "long" else TradeType.SELL

        order_id = self.place_order(
            pair=self.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.OPEN,
            strategy="momentum"
        )

        if order_id:
            self.position = direction
            self.entry_price = price
            self.entry_signals = [k for k, v in signals.items() if v]

            # Generate trade ID for ML tracking
            self._current_trade_id = str(uuid.uuid4())[:8]

            # Record entry for ML online learning
            if self._ml_model and ml_features:
                self._ml_model.record_entry(
                    trade_id=self._current_trade_id,
                    features=ml_features,
                    entry_price=float(price),
                )

            # Build signal summary
            signal_str = ", ".join(self.entry_signals)
            ml_info = " [ML+]" if ml_features else ""
            self.logger.info(
                f"MOMENTUM: Opened {direction.upper()} @ {price:.2f} "
                f"(RSI: {rsi:.1f}, ${position_size} x {leverage}x) "
                f"Signals: [{signal_str}]{extra_info}{ml_info}"
            )
            return True

        return False

    def close_position(
        self,
        price: Decimal,
        reason: str,
        global_pnl_updater: Callable[[Decimal], None],
    ) -> bool:
        """
        Close momentum position with enhanced P&L tracking and ML learning.

        Args:
            price: Exit price
            reason: Reason for closing (e.g., "take_profit", "stop_loss")
            global_pnl_updater: Callback to update global P&L

        Returns:
            True if position was closed successfully
        """
        if not self.position:
            return False

        position_size = self.config.momentum_position_size
        leverage = self.config.momentum_leverage

        # Notional = margin * leverage, then convert to asset quantity
        notional = position_size * leverage
        amount = notional / price
        trade_type = TradeType.SELL if self.position == "long" else TradeType.BUY

        order_id = self.place_order(
            pair=self.pair,
            side=trade_type,
            amount=amount,
            price=price,
            leverage=leverage,
            position_action=PositionAction.CLOSE,
            strategy="momentum"
        )

        if order_id and self.entry_price:
            if self.position == "long":
                pnl = (price - self.entry_price) * amount * leverage
            else:
                pnl = (self.entry_price - price) * amount * leverage

            self.metrics.realized_pnl += pnl
            global_pnl_updater(pnl)

            # Track win/loss
            if pnl > 0:
                self.metrics.winning_trades += 1

            # === GOD MODE Phase 4: Record exit for ML online learning ===
            ml_learned = ""
            if self._ml_model and self._current_trade_id:
                self._ml_model.record_exit(
                    trade_id=self._current_trade_id,
                    exit_price=float(price),
                    exit_reason=reason,
                )
                ml_learned = " [ML]"  # Learning indicator

            pnl_pct = pnl / position_size * 100
            signals_str = ", ".join(self.entry_signals) if self.entry_signals else "N/A"

            self.logger.info(
                f"MOMENTUM: Closed {self.position.upper()} @ {price:.2f} "
                f"({reason}) P&L: ${pnl:.4f} ({pnl_pct:.2f}%) "
                f"Entry signals: [{signals_str}]{ml_learned}"
            )

            self._reset_position()
            return True

        return False

    def close_position_for_shutdown(
        self,
        global_pnl_updater: Callable[[Decimal], None],
    ):
        """
        Close position for shutdown.

        Args:
            global_pnl_updater: Callback to update global P&L
        """
        if self.position:
            price = self.connector.get_price_by_type(self.pair, PriceType.MidPrice)
            if price:
                self.close_position(price, "shutdown", global_pnl_updater)

    def _reset_position(self):
        """Reset position state after closing."""
        self.position = None
        self.entry_price = None
        self.entry_signals = []
        self._current_trade_id = None

    def set_ml_model(self, ml_model: Optional["SignalConfirmationModel"]):
        """Set the ML model for tracking."""
        self._ml_model = ml_model
