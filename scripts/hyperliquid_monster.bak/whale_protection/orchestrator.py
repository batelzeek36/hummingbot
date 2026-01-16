"""
Whale Protection Orchestrator - Initializes and coordinates all whale protection components.

This module handles:
- Factory creation of all protection components
- Callback wiring between components and pause manager
- Running periodic protection checks
- Position synchronization with circuit breaker
"""

from typing import Callable, Dict, Optional, Set
from decimal import Decimal

from .circuit_breaker import CircuitBreaker
from .grid_protection import GridProtection
from .trailing_stop import TrailingStopManager, TrailingStopMode
from .dynamic_risk import DynamicRiskManager
from .early_warning import EarlyWarningSystem


class WhaleProtectionOrchestrator:
    """
    Orchestrates all whale protection components.
    
    Provides a clean interface for the main bot to interact with whale protection
    without managing individual components.
    """
    
    def __init__(
        self,
        config,
        pause_manager,
        logger: Optional[Callable] = None,
    ):
        """
        Initialize WhaleProtectionOrchestrator.
        
        Args:
            config: Bot configuration object with whale protection settings
            pause_manager: PauseManager instance for triggering pauses
            logger: Logger function
        """
        self.config = config
        self.pause_manager = pause_manager
        self._logger = logger
        
        # Protection components
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.grid_protection: Optional[GridProtection] = None
        self.trailing_stop: Optional[TrailingStopManager] = None
        self.dynamic_risk: Optional[DynamicRiskManager] = None
        self.early_warning: Optional[EarlyWarningSystem] = None
        
        # Emergency close callback (set by bot)
        self._emergency_close_fn: Optional[Callable[[str], None]] = None
    
    def _log(self, level: str, message: str):
        """Log a message if logger is available."""
        if self._logger:
            log_fn = getattr(self._logger(), level, self._logger().info)
            log_fn(message)
    
    def set_emergency_close_fn(self, fn: Callable[[str], None]):
        """Set the emergency close position function."""
        self._emergency_close_fn = fn
    
    def initialize(self):
        """Initialize all enabled whale protection components."""
        if not self.config.whale_protection_enabled:
            self._log("info", "Whale protection DISABLED by config")
            return
        
        self._log("info", "Initializing Whale Protection System...")
        
        # Circuit Breaker
        if self.config.circuit_breaker_enabled:
            self.circuit_breaker = CircuitBreaker(
                pause_threshold_30s=float(self.config.cb_pause_threshold_30s),
                pause_threshold_60s=float(self.config.cb_pause_threshold_60s),
                pause_threshold_120s=float(self.config.cb_pause_threshold_120s),
                flatten_threshold_30s=float(self.config.cb_flatten_threshold_30s),
                flatten_threshold_60s=float(self.config.cb_flatten_threshold_60s),
                flatten_threshold_120s=float(self.config.cb_flatten_threshold_120s),
                cooldown_seconds=self.config.cb_cooldown_seconds,
                on_pause=self._on_circuit_breaker_pause,
                on_flatten=self._on_circuit_breaker_flatten,
                on_resume=self._on_circuit_breaker_resume,
                logger=self._logger,
                enabled=True,
            )
            self._log("info", "  Circuit Breaker: ENABLED")
        
        # Grid Protection
        if self.config.grid_protection_enabled:
            self.grid_protection = GridProtection(
                max_one_sided_fills=self.config.gp_max_one_sided_fills,
                imbalance_ratio_threshold=float(self.config.gp_imbalance_ratio_threshold),
                window_seconds=self.config.gp_window_seconds,
                cooldown_seconds=self.config.gp_cooldown_seconds,
                on_pause=self._on_grid_protection_pause,
                on_resume=self._on_grid_protection_resume,
                logger=self._logger,
                enabled=True,
            )
            self._log("info", "  Grid Protection: ENABLED")
        
        # Trailing Stop
        if self.config.trailing_stop_enabled:
            mode_map = {
                "disabled": TrailingStopMode.DISABLED,
                "percentage": TrailingStopMode.PERCENTAGE,
                "breakeven_then_trail": TrailingStopMode.BREAKEVEN_THEN_TRAIL,
            }
            ts_mode = mode_map.get(self.config.ts_mode, TrailingStopMode.BREAKEVEN_THEN_TRAIL)
            
            self.trailing_stop = TrailingStopManager(
                mode=ts_mode,
                trail_distance_pct=self.config.ts_trail_distance_pct,
                activation_profit_pct=self.config.ts_activation_profit_pct,
                breakeven_activation_pct=self.config.ts_breakeven_activation_pct,
                tighten_at_profit_pct=self.config.ts_tighten_at_profit_pct,
                tightened_trail_pct=self.config.ts_tightened_trail_pct,
                logger=self._logger,
                enabled=True,
            )
            self._log("info", f"  Trailing Stop: ENABLED (mode: {ts_mode.value})")
        
        # Dynamic Risk
        if self.config.dynamic_risk_enabled:
            self.dynamic_risk = DynamicRiskManager(
                volatility_lookback=self.config.dr_volatility_lookback,
                emergency_loss_threshold=float(self.config.dr_emergency_loss_threshold),
                critical_loss_threshold=float(self.config.dr_critical_loss_threshold),
                on_emergency=self._on_emergency_action,
                logger=self._logger,
                enabled=True,
            )
            self._log("info", "  Dynamic Risk: ENABLED")
        
        # Early Warning
        if self.config.early_warning_enabled:
            # Convert APR to hourly rate for thresholds
            warning_rate = float(self.config.ew_funding_warning_apr) / 8760 / 100
            danger_rate = float(self.config.ew_funding_danger_apr) / 8760 / 100
            
            self.early_warning = EarlyWarningSystem(
                ob_imbalance_warning=float(self.config.ew_ob_imbalance_warning),
                ob_imbalance_danger=float(self.config.ew_ob_imbalance_danger),
                funding_warning_rate=warning_rate,
                funding_danger_rate=danger_rate,
                logger=self._logger,
                enabled=True,
            )
            self._log("info", "  Early Warning: ENABLED")
        
        self._log("info", "Whale Protection System initialized!")
    
    # =========================================================================
    # CALLBACKS - Bridge between protection components and pause manager
    # =========================================================================
    
    def _on_circuit_breaker_pause(self):
        """Called when circuit breaker triggers PAUSE - cancel orders, hold positions."""
        self.pause_manager.trigger_pause(
            system="circuit_breaker",
            reason="Rapid price movement detected - PAUSE mode",
            cancel_orders=True,
            close_positions=False
        )
    
    def _on_circuit_breaker_flatten(self):
        """Called when circuit breaker triggers FLATTEN - close everything."""
        self.pause_manager.trigger_pause(
            system="circuit_breaker",
            reason="Extreme price movement detected - FLATTEN mode",
            cancel_orders=True,
            close_positions=True
        )
    
    def _on_circuit_breaker_resume(self):
        """Called when circuit breaker cooldown ends."""
        self.pause_manager.clear_pause("circuit_breaker")
    
    def _on_grid_protection_pause(self):
        """Called when grid protection triggers pause - one-sided fills detected."""
        self.pause_manager.trigger_pause(
            system="grid_protection",
            reason="One-sided grid fills detected - possible directional move",
            cancel_orders=True,
            close_positions=False
        )
    
    def _on_grid_protection_resume(self):
        """Called when grid protection cooldown ends."""
        self.pause_manager.clear_pause("grid_protection")
    
    def _on_early_warning_danger(self, pair: str, level: str, reasons: list):
        """Called when early warning reaches DANGER or CRITICAL level."""
        self.pause_manager.trigger_pause(
            system="early_warning",
            reason=f"{pair} {level.upper()}: {', '.join(reasons)}",
            cancel_orders=True,
            close_positions=False
        )
    
    def _on_emergency_action(self, pair: str, action, reason: str):
        """Called when dynamic risk triggers emergency action."""
        self.pause_manager.trigger_pause(
            system="dynamic_risk",
            reason=f"Emergency on {pair}: {reason}",
            cancel_orders=True,
            close_positions=True  # Dynamic risk emergencies are serious
        )
        if self.config.dr_use_market_orders_emergency and self._emergency_close_fn:
            self._emergency_close_fn(pair)
    
    # =========================================================================
    # PROTECTION CHECKS - Called from main bot tick
    # =========================================================================
    
    def run_checks(
        self,
        connector,
        pairs: Set[str],
        current_timestamp: float,
    ):
        """
        Run all whale protection checks.
        
        Args:
            connector: Exchange connector
            pairs: Set of pairs to monitor
            current_timestamp: Current timestamp
        """
        from hummingbot.core.data_type.common import PriceType
        
        for pair in pairs:
            try:
                price = connector.get_price_by_type(pair, PriceType.MidPrice)
                if price is None:
                    continue
                
                price_float = float(price)
                
                # Update circuit breaker (callbacks handle pause/resume)
                if self.circuit_breaker:
                    self.circuit_breaker.update_price(pair, price_float, current_timestamp)
                
                # Update dynamic risk (for volatility tracking)
                if self.dynamic_risk:
                    self.dynamic_risk.update_price(pair, price_float, current_timestamp)
                
                # Update early warning with funding data and check for DANGER level
                if self.early_warning:
                    try:
                        funding_info = connector.get_funding_info(pair)
                        if funding_info:
                            self.early_warning.update_funding(
                                pair,
                                float(funding_info.rate),
                                current_timestamp
                            )
                    except Exception:
                        pass  # Funding info not always available
                    
                    # Check warning level - only PAUSE on CRITICAL, just warn on DANGER
                    if self.config.ew_block_entries_on_danger:
                        warning_level = self.early_warning.get_warning_level(pair)
                        if warning_level:
                            if warning_level.value == "critical":
                                # CRITICAL = definite danger, pause everything
                                if "early_warning" not in self.pause_manager.pause_reasons:
                                    reasons = []
                                    pair_warnings = self.early_warning.get_active_warnings(pair)
                                    for w in pair_warnings:
                                        reasons.append(f"{w.category}: {w.message}")
                                    self._on_early_warning_danger(pair, warning_level.value, reasons)
                            elif warning_level.value == "danger":
                                # DANGER = be careful but don't stop winning trades
                                self._log(
                                    "warning",
                                    f"EARLY WARNING DANGER on {pair} - monitoring closely but not pausing"
                                )
            
            except Exception as e:
                self._log("debug", f"Whale protection check error for {pair}: {e}")
    
    def sync_positions_to_circuit_breaker(
        self,
        momentum_strategy=None,
        funding_strategy=None,
        grid_strategy=None,
    ):
        """
        Sync current positions to circuit breaker so it only triggers on ADVERSE moves.
        This prevents the circuit breaker from stopping us out of winning trades.
        
        Args:
            momentum_strategy: MomentumStrategy instance
            funding_strategy: FundingHunterStrategy instance  
            grid_strategy: GridStrategy instance
        """
        if not self.circuit_breaker:
            return
        
        # Check momentum position
        if momentum_strategy:
            pos_info = momentum_strategy.get_position_info()
            if pos_info:
                direction, _ = pos_info
                self.circuit_breaker.register_position(self.config.momentum_pair, direction)
        
        # Check funding positions
        if funding_strategy:
            funding_positions = funding_strategy.get_active_positions()
            for pair, pos_data in funding_positions.items():
                direction = pos_data.get("direction", "long")
                self.circuit_breaker.register_position(pair, direction)
        
        # Check grid position (grid can have net long or short)
        if grid_strategy:
            grid_direction = grid_strategy.get_net_direction()
            if grid_direction:
                self.circuit_breaker.register_position(self.config.grid_pair, grid_direction)
    
    def update_trailing_stop(
        self,
        momentum_strategy,
        connector,
        momentum_pair: str,
        stop_loss_pct: Decimal,
        close_position_fn: Callable,
    ):
        """
        Update trailing stop for momentum position.
        
        Args:
            momentum_strategy: MomentumStrategy instance
            connector: Exchange connector
            momentum_pair: The momentum trading pair
            stop_loss_pct: Stop loss percentage from config
            close_position_fn: Function to close position
        """
        if not self.trailing_stop or not momentum_strategy:
            return
        
        from hummingbot.core.data_type.common import PriceType
        
        pos_info = momentum_strategy.get_position_info()
        if not pos_info:
            # No position - remove any trailing stop
            if self.trailing_stop.has_stop(momentum_pair):
                self.trailing_stop.remove_stop(momentum_pair)
            return
        
        direction, entry_price = pos_info
        current_price = connector.get_price_by_type(momentum_pair, PriceType.MidPrice)
        
        if current_price is None:
            return
        
        # Create trailing stop if not exists
        if not self.trailing_stop.has_stop(momentum_pair):
            self.trailing_stop.create_stop(
                pair=momentum_pair,
                direction=direction,
                entry_price=entry_price,
                initial_stop_pct=stop_loss_pct,
            )
        
        # Update with current price
        stop_price, should_exit = self.trailing_stop.update_price(
            momentum_pair,
            current_price
        )
        
        if should_exit:
            self._log("warning", f"TRAILING STOP: Exiting momentum position at {current_price}")
            close_position_fn()
            self.trailing_stop.remove_stop(momentum_pair)
    
    def record_grid_fill(self, pair: str, side: str, price: float, amount: float, order_id: str, timestamp: float):
        """Record a grid fill for grid protection tracking."""
        if self.grid_protection:
            self.grid_protection.record_fill(
                pair=pair,
                side=side,
                price=price,
                amount=amount,
                order_id=order_id,
                timestamp=timestamp
            )
    
    def check_auto_resume(self):
        """Check if we can auto-resume from pause."""
        self.pause_manager.check_safety_and_maybe_resume(
            circuit_breaker=self.circuit_breaker,
            grid_protection=self.grid_protection,
            early_warning=self.early_warning,
            dynamic_risk=self.dynamic_risk,
        )
    
    def is_grid_paused(self) -> bool:
        """Check if grid protection has paused grid trading."""
        return self.grid_protection.is_paused() if self.grid_protection else False
