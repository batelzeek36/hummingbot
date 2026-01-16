"""
Pause Manager - Unified pause/resume system for trading bot.

Handles pause state from multiple protection systems (circuit breaker,
grid protection, early warning, dynamic risk) and coordinates auto-resume.
"""

from typing import Callable, Dict, Optional


class PauseManager:
    """
    Manages unified pause/resume state across multiple protection systems.
    
    Trading only resumes when ALL systems have cleared their pause reasons.
    """
    
    def __init__(
        self,
        cancel_orders_fn: Callable[[], None],
        close_positions_fn: Callable[[], None],
        get_timestamp_fn: Callable[[], float],
        logger: Optional[Callable] = None,
    ):
        """
        Initialize PauseManager.
        
        Args:
            cancel_orders_fn: Function to cancel all open orders
            close_positions_fn: Function to close all positions
            get_timestamp_fn: Function to get current timestamp
            logger: Logger function
        """
        self._cancel_orders_fn = cancel_orders_fn
        self._close_positions_fn = close_positions_fn
        self._get_timestamp = get_timestamp_fn
        self._logger = logger
        
        # Unified pause tracking - tracks WHY we're paused for auto-resume
        self._pause_reasons: Dict[str, str] = {}  # system -> reason
        self._last_pause_time: float = 0
        self._last_resume_time: float = 0
    
    def _log(self, level: str, message: str):
        """Log a message if logger is available."""
        if self._logger:
            log_fn = getattr(self._logger(), level, self._logger().info)
            log_fn(message)
    
    @property
    def is_paused(self) -> bool:
        """Check if trading is paused for any reason."""
        return len(self._pause_reasons) > 0
    
    @property
    def pause_reasons(self) -> Dict[str, str]:
        """Get current pause reasons."""
        return self._pause_reasons.copy()
    
    @property
    def last_pause_time(self) -> float:
        """Get timestamp of last pause."""
        return self._last_pause_time
    
    @property
    def last_resume_time(self) -> float:
        """Get timestamp of last resume."""
        return self._last_resume_time
    
    def trigger_pause(
        self,
        system: str,
        reason: str,
        cancel_orders: bool = True,
        close_positions: bool = False
    ):
        """
        Trigger a trading pause from any protection system.
        
        Args:
            system: Name of system triggering pause (circuit_breaker, grid_protection, etc.)
            reason: Human-readable reason for the pause
            cancel_orders: Whether to cancel all open orders
            close_positions: Whether to close all positions (for severe events)
        """
        was_paused = self.is_paused
        self._pause_reasons[system] = reason
        self._last_pause_time = self._get_timestamp()
        
        self._log(
            "error",
            f"PAUSE TRIGGERED by {system.upper()}\n"
            f"   Reason: {reason}\n"
            f"   Cancel orders: {cancel_orders} | Close positions: {close_positions}\n"
            f"   Active pause reasons: {list(self._pause_reasons.keys())}"
        )
        
        if cancel_orders:
            self._cancel_orders_fn()
        
        if close_positions:
            self._close_positions_fn()
        
        if not was_paused:
            self._log("error", ">>> ALL TRADING PAUSED - Waiting for conditions to normalize <<<")
    
    def clear_pause(self, system: str):
        """
        Clear a pause reason from a specific system.
        Trading only resumes when ALL systems are clear.
        
        Args:
            system: Name of system clearing its pause
        """
        if system in self._pause_reasons:
            old_reason = self._pause_reasons.pop(system)
            self._log(
                "info",
                f"{system.upper()} cleared: {old_reason}\n"
                f"   Remaining pause reasons: {list(self._pause_reasons.keys()) or 'NONE'}"
            )
            
            if not self.is_paused:
                self._on_all_clear()
    
    def _on_all_clear(self):
        """Called when all pause reasons are cleared - resume trading."""
        self._last_resume_time = self._get_timestamp()
        pause_duration = self._last_resume_time - self._last_pause_time
        
        self._log(
            "info",
            f"ALL CLEAR - Resuming trading\n"
            f"   Pause duration: {pause_duration:.1f} seconds\n"
            f"   Strategies will resume on next tick"
        )
    
    def check_safety_and_maybe_resume(
        self,
        circuit_breaker=None,
        grid_protection=None,
        early_warning=None,
        dynamic_risk=None,
    ):
        """
        Check all protection systems and auto-resume if safe.
        Called periodically to handle systems that don't have explicit resume callbacks.
        
        Args:
            circuit_breaker: CircuitBreaker instance (optional)
            grid_protection: GridProtection instance (optional)
            early_warning: EarlyWarningSystem instance (optional)
            dynamic_risk: DynamicRiskManager instance (optional)
        """
        if not self.is_paused:
            return  # Not paused, nothing to check
        
        current_time = self._get_timestamp()
        
        # Check circuit breaker
        if "circuit_breaker" in self._pause_reasons:
            if circuit_breaker:
                state = circuit_breaker.get_status()["state"]
                if state == "normal":
                    self.clear_pause("circuit_breaker")
        
        # Check grid protection
        if "grid_protection" in self._pause_reasons:
            if grid_protection and not grid_protection.is_paused():
                self.clear_pause("grid_protection")
        
        # Check early warning
        if "early_warning" in self._pause_reasons:
            if early_warning:
                # Only clear if no DANGER or CRITICAL warnings
                status = early_warning.get_status()
                max_level = "clear"
                for pair_status in status.get("pair_status", {}).values():
                    level = pair_status.get("level", "clear")
                    if level in ("danger", "critical"):
                        max_level = level
                        break
                    elif level == "warning" and max_level == "clear":
                        max_level = "warning"
                if max_level not in ("danger", "critical"):
                    self.clear_pause("early_warning")
        
        # Check dynamic risk (emergency situations)
        if "dynamic_risk" in self._pause_reasons:
            # Dynamic risk pauses are more serious - require longer cooldown
            time_since_pause = current_time - self._last_pause_time
            if time_since_pause > 300:  # 5 minute cooldown for emergency situations
                self.clear_pause("dynamic_risk")
    
    def get_status(self) -> dict:
        """Get current pause status."""
        current_time = self._get_timestamp()
        return {
            "is_paused": self.is_paused,
            "pause_reasons": self._pause_reasons.copy(),
            "pause_duration": current_time - self._last_pause_time if self.is_paused else 0,
            "last_pause_time": self._last_pause_time,
            "last_resume_time": self._last_resume_time,
        }
