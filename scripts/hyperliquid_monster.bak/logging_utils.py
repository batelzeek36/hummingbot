"""
Logging Utilities - Startup banner and logging helpers.
"""

from typing import Callable, Optional


def log_startup_banner(config, logger: Optional[Callable] = None):
    """
    Log the startup banner with bot configuration.
    
    Args:
        config: Bot configuration object
        logger: Logger function (must return logger object with .info method)
    """
    if not logger:
        return
    
    log = logger()
    
    log.info("=" * 70)
    log.info("  HYPERLIQUID MONSTER BOT V2.5 - WHALE PROTECTION")
    log.info("=" * 70)
    log.info(f"  Capital: ${config.total_capital}")
    log.info(f"  Smart Leverage: ENABLED (volatility-based)")
    log.info(f"  Grid Leverage: {config.grid_leverage}x (SOL)")
    log.info(f"  Momentum Leverage: {config.momentum_leverage}x (BTC)")
    log.info(f"  Min Funding APR: {config.min_funding_apr}%")
    log.info(f"  Scanning: {config.funding_scan_pairs}")
    
    log.info("-" * 70)
    log.info("  ENHANCED MOMENTUM INDICATORS:")
    log.info(f"    Trend Filter (EMA {config.ema_short_period}/{config.ema_long_period}): {'ON' if config.use_trend_filter else 'OFF'}")
    log.info(f"    Volume Filter: {'ON' if config.use_volume_filter else 'OFF'}")
    log.info(f"    Funding Sentiment: {'ON' if config.use_funding_sentiment else 'OFF'}")
    log.info(f"    MACD ({config.macd_fast}/{config.macd_slow}/{config.macd_signal}): {'ON' if config.use_macd_filter else 'OFF'}")
    log.info(f"    Min Confirming Signals: {config.min_signals_required}")
    
    log.info("-" * 70)
    log.info("  ENHANCED GRID SETTINGS:")
    log.info(f"    Trend Pause (EMA {config.grid_trend_ema_short}/{config.grid_trend_ema_long}): {'ON' if config.grid_trend_pause else 'OFF'}")
    log.info(f"    Adaptive Spacing: {'ON' if config.grid_adaptive_spacing else 'OFF'} ({float(config.grid_min_spacing_pct)*100:.1f}%-{float(config.grid_max_spacing_pct)*100:.1f}%)")
    log.info(f"    Inventory Management: {'ON' if config.grid_inventory_management else 'OFF'} (max skew: {config.grid_max_inventory_skew})")
    
    log.info("-" * 70)
    log.info("  FUNDING PROTECTION:")
    log.info(f"    Max Loss Exit: {'ON' if config.funding_max_loss_exit else 'OFF'} (volatility-scaled)")
    if config.funding_max_loss_exit:
        log.info("    Thresholds: SAFE:0.3% | MEDIUM:0.5% | HIGH:1.0% | EXTREME:2.0%")
    
    log.info("-" * 70)
    log.info("  WHALE PROTECTION:")
    log.info(f"    Master Switch: {'ON' if config.whale_protection_enabled else 'OFF'}")
    if config.whale_protection_enabled:
        log.info(f"    Circuit Breaker: {'ON' if config.circuit_breaker_enabled else 'OFF'}")
        log.info(f"    Grid Protection: {'ON' if config.grid_protection_enabled else 'OFF'}")
        log.info(f"    Trailing Stop: {'ON' if config.trailing_stop_enabled else 'OFF'}")
        log.info(f"    Dynamic Risk: {'ON' if config.dynamic_risk_enabled else 'OFF'}")
        log.info(f"    Early Warning: {'ON' if config.early_warning_enabled else 'OFF'}")
        log.info(f"    Risk Check Interval: {config.risk_check_interval}s")
