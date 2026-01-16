"""
Configuration for Hyperliquid Monster Bot v2.

Pydantic-based configuration class for all bot settings.
"""

from decimal import Decimal

from pydantic import Field

from hummingbot.client.config.config_data_types import BaseClientModel


class HyperliquidMonsterV2Config(BaseClientModel):
    """Configuration for Hyperliquid Monster Bot V2 - AGGRESSIVE."""

    script_file_name: str = Field(default="hyperliquid_monster_bot_v2.py")

    # === EXCHANGE SETTINGS ===
    exchange: str = Field(
        default="hyperliquid_perpetual",
        description="Exchange connector"
    )

    # === PAIRS TO SCAN FOR FUNDING ===
    # All pairs we'll scan for funding opportunities
    # Start with known high-funding pairs, expand as needed
    funding_scan_pairs: str = Field(
        default="ETH-USD,SOL-USD,DOGE-USD,TAO-USD,HYPE-USD,AVNT-USD,kPEPE-USD,kBONK-USD,WIF-USD,HYPER-USD,IP-USD",
        description="Pairs to scan for funding rate opportunities (VVV removed - 9.5% buffer too tight)"
    )

    # Maximum simultaneous funding positions
    max_funding_positions: int = Field(
        default=3,
        description="Maximum number of simultaneous funding positions"
    )

    # === STATIC PAIRS FOR GRID/MOMENTUM ===
    grid_pair: str = Field(
        default="SOL-USD",
        description="Pair for grid trading (needs good liquidity)"
    )
    momentum_pair: str = Field(
        default="BTC-USD",
        description="Pair for momentum/directional trading"
    )

    # === CAPITAL ALLOCATION - MORE TO FUNDING ===
    total_capital: Decimal = Field(
        default=Decimal("152"),
        description="Total capital in USDC"
    )
    funding_capital_pct: Decimal = Field(
        default=Decimal("45"),  # UP from 40%
        description="% of capital for funding harvesting"
    )
    grid_capital_pct: Decimal = Field(
        default=Decimal("35"),
        description="% of capital for grid trading"
    )
    momentum_capital_pct: Decimal = Field(
        default=Decimal("20"),  # DOWN from 25%
        description="% of capital for momentum trading"
    )

    # === LEVERAGE SETTINGS - SMART (based on volatility analysis) ===
    # These are MAX values - actual leverage is determined by coin volatility
    # BTC: 8x, SOL: 5x, DOGE/HYPE/MOG: 3x, PEPE/BONK/WIF/ANIME: 2x
    funding_leverage_max: int = Field(default=8, description="Max leverage for funding (actual varies by coin)")
    grid_leverage: int = Field(default=5, description="Leverage for grid trading (SOL = medium volatility)")
    momentum_leverage: int = Field(default=8, description="Leverage for momentum trades (BTC = low volatility)")
    max_leverage: int = Field(default=8, description="Absolute maximum leverage cap")
    use_smart_leverage: bool = Field(default=True, description="Use volatility-based leverage per coin")

    # === FUNDING HARVESTING SETTINGS - AGGRESSIVE ===
    funding_enabled: bool = Field(default=True)

    # Minimum 30% APR to enter (filters out weak opportunities)
    min_funding_apr: Decimal = Field(
        default=Decimal("30"),  # 30% APR minimum
        description="Minimum annualized funding rate to open position"
    )

    # Auto-rotate if better opportunity is 20% higher APR
    funding_rotation_threshold: Decimal = Field(
        default=Decimal("20"),  # Switch if new opp is 20% APR better
        description="APR improvement needed to rotate positions"
    )

    funding_position_size: Decimal = Field(
        default=Decimal("12"),  # $12 per position (with 8x = $96 exposure)
        description="Position size per funding pair"
    )

    minutes_before_funding: int = Field(
        default=10,  # UP from 5 - more time to position
        description="Open position this many minutes before funding"
    )

    # Scan interval for new opportunities
    funding_scan_interval: int = Field(
        default=60,  # Scan every 60 seconds
        description="Seconds between funding rate scans"
    )

    # Max loss exit - exit early if price move exceeds expected funding
    funding_max_loss_exit: bool = Field(
        default=True,
        description="Exit funding position early if unrealized loss exceeds expected funding payment"
    )
    funding_loss_multiplier: Decimal = Field(
        default=Decimal("1.5"),  # Exit if loss > 1.5x expected funding
        description="Exit if unrealized loss exceeds this multiple of expected funding payment"
    )

    # === FUNDING OI LEADING INDICATORS ===
    # Predict funding rate changes using Open Interest data
    funding_use_oi: bool = Field(
        default=True,
        description="Use OI to predict funding rate sustainability"
    )
    funding_oi_exit_threshold: Decimal = Field(
        default=Decimal("-3.0"),
        description="Exit early if OI drops more than this % (funding likely to drop)"
    )
    funding_oi_entry_boost: bool = Field(
        default=True,
        description="Prefer positions where OI is rising (funding more sustainable)"
    )
    funding_oi_rising_bonus: Decimal = Field(
        default=Decimal("10.0"),
        description="APR bonus for scoring when OI is rising (makes OI-confirmed positions more attractive)"
    )

    # === GRID TRADING SETTINGS - TIGHTER ===
    grid_enabled: bool = Field(default=True)
    grid_levels: int = Field(default=6, description="Number of grid levels each side")  # UP from 5
    grid_spacing_pct: Decimal = Field(
        default=Decimal("0.004"),  # 0.4% spacing (tighter than 0.5%)
        description="Base spacing between grid levels (adjusted by volatility)"
    )
    grid_order_size: Decimal = Field(
        default=Decimal("12"),  # $12 per grid order (Hyperliquid $10 min + buffer for rounding)
        description="Size per grid order in USD"
    )
    grid_rebalance_pct: Decimal = Field(
        default=Decimal("0.025"),  # 2.5% rebalance trigger (tighter)
        description="Rebalance grid when price moves this much"
    )

    # === GRID ENHANCED SETTINGS ===
    # Trend pause - stop grid during strong trends
    grid_trend_pause: bool = Field(
        default=True,
        description="Pause grid during strong trends to avoid accumulating losses"
    )
    grid_trend_ema_short: int = Field(default=20, description="Short EMA for grid trend detection")
    grid_trend_ema_long: int = Field(default=50, description="Long EMA for grid trend detection")
    grid_trend_strength_threshold: Decimal = Field(
        default=Decimal("0.5"),  # 0.5% EMA separation = strong trend
        description="EMA separation % to consider trend strong enough to pause"
    )

    # Volatility-adaptive spacing
    grid_adaptive_spacing: bool = Field(
        default=True,
        description="Adjust grid spacing based on recent volatility"
    )
    grid_volatility_lookback: int = Field(
        default=20,
        description="Periods to calculate volatility for adaptive spacing"
    )
    grid_min_spacing_pct: Decimal = Field(
        default=Decimal("0.003"),  # 0.3% minimum spacing
        description="Minimum grid spacing in low volatility"
    )
    grid_max_spacing_pct: Decimal = Field(
        default=Decimal("0.008"),  # 0.8% maximum spacing
        description="Maximum grid spacing in high volatility"
    )

    # Inventory/position skew management
    grid_inventory_management: bool = Field(
        default=True,
        description="Reduce position sizing when inventory gets skewed"
    )
    grid_max_inventory_skew: int = Field(
        default=4,  # Max 4 more positions on one side
        description="Maximum inventory imbalance before reducing new orders"
    )
    grid_skew_reduction_factor: Decimal = Field(
        default=Decimal("0.5"),  # Reduce to 50% size when skewed
        description="Order size multiplier when inventory is skewed"
    )

    # === MOMENTUM SETTINGS ===
    momentum_enabled: bool = Field(default=True)
    momentum_lookback: int = Field(default=20, description="Candles for RSI calculation")
    rsi_oversold: Decimal = Field(default=Decimal("25"), description="RSI oversold threshold")  # More extreme
    rsi_overbought: Decimal = Field(default=Decimal("75"), description="RSI overbought threshold")  # More extreme
    momentum_take_profit: Decimal = Field(
        default=Decimal("0.025"),  # 2.5% TP (with 5x = 12.5% gain)
        description="Take profit percentage"
    )
    momentum_stop_loss: Decimal = Field(
        default=Decimal("0.015"),  # 1.5% SL (with 5x = 7.5% loss)
        description="Stop loss percentage"
    )
    momentum_position_size: Decimal = Field(
        default=Decimal("12"),  # $12 per momentum trade (Hyperliquid $10 min + buffer)
        description="Position size for momentum trades"
    )

    # === MOMENTUM ENHANCED INDICATORS ===
    # Trend filter (EMA 50/200)
    use_trend_filter: bool = Field(
        default=True,
        description="Enable trend filter - only trade with the trend"
    )
    ema_short_period: int = Field(default=50, description="Short-term EMA period for trend")
    ema_long_period: int = Field(default=200, description="Long-term EMA period for trend")

    # Volume confirmation
    use_volume_filter: bool = Field(
        default=True,
        description="Enable volume confirmation for entries"
    )
    volume_lookback: int = Field(default=20, description="Periods for average volume calculation")
    min_volume_ratio: Decimal = Field(
        default=Decimal("1.0"),
        description="Minimum volume ratio (current/average) for valid signal"
    )
    strong_volume_ratio: Decimal = Field(
        default=Decimal("1.5"),
        description="Volume ratio for strong signal confirmation"
    )

    # Funding rate sentiment
    use_funding_sentiment: bool = Field(
        default=True,
        description="Enable funding rate sentiment for entry confirmation"
    )
    funding_sentiment_threshold: Decimal = Field(
        default=Decimal("0.0001"),  # 0.01% hourly = significant
        description="Funding rate threshold for sentiment signal"
    )

    # MACD confirmation
    use_macd_filter: bool = Field(
        default=True,
        description="Enable MACD confirmation for entries"
    )
    macd_fast: int = Field(default=12, description="MACD fast EMA period")
    macd_slow: int = Field(default=26, description="MACD slow EMA period")
    macd_signal: int = Field(default=9, description="MACD signal line period")

    # Multi-indicator confluence
    min_signals_required: int = Field(
        default=2,
        description="Minimum confirming signals required (RSI always required, plus N of: trend, volume, funding, MACD, OI)"
    )

    # === LEADING INDICATORS (Open Interest) ===
    # These are TRUE leading indicators from Hyperliquid API
    use_oi_filter: bool = Field(
        default=True,
        description="Enable Open Interest confirmation for entries"
    )
    oi_lookback_periods: int = Field(
        default=12,
        description="Number of OI snapshots to keep for analysis"
    )
    oi_fetch_interval: int = Field(
        default=30,
        description="Seconds between Hyperliquid API fetches"
    )
    oi_change_threshold: Decimal = Field(
        default=Decimal("2.0"),
        description="OI change % to be considered significant"
    )
    oi_block_exhaustion_entries: bool = Field(
        default=True,
        description="Block entries during OI exhaustion (squeeze/capitulation)"
    )
    oi_require_confirmation: bool = Field(
        default=False,
        description="Require OI confirmation for all entries (stricter)"
    )

    # === HOLY GRAIL - Combined Leading Indicator Signal ===
    # Combines OI + Premium + Funding Velocity + Volume into one directional signal
    use_holy_grail_signal: bool = Field(
        default=True,
        description="Use the combined 'Holy Grail' directional signal for entries"
    )
    holy_grail_min_confidence: Decimal = Field(
        default=Decimal("40"),
        description="Minimum confidence score (0-100) to act on Holy Grail signal"
    )
    holy_grail_block_contrary: bool = Field(
        default=True,
        description="Block entries that go against the Holy Grail direction"
    )
    holy_grail_require_strong: bool = Field(
        default=False,
        description="Only enter on 'strong_long' or 'strong_short' signals"
    )

    # Component weights (should sum to 1.0)
    holy_grail_oi_weight: Decimal = Field(
        default=Decimal("0.35"),
        description="Weight for OI momentum in Holy Grail calculation"
    )
    holy_grail_premium_weight: Decimal = Field(
        default=Decimal("0.25"),
        description="Weight for Premium pressure in Holy Grail calculation"
    )
    holy_grail_funding_vel_weight: Decimal = Field(
        default=Decimal("0.25"),
        description="Weight for Funding velocity in Holy Grail calculation"
    )
    holy_grail_volume_weight: Decimal = Field(
        default=Decimal("0.15"),
        description="Weight for Volume surge in Holy Grail calculation"
    )

    # === RISK MANAGEMENT - SLIGHTLY LOOSER FOR AGGRESSIVE MODE ===
    max_drawdown_pct: Decimal = Field(
        default=Decimal("20"),  # UP from 15% - more room before kill
        description="Kill all strategies at this drawdown %"
    )
    max_position_size_pct: Decimal = Field(
        default=Decimal("60"),  # UP from 50%
        description="Max position as % of capital (leveraged)"
    )
    daily_loss_limit: Decimal = Field(
        default=Decimal("12"),  # UP from 10
        description="Max daily loss in USD"
    )

    # ==========================================================================
    # WHALE PROTECTION SETTINGS
    # ==========================================================================
    # Circuit Breaker - detects rapid price spikes and pauses/flattens
    # Conservative defaults to avoid false triggers

    whale_protection_enabled: bool = Field(
        default=True,
        description="Master switch for whale protection system"
    )

    # --- Circuit Breaker ---
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker for rapid price movement detection"
    )
    # Pause thresholds - price move % that triggers PAUSE (hold positions, cancel orders)
    cb_pause_threshold_30s: Decimal = Field(
        default=Decimal("0.025"),  # 2.5% in 30s
        description="Price change in 30s that triggers pause"
    )
    cb_pause_threshold_60s: Decimal = Field(
        default=Decimal("0.035"),  # 3.5% in 60s
        description="Price change in 60s that triggers pause"
    )
    cb_pause_threshold_120s: Decimal = Field(
        default=Decimal("0.05"),  # 5% in 2min
        description="Price change in 120s that triggers pause"
    )
    # Flatten thresholds - price move % that triggers FLATTEN (close all positions)
    cb_flatten_threshold_30s: Decimal = Field(
        default=Decimal("0.05"),  # 5% in 30s
        description="Price change in 30s that triggers flatten"
    )
    cb_flatten_threshold_60s: Decimal = Field(
        default=Decimal("0.07"),  # 7% in 60s
        description="Price change in 60s that triggers flatten"
    )
    cb_flatten_threshold_120s: Decimal = Field(
        default=Decimal("0.10"),  # 10% in 2min
        description="Price change in 120s that triggers flatten"
    )
    cb_cooldown_seconds: int = Field(
        default=300,  # 5 minutes
        description="Cooldown period after circuit breaker triggers"
    )

    # --- Grid Protection ---
    grid_protection_enabled: bool = Field(
        default=True,
        description="Enable grid fill rate protection"
    )
    gp_max_one_sided_fills: int = Field(
        default=4,
        description="Max consecutive fills on one side before pausing grid"
    )
    gp_imbalance_ratio_threshold: Decimal = Field(
        default=Decimal("3.0"),  # 3:1 buy:sell ratio
        description="Fill ratio that triggers grid pause"
    )
    gp_window_seconds: int = Field(
        default=120,  # 2 minutes
        description="Time window for grid fill counting"
    )
    gp_cooldown_seconds: int = Field(
        default=180,  # 3 minutes
        description="Cooldown after grid protection triggers"
    )

    # --- Trailing Stop (for Momentum) ---
    trailing_stop_enabled: bool = Field(
        default=True,
        description="Enable trailing stop for momentum positions"
    )
    ts_mode: str = Field(
        default="breakeven_then_trail",
        description="Trailing stop mode: disabled, percentage, breakeven_then_trail"
    )
    ts_trail_distance_pct: Decimal = Field(
        default=Decimal("0.015"),  # 1.5% trail distance
        description="Trailing distance as percentage of price"
    )
    ts_activation_profit_pct: Decimal = Field(
        default=Decimal("0.01"),  # 1% profit before trailing activates
        description="Minimum profit before trailing stop activates"
    )
    ts_breakeven_activation_pct: Decimal = Field(
        default=Decimal("0.008"),  # 0.8% profit to move to breakeven
        description="Profit level to move stop to breakeven"
    )
    ts_tighten_at_profit_pct: Decimal = Field(
        default=Decimal("0.025"),  # Tighten at 2.5% profit
        description="Profit level to tighten trailing distance"
    )
    ts_tightened_trail_pct: Decimal = Field(
        default=Decimal("0.01"),  # Tighten to 1% trail
        description="Tightened trail distance"
    )

    # --- Dynamic Risk (Volatility-Scaled) ---
    dynamic_risk_enabled: bool = Field(
        default=True,
        description="Enable volatility-scaled risk management"
    )
    dr_volatility_lookback: int = Field(
        default=30,
        description="Periods for volatility calculation"
    )
    # Emergency thresholds (unrealized loss %)
    dr_emergency_loss_threshold: Decimal = Field(
        default=Decimal("0.05"),  # 5% unrealized loss
        description="Loss threshold that triggers emergency action"
    )
    dr_critical_loss_threshold: Decimal = Field(
        default=Decimal("0.08"),  # 8% unrealized loss
        description="Loss threshold that triggers immediate market close"
    )
    dr_use_market_orders_emergency: bool = Field(
        default=True,
        description="Use market orders for emergency exits"
    )

    # --- Early Warning System ---
    early_warning_enabled: bool = Field(
        default=True,
        description="Enable early warning system (order book imbalance, funding spikes)"
    )
    ew_ob_imbalance_warning: Decimal = Field(
        default=Decimal("2.0"),  # 2:1 imbalance
        description="Order book imbalance ratio for WARNING level"
    )
    ew_ob_imbalance_danger: Decimal = Field(
        default=Decimal("3.5"),  # 3.5:1 imbalance
        description="Order book imbalance ratio for DANGER level"
    )
    ew_funding_warning_apr: Decimal = Field(
        default=Decimal("438"),  # ~438% APR (0.05% hourly)
        description="Funding rate APR for WARNING level"
    )
    ew_funding_danger_apr: Decimal = Field(
        default=Decimal("876"),  # ~876% APR (0.1% hourly)
        description="Funding rate APR for DANGER level"
    )
    ew_block_entries_on_danger: bool = Field(
        default=True,
        description="Block new entries when DANGER level warnings active"
    )

    # --- Risk Tick Interval ---
    risk_check_interval: int = Field(
        default=3,  # Check every 3 seconds (down from 10)
        description="Seconds between risk checks (faster = more responsive)"
    )

    # ==========================================================================
    # GOD MODE PHASE 2: COINGLASS INTEGRATION
    # ==========================================================================
    # CVD (Cumulative Volume Delta), Liquidation Heatmaps, Long/Short Ratios

    coinglass_enabled: bool = Field(
        default=True,
        description="Enable Coinglass API integration for CVD and liquidation data"
    )
    coinglass_api_key: str = Field(
        default="",
        description="Coinglass API key (or set COINGLASS_API_KEY env var)"
    )

    # --- CVD Settings ---
    use_cvd_signals: bool = Field(
        default=True,
        description="Use CVD (Cumulative Volume Delta) for entry signals"
    )
    cvd_divergence_threshold: Decimal = Field(
        default=Decimal("70"),
        description="Minimum divergence strength (0-100) to act on spot/perp divergence"
    )
    cvd_block_contrary_entries: bool = Field(
        default=True,
        description="Block entries that go against CVD divergence signals"
    )

    # --- Liquidation Heatmap Settings ---
    use_liquidation_heatmap: bool = Field(
        default=True,
        description="Use liquidation heatmap data for entry/exit decisions"
    )
    liq_proximity_warning_pct: Decimal = Field(
        default=Decimal("2.0"),
        description="Warn when price is within this % of major liquidation cluster"
    )
    liq_magnetic_weight: Decimal = Field(
        default=Decimal("0.5"),
        description="Weight for liquidation magnetic direction in signals (0-1)"
    )

    # --- Long/Short Ratio Settings ---
    use_long_short_ratio: bool = Field(
        default=True,
        description="Use long/short ratio for contrarian signals"
    )
    ls_crowding_threshold: Decimal = Field(
        default=Decimal("65"),
        description="Long or short % that indicates crowding"
    )
    ls_extreme_threshold: Decimal = Field(
        default=Decimal("75"),
        description="Long or short % that indicates extreme crowding (block entries)"
    )

    # --- Coinglass Rate Limiting ---
    coinglass_request_interval: Decimal = Field(
        default=Decimal("2.0"),
        description="Minimum seconds between Coinglass API requests (30/min limit)"
    )
    coinglass_cache_ttl: int = Field(
        default=60,
        description="Cache TTL in seconds for Coinglass data"
    )

    # --- Funding Strategy Coinglass Integration ---
    funding_use_coinglass: bool = Field(
        default=True,
        description="Use Coinglass data for funding entry/exit decisions"
    )
    funding_coinglass_block_crowded: bool = Field(
        default=True,
        description="Block funding entries when position is crowded (>65% one side)"
    )
    funding_coinglass_crowding_threshold: Decimal = Field(
        default=Decimal("65"),
        description="L/S ratio % to consider crowded"
    )
    funding_coinglass_block_divergence: bool = Field(
        default=True,
        description="Block entries against strong spot/perp divergence"
    )
    funding_coinglass_divergence_threshold: Decimal = Field(
        default=Decimal("70"),
        description="Divergence signal strength to block entry"
    )
    funding_coinglass_exit_on_squeeze: bool = Field(
        default=True,
        description="Exit early if liquidation clusters approach"
    )
    funding_liq_proximity_exit_pct: Decimal = Field(
        default=Decimal("3.0"),
        description="Exit if liquidation cluster within this % of price"
    )

    # ==========================================================================
    # GOD MODE PHASE 3: MULTI-TIMEFRAME CONFLUENCE + LIQUIDATION MAGNETS
    # ==========================================================================
    # Higher timeframes carry more weight. Only enter when 3+ timeframes agree.

    # --- Multi-Timeframe Confluence ---
    use_mtf_confluence: bool = Field(
        default=True,
        description="Enable Multi-Timeframe confluence analysis for entries"
    )
    mtf_min_timeframes: int = Field(
        default=3,
        description="Minimum timeframes that must agree for confluence (out of 4: 1m, 5m, 15m, 1h)"
    )
    mtf_block_contrary: bool = Field(
        default=True,
        description="Block entries that go against strong MTF confluence"
    )
    mtf_require_htf_alignment: bool = Field(
        default=True,
        description="Require 1H timeframe to align with entry direction"
    )

    # Timeframe weights (should sum to 1.0)
    mtf_weight_1m: Decimal = Field(
        default=Decimal("0.10"),
        description="Weight for 1-minute timeframe signals"
    )
    mtf_weight_5m: Decimal = Field(
        default=Decimal("0.20"),
        description="Weight for 5-minute timeframe signals"
    )
    mtf_weight_15m: Decimal = Field(
        default=Decimal("0.30"),
        description="Weight for 15-minute timeframe signals"
    )
    mtf_weight_1h: Decimal = Field(
        default=Decimal("0.40"),
        description="Weight for 1-hour timeframe signals (king timeframe)"
    )

    # MTF indicator settings
    mtf_ema_short: int = Field(
        default=9,
        description="Short EMA period for MTF trend detection"
    )
    mtf_ema_long: int = Field(
        default=21,
        description="Long EMA period for MTF trend detection"
    )
    mtf_strong_trend_threshold: Decimal = Field(
        default=Decimal("1.0"),
        description="EMA separation % to consider trend 'strong'"
    )

    # --- Liquidation Magnet Detection ---
    use_liq_magnet: bool = Field(
        default=True,
        description="Use liquidation cluster data for entry/exit decisions"
    )
    liq_magnet_boost_entries: bool = Field(
        default=True,
        description="Boost entries in direction of liquidation magnet"
    )
    liq_magnet_block_contrary: bool = Field(
        default=True,
        description="Block entries against strong liquidation magnet direction"
    )
    liq_magnet_min_imbalance: Decimal = Field(
        default=Decimal("1.5"),
        description="Minimum liquidity imbalance ratio to consider magnetic (1.5 = 50% more one side)"
    )
    liq_cluster_proximity_pct: Decimal = Field(
        default=Decimal("3.0"),
        description="Distance % from current price to consider liquidation cluster 'nearby'"
    )
    liq_cluster_warning_pct: Decimal = Field(
        default=Decimal("1.5"),
        description="Distance % to warn about imminent liquidation cluster"
    )

    # ==========================================================================
    # GOD MODE PHASE 4: ML MODELS - SIGNAL CONFIRMATION + FUNDING PREDICTION
    # ==========================================================================
    # XGBoost-based signal confirmation and funding rate prediction

    # --- ML Signal Confirmation ---
    use_ml_confirmation: bool = Field(
        default=True,
        description="Enable ML model for signal confirmation (requires xgboost)"
    )
    ml_min_confidence: Decimal = Field(
        default=Decimal("60"),
        description="Minimum ML confidence (0-100) to confirm trade"
    )
    ml_min_training_samples: int = Field(
        default=50,
        description="Minimum trades before ML model is used"
    )
    ml_block_low_confidence: bool = Field(
        default=True,
        description="Block trades with low ML confidence"
    )
    ml_model_path: str = Field(
        default="hyperliquid_monster_ml_model.pkl",
        description="Path to save/load ML model"
    )
    ml_data_path: str = Field(
        default="hyperliquid_monster_ml_data.json",
        description="Path to save trade outcome data"
    )

    # --- Funding Rate Prediction ---
    use_funding_prediction: bool = Field(
        default=True,
        description="Enable ML funding rate prediction"
    )
    funding_prediction_min_samples: int = Field(
        default=100,
        description="Minimum samples before funding prediction is used"
    )
    funding_prediction_horizon_hours: Decimal = Field(
        default=Decimal("1.0"),
        description="How far ahead to predict funding (hours)"
    )
    funding_early_entry_threshold: Decimal = Field(
        default=Decimal("1.5"),
        description="Enter early if predicted funding > current * this"
    )
    funding_prediction_model_path: str = Field(
        default="hyperliquid_monster_funding_model.json",
        description="Path to save/load funding prediction model"
    )

    # ==========================================================================
    # GOD MODE PHASE 5: NEWS EVENTS + ATTRIBUTION + BACKTEST
    # ==========================================================================
    # Auto-pause trading around major economic events (FOMC, CPI, NFP)

    # --- News Event Detection ---
    news_events_enabled: bool = Field(
        default=True,
        description="Enable news event detection and auto-pause"
    )
    news_pause_on_high_impact: bool = Field(
        default=True,
        description="Pause trading for HIGH impact events (CPI, NFP)"
    )
    news_pause_on_critical_only: bool = Field(
        default=False,
        description="Only pause for CRITICAL events (FOMC)"
    )
    news_warning_minutes: int = Field(
        default=120,
        description="Warn about events this many minutes ahead"
    )
    news_close_on_critical: bool = Field(
        default=False,
        description="Close positions before CRITICAL events"
    )

    # --- Performance Attribution ---
    attribution_enabled: bool = Field(
        default=True,
        description="Enable signal attribution tracking"
    )
    attribution_data_path: str = Field(
        default="hyperliquid_monster_attribution.json",
        description="Path to save attribution data"
    )
    attribution_auto_save: bool = Field(
        default=True,
        description="Automatically save attribution after each trade"
    )

    # --- Backtest Framework ---
    backtest_cache_enabled: bool = Field(
        default=True,
        description="Cache historical data for faster backtests"
    )
    backtest_cache_dir: str = Field(
        default=".backtest_cache",
        description="Directory for backtest data cache"
    )
    backtest_default_leverage: Decimal = Field(
        default=Decimal("5.0"),
        description="Default leverage for backtests"
    )
    backtest_default_position_pct: Decimal = Field(
        default=Decimal("10.0"),
        description="Default position size as % of capital"
    )
