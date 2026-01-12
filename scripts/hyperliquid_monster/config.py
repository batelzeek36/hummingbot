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
        default="ETH-USD,SOL-USD,DOGE-USD,TAO-USD,HYPE-USD,AVNT-USD,kPEPE-USD,kBONK-USD,WIF-USD,VVV-USD,HYPER-USD,IP-USD",
        description="Pairs to scan for funding rate opportunities"
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
        default=Decimal("78"),
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

    # === GRID TRADING SETTINGS - TIGHTER ===
    grid_enabled: bool = Field(default=True)
    grid_levels: int = Field(default=6, description="Number of grid levels each side")  # UP from 5
    grid_spacing_pct: Decimal = Field(
        default=Decimal("0.004"),  # 0.4% spacing (tighter than 0.5%)
        description="Spacing between grid levels"
    )
    grid_order_size: Decimal = Field(
        default=Decimal("12"),  # $12 per grid order (Hyperliquid $10 min + buffer for rounding)
        description="Size per grid order in USD"
    )
    grid_rebalance_pct: Decimal = Field(
        default=Decimal("0.025"),  # 2.5% rebalance trigger (tighter)
        description="Rebalance grid when price moves this much"
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
