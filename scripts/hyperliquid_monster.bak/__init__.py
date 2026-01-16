"""
Hyperliquid Monster Bot v2 - Modular Package

A multi-strategy perpetual trading bot for Hyperliquid with:
- Dynamic funding rate hunting
- Grid trading
- Enhanced momentum trading with multi-indicator confluence
- Smart volatility-based leverage
- **Open Interest (OI) LEADING INDICATORS from Hyperliquid API** (v2.6)
- **GOD MODE Phase 1: Spike Detection, Market Regime, Order Book Velocity** (v2.8)
- **GOD MODE Phase 2: Coinglass CVD, Liquidation Heatmap, Long/Short Ratio** (v2.9)
- **GOD MODE Phase 3: Multi-Timeframe Confluence, Liquidation Magnets** (v2.10)
- **GOD MODE Phase 4: ML Signal Confirmation, Online Learning** (v2.11)
- **GOD MODE Phase 5: News Events, Performance Attribution, Backtesting** (v2.12)

Author: Dollar-A-Day Project
Version: 2.12
"""

from .config import HyperliquidMonsterV2Config
from .indicators import (
    MACDResult,
    TechnicalIndicators,
    TrendInfo,
    VolumeAnalysis,
)
from .leading_indicators import (
    DirectionalSignal,
    FundingVelocity,
    HyperliquidLeadingIndicators,
    OIAnalysis,
    OIMomentum,
    OISnapshot,
    PremiumAnalysis,
    VolumeSurge,
    # Phase 1 GOD MODE
    OISpikeResult,
    PremiumSpikeResult,
    MarketRegime,
)
from .models import FundingOpportunity, PositionInfo, StrategyMetrics, StrategyMode
from .performance import CoinPerformance, PerformanceTracker
from .volatility import (
    COIN_VOLATILITY,
    VOLATILITY_LEVERAGE,
    VOLATILITY_MIN_LOSS_THRESHOLD,
    VOLATILITY_GRID_TREND_THRESHOLD,
    CoinVolatility,
    get_safe_leverage,
    get_min_loss_threshold,
    get_volatility_class,
    get_grid_trend_threshold,
)
from .strategies import FundingHunterStrategy, GridStrategy, MomentumStrategy
from .whale_protection import (
    CircuitBreaker,
    CircuitBreakerState,
    GridProtection,
    GridProtectionState,
    TrailingStopManager,
    DynamicRiskManager,
    EarlyWarningSystem,
    WarningLevel,
    WhaleProtectionOrchestrator,
    # Phase 1 GOD MODE
    OrderBookVelocity,
)
from .pause_manager import PauseManager
from .status_formatter import StatusFormatter
from .logging_utils import log_startup_banner
# Phase 2 GOD MODE - Coinglass Integration
from .coinglass import (
    CoinglassAPI,
    CVDSnapshot,
    CVDDirection,
    SpotPerpDivergence,
    LiquidationHeatmap,
    LiquidationCluster,
    LongShortRatio,
)
# Phase 3 GOD MODE - Multi-Timeframe Confluence
from .multi_timeframe import (
    MTFConfluence,
    MultiTimeframeAnalyzer,
    TimeframeSignal,
    TimeframeTrend,
)
# Phase 4 GOD MODE - ML Signal Confirmation
from .ml_models import (
    MLFeatures,
    MLPrediction,
    SignalConfirmationModel,
    TradeOutcome,
    FeatureExtractor,
)
from .funding_predictor import (
    FundingFeatures,
    FundingPrediction,
    FundingPredictor,
    FundingHistoryTracker,
)
# Phase 5 GOD MODE - News Events, Attribution, Backtest
from .news_events import (
    EconomicEvent,
    EventImpact,
    EventType,
    MarketState,
    EventCheckResult,
    EconomicCalendar,
    NewsEventDetector,
)
from .attribution import (
    SignalSnapshot,
    SignalType,
    TradeAttribution,
    SignalPerformance,
    AttributionReport,
    AttributionTracker,
    AttributionAnalyzer,
)
from .backtest import (
    Candle,
    HistoricalDataPoint,
    BacktestTrade,
    BacktestResult,
    BacktestConfig,
    HyperliquidHistoricalData,
    DataCache,
    BacktestEngine,
    BacktestMetrics,
)

__all__ = [
    # Config
    "HyperliquidMonsterV2Config",
    # Indicators (Lagging)
    "TechnicalIndicators",
    "MACDResult",
    "TrendInfo",
    "VolumeAnalysis",
    # Leading Indicators (from Hyperliquid API)
    "HyperliquidLeadingIndicators",
    "OIAnalysis",
    "OIMomentum",
    "OISnapshot",
    "PremiumAnalysis",
    "FundingVelocity",
    "VolumeSurge",
    "DirectionalSignal",
    # Phase 1 GOD MODE - Spike Detection & Market Regime
    "OISpikeResult",
    "PremiumSpikeResult",
    "MarketRegime",
    "OrderBookVelocity",
    # Models
    "FundingOpportunity",
    "PositionInfo",
    "StrategyMetrics",
    "StrategyMode",
    # Performance
    "CoinPerformance",
    "PerformanceTracker",
    # Volatility
    "CoinVolatility",
    "COIN_VOLATILITY",
    "VOLATILITY_LEVERAGE",
    "VOLATILITY_MIN_LOSS_THRESHOLD",
    "VOLATILITY_GRID_TREND_THRESHOLD",
    "get_safe_leverage",
    "get_min_loss_threshold",
    "get_volatility_class",
    "get_grid_trend_threshold",
    # Strategies
    "FundingHunterStrategy",
    "GridStrategy",
    "MomentumStrategy",
    # Whale Protection
    "CircuitBreaker",
    "CircuitBreakerState",
    "GridProtection",
    "GridProtectionState",
    "TrailingStopManager",
    "DynamicRiskManager",
    "EarlyWarningSystem",
    "WarningLevel",
    "WhaleProtectionOrchestrator",
    # Orchestration
    "PauseManager",
    "StatusFormatter",
    "log_startup_banner",
    # Phase 2 GOD MODE - Coinglass
    "CoinglassAPI",
    "CVDSnapshot",
    "CVDDirection",
    "SpotPerpDivergence",
    "LiquidationHeatmap",
    "LiquidationCluster",
    "LongShortRatio",
    # Phase 3 GOD MODE - Multi-Timeframe Confluence
    "MTFConfluence",
    "MultiTimeframeAnalyzer",
    "TimeframeSignal",
    "TimeframeTrend",
    # Phase 4 GOD MODE - ML Signal Confirmation
    "MLFeatures",
    "MLPrediction",
    "SignalConfirmationModel",
    "TradeOutcome",
    "FeatureExtractor",
    "FundingFeatures",
    "FundingPrediction",
    "FundingPredictor",
    "FundingHistoryTracker",
    # Phase 5 GOD MODE - News Events
    "EconomicEvent",
    "EventImpact",
    "EventType",
    "MarketState",
    "EventCheckResult",
    "EconomicCalendar",
    "NewsEventDetector",
    # Phase 5 GOD MODE - Attribution
    "SignalSnapshot",
    "SignalType",
    "TradeAttribution",
    "SignalPerformance",
    "AttributionReport",
    "AttributionTracker",
    "AttributionAnalyzer",
    # Phase 5 GOD MODE - Backtest
    "Candle",
    "HistoricalDataPoint",
    "BacktestTrade",
    "BacktestResult",
    "BacktestConfig",
    "HyperliquidHistoricalData",
    "DataCache",
    "BacktestEngine",
    "BacktestMetrics",
]

__version__ = "2.12"
