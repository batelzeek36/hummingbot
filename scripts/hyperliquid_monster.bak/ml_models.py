"""
Machine Learning Models for Hyperliquid Monster Bot v2.11

GOD MODE Phase 4: ML-Based Signal Confirmation

Features:
- XGBoost Signal Confirmation: Scores entry setups, filters false positives
- Feature Engineering: Converts all indicators to ML-ready features
- Online Learning: Model improves from each trade outcome
- Trade Outcome Tracking: Logs features and outcomes for training

The ML model acts as a final gate AFTER all other signals pass.
It doesn't generate signals - it confirms/rejects them.

Note: Uses pickle for model persistence (standard ML practice for local models).
The model file is created and loaded locally by the bot.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import json
import logging
import os
import pickle  # Used for XGBoost model persistence (standard ML practice)
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# Try to import XGBoost - graceful fallback if not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None


@dataclass
class MLFeatures:
    """
    Feature vector for ML model input.

    All features are normalized to similar scales for better model performance.
    """
    timestamp: float
    symbol: str
    direction: str  # "long" or "short"

    # Core indicators (normalized)
    rsi: float                    # 0-100
    rsi_distance_from_extreme: float  # How far from oversold/overbought
    trend_strength: float         # -100 to +100 (EMA separation)
    macd_histogram: float         # Normalized
    volume_ratio: float           # Current/average

    # Leading indicators
    oi_change_pct: float          # % change
    oi_momentum_score: float      # -100 to +100
    premium_pct: float            # Premium/discount %
    funding_rate: float           # Current funding (normalized)
    funding_velocity: float       # Rate of change

    # Phase 2: Coinglass
    cvd_direction: float          # -1, 0, +1
    cvd_divergence_strength: float  # 0-100
    long_short_ratio: float       # Normalized around 50
    liq_imbalance: float          # -1 to +1 (negative = more long liqs)

    # Phase 3: Multi-timeframe
    mtf_weighted_score: float     # -100 to +100
    mtf_confluence_count: int     # 0-4
    mtf_1h_direction: float       # -1, 0, +1

    # Holy Grail
    holy_grail_confidence: float  # 0-100
    holy_grail_direction: float   # -1, 0, +1

    # Time features (cyclical)
    hour_sin: float               # sin(hour * 2pi / 24)
    hour_cos: float               # cos(hour * 2pi / 24)
    day_of_week: int              # 0-6

    # Market context
    btc_trend: float              # -1, 0, +1 (BTC direction affects alts)
    volatility_regime: float      # Low/medium/high (0, 0.5, 1)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.rsi,
            self.rsi_distance_from_extreme,
            self.trend_strength,
            self.macd_histogram,
            self.volume_ratio,
            self.oi_change_pct,
            self.oi_momentum_score,
            self.premium_pct,
            self.funding_rate,
            self.funding_velocity,
            self.cvd_direction,
            self.cvd_divergence_strength,
            self.long_short_ratio,
            self.liq_imbalance,
            self.mtf_weighted_score,
            self.mtf_confluence_count,
            self.mtf_1h_direction,
            self.holy_grail_confidence,
            self.holy_grail_direction,
            self.hour_sin,
            self.hour_cos,
            self.day_of_week,
            self.btc_trend,
            self.volatility_regime,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model interpretation."""
        return [
            "rsi",
            "rsi_distance_from_extreme",
            "trend_strength",
            "macd_histogram",
            "volume_ratio",
            "oi_change_pct",
            "oi_momentum_score",
            "premium_pct",
            "funding_rate",
            "funding_velocity",
            "cvd_direction",
            "cvd_divergence_strength",
            "long_short_ratio",
            "liq_imbalance",
            "mtf_weighted_score",
            "mtf_confluence_count",
            "mtf_1h_direction",
            "holy_grail_confidence",
            "holy_grail_direction",
            "hour_sin",
            "hour_cos",
            "day_of_week",
            "btc_trend",
            "volatility_regime",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "direction": self.direction,
            "features": dict(zip(self.feature_names(), self.to_array().tolist())),
        }


@dataclass
class TradeOutcome:
    """Recorded trade outcome for online learning."""
    features: MLFeatures
    entry_price: float
    exit_price: float
    pnl_pct: float
    win: bool  # True if profitable
    exit_reason: str  # "take_profit", "stop_loss", "signal_exit", etc.
    hold_time_seconds: float


@dataclass
class MLPrediction:
    """ML model prediction result."""
    should_take_trade: bool
    confidence: float           # 0-100
    win_probability: float      # 0-1
    expected_pnl_score: float   # Model's score
    feature_importances: Dict[str, float]  # Top contributing features
    reasoning: str


class FeatureExtractor:
    """
    Extracts ML features from strategy indicators.

    Normalizes all features to similar scales for model performance.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._btc_ema_short: Optional[float] = None
        self._btc_ema_long: Optional[float] = None
        self._btc_prices: Deque[float] = deque(maxlen=200)

    def update_btc_price(self, price: float):
        """Update BTC price for market context."""
        self._btc_prices.append(price)

        if len(self._btc_prices) >= 50:
            prices = list(self._btc_prices)
            self._btc_ema_short = self._ema(prices, 20)
            self._btc_ema_long = self._ema(prices, 50)

    def _ema(self, prices: List[float], period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1]

        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def get_btc_trend(self) -> float:
        """Get BTC trend direction (-1, 0, +1)."""
        if self._btc_ema_short is None or self._btc_ema_long is None:
            return 0.0

        diff_pct = (self._btc_ema_short - self._btc_ema_long) / self._btc_ema_long * 100

        if diff_pct > 0.5:
            return 1.0
        elif diff_pct < -0.5:
            return -1.0
        return 0.0

    def extract_features(
        self,
        symbol: str,
        direction: str,
        rsi: float,
        trend_info: Optional[Any],
        macd_info: Optional[Any],
        volume_info: Optional[Any],
        oi_analysis: Optional[Any],
        holy_grail: Optional[Any],
        mtf_confluence: Optional[Any],
        liq_heatmap: Optional[Any],
        cvd_divergence: Optional[Any],
        funding_rate: Optional[float] = None,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ) -> MLFeatures:
        """
        Extract normalized features from all available indicators.

        Args:
            All the indicator objects from the strategy

        Returns:
            MLFeatures object ready for model input
        """
        now = datetime.now()

        # RSI features
        if direction == "long":
            rsi_distance = rsi - rsi_oversold  # Negative when oversold
        else:
            rsi_distance = rsi_overbought - rsi  # Negative when overbought

        # Trend features
        trend_strength = 0.0
        if trend_info:
            trend_strength = getattr(trend_info, 'strength', 0.0)
            if trend_info.direction == "downtrend":
                trend_strength = -abs(trend_strength)

        # MACD features
        macd_histogram = 0.0
        if macd_info:
            macd_histogram = getattr(macd_info, 'histogram', 0.0)
            # Normalize to roughly -100 to +100
            macd_histogram = max(-100, min(100, macd_histogram * 10000))

        # Volume features
        volume_ratio = 1.0
        if volume_info:
            volume_ratio = getattr(volume_info, 'volume_ratio', 1.0)

        # OI features
        oi_change_pct = 0.0
        oi_momentum_score = 0.0
        if oi_analysis:
            oi_change_pct = getattr(oi_analysis, 'oi_change_pct', 0.0)
            # Convert momentum enum to score
            momentum = getattr(oi_analysis, 'momentum', None)
            if momentum:
                momentum_str = str(momentum.value) if hasattr(momentum, 'value') else str(momentum)
                if 'bullish' in momentum_str.lower():
                    oi_momentum_score = 50.0
                elif 'bearish' in momentum_str.lower():
                    oi_momentum_score = -50.0

        # Holy Grail features
        hg_confidence = 0.0
        hg_direction = 0.0
        premium_pct = 0.0
        funding_velocity = 0.0
        if holy_grail:
            hg_confidence = getattr(holy_grail, 'confidence', 0.0)
            dir_str = getattr(holy_grail, 'direction', 'neutral')
            if 'long' in dir_str:
                hg_direction = 1.0
            elif 'short' in dir_str:
                hg_direction = -1.0
            premium_pct = getattr(holy_grail, 'premium_score', 0.0)
            funding_velocity = getattr(holy_grail, 'funding_velocity_score', 0.0)

        # MTF Confluence features
        mtf_score = 0.0
        mtf_count = 0
        mtf_1h = 0.0
        if mtf_confluence:
            mtf_score = getattr(mtf_confluence, 'weighted_score', 0.0)
            mtf_count = getattr(mtf_confluence, 'bullish_count', 0) + getattr(mtf_confluence, 'bearish_count', 0)
            signals = getattr(mtf_confluence, 'signals', {})
            if '1h' in signals:
                trend_val = signals['1h'].trend.value if hasattr(signals['1h'].trend, 'value') else str(signals['1h'].trend)
                if 'bull' in trend_val.lower():
                    mtf_1h = 1.0
                elif 'bear' in trend_val.lower():
                    mtf_1h = -1.0

        # CVD features
        cvd_dir = 0.0
        cvd_strength = 0.0
        if cvd_divergence:
            div_type = getattr(cvd_divergence, 'divergence_type', '')
            if 'bullish' in str(div_type).lower():
                cvd_dir = 1.0
            elif 'bearish' in str(div_type).lower():
                cvd_dir = -1.0
            cvd_strength = getattr(cvd_divergence, 'signal_strength', 0.0)

        # Liquidation features
        liq_imbalance = 0.0
        if liq_heatmap:
            long_total = sum(c.estimated_size_usd for c in getattr(liq_heatmap, 'long_clusters', [])[:3])
            short_total = sum(c.estimated_size_usd for c in getattr(liq_heatmap, 'short_clusters', [])[:3])
            total = long_total + short_total
            if total > 0:
                # -1 = all long liqs (bearish), +1 = all short liqs (bullish)
                liq_imbalance = (short_total - long_total) / total

        # Long/short ratio (placeholder - would come from Coinglass)
        ls_ratio = 50.0  # Neutral

        # Funding rate
        funding = funding_rate if funding_rate else 0.0
        # Normalize to roughly -100 to +100
        funding_normalized = max(-100, min(100, funding * 100000))

        # Time features (cyclical encoding)
        hour = now.hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_of_week = now.weekday()

        # Volatility regime (simplified)
        volatility_regime = 0.5  # Medium by default
        if volume_ratio > 2.0:
            volatility_regime = 1.0
        elif volume_ratio < 0.5:
            volatility_regime = 0.0

        return MLFeatures(
            timestamp=time.time(),
            symbol=symbol,
            direction=direction,
            rsi=rsi,
            rsi_distance_from_extreme=rsi_distance,
            trend_strength=trend_strength,
            macd_histogram=macd_histogram,
            volume_ratio=volume_ratio,
            oi_change_pct=oi_change_pct,
            oi_momentum_score=oi_momentum_score,
            premium_pct=premium_pct,
            funding_rate=funding_normalized,
            funding_velocity=funding_velocity,
            cvd_direction=cvd_dir,
            cvd_divergence_strength=cvd_strength,
            long_short_ratio=ls_ratio,
            liq_imbalance=liq_imbalance,
            mtf_weighted_score=mtf_score,
            mtf_confluence_count=mtf_count,
            mtf_1h_direction=mtf_1h,
            holy_grail_confidence=hg_confidence,
            holy_grail_direction=hg_direction,
            hour_sin=float(hour_sin),
            hour_cos=float(hour_cos),
            day_of_week=day_of_week,
            btc_trend=self.get_btc_trend(),
            volatility_regime=volatility_regime,
        )


class SignalConfirmationModel:
    """
    XGBoost model to confirm/reject trading signals.

    The model learns from trade outcomes to filter false positives.
    It doesn't generate signals - it acts as a final quality gate.

    Training:
    - Features: All indicator scores + time features
    - Target: Did the trade win (1) or lose (0)?
    - Online learning: Retrained periodically with new outcomes
    """

    DEFAULT_MODEL_PATH = "hyperliquid_monster_ml_model.pkl"
    DEFAULT_DATA_PATH = "hyperliquid_monster_ml_data.json"

    def __init__(
        self,
        min_confidence: float = 60.0,
        min_training_samples: int = 50,
        model_path: Optional[str] = None,
        data_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Signal Confirmation Model.

        Args:
            min_confidence: Minimum confidence to confirm trade (0-100)
            min_training_samples: Minimum trades before model is used
            model_path: Path to save/load model
            data_path: Path to save trade outcomes
            logger: Logger instance
        """
        self.min_confidence = min_confidence
        self.min_training_samples = min_training_samples
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self.logger = logger or logging.getLogger(__name__)

        # Model state
        self.model: Optional[Any] = None
        self.is_trained = False
        self.training_samples = 0

        # Trade outcome storage for online learning
        self.trade_outcomes: List[TradeOutcome] = []
        self.pending_trades: Dict[str, Tuple[MLFeatures, float, float]] = {}  # trade_id -> (features, entry_price, entry_time)

        # Feature extractor
        self.feature_extractor = FeatureExtractor(logger)

        # Stats
        self.predictions_made = 0
        self.trades_confirmed = 0
        self.trades_rejected = 0

        # Load existing model and data
        self._load_model()
        self._load_data()

        if not XGBOOST_AVAILABLE:
            self.logger.warning(
                "XGBoost not installed. ML confirmation disabled. "
                "Install with: pip install xgboost"
            )

    def _load_model(self):
        """Load trained model from disk (pickle used for XGBoost compatibility)."""
        if not XGBOOST_AVAILABLE:
            return

        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    saved = pickle.load(f)
                    self.model = saved.get('model')
                    self.is_trained = saved.get('is_trained', False)
                    self.training_samples = saved.get('training_samples', 0)
                self.logger.info(
                    f"ML MODEL: Loaded from {self.model_path} "
                    f"({self.training_samples} training samples)"
                )
        except Exception as e:
            self.logger.warning(f"ML MODEL: Could not load model - {e}")

    def _save_model(self):
        """Save trained model to disk (pickle used for XGBoost compatibility)."""
        if not XGBOOST_AVAILABLE or self.model is None:
            return

        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'is_trained': self.is_trained,
                    'training_samples': self.training_samples,
                }, f)
            self.logger.debug(f"ML MODEL: Saved to {self.model_path}")
        except Exception as e:
            self.logger.warning(f"ML MODEL: Could not save model - {e}")

    def _load_data(self):
        """Load trade outcome data."""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    # We just load the count, actual outcomes are reconstructed from model training
                    self.training_samples = data.get('total_outcomes', 0)
        except Exception as e:
            self.logger.debug(f"ML MODEL: Could not load data - {e}")

    def _save_data(self, outcome: TradeOutcome):
        """Append trade outcome to data file."""
        try:
            data = {"outcomes": [], "total_outcomes": 0}
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r') as f:
                    data = json.load(f)

            # Append new outcome
            data["outcomes"].append({
                "features": outcome.features.to_dict(),
                "entry_price": outcome.entry_price,
                "exit_price": outcome.exit_price,
                "pnl_pct": outcome.pnl_pct,
                "win": outcome.win,
                "exit_reason": outcome.exit_reason,
                "hold_time": outcome.hold_time_seconds,
            })
            data["total_outcomes"] = len(data["outcomes"])

            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.debug(f"ML MODEL: Could not save outcome - {e}")

    def record_entry(
        self,
        trade_id: str,
        features: MLFeatures,
        entry_price: float,
    ):
        """
        Record trade entry for later outcome tracking.

        Args:
            trade_id: Unique trade identifier
            features: Features at entry time
            entry_price: Entry price
        """
        self.pending_trades[trade_id] = (features, entry_price, time.time())
        self.logger.debug(f"ML MODEL: Recorded entry for trade {trade_id}")

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ):
        """
        Record trade exit and outcome for online learning.

        Args:
            trade_id: Trade identifier from entry
            exit_price: Exit price
            exit_reason: Why trade was closed
        """
        if trade_id not in self.pending_trades:
            self.logger.debug(f"ML MODEL: No entry found for trade {trade_id}")
            return

        features, entry_price, entry_time = self.pending_trades.pop(trade_id)

        # Calculate outcome
        if features.direction == "long":
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        outcome = TradeOutcome(
            features=features,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            win=pnl_pct > 0,
            exit_reason=exit_reason,
            hold_time_seconds=time.time() - entry_time,
        )

        self.trade_outcomes.append(outcome)
        self._save_data(outcome)

        self.logger.info(
            f"ML MODEL: Recorded outcome - {'WIN' if outcome.win else 'LOSS'} "
            f"{pnl_pct:+.2f}% ({exit_reason})"
        )

        # Trigger retraining if enough new samples
        if len(self.trade_outcomes) >= 10 and len(self.trade_outcomes) % 10 == 0:
            self._retrain()

    def _retrain(self):
        """Retrain model with accumulated outcomes."""
        if not XGBOOST_AVAILABLE:
            return

        if len(self.trade_outcomes) < self.min_training_samples:
            self.logger.debug(
                f"ML MODEL: Not enough samples to train "
                f"({len(self.trade_outcomes)}/{self.min_training_samples})"
            )
            return

        try:
            # Prepare training data
            X = np.array([o.features.to_array() for o in self.trade_outcomes])
            y = np.array([1 if o.win else 0 for o in self.trade_outcomes])

            # Train XGBoost classifier
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
            )
            self.model.fit(X, y)

            self.is_trained = True
            self.training_samples = len(self.trade_outcomes)
            self._save_model()

            # Calculate training accuracy
            train_pred = self.model.predict(X)
            accuracy = (train_pred == y).mean() * 100

            self.logger.info(
                f"ML MODEL: Retrained with {self.training_samples} samples "
                f"(accuracy: {accuracy:.1f}%)"
            )

        except Exception as e:
            self.logger.error(f"ML MODEL: Training failed - {e}")

    def predict(self, features: MLFeatures) -> MLPrediction:
        """
        Get model prediction for a potential trade.

        Args:
            features: Extracted ML features

        Returns:
            MLPrediction with confidence and recommendation
        """
        self.predictions_made += 1

        # If model not ready, return default (allow trade)
        if not XGBOOST_AVAILABLE or not self.is_trained or self.model is None:
            return MLPrediction(
                should_take_trade=True,
                confidence=50.0,
                win_probability=0.5,
                expected_pnl_score=0.0,
                feature_importances={},
                reasoning="ML model not trained yet - defaulting to allow",
            )

        try:
            X = features.to_array().reshape(1, -1)

            # Get probability prediction
            proba = self.model.predict_proba(X)[0]
            win_prob = proba[1]  # Probability of winning

            # Convert to confidence (0-100)
            confidence = abs(win_prob - 0.5) * 200  # 0-100 scale

            # Get feature importances for this prediction
            importances = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_names = MLFeatures.feature_names()
                for i, imp in enumerate(self.model.feature_importances_):
                    if imp > 0.05:  # Only significant features
                        importances[feature_names[i]] = float(imp)

            # Decision
            should_take = win_prob >= 0.5 and confidence >= (self.min_confidence / 100)

            if should_take:
                self.trades_confirmed += 1
            else:
                self.trades_rejected += 1

            # Build reasoning
            top_features = sorted(importances.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join([f"{k}:{v:.2f}" for k, v in top_features])

            reasoning = (
                f"Win probability: {win_prob:.1%}, Confidence: {confidence:.0f}% "
                f"[Top features: {top_str}]"
            )

            return MLPrediction(
                should_take_trade=should_take,
                confidence=confidence,
                win_probability=win_prob,
                expected_pnl_score=win_prob * 100 - 50,  # -50 to +50
                feature_importances=importances,
                reasoning=reasoning,
            )

        except Exception as e:
            self.logger.warning(f"ML MODEL: Prediction failed - {e}")
            return MLPrediction(
                should_take_trade=True,
                confidence=50.0,
                win_probability=0.5,
                expected_pnl_score=0.0,
                feature_importances={},
                reasoning=f"Prediction error: {e}",
            )

    def should_confirm_trade(
        self,
        symbol: str,
        direction: str,
        rsi: float,
        trend_info: Optional[Any] = None,
        macd_info: Optional[Any] = None,
        volume_info: Optional[Any] = None,
        oi_analysis: Optional[Any] = None,
        holy_grail: Optional[Any] = None,
        mtf_confluence: Optional[Any] = None,
        liq_heatmap: Optional[Any] = None,
        cvd_divergence: Optional[Any] = None,
        funding_rate: Optional[float] = None,
    ) -> Tuple[bool, str, Optional[MLFeatures]]:
        """
        High-level method to check if ML confirms a trade.

        Args:
            All indicator values from the strategy

        Returns:
            Tuple of (should_take, reason, features)
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            symbol=symbol,
            direction=direction,
            rsi=rsi,
            trend_info=trend_info,
            macd_info=macd_info,
            volume_info=volume_info,
            oi_analysis=oi_analysis,
            holy_grail=holy_grail,
            mtf_confluence=mtf_confluence,
            liq_heatmap=liq_heatmap,
            cvd_divergence=cvd_divergence,
            funding_rate=funding_rate,
        )

        # Get prediction
        prediction = self.predict(features)

        return prediction.should_take_trade, prediction.reasoning, features

    def get_status(self) -> Dict[str, Any]:
        """Get model status for display."""
        return {
            "xgboost_available": XGBOOST_AVAILABLE,
            "is_trained": self.is_trained,
            "training_samples": self.training_samples,
            "min_samples_required": self.min_training_samples,
            "predictions_made": self.predictions_made,
            "trades_confirmed": self.trades_confirmed,
            "trades_rejected": self.trades_rejected,
            "pending_trades": len(self.pending_trades),
            "confirmation_rate": (
                self.trades_confirmed / max(1, self.trades_confirmed + self.trades_rejected) * 100
            ),
        }
