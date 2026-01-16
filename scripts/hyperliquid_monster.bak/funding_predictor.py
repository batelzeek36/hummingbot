"""
Funding Rate Prediction Model for Hyperliquid Monster Bot v2.11

GOD MODE Phase 4: ML-Based Funding Prediction

Predicts funding rate 1 hour ahead using current market state.
Allows entering funding positions BEFORE the rate spikes.

Current Flow (reactive):
    Funding changes -> You detect it -> You enter position -> Collect funding

God Mode Flow (predictive):
    ML predicts funding spike -> You enter BEFORE spike -> Collect MORE funding

Features used for prediction:
- OI change (1h, 4h)
- Premium/discount
- Premium velocity
- Volume (24h, ratio)
- Price change (1h)
- Current funding rate
- Hour of day (funding has time patterns)

Based on research showing 67%+ accuracy on crypto funding prediction.

Author: Dollar-A-Day Project
Date: 2026-01-12
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
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
class FundingFeatures:
    """Features for funding rate prediction."""
    timestamp: float
    symbol: str

    # OI features
    oi_change_1h: float       # % change in last hour
    oi_change_4h: float       # % change in last 4 hours

    # Premium features
    premium: float            # Current premium/discount %
    premium_velocity: float   # Rate of change

    # Volume features
    volume_24h: float         # 24h volume (normalized)
    volume_ratio: float       # Current vs average

    # Price features
    price_change_1h: float    # % price change in last hour

    # Current funding
    current_funding: float    # Current funding rate

    # Time features
    hour_of_day: int          # 0-23
    hour_sin: float           # sin(hour * 2pi / 24)
    hour_cos: float           # cos(hour * 2pi / 24)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.oi_change_1h,
            self.oi_change_4h,
            self.premium,
            self.premium_velocity,
            self.volume_24h,
            self.volume_ratio,
            self.price_change_1h,
            self.current_funding,
            self.hour_sin,
            self.hour_cos,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "oi_change_1h",
            "oi_change_4h",
            "premium",
            "premium_velocity",
            "volume_24h",
            "volume_ratio",
            "price_change_1h",
            "current_funding",
            "hour_sin",
            "hour_cos",
        ]


@dataclass
class FundingPrediction:
    """Funding rate prediction result."""
    predicted_funding_1h: float     # Predicted funding rate in 1 hour
    predicted_apr: float            # Predicted annualized rate
    current_funding: float          # Current funding rate
    current_apr: float              # Current annualized rate
    change_direction: str           # "increasing", "decreasing", "stable"
    change_magnitude: float         # % change predicted
    confidence: float               # 0-100
    should_enter_early: bool        # True if predicted > current significantly
    reasoning: str


class FundingHistoryTracker:
    """
    Tracks historical funding data for model training.

    Stores funding rates, OI, premium, and volume data over time.
    """

    def __init__(
        self,
        history_size: int = 1000,
        logger: Optional[logging.Logger] = None,
    ):
        self.history_size = history_size
        self.logger = logger or logging.getLogger(__name__)

        # Historical data per symbol
        self._funding_history: Dict[str, Deque[Tuple[float, float]]] = {}  # timestamp, rate
        self._oi_history: Dict[str, Deque[Tuple[float, float]]] = {}       # timestamp, oi
        self._premium_history: Dict[str, Deque[Tuple[float, float]]] = {}  # timestamp, premium
        self._price_history: Dict[str, Deque[Tuple[float, float]]] = {}    # timestamp, price
        self._volume_history: Dict[str, Deque[Tuple[float, float]]] = {}   # timestamp, volume

    def update(
        self,
        symbol: str,
        funding_rate: float,
        oi: Optional[float] = None,
        premium: Optional[float] = None,
        price: Optional[float] = None,
        volume: Optional[float] = None,
    ):
        """Update historical data for a symbol."""
        now = time.time()

        # Initialize if needed
        if symbol not in self._funding_history:
            self._funding_history[symbol] = deque(maxlen=self.history_size)
            self._oi_history[symbol] = deque(maxlen=self.history_size)
            self._premium_history[symbol] = deque(maxlen=self.history_size)
            self._price_history[symbol] = deque(maxlen=self.history_size)
            self._volume_history[symbol] = deque(maxlen=self.history_size)

        self._funding_history[symbol].append((now, funding_rate))

        if oi is not None:
            self._oi_history[symbol].append((now, oi))
        if premium is not None:
            self._premium_history[symbol].append((now, premium))
        if price is not None:
            self._price_history[symbol].append((now, price))
        if volume is not None:
            self._volume_history[symbol].append((now, volume))

    def get_oi_change(self, symbol: str, hours: float) -> float:
        """Get OI % change over the last N hours."""
        if symbol not in self._oi_history:
            return 0.0

        history = self._oi_history[symbol]
        if len(history) < 2:
            return 0.0

        cutoff = time.time() - (hours * 3600)
        current = history[-1][1]

        # Find value from N hours ago
        past_val = current
        for ts, val in reversed(history):
            if ts < cutoff:
                past_val = val
                break

        if past_val == 0:
            return 0.0

        return ((current - past_val) / past_val) * 100

    def get_premium_velocity(self, symbol: str, periods: int = 10) -> float:
        """Get rate of change in premium."""
        if symbol not in self._premium_history:
            return 0.0

        history = list(self._premium_history[symbol])
        if len(history) < periods + 1:
            return 0.0

        # Simple linear regression slope
        recent = [p for _, p in history[-periods:]]
        if len(recent) < 2:
            return 0.0

        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        return slope * 100  # Normalize

    def get_price_change(self, symbol: str, hours: float) -> float:
        """Get price % change over the last N hours."""
        if symbol not in self._price_history:
            return 0.0

        history = self._price_history[symbol]
        if len(history) < 2:
            return 0.0

        cutoff = time.time() - (hours * 3600)
        current = history[-1][1]

        # Find value from N hours ago
        past_val = current
        for ts, val in reversed(history):
            if ts < cutoff:
                past_val = val
                break

        if past_val == 0:
            return 0.0

        return ((current - past_val) / past_val) * 100

    def get_volume_ratio(self, symbol: str, lookback: int = 20) -> float:
        """Get current volume vs average volume."""
        if symbol not in self._volume_history:
            return 1.0

        history = [v for _, v in self._volume_history[symbol]]
        if len(history) < lookback:
            return 1.0

        avg = np.mean(history[-lookback:])
        current = history[-1]

        if avg == 0:
            return 1.0

        return current / avg

    def get_current(self, symbol: str, field: str) -> Optional[float]:
        """Get current value of a field."""
        history_map = {
            "funding": self._funding_history,
            "oi": self._oi_history,
            "premium": self._premium_history,
            "price": self._price_history,
            "volume": self._volume_history,
        }

        history = history_map.get(field, {}).get(symbol)
        if history and len(history) > 0:
            return history[-1][1]
        return None


class FundingPredictor:
    """
    Predicts funding rates 1 hour ahead.

    Uses XGBoost regression trained on historical funding patterns.
    Allows entering positions BEFORE funding spikes.
    """

    DEFAULT_MODEL_PATH = "hyperliquid_monster_funding_model.json"
    DEFAULT_DATA_PATH = "hyperliquid_monster_funding_data.json"

    def __init__(
        self,
        min_training_samples: int = 100,
        prediction_horizon_hours: float = 1.0,
        entry_threshold_multiplier: float = 1.5,
        model_path: Optional[str] = None,
        data_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize Funding Predictor.

        Args:
            min_training_samples: Minimum samples before model is used
            prediction_horizon_hours: How far ahead to predict
            entry_threshold_multiplier: Enter if predicted > current * this
            model_path: Path to save/load model
            data_path: Path to save training data
            logger: Logger instance
        """
        self.min_training_samples = min_training_samples
        self.prediction_horizon = prediction_horizon_hours
        self.entry_threshold = entry_threshold_multiplier
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.data_path = data_path or self.DEFAULT_DATA_PATH
        self.logger = logger or logging.getLogger(__name__)

        # Model state
        self.model: Optional[Any] = None
        self.is_trained = False
        self.training_samples = 0

        # History tracker
        self.history = FundingHistoryTracker(logger=logger)

        # Training data storage
        self._training_data: List[Tuple[FundingFeatures, float]] = []  # (features, actual_1h_funding)
        self._pending_predictions: Dict[str, Tuple[FundingFeatures, float]] = {}  # symbol -> (features, timestamp)

        # Stats
        self.predictions_made = 0
        self.mean_absolute_error: Optional[float] = None

        # Load existing model
        self._load_model()

        if not XGBOOST_AVAILABLE:
            self.logger.warning(
                "XGBoost not installed. Funding prediction disabled. "
                "Install with: pip install xgboost"
            )

    def _load_model(self):
        """Load trained model from disk."""
        if not XGBOOST_AVAILABLE:
            return

        try:
            if os.path.exists(self.model_path):
                self.model = xgb.XGBRegressor()
                self.model.load_model(self.model_path)
                self.is_trained = True
                self.logger.info(f"FUNDING PREDICTOR: Loaded model from {self.model_path}")
        except Exception as e:
            self.logger.warning(f"FUNDING PREDICTOR: Could not load model - {e}")

    def _save_model(self):
        """Save trained model to disk."""
        if not XGBOOST_AVAILABLE or self.model is None:
            return

        try:
            self.model.save_model(self.model_path)
            self.logger.debug(f"FUNDING PREDICTOR: Saved model to {self.model_path}")
        except Exception as e:
            self.logger.warning(f"FUNDING PREDICTOR: Could not save model - {e}")

    def update(
        self,
        symbol: str,
        funding_rate: float,
        oi: Optional[float] = None,
        premium: Optional[float] = None,
        price: Optional[float] = None,
        volume: Optional[float] = None,
    ):
        """
        Update with new market data.

        Call this regularly with current funding rate and market state.
        The predictor will track history and collect training data.
        """
        self.history.update(symbol, funding_rate, oi, premium, price, volume)

        # Check if any pending predictions can now be evaluated
        self._check_pending_predictions(symbol, funding_rate)

        # Create a prediction snapshot for future evaluation
        features = self._extract_features(symbol)
        if features:
            self._pending_predictions[symbol] = (features, time.time())

    def _check_pending_predictions(self, symbol: str, actual_funding: float):
        """Check if any pending predictions are ready to be evaluated."""
        if symbol not in self._pending_predictions:
            return

        features, timestamp = self._pending_predictions[symbol]
        elapsed = time.time() - timestamp

        # If enough time has passed, we can evaluate the prediction
        if elapsed >= self.prediction_horizon * 3600:
            self._training_data.append((features, actual_funding))
            del self._pending_predictions[symbol]

            # Retrain periodically
            if len(self._training_data) >= self.min_training_samples:
                if len(self._training_data) % 20 == 0:
                    self._retrain()

    def _extract_features(self, symbol: str) -> Optional[FundingFeatures]:
        """Extract features for a symbol."""
        current_funding = self.history.get_current(symbol, "funding")
        if current_funding is None:
            return None

        premium = self.history.get_current(symbol, "premium") or 0.0
        volume = self.history.get_current(symbol, "volume") or 0.0

        now = datetime.now()
        hour = now.hour

        return FundingFeatures(
            timestamp=time.time(),
            symbol=symbol,
            oi_change_1h=self.history.get_oi_change(symbol, 1.0),
            oi_change_4h=self.history.get_oi_change(symbol, 4.0),
            premium=premium,
            premium_velocity=self.history.get_premium_velocity(symbol),
            volume_24h=volume,
            volume_ratio=self.history.get_volume_ratio(symbol),
            price_change_1h=self.history.get_price_change(symbol, 1.0),
            current_funding=current_funding,
            hour_of_day=hour,
            hour_sin=float(np.sin(2 * np.pi * hour / 24)),
            hour_cos=float(np.cos(2 * np.pi * hour / 24)),
        )

    def _retrain(self):
        """Retrain model with accumulated data."""
        if not XGBOOST_AVAILABLE:
            return

        if len(self._training_data) < self.min_training_samples:
            return

        try:
            X = np.array([f.to_array() for f, _ in self._training_data])
            y = np.array([actual for _, actual in self._training_data])

            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='reg:squarederror',
                random_state=42,
            )
            self.model.fit(X, y)

            self.is_trained = True
            self.training_samples = len(self._training_data)
            self._save_model()

            # Calculate MAE on training data
            predictions = self.model.predict(X)
            self.mean_absolute_error = float(np.mean(np.abs(predictions - y)))

            self.logger.info(
                f"FUNDING PREDICTOR: Retrained with {self.training_samples} samples "
                f"(MAE: {self.mean_absolute_error:.6f})"
            )

        except Exception as e:
            self.logger.error(f"FUNDING PREDICTOR: Training failed - {e}")

    def predict(self, symbol: str) -> Optional[FundingPrediction]:
        """
        Predict funding rate 1 hour ahead.

        Args:
            symbol: Trading pair

        Returns:
            FundingPrediction or None if not enough data
        """
        self.predictions_made += 1

        features = self._extract_features(symbol)
        if features is None:
            return None

        current = features.current_funding
        current_apr = current * 3 * 365 * 100  # Hourly -> APR

        # If model not ready, return current + trend estimate
        if not XGBOOST_AVAILABLE or not self.is_trained or self.model is None:
            # Simple heuristic: premium predicts funding direction
            trend = features.premium * 0.1  # Premium roughly predicts funding
            predicted = current + trend * current

            return FundingPrediction(
                predicted_funding_1h=predicted,
                predicted_apr=predicted * 3 * 365 * 100,
                current_funding=current,
                current_apr=current_apr,
                change_direction="increasing" if predicted > current else "decreasing",
                change_magnitude=abs(predicted - current) / max(abs(current), 0.0001) * 100,
                confidence=30.0,  # Low confidence without ML
                should_enter_early=False,
                reasoning="ML model not trained - using heuristic",
            )

        try:
            X = features.to_array().reshape(1, -1)
            predicted = float(self.model.predict(X)[0])
            predicted_apr = predicted * 3 * 365 * 100

            # Determine direction
            if predicted > current * 1.1:
                direction = "increasing"
            elif predicted < current * 0.9:
                direction = "decreasing"
            else:
                direction = "stable"

            # Calculate change magnitude
            change_pct = abs(predicted - current) / max(abs(current), 0.0001) * 100

            # Confidence based on model MAE
            confidence = 70.0  # Default
            if self.mean_absolute_error is not None:
                # Lower MAE = higher confidence
                confidence = max(30, min(90, 90 - self.mean_absolute_error * 10000))

            # Should enter early?
            should_enter = abs(predicted) > abs(current) * self.entry_threshold

            reasoning = (
                f"Current: {current:.6f} ({current_apr:+.1f}% APR) -> "
                f"Predicted: {predicted:.6f} ({predicted_apr:+.1f}% APR) "
                f"[{direction}, {change_pct:.1f}% change]"
            )

            return FundingPrediction(
                predicted_funding_1h=predicted,
                predicted_apr=predicted_apr,
                current_funding=current,
                current_apr=current_apr,
                change_direction=direction,
                change_magnitude=change_pct,
                confidence=confidence,
                should_enter_early=should_enter,
                reasoning=reasoning,
            )

        except Exception as e:
            self.logger.warning(f"FUNDING PREDICTOR: Prediction failed - {e}")
            return None

    def should_enter_early(self, symbol: str, min_predicted_apr: float = 30.0) -> Tuple[bool, str]:
        """
        Check if we should enter a funding position early.

        Args:
            symbol: Trading pair
            min_predicted_apr: Minimum predicted APR to consider

        Returns:
            Tuple of (should_enter, reason)
        """
        prediction = self.predict(symbol)
        if prediction is None:
            return False, "No prediction available"

        if prediction.predicted_apr < min_predicted_apr:
            return False, f"Predicted APR {prediction.predicted_apr:.1f}% below threshold {min_predicted_apr}%"

        if prediction.should_enter_early:
            return True, (
                f"Funding predicted to increase: {prediction.current_apr:.1f}% -> "
                f"{prediction.predicted_apr:.1f}% APR ({prediction.confidence:.0f}% confidence)"
            )

        return False, f"Predicted APR {prediction.predicted_apr:.1f}% not significantly higher than current"

    def get_status(self) -> Dict[str, Any]:
        """Get predictor status for display."""
        return {
            "xgboost_available": XGBOOST_AVAILABLE,
            "is_trained": self.is_trained,
            "training_samples": self.training_samples,
            "min_samples_required": self.min_training_samples,
            "predictions_made": self.predictions_made,
            "mean_absolute_error": self.mean_absolute_error,
            "pending_evaluations": len(self._pending_predictions),
            "history_symbols": list(self.history._funding_history.keys()),
        }
