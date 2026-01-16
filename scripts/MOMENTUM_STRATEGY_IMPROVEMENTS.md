# Momentum Strategy Improvement Ideas

## Current Implementation: RSI Only

The current momentum strategy uses a simple RSI (Relative Strength Index) approach:
- RSI < 25 (oversold) → Go LONG
- RSI > 75 (overbought) → Go SHORT
- Take profit: 2.5%
- Stop loss: 1.5%

### RSI Limitations:
- Lags behind price moves
- Gets destroyed in strong trends (keeps signaling oversold as price falls)
- Everyone uses it = edges arbitraged away
- Ignores volume and market context

---

## Improvement Options (in order of sophistication)

### 1. Multi-Indicator Confirmation (Easy)

Don't trade RSI alone - require 2-3 signals to agree:

```python
# Example: RSI + MACD + EMA
def should_go_long(self):
    rsi_signal = self.rsi < 25
    macd_signal = self.macd_line > self.signal_line  # MACD crossover
    ema_signal = self.price > self.ema_200  # Above long-term trend

    # Require at least 2 of 3 signals
    return sum([rsi_signal, macd_signal, ema_signal]) >= 2
```

**Indicators to consider adding:**
- MACD (Moving Average Convergence Divergence)
- EMA crossovers (9/21 or 12/26)
- Bollinger Bands (price at lower band = oversold)
- Stochastic RSI (RSI of RSI, more sensitive)

---

### 2. Volume-Weighted Signals (Medium)

RSI ignores volume. High volume = stronger signal.

```python
def calculate_volume_confirmed_rsi(self):
    rsi = self.calculate_rsi()
    avg_volume = self.get_average_volume(periods=20)
    current_volume = self.get_current_volume()

    volume_multiplier = current_volume / avg_volume

    # Only trade if volume is above average
    if volume_multiplier < 1.0:
        return None  # Skip weak volume signals

    # Stronger signal with higher volume
    if rsi < 25 and volume_multiplier > 1.5:
        return "strong_long"
    elif rsi < 30 and volume_multiplier > 1.2:
        return "weak_long"

    return None
```

---

### 3. Trend Filter (Medium)

Don't go long in a downtrend, don't go short in an uptrend.

```python
def get_trend(self):
    ema_50 = self.calculate_ema(50)
    ema_200 = self.calculate_ema(200)

    if ema_50 > ema_200:
        return "uptrend"
    else:
        return "downtrend"

def should_trade(self, direction):
    trend = self.get_trend()

    # Only trade WITH the trend
    if direction == "long" and trend == "downtrend":
        return False  # Don't catch falling knives
    if direction == "short" and trend == "uptrend":
        return False  # Don't short strength

    return True
```

---

### 4. Support/Resistance Levels (Medium-Hard)

Trade RSI signals only near key price levels.

```python
def find_support_resistance(self, lookback=100):
    highs = self.get_recent_highs(lookback)
    lows = self.get_recent_lows(lookback)

    # Cluster nearby levels
    resistance_levels = self.cluster_levels(highs)
    support_levels = self.cluster_levels(lows)

    return support_levels, resistance_levels

def near_support(self, price, support_levels, threshold=0.01):
    for level in support_levels:
        if abs(price - level) / level < threshold:
            return True
    return False
```

---

### 5. Funding Rate Integration (Recommended)

Use Hyperliquid's funding rate as a sentiment indicator:

```python
def get_funding_sentiment(self, pair):
    funding_rate = self.connector.get_funding_info(pair).rate

    # Positive funding = longs paying shorts = too many longs = bearish
    # Negative funding = shorts paying longs = too many shorts = bullish

    if funding_rate > 0.0001:  # 0.01% = very positive
        return "bearish"  # Crowded long, expect pullback
    elif funding_rate < -0.0001:
        return "bullish"  # Crowded short, expect squeeze
    else:
        return "neutral"

def enhanced_entry_signal(self):
    rsi = self.calculate_rsi()
    funding_sentiment = self.get_funding_sentiment(self.pair)

    # RSI oversold + negative funding = strong long signal
    if rsi < 25 and funding_sentiment == "bullish":
        return "strong_long"

    # RSI overbought + positive funding = strong short signal
    if rsi > 75 and funding_sentiment == "bearish":
        return "strong_short"

    return None
```

---

### 6. Liquidation Level Awareness (Advanced)

Hyperliquid publishes liquidation data. Large liquidation clusters act as magnets.

```python
def get_liquidation_levels(self):
    # Would need to fetch from Hyperliquid API or third-party
    # Large clusters of liquidations above = resistance (shorts get liquidated)
    # Large clusters below = support (longs get liquidated)
    pass
```

---

### 7. Machine Learning Approach (Hard)

Train a model on historical data:

```python
# Features to consider:
features = [
    'rsi_14',
    'rsi_7',
    'macd',
    'macd_signal',
    'volume_ratio',
    'price_vs_ema_50',
    'price_vs_ema_200',
    'funding_rate',
    'open_interest_change',
    'hour_of_day',  # Time-based patterns
    'day_of_week',
]

# Target: price change in next N candles
# Model: XGBoost, LSTM, or simple logistic regression
```

**Warning:** Overfitting is a major risk. Always use walk-forward validation.

---

## Recommended Priority

1. **First:** Add trend filter (don't trade against the trend)
2. **Second:** Add volume confirmation
3. **Third:** Integrate funding rate sentiment
4. **Fourth:** Add MACD or EMA crossover confirmation

These four additions would significantly improve the strategy without overcomplicating it.

---

## Implementation Notes

- All indicators should use the same price data source
- Consider using `ta-lib` or `pandas-ta` for indicator calculations
- Backtest thoroughly before live trading
- Start with paper trading to validate

---

## Files to Modify

- `/scripts/hyperliquid_monster/strategies/momentum.py` - Main strategy logic
- `/scripts/hyperliquid_monster/config.py` - Add new config parameters
- Consider creating `/scripts/hyperliquid_monster/indicators.py` for reusable indicator calculations
