# Option 3: Multi-Coin Strategy Optimization Plan

## Overview

Currently, the `multi_strategy_bot.py` uses XRP-USD for all three strategies. This is suboptimal because different strategies perform best in different market conditions:

| Strategy | Optimal Market Condition | Current XRP Performance |
|----------|-------------------------|------------------------|
| Market Making | High volume, tight spreads, ranging | Suboptimal (trending) |
| Grid Trading | High oscillation, ranging, mean-reverting | Suboptimal (trending) |
| Momentum Breakout | Strong directional trends | Moderate |

**Goal:** Assign each strategy to the coin that best matches its requirements on Kraken.

---

## Task 1: Analyze Kraken Coins

### Coins to Analyze
Pull data for all major USD pairs on Kraken:
- BTC-USD, ETH-USD, XRP-USD, SOL-USD, DOGE-USD
- ADA-USD, DOT-USD, LINK-USD, AVAX-USD, MATIC-USD
- LTC-USD, ATOM-USD, UNI-USD, XLM-USD, ALGO-USD

### Metrics to Calculate (30-day data)

#### For Market Making - Need:
1. **Spread tightness** - Average bid-ask spread %
2. **Volume** - 24h trading volume (higher = better liquidity)
3. **Volatility** - Standard deviation of returns (moderate is best, ~1-3%)
4. **Trend strength** - Absolute value of 30-day return (LOWER is better for MM)

**Scoring Formula for MM:**
```
MM_Score = (Volume_Rank * 0.3) + (Spread_Tightness_Rank * 0.3) +
           (Moderate_Volatility_Rank * 0.2) + (Low_Trend_Rank * 0.2)
```

#### For Grid Trading - Need:
1. **Oscillation score** - Count of times price crosses its 20-period MA
2. **Range-bound behavior** - (High - Low) / Average price over 30 days
3. **Mean reversion** - Correlation of returns with lagged returns (negative = good)
4. **Volatility** - Higher volatility = wider grid spacing opportunities

**Scoring Formula for Grid:**
```
Grid_Score = (Oscillation_Rank * 0.35) + (Range_Bound_Rank * 0.25) +
             (Mean_Reversion_Rank * 0.25) + (Volatility_Rank * 0.15)
```

#### For Momentum Breakout - Need:
1. **Trend strength** - Absolute 30-day return (HIGHER is better)
2. **Momentum persistence** - Autocorrelation of returns (positive = good)
3. **Breakout frequency** - Count of >3% moves in 24h periods
4. **Clean trends** - Low noise ratio (trend / volatility)

**Scoring Formula for Momentum:**
```
Momentum_Score = (Trend_Strength_Rank * 0.35) + (Momentum_Persistence_Rank * 0.25) +
                 (Breakout_Frequency_Rank * 0.25) + (Clean_Trend_Rank * 0.15)
```

---

## Task 2: Python Analysis Script

Create and run this analysis using ccxt:

```python
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize Kraken
exchange = ccxt.kraken()

# Coins to analyze (Kraken symbols)
COINS = [
    'BTC/USD', 'ETH/USD', 'XRP/USD', 'SOL/USD', 'DOGE/USD',
    'ADA/USD', 'DOT/USD', 'LINK/USD', 'AVAX/USD', 'MATIC/USD',
    'LTC/USD', 'ATOM/USD', 'UNI/USD', 'XLM/USD', 'ALGO/USD'
]

def analyze_coin(symbol):
    """Analyze a single coin for all strategies."""
    try:
        # Fetch 30 days of hourly data
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=720)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Calculate metrics
        df['returns'] = df['close'].pct_change()

        metrics = {
            'symbol': symbol,
            # Volume
            'avg_volume_usd': (df['volume'] * df['close']).mean(),
            # Volatility
            'volatility': df['returns'].std() * 100,
            # Trend strength (absolute return over period)
            'trend_strength': abs((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100),
            # Oscillation (MA crossings)
            'ma_20': df['close'].rolling(20).mean(),
        }

        # Count MA crossings
        df['above_ma'] = df['close'] > df['close'].rolling(20).mean()
        metrics['oscillation_count'] = df['above_ma'].diff().abs().sum()

        # Mean reversion (negative autocorrelation = good)
        metrics['autocorr'] = df['returns'].autocorr(lag=1)

        # Breakout frequency (>3% moves in rolling 24h)
        df['rolling_24h_return'] = df['close'].pct_change(24) * 100
        metrics['breakout_count'] = (abs(df['rolling_24h_return']) > 3).sum()

        # Range (high-low / avg)
        metrics['range_pct'] = (df['high'].max() - df['low'].min()) / df['close'].mean() * 100

        return metrics
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

# Analyze all coins
results = []
for coin in COINS:
    print(f"Analyzing {coin}...")
    metrics = analyze_coin(coin)
    if metrics:
        results.append(metrics)

# Create DataFrame and rank
df_results = pd.DataFrame(results)

# Score for Market Making (want: high volume, moderate volatility, LOW trend)
df_results['mm_score'] = (
    df_results['avg_volume_usd'].rank(pct=True) * 30 +
    (1 - abs(df_results['volatility'] - 2).rank(pct=True)) * 30 +  # Prefer ~2% volatility
    (1 - df_results['trend_strength'].rank(pct=True)) * 40  # Prefer LOW trend
)

# Score for Grid (want: high oscillation, high range, negative autocorr)
df_results['grid_score'] = (
    df_results['oscillation_count'].rank(pct=True) * 35 +
    df_results['range_pct'].rank(pct=True) * 25 +
    (1 - df_results['autocorr'].rank(pct=True)) * 25 +  # Prefer negative autocorr
    df_results['volatility'].rank(pct=True) * 15
)

# Score for Momentum (want: high trend, high breakouts, positive autocorr)
df_results['momentum_score'] = (
    df_results['trend_strength'].rank(pct=True) * 35 +
    df_results['breakout_count'].rank(pct=True) * 30 +
    df_results['autocorr'].rank(pct=True) * 20 +  # Prefer positive autocorr
    df_results['volatility'].rank(pct=True) * 15
)

# Print results
print("\n" + "="*70)
print("BEST COINS PER STRATEGY")
print("="*70)

print("\nBEST FOR MARKET MAKING:")
print(df_results.nlargest(3, 'mm_score')[['symbol', 'mm_score', 'avg_volume_usd', 'volatility', 'trend_strength']])

print("\nBEST FOR GRID TRADING:")
print(df_results.nlargest(3, 'grid_score')[['symbol', 'grid_score', 'oscillation_count', 'range_pct', 'autocorr']])

print("\nBEST FOR MOMENTUM BREAKOUT:")
print(df_results.nlargest(3, 'momentum_score')[['symbol', 'momentum_score', 'trend_strength', 'breakout_count', 'autocorr']])
```

---

## Task 3: Update multi_strategy_bot.py

After determining the best coins, update the config:

### Current Config (single coin):
```python
mm_trading_pair: str = Field(default="XRP-USD")
grid_trading_pair: str = Field(default="XRP-USD")
momentum_trading_pair: str = Field(default="XRP-USD")
```

### New Config (optimized per strategy):
```python
# Example - replace with actual analysis results
mm_trading_pair: str = Field(
    default="BTC-USD",  # Best for MM: high volume, tight spread
    description="Trading pair for Market Making"
)
grid_trading_pair: str = Field(
    default="LTC-USD",  # Best for Grid: high oscillation
    description="Trading pair for Grid Trading"
)
momentum_trading_pair: str = Field(
    default="SOL-USD",  # Best for Momentum: strong trends
    description="Trading pair for Momentum Breakout"
)
```

### Update markets definition:
```python
markets = {"kraken": {"BTC-USD", "LTC-USD", "SOL-USD"}}

@classmethod
def init_markets(cls, config: MultiStrategyConfig):
    pairs = {config.mm_trading_pair, config.grid_trading_pair, config.momentum_trading_pair}
    cls.markets = {config.exchange: pairs}
```

---

## Task 4: Capital Allocation Considerations

With multiple coins, consider:

1. **Balance requirements** - Each coin needs minimum balance for orders
2. **Kraken minimums** - Check minimum order sizes for each pair
3. **Capital split** - May need to adjust percentages:
   - MM: 30% → dedicated to MM coin
   - Grid: 40% → dedicated to Grid coin
   - Momentum: 30% → dedicated to Momentum coin

### Important: Check Kraken Minimum Order Sizes
```python
# Kraken minimums (approximate, verify current values)
KRAKEN_MINS = {
    'BTC-USD': 0.0001,  # ~$10 at $100k
    'ETH-USD': 0.004,   # ~$15 at $3.5k
    'XRP-USD': 1.65,    # ~$3.50 at $2.10
    'SOL-USD': 0.02,    # ~$4 at $200
    'LTC-USD': 0.03,    # ~$3 at $100
    'DOGE-USD': 15,     # ~$5 at $0.35
}
```

---

## Task 5: Verification Checklist

After implementation:

- [ ] Run the analysis script to identify best coins
- [ ] Verify Kraken supports all selected pairs
- [ ] Check minimum order sizes for each pair
- [ ] Update config with new trading pairs
- [ ] Update markets definition
- [ ] Test with paper trading or small amounts first
- [ ] Monitor for 24-48 hours before full deployment

---

## Expected Outcome

By matching each strategy to its optimal coin:

| Strategy | Current (XRP) | Expected (Optimized) |
|----------|---------------|---------------------|
| Market Making | ~0% (trending hurts) | +1-2% (ranging coin) |
| Grid Trading | ~0% (trending hurts) | +2-3% (oscillating coin) |
| Momentum | +10% | +10-15% (trending coin) |
| **Total** | **~+5%** | **~+15-20%** |

---

## Files to Modify

1. `/Users/kingkamehameha/Documents/hummingbot/scripts/multi_strategy_bot.py`
   - Update default trading pairs in config
   - Verify markets definition includes all pairs

2. `/Users/kingkamehameha/Documents/hummingbot/scripts/strategies/multi_strategy_bot.py`
   - Same changes (keep in sync)

---

## Summary for Claude Code Instance

1. **Run the analysis script** to find best coins for each strategy on Kraken
2. **Report the findings** with scores and reasoning
3. **Update the config** in multi_strategy_bot.py with the optimal pairs
4. **Verify** Kraken minimum order sizes work with current capital (~$100)
5. **Test syntax** with py_compile

The goal is to maximize returns by letting each strategy trade the coin it performs best on, rather than forcing all strategies onto one coin that may not suit them all.
