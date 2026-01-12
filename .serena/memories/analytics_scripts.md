# Analytics Scripts for Hyperliquid Bot

Location: `scripts/analytics/`

## Available Scripts

### 1. quick_strategy_check.py (RECOMMENDED)
**Purpose**: Analyzes all liquid Hyperliquid coins for optimal strategy selection

**Usage**:
```bash
python scripts/analytics/quick_strategy_check.py
python scripts/analytics/quick_strategy_check.py --min-volume 500000
python scripts/analytics/quick_strategy_check.py --include NEWCOIN1,NEWCOIN2
```

**What it does**:
- Fetches all coins from Hyperliquid API
- Filters by volume threshold ($1M default) + force-includes config coins
- Fetches 7-day OHLCV via ccxt for volatility analysis
- Calculates safe leverage per coin
- Scores coins for: Funding, Grid, Momentum strategies
- Compares to current bot config and recommends changes

**Output sections**:
- Funding Harvesting Analysis (ranked by effective APR)
- Grid Trading Analysis (ranked by grid score)
- Momentum Trading Analysis (ranked by momentum score)
- Summary with recommended actions

### 2. funding_rate_checker.py
**Purpose**: Quick check of current funding rates only (no volatility analysis)

**Usage**:
```bash
python scripts/analytics/funding_rate_checker.py
```

**What it does**:
- Fetches funding rates from Hyperliquid API
- Shows APR, direction, effective APR at safe leverage
- Compares memecoins vs majors

### 3. volatility_analysis.py
**Purpose**: Deep volatility analysis using Binance/Bybit data

**Usage**:
```bash
python scripts/analytics/volatility_analysis.py
```

**Note**: Uses external exchanges (Binance/Bybit) for historical data since some coins aren't on Hyperliquid long enough.

## Data Sources

- **Hyperliquid API** (direct): `https://api.hyperliquid.xyz/info`
  - Funding rates, open interest, mark prices, 24h volume
  - Used by: funding_rate_checker.py, quick_strategy_check.py

- **CCXT Hyperliquid**: `ccxt.hyperliquid()`
  - OHLCV candle data for volatility calculation
  - Symbol format: `COIN/USDC:USDC`
  - Used by: quick_strategy_check.py

## When to Run

- **Before deploying**: Run quick_strategy_check.py to verify coins are still optimal
- **Weekly**: Run to check if funding rates have shifted significantly
- **After market events**: Major moves can change volatility classifications
