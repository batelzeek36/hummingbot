# Dollar-A-Day Market Making Strategy

## Overview
Conservative market making strategy designed for consistent small profits ($1/day target).
Uses KuCoin paper trading for validation before real funds.

## Files
- **Strategy**: `scripts/dollar_a_day_pmm.py`
- **Config**: `conf/scripts/conf_dollar_a_day_pmm.yml`

## Current Configuration
```yaml
exchange: kucoin_paper_trade
trading_pair: ATOM-USDT
order_amount: 5
base_bid_spread: 0.003  # 0.3%
base_ask_spread: 0.003  # 0.3%
order_refresh_time: 30
inventory_skew_enabled: true
target_base_ratio: 0.5
max_inventory_ratio: 0.8
```

## Key Features
1. **Inventory Skew Management**: Adjusts spreads based on holdings to prevent imbalance
2. **Max Inventory Limits**: Stops buying/selling when too imbalanced (80% threshold)
3. **P&L Tracking**: Tracks buy/sell volumes and estimated profit
4. **Status Dashboard**: Shows balances, orders, spreads, and trading metrics

## Fixes Applied (Jan 2025)
1. Added default `markets` class variable (required by ScriptStrategyBase)
2. Made `config` parameter optional in `__init__` with default fallback
3. Capped spread adjustment to ±0.5% max to prevent extreme spreads
4. Capped ask spread to 1% max
5. Changed rate oracle from Binance to KuCoin (Binance geo-restricted, HTTP 451)

## How to Run
```bash
# In Docker container
start --script dollar_a_day_pmm.py
status  # View dashboard
stop    # Stop strategy
```

## Rate Oracle Fix
In `conf/conf_client.yml`:
```yaml
rate_oracle_source:
  name: kucoin  # Changed from binance due to geo-restriction
```

## Paper Trading Status
- Initial test: Strategy runs successfully
- Orders placed: Bid and Ask around mid-price
- Spread adjustment working correctly
- Next step: Run for extended period to validate profitability

## Analytics Module

**Location**: `scripts/analytics/performance_tracker.py`

Automatically tracks all performance metrics from Hummingbot's trade database.

### How to Run
```bash
# From project root (outside Docker)
python scripts/analytics/performance_tracker.py --strategy dollar_a_day --days 7 --print

# Options
--strategy NAME    Filter by strategy
--market NAME      Filter by exchange  
--pair PAIR        Filter by trading pair
--days N           Analyze last N days (default: 30)
--format FORMAT    json, markdown, or both (default: both)
--print            Print report to console
```

### Metrics Tracked
- **P&L**: Total, realized, unrealized, per-trade average
- **Returns**: Total %, daily %, best/worst day
- **Risk**: Sharpe ratio, max drawdown, drawdown duration
- **Trade Quality**: Win rate, profit factor, spread captured
- **Volume**: Buy/sell volumes, fees paid

### Output Files
Reports saved to `data/reports/`:
- `performance_YYYYMMDD_HHMMSS.json` (machine-readable)
- `performance_YYYYMMDD_HHMMSS.md` (human-readable)

---

## Graduation Path
1. Paper trade for 1 week
2. Analyze fills and P&L with analytics module
3. If profitable, switch to real exchange with small capital ($100-200)
4. Change `exchange: kucoin` (remove _paper_trade suffix)