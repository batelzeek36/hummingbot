# Multi-Strategy Trading Bot

## Overview
Advanced trading bot that runs 3 strategies simultaneously with independent P&L tracking and kill switches.

## Files
- **Script**: `scripts/multi_strategy_bot.py`
- **Config**: `conf/scripts/conf_multi_strategy_bot.yml`

## Strategies Included

### 1. Market Making (MM)
- Places bid/ask orders around mid-price
- Captures spread from volatility
- Inventory skew adjustment
- Expected: 0.3-0.5% daily

### 2. Grid Trading
- Places orders at fixed price levels
- Profits from price oscillation
- Auto-replaces filled orders with opposite side
- Expected: 0.2-0.5% daily

### 3. RSI Directional
- Buys when RSI < 30 (oversold)
- Sells when RSI > 70 or take-profit/stop-loss
- Momentum-based entries
- Expected: 0.5-2% per signal

## Configuration Summary

```yaml
# Trading Pairs
mm_trading_pair: ATOM-USD
grid_trading_pair: ATOM-USD
rsi_trading_pair: SOL-USD

# Capital Allocation
mm_capital_pct: 30
grid_capital_pct: 40
rsi_capital_pct: 30

# Kill Switches
mm_max_loss: 20
grid_max_loss: 30
rsi_max_loss: 25
global_max_loss: 50
global_max_drawdown_pct: 10
daily_loss_limit: 25
```

## How to Run

```bash
start --script multi_strategy_bot.py
status  # View unified dashboard
stop    # Stop all strategies
```

## Status Dashboard Features
- Global P&L and drawdown
- Per-strategy P&L, drawdown, and trade count
- Active order counts per strategy
- Kill switch status indicators
- Runtime tracking

## Kill Switch Behavior
1. **Per-strategy kill**: Stops only that strategy
2. **Global kill**: Stops everything if total loss exceeds limit
3. **Daily pause**: Pauses until midnight if daily loss exceeded
4. **Auto-resume**: Strategies resume on new day after daily pause

## Paper Trade Requirements
In `conf/conf_client.yml`:
```yaml
paper_trade_account_balance:
  ATOM: 100.0
  SOL: 5.0
  USD: 500.0
```
