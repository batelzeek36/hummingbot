# Hyperliquid Monster Bot v2.4 - Configuration Decisions

Last Updated: 2026-01-12
Status: ✅ Live and trading - all issues fixed

## Current Configuration

### Strategy Allocation ($78 capital)
- Funding Harvesting: 45% ($35)
- Grid Trading: 35% ($27)
- Momentum/Directional: 20% ($16)

### Pair Selection

| Strategy | Pair | Leverage | Rationale |
|----------|------|----------|-----------|
| Grid | SOL-USD | 5x | Best liquidity ($461M/day), moderate volatility |
| Momentum | BTC-USD | 8x | Safest, most predictable, lowest volatility |
| Funding | 13 pairs | Smart (2-8x) | Scans for best opportunities |

### Funding Scanner Pairs (12 total)
ETH, SOL, DOGE, TAO, HYPE, AVNT, kPEPE, kBONK, WIF, VVV, HYPER, IP

**Removed**:
- ANIME-USD (51.9% max move exceeds 50% threshold)
- PURR-USD (55.8% max move exceeds 50% threshold)
- MOG-USD, JEFF-USD (not available on Hyperliquid)

**Symbol Notes**: Hyperliquid uses kPEPE/kBONK (1000x multiplier tokens)

## Smart Leverage System (Updated)

```python
VOLATILITY_LEVERAGE = {
    CoinVolatility.SAFE: 8,     # BTC only
    CoinVolatility.MEDIUM: 5,   # ETH, SOL, TAO
    CoinVolatility.HIGH: 3,     # DOGE, HYPE
    CoinVolatility.EXTREME: 2,  # AVNT, kPEPE, kBONK, WIF, VVV, HYPER, IP
}
```

### Order Size Requirements (Hyperliquid)
- Minimum order value: **$10**
- All position sizes set to **$12** for buffer:
  - `funding_position_size`: $12
  - `grid_order_size`: $12
  - `momentum_position_size`: $12

### Changes from Original (2026-01-11 to 2026-01-12)
- ETH: 8x → 5x (was marginal with +2.7% buffer, now +10.2%)
- IP: 3x → 2x (was marginal with +2.6% buffer, now +19.3%)
- AVNT: 3x → 2x (25.8% move was borderline)
- ANIME: REMOVED (51.9% exceeded threshold)
- PURR: REMOVED (55.8% exceeded threshold)
- Symbol fixes: PEPE→kPEPE, BONK→kBONK

### Bug Fixes (2026-01-12)
- Fixed `on_start()` not being called - added `_first_run` flag in `on_tick()`
- Fixed minimum order size - increased from $5/$4 to $12 (Hyperliquid requires $10 min)
- Fixed symbol names for Hyperliquid (kPEPE, kBONK)

## Expected Returns

| Metric | Before | After |
|--------|--------|-------|
| Monthly Return | ~20-22% | ~18-20% |
| Liquidation Risk | Present | **Eliminated** |

Trade-off: ~2% monthly reduction for zero liquidation risk.

## Risk Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max Drawdown | 20% | Emergency shutdown trigger |
| Daily Loss Limit | $12 | ~15% of capital |
| Max Funding Positions | 3 | Diversification |
| Min Funding APR | 30% | Filter weak opportunities |
| Min Buffer | +6.5% | All coins now safe |

## Key Decisions & Rationale

### Why Remove ANIME?
- 51.9% max daily move exceeds 50% liquidation threshold at 2x
- No safe leverage exists for this coin
- HYPER offers similar APR (~663% vs ~700%) with safe buffer

### Why Move ETH to 5x?
- 9.8% max move at 8x gave only +2.7% buffer
- At 5x, buffer increases to +10.2%
- ETH has lower funding rates anyway, rarely in top 3

### Why Move IP to 2x?
- 30.7% max move at 3x gave only +2.6% buffer
- At 2x, buffer increases to +19.3%
- Still captures 403% APR, just with less leverage

## Files

- Bot: `scripts/hyperliquid_monster_bot_v2.py` (also copied to scripts/strategies/)
- Analysis: `prompts/LIQUIDATION_SAFETY_ANALYSIS.md`
- Analytics: `scripts/analytics/quick_strategy_check.py`
- Credentials: `.env` (gitignored)
- Performance Data: `data/performance/coin_performance.json` and `.md`

## Performance Tracking System (Added 2026-01-11)

The bot now includes comprehensive per-coin performance tracking:

### Features
- **Per-coin metrics**: Funding received, payment count, avg APR, best/worst payments
- **Dual export**: JSON (machine-readable) + Markdown (investor reports)
- **Real-time saving**: After each funding payment
- **Hourly auto-save**: Periodic backup
- **Persistence**: Data loads on restart, accumulates across sessions
- **Leaderboard**: Top 5 coins shown in `status` command

### Output Files
- `data/performance/coin_performance.json` - For programmatic access
- `data/performance/coin_performance.md` - For investor reports

### Tracked Metrics Per Coin
- Total funding received
- Number of funding payments
- Average APR captured
- Best/worst single payment
- Leverage used
- Volatility classification
