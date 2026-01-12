# Handoff: Optimization Opportunities for Hyperliquid Bot

## Session Date: 2026-01-11

## Current State

Bot V2.1 is ready to deploy with smart leverage. However, there are optimization opportunities that weren't fully explored due to time.

---

## What Was Completed

1. **V1 Bot** - Basic multi-strategy bot (funding, grid, momentum)
2. **V2 Bot** - Added dynamic funding hunter, auto-rotation
3. **Volatility Analysis** - Discovered memecoins too volatile for 8x leverage
4. **V2.1 Bot** - Smart leverage per coin (2x-8x based on volatility)
5. **Documentation** - Full handoff docs created

---

## Outstanding Optimization Questions

### 1. Verify Current Funding Rates

**Problem:** Funding rates change constantly. ANIME was -110% APR when analyzed, but could be different now.

**Action needed:**
```bash
# Run this to check current rates (NEW SCRIPT CREATED)
python scripts/analytics/funding_rate_checker.py
```

**LATEST CHECK (2026-01-11 14:26):**
| Coin | APR | Direction | Leverage | Effective APR |
|------|-----|-----------|----------|---------------|
| ANIME | 668.2% | SHORT | 2x | 1336.3% |
| VVV | 47.5% | SHORT | 3x | 142.5% |
| BTC | 10.9% | LONG | 8x | 87.6% |
| ETH | 10.9% | LONG | 8x | 87.6% |

**Conclusion:** ANIME and VVV are currently the best opportunities. ANIME has extremely high negative funding (shorts receive 668% APR). Even at 2x leverage, effective return is massive.

---

### 2. Add ETH to Strategy Mix ✅ DONE

**Current state:** ETH is classified as SAFE (8x leverage OK) but not used anywhere.

**Opportunity:**
- Add `ETH-USD` to funding scan pairs
- Consider ETH as alternative grid pair (8x vs SOL's 5x)
- Could increase returns if ETH funding rates are favorable

**IMPLEMENTED (2026-01-11):**
```python
# In hyperliquid_monster_bot_v2.py:
funding_scan_pairs: str = Field(
    default="ETH-USD,ANIME-USD,VVV-USD,HYPE-USD,PURR-USD,JEFF-USD,MOG-USD,WIF-USD,PEPE-USD,BONK-USD,DOGE-USD",
    ...
)
```

ETH now included in funding scan. At 10.9% APR with 8x leverage = 87.6% effective APR.

---

### 3. Dynamic Grid Pair Selection

**Current state:** Grid pair is hardcoded to SOL-USD.

**Opportunity:** Bot could dynamically select best grid pair based on:
- Current volatility (want medium - not too high, not too low)
- Liquidity (need order fills)
- Recent price action (ranging = good for grid, trending = bad)

**Complexity:** Medium-high. Would need to add scanning logic similar to funding hunter.

---

### 4. Funding Rate Math Verification

**The question:** At reduced leverage, are memecoins still optimal?

**Analysis done:**
```
ANIME: 100% APR × 2x leverage = 200% effective
BTC:   15% APR × 8x leverage  = 120% effective

Conclusion: Memecoins still win even at 2x leverage
```

**But:** This assumes funding rates stay high. If ANIME drops to 30% APR:
```
ANIME: 30% APR × 2x = 60% effective
BTC:   15% APR × 8x = 120% effective

Now BTC wins!
```

**Action needed:** Monitor funding rates. If memecoin rates normalize, may need to shift strategy to higher-leverage stable coins.

---

### 5. Consider Longer Hold Periods

**Current logic:** Enter before funding, exit after collecting.

**Alternative:** If funding rate stays favorable, hold position longer to collect multiple funding payments.

**Risk:** Price movement during hold could eat into profits.

**Question:** What's the optimal hold period? 1 funding? 3? 24 hours?

---

### 6. Backtest Before Live

**Not done:** No historical backtesting was performed.

**Why it matters:** All profit estimates are theoretical. Real performance could vary.

**Tools available:**
- Hummingbot has backtesting framework
- Could use historical OHLCV data from `volatility_analysis.py` approach

---

## Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `scripts/strategies/hyperliquid_monster_bot_v2.py` | V2.1 Smart Leverage bot | READY |
| `scripts/strategies/hyperliquid_monster_bot.py` | V1 Balanced bot | READY |
| `scripts/analytics/volatility_analysis.py` | Volatility checker | READY |
| `scripts/analytics/funding_rate_checker.py` | Live funding rate checker | NEW |
| `.env` | API credentials | CONFIGURED |
| `prompts/HANDOFF_HYPERLIQUID_BOT.md` | Main handoff doc | UPDATED |

---

## Quick Wins (Easy to Implement)

1. ~~**Add ETH to funding scan**~~ ✅ DONE - ETH added to funding_scan_pairs
2. ~~**Run live funding rate check**~~ ✅ DONE - Created `funding_rate_checker.py`
3. **Adjust min APR threshold** - Currently 30%, could tune based on current market (most coins at ~11% APR now, only ANIME/VVV above threshold)

---

## Bigger Improvements (More Work)

1. **Dynamic grid pair selection** - Needs new scanning logic
2. **Backtesting framework** - Needs historical data + simulation
3. **Liquidation price monitoring** - Add hard stops based on margin
4. **Trailing stops for momentum** - Discussed but not implemented (user decided against for now)

---

## User Preferences Noted

- **No Binance** - User doesn't want to use Binance (cross-exchange arb not possible)
- **Prefers safety** - Chose smart leverage over aggressive 8x everywhere
- **No trailing stops for now** - Decided fixed TP is safer for small account
- **Wants ~20% monthly** - But accepted 12-16% for lower risk

---

## Credentials

**Stored securely in:** `/Users/kingkamehameha/Documents/hummingbot/.env`

Capital: ~$78.82 USDC on Hyperliquid

> ⚠️ Never store API secrets in documentation files. Always use `.env` files that are gitignored.

---

## Risk Management Guidelines

### Position Limits
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max Drawdown | 20% | Emergency shutdown trigger |
| Daily Loss Limit | $12 | ~15% of capital |
| Max Position Size | 60% of capital | Leaves margin buffer |
| Max Funding Positions | 3 simultaneous | Diversification |

### Leverage by Volatility Class
| Class | Coins | Max Leverage | Liquidation Buffer |
|-------|-------|--------------|-------------------|
| SAFE | BTC, ETH | 8x | 12.5% adverse move |
| MEDIUM | SOL | 5x | 20% adverse move |
| HIGH | DOGE, HYPE, MOG, VVV | 3x | 33% adverse move |
| EXTREME | ANIME, PEPE, BONK, WIF | 2x | 50% adverse move |

### Emergency Procedures
1. **If drawdown hits 15%**: Review open positions, consider manual intervention
2. **If drawdown hits 20%**: Bot auto-kills all strategies, closes all positions
3. **If funding rate flips**: Bot auto-closes affected positions
4. **If API connection lost**: Positions remain open until reconnection

### What to Watch For
- Sudden funding rate changes (can flip from +100% to -100% APR quickly)
- Exchange maintenance windows (Hyperliquid announces on Discord)
- Major market events (FOMC, CPI releases cause volatility spikes)
- Liquidation cascade events (can cause slippage beyond expected)

---

## 48-Hour Monitoring Checklist

### Hour 0-4 (Critical Launch Period)
- [ ] Bot started without errors
- [ ] Positions opening at expected leverage
- [ ] Funding scanner detecting opportunities
- [ ] Grid orders placed correctly
- [ ] No unexpected error logs

### Hour 4-12 (First Funding Cycles)
- [ ] At least one funding payment received
- [ ] Positions rotating correctly when rates change
- [ ] Grid fills generating expected P&L
- [ ] Drawdown staying under 5%

### Hour 12-24 (Full Day Validation)
- [ ] Total funding collected matches expectations
- [ ] No stuck positions (positions not closing when they should)
- [ ] Momentum strategy triggered at least once (if RSI conditions met)
- [ ] Session P&L positive or within acceptable loss range

### Hour 24-48 (Extended Validation)
- [ ] Bot survived overnight without intervention
- [ ] Consistent funding collection pattern
- [ ] No memory leaks or performance degradation
- [ ] Actual returns within ±50% of projected

### Key Metrics to Log
```
| Metric | Hour 4 | Hour 12 | Hour 24 | Hour 48 |
|--------|--------|---------|---------|---------|
| Total Funding Collected | $__ | $__ | $__ | $__ |
| Grid P&L | $__ | $__ | $__ | $__ |
| Momentum P&L | $__ | $__ | $__ | $__ |
| Max Drawdown Hit | __% | __% | __% | __% |
| Error Count | __ | __ | __ | __ |
```

### Red Flags (Stop and Investigate)
- Drawdown exceeds 10% in first 24 hours
- More than 3 consecutive losing trades on momentum
- Funding payments not appearing after expected collection time
- Position sizes different from configured amounts

---

## Recommended Next Steps

1. **Verify funding rates are still favorable** before deploying
2. **Add ETH to funding scan** (quick win)
3. **Deploy and monitor** for 24-48 hours
4. **Analyze actual performance** vs estimates
5. **Iterate** based on real data

---

## Expected Performance (V2.1)

| Scenario | Monthly % | Monthly $ |
|----------|-----------|-----------|
| Pessimistic | 10-12% | $8-10 |
| Realistic | 12-16% | $10-13 |
| Optimistic | 16-20% | $13-16 |

---

*Created: 2026-01-11*
*For: Next agent session*
*Context: Hyperliquid perpetual trading bot optimization*
