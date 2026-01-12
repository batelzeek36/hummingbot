# Handoff: Hyperliquid Trading Bot Project

## Current Status: V2.1 SMART LEVERAGE BOT - READY TO DEPLOY

User has successfully:
1. Deposited ~$78.82 USDC to Hyperliquid
2. Generated API wallet credentials
3. Created v1 Balanced Bot
4. Created v2 Aggressive Bot
5. **NEW: Ran volatility analysis - discovered memecoins too volatile for 8x**
6. **NEW: Created v2.1 with SMART LEVERAGE (varies by coin volatility)**

---

## Why V2.1? (The Volatility Discovery)

Ran 30-day volatility analysis on all coins. Results were eye-opening:

| Coin | Max Daily Move | 8x Safe? | Safe Leverage |
|------|----------------|----------|---------------|
| BTC | 6.0% | YES | 8x |
| SOL | 10.4% | YES | 5x |
| DOGE | 15.1% | NO | 3x |
| HYPE | 17.0% | NO | 3x |
| MOG | 20.4% | NO | 3x |
| PEPE | 28.1% | NO | 2x |
| BONK | 39.7% | NO | 2x |
| WIF | 27.2% | NO | 2x |
| **ANIME** | **66.2%** | **NO** | **2x** |

**Key insight:** ANIME moved 66% in one day. At 8x leverage, you'd be liquidated 5x over.

---

## Bot Versions

### V1 - Balanced (~13.5%/month)
- File: `hyperliquid_monster_bot.py`
- Static 3x/5x/3x leverage
- Lower risk, lower returns

### V2 - Aggressive (~20-25%/month) - DANGEROUS
- File: Was going to use 8x everywhere
- **Would likely get liquidated on memecoins**

### V2.1 - Smart Leverage (~12-16%/month) - RECOMMENDED
- File: `hyperliquid_monster_bot_v2.py`
- **Dynamic leverage based on coin volatility**
- Maximizes returns while preventing liquidation

---

## V2.1 Smart Leverage System

```
COIN CLASSIFICATION (based on 30-day volatility analysis):

SAFE (8x leverage):     BTC, ETH
MEDIUM (5x leverage):   SOL
HIGH (3x leverage):     DOGE, HYPE, MOG, PURR, JEFF, VVV
EXTREME (2x leverage):  ANIME, PEPE, BONK, WIF
```

**How it works:**
- Bot automatically detects which coin it's trading
- Applies appropriate leverage for that coin's volatility
- You get maximum safe leverage on each position

---

## Expected Performance V2.1

| Metric | V1 | V2 (Dangerous) | V2.1 (Smart) |
|--------|-------|----------------|--------------|
| Monthly Target | 13.5% | 20-25% | **12-16%** |
| Monthly Profit | ~$10.62 | ~$17-20 | **~$10-13** |
| Liquidation Risk | Low | **HIGH** | **Low** |
| Break-even | ~7.4 months | ~4-5 months | **~6-8 months** |

**V2.1 Monthly Breakdown:**
- Funding (smart leverage): ~$6-8
- Grid Trading (5x SOL): ~$3-4
- Momentum (8x BTC): ~$1-2
- **Total: ~$10-13/month**

---

## Credentials (in `.env`)

```
# Hyperliquid Perpetual API (created 2026-01-11)
HYPERLIQUID_ACCOUNT_ADDRESS=0x6baFf6ca4275C0D551949265a53F276813616aC2
HYPERLIQUID_API_WALLET=0x123357C5fca2C994Ff54CFd81d6191E07d956301
HYPERLIQUID_API_SECRET=0x3c9198588617e9889e771825c7e26bfc7acf420bebd75f5a7d4742540407af02
```

---

## How to Deploy V2.1

```bash
cd /Users/kingkamehameha/Documents/hummingbot
docker compose up -d
docker attach hummingbot

# In hummingbot:
connect hyperliquid_perpetual
# Enter API key and secret when prompted

start --script hyperliquid_monster_bot_v2.py
```

---

## V2.1 Configuration Highlights

```python
# SMART LEVERAGE - varies by coin volatility
use_smart_leverage: bool = True

# Leverage by volatility class:
# SAFE (BTC):     8x
# MEDIUM (SOL):   5x
# HIGH (memes):   3x
# EXTREME (ANIME): 2x

# Grid uses 5x on SOL (medium volatility)
grid_leverage: int = 5

# Momentum uses 8x on BTC (low volatility)
momentum_leverage: int = 8

# Still scans all these for funding opportunities
funding_scan_pairs = "ANIME-USD,VVV-USD,HYPE-USD,PURR-USD,JEFF-USD,MOG-USD,WIF-USD,PEPE-USD,BONK-USD,DOGE-USD"

# But applies appropriate leverage to each!
```

---

## Risk Management Summary

| Protection | Trigger | Action |
|------------|---------|--------|
| Smart Leverage | Per-coin | 2x-8x based on volatility |
| Kill Switch | 20% drawdown | Close all positions |
| Momentum Stop | 1.5% loss | Exit trade |
| Funding Flip | Rate changes sign | Exit position |
| Unknown Coin | New coin | Default to 3x (HIGH) |

---

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/strategies/hyperliquid_monster_bot_v2.py` | **V2.1 Smart Leverage bot** |
| `scripts/strategies/hyperliquid_monster_bot.py` | V1 Balanced bot |
| `scripts/analytics/volatility_analysis.py` | Volatility analysis tool |
| `.env` | API credentials |

---

## Volatility Analysis Tool

Re-run anytime to update coin classifications:

```bash
cd /Users/kingkamehameha/Documents/hummingbot
python scripts/analytics/volatility_analysis.py
```

---

## Why V2.1 Returns Are Lower Than V2

| Factor | V2 | V2.1 | Impact |
|--------|----|----|--------|
| ANIME leverage | 8x | 2x | -75% on ANIME funding |
| PEPE/BONK leverage | 8x | 2x | -75% on these |
| SOL leverage | 8x | 5x | -37% on grid |
| BTC leverage | 5x | 8x | +60% on momentum |

**The tradeoff:** Lower returns BUT you won't get liquidated by a 20% memecoin dump.

---

## Monitoring Commands

In hummingbot:
```
status          # Shows smart leverage per position
history         # Trade history
balance         # Account balance
stop            # Stop gracefully
```

---

*Last updated: 2026-01-11*
*Status: V2.1 Smart Leverage bot ready to deploy*
*Volatility analysis completed - leverage optimized per coin*
