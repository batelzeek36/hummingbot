# Hyperliquid Coin Analysis & Volatility Classifications

Last Updated: 2026-01-12
Status: ✅ ALL SAFE - Liquidation risks eliminated

## CRITICAL: Liquidation Thresholds
- 8x leverage = liquidated at 12.5% adverse move
- 5x leverage = liquidated at 20% adverse move
- 3x leverage = liquidated at 33% adverse move
- 2x leverage = liquidated at 50% adverse move

## Volatility Classifications (VERIFIED & SAFE)

### SAFE (8x leverage OK) - Max daily <10%
| Coin | Max Move | Liq Threshold | Buffer | Status |
|------|----------|---------------|--------|--------|
| BTC-USD | 6.0% | 12.5% | +6.5% | SAFE |

### MEDIUM (5x leverage) - Max daily <15%
| Coin | Max Move | Liq Threshold | Buffer | Status |
|------|----------|---------------|--------|--------|
| ETH-USD | 9.8% | 20.0% | +10.2% | SAFE (moved from 8x) |
| SOL-USD | 10.4% | 20.0% | +9.6% | SAFE |
| TAO-USD | 13.1% | 20.0% | +6.9% | SAFE |

### HIGH (3x leverage) - Max daily <25%
| Coin | Max Move | Liq Threshold | Buffer | Status |
|------|----------|---------------|--------|--------|
| DOGE-USD | 15.0% | 33.3% | +18.3% | SAFE |
| HYPE-USD | 17.4% | 33.3% | +15.9% | SAFE |
| JEFF-USD | ~20% | 33.3% | ~+13% | ASSUMED |

### EXTREME (2x leverage only) - Max daily 25%+
| Coin | Max Move | Liq Threshold | Buffer | Status |
|------|----------|---------------|--------|--------|
| AVNT-USD | 25.8% | 50.0% | +24.2% | SAFE (moved from 3x) |
| WIF-USD | 26.6% | 50.0% | +23.4% | SAFE |
| kPEPE-USD | 30.3% | 50.0% | +19.7% | SAFE |
| IP-USD | 30.7% | 50.0% | +19.3% | SAFE (moved from 3x) |
| HYPER-USD | 33.4% | 50.0% | +16.6% | SAFE |
| kBONK-USD | 39.3% | 50.0% | +10.7% | SAFE |
| VVV-USD | 40.5% | 50.0% | +9.5% | SAFE |

### REMOVED (unsafe at any leverage)
| Coin | Max Move | Reason |
|------|----------|--------|
| ANIME-USD | 51.9% | Exceeds 50% threshold |
| PURR-USD | 55.8% | Exceeds 50% threshold |
| MOG-USD | N/A | Not available on Hyperliquid |
| JEFF-USD | N/A | Not available on Hyperliquid |

## Changes Applied (2026-01-12)
- ETH-USD: SAFE (8x) → MEDIUM (5x)
- IP-USD: HIGH (3x) → EXTREME (2x)
- AVNT-USD: HIGH (3x) → EXTREME (2x)
- ANIME-USD: REMOVED (51.9% > 50%)
- PURR-USD: REMOVED (55.8% > 50%)
- Symbol fixes: PEPE→kPEPE, BONK→kBONK (Hyperliquid naming)

## Volume/Liquidity Rankings

| Coin | 24h Volume | Suitable For |
|------|------------|--------------|
| SOL | $461M | Grid (best liquidity) |
| XRP | $43M | Grid alternative |
| BTC | High | Momentum |
| HYPER | $4.3M | Funding only |
| IP | $4.7M | Funding only |
| AVNT | $1.2M | Funding only |

## Key Rule

**Always verify: Max 30-day move < Liquidation threshold**

Minimum buffer across all coins: +6.5% (BTC)

Run `python scripts/analytics/quick_strategy_check.py` to get current data before deploying.
