# Dollar-A-Day Project Evolution

## Project Timeline

### Phase 1: Dollar-A-Day PMM (Original)
- **Goal**: $1/day from market making on KuCoin
- **Strategy**: Simple PMM with inventory skew
- **Status**: Validated, but moved to Hyperliquid for better opportunities

### Phase 2: Multi-Strategy Bot
- **Goal**: Diversification across strategies
- **Strategies**: Market Making, Grid, RSI Directional
- **Status**: Framework built, superseded by Hyperliquid focus

### Phase 3: Hyperliquid Monster Bot v1
- **Goal**: Leverage perpetual trading on Hyperliquid
- **Strategies**: Funding Harvesting, Grid, Momentum
- **Capital**: ~$78 USDC
- **Status**: Working, but used uniform 8x leverage (risky)

### Phase 4: Hyperliquid Monster Bot v2.1
- **Key Upgrade**: Smart leverage by volatility
- **Insight**: ANIME had 66% daily move - would liquidate 8x position
- **Solution**: 2x on volatile coins, 8x on safe coins
- **Status**: Production ready

### Phase 5: Hyperliquid Monster Bot v2.4
- **Key Upgrades**:
  - Added AVNT, HYPER, IP to funding scanner (high APR opportunities)
  - Expanded volatility classifications
  - Created analytics scripts for ongoing optimization
  - Performance tracking system with per-coin metrics
- **Status**: Superseded by v2.5

### Phase 6: Hyperliquid Monster Bot v2.5 - Whale Protection (Current)
- **Key Upgrades**:
  - Whale Protection System (Circuit Breaker, Grid Protection, Trailing Stop, Dynamic Risk, Early Warning)
  - Full modular refactoring - main file reduced from 1114 to 497 lines
  - Extracted: PauseManager, WhaleProtectionOrchestrator, StatusFormatter, logging_utils
- **Bot File**: `scripts/hyperliquid_monster_bot_v2.py`
- **Status**: Production ready

## Current Strategy (v2.5)

| Strategy | Pair | Leverage | Capital | Target |
|----------|------|----------|---------|--------|
| Funding | 12 pairs (smart) | 2-8x | 45% ($35) | Capture high APRs |
| Grid | SOL-USD | 5x | 35% ($27) | Range trading |
| Momentum | BTC-USD | 8x | 20% ($16) | RSI-based entries |

## Key Learnings

1. **Leverage must match volatility** - 8x on memecoins = liquidation
2. **Liquidity matters for grid** - SOL's $461M volume beats TAO's $3.8M
3. **Funding rates change fast** - Run analytics before deploying
4. **Smart leverage preserves capital** - Even 2x on 700% APR beats 8x on 11% APR
5. **Modular code scales** - Extract concerns early to prevent monolithic files

## Files Structure

```
scripts/
├── hyperliquid_monster_bot_v2.py      # Main orchestrator (497 lines)
├── hyperliquid_monster/               # Modular package
│   ├── __init__.py
│   ├── config.py                      # HyperliquidMonsterV2Config
│   ├── models.py                      # Data models
│   ├── indicators.py                  # Technical indicators
│   ├── performance.py                 # Performance tracking
│   ├── volatility.py                  # Volatility classification
│   ├── pause_manager.py               # Unified pause/resume system
│   ├── status_formatter.py            # Status display formatting
│   ├── logging_utils.py               # Startup banner
│   ├── strategies/
│   │   ├── funding_hunter.py          # Funding rate strategy
│   │   ├── grid_trading.py            # Grid trading strategy
│   │   └── momentum.py                # RSI momentum strategy
│   └── whale_protection/
│       ├── orchestrator.py            # Protection coordinator
│       ├── circuit_breaker.py         # Rapid price spike detection
│       ├── grid_protection.py         # One-sided fill detection
│       ├── trailing_stop.py           # Dynamic stop loss
│       ├── dynamic_risk.py            # Volatility-scaled risk
│       └── early_warning.py           # Order book/funding alerts
├── analytics/
│   ├── quick_strategy_check.py        # Strategy optimizer
│   ├── funding_rate_checker.py        # Quick funding check
│   └── volatility_analysis.py         # Deep volatility analysis
└── strategies/
    └── hyperliquid_monster_bot_v2.py  # Copy of main bot

data/performance/
├── coin_performance.json              # Per-coin metrics (machine-readable)
└── coin_performance.md                # Per-coin metrics (investor reports)

prompts/
├── HANDOFF_HYPERLIQUID_BOT.md         # Main handoff doc
└── LIQUIDATION_SAFETY_ANALYSIS.md     # Leverage safety analysis
```

## Credentials
Stored in `.env` (gitignored):
- HYPERLIQUID_ACCOUNT_ADDRESS
- HYPERLIQUID_API_WALLET
- HYPERLIQUID_API_SECRET
