# Hyperliquid Monster Bot v2 Modularization Plan

## Phase 1: Git Setup & Fork Creation

### Step 1.1: Fork the hummingbot repo to your GitHub account
```bash
gh repo fork hummingbot/hummingbot --clone=false
```

### Step 1.2: Add your fork as a remote
```bash
git remote add myfork https://github.com/batelzeek36/hummingbot.git
```

### Step 1.3: Create a feature branch for the modularization
```bash
git checkout -b monster-bot-modular
```

### Step 1.4: Stage and commit current work (before modularization)
```bash
git add scripts/hyperliquid_monster_bot_v2.py
git add data/performance/  # if exists
git commit -m "Add Hyperliquid Monster Bot v2.3 - monolithic version"
```

### Step 1.5: Push to your fork
```bash
git push -u myfork monster-bot-modular
```

---

## Phase 2: Modularization

### Target Structure
```
scripts/
├── hyperliquid_monster_bot_v2.py              # Main orchestrator (slim, ~200 lines)
└── hyperliquid_monster/                        # New package
    ├── __init__.py                             # Package exports
    ├── models.py                               # Dataclasses (~70 lines)
    ├── config.py                               # Configuration class (~150 lines)
    ├── volatility.py                           # Volatility/leverage logic (~60 lines)
    ├── performance.py                          # Performance tracking (~200 lines)
    ├── strategies/
    │   ├── __init__.py
    │   ├── base.py                             # Base strategy class
    │   ├── funding_hunter.py                   # Funding rate strategy (~200 lines)
    │   ├── grid_trading.py                     # Grid trading strategy (~120 lines)
    │   └── momentum.py                         # Momentum/RSI strategy (~150 lines)
    └── risk.py                                 # Risk management (~50 lines)
```

### Step 2.1: Create package directory structure
Create `scripts/hyperliquid_monster/` and subdirectories.

### Step 2.2: Extract `models.py`
Move these dataclasses:
- `FundingOpportunity`
- `PositionInfo`
- `StrategyMetrics`
- `StrategyMode` (enum)

**Lines to extract:** 48-55, 115-151

### Step 2.3: Extract `volatility.py`
Move volatility-related code:
- `CoinVolatility` enum
- `COIN_VOLATILITY` dictionary
- `VOLATILITY_LEVERAGE` dictionary
- `get_safe_leverage()` function

**Lines to extract:** 61-113

### Step 2.4: Extract `config.py`
Move the configuration class:
- `HyperliquidMonsterV2Config`

**Lines to extract:** 189-328

### Step 2.5: Extract `performance.py`
Move performance tracking:
- `CoinPerformance` dataclass
- Performance load/save methods (will become standalone functions)

**Lines to extract:** 154-183, 425-585

### Step 2.6: Extract `strategies/funding_hunter.py`
Create a `FundingHunterStrategy` class encapsulating:
- `_run_funding_hunter()`
- `_scan_funding_opportunities()`
- `_manage_funding_positions()`
- `_open_best_funding_positions()`
- `_open_funding_position()`
- `_close_funding_position()`

**Lines to extract:** 715-966

### Step 2.7: Extract `strategies/grid_trading.py`
Create a `GridStrategy` class encapsulating:
- `_run_grid_strategy()`
- `_initialize_grid()`
- `_cancel_grid_orders()`
- Grid fill handling logic

**Lines to extract:** 974-1071, 1287-1323

### Step 2.8: Extract `strategies/momentum.py`
Create a `MomentumStrategy` class encapsulating:
- `_run_momentum_strategy()`
- `_calculate_rsi()`
- `_open_momentum_position()`
- `_check_momentum_exit()`
- `_close_momentum_position()`

**Lines to extract:** 1077-1226

### Step 2.9: Extract `risk.py`
Move risk management:
- `_check_risk_limits()`
- `_kill_all()`

**Lines to extract:** 1329-1354

### Step 2.10: Refactor main bot class
The main `HyperliquidMonsterBotV2` class becomes a slim orchestrator:
- Imports all modules
- Instantiates strategy helpers in `__init__`
- Delegates to strategies in `on_tick()`
- Handles Hummingbot lifecycle (`on_start`, `on_stop`, event handlers)
- Keeps `format_status()` for UI

---

## Phase 3: Testing & Commit

### Step 3.1: Verify imports work
```bash
cd /Users/kingkamehameha/Documents/hummingbot
python -c "from scripts.hyperliquid_monster_bot_v2 import HyperliquidMonsterBotV2; print('OK')"
```

### Step 3.2: Commit modularization
```bash
git add scripts/hyperliquid_monster/
git add scripts/hyperliquid_monster_bot_v2.py
git commit -m "Refactor: Modularize Monster Bot v2 into separate modules

- Extract models, config, volatility logic to separate files
- Create strategy classes for funding, grid, momentum
- Extract performance tracking to dedicated module
- Main bot class now thin orchestration layer
- No functional changes, same behavior"
```

### Step 3.3: Push to fork
```bash
git push myfork monster-bot-modular
```

---

## Benefits of This Approach

1. **Hummingbot Compatibility**: Main class still inherits `ScriptStrategyBase` with all required methods
2. **No Code Loss**: All logic preserved, just reorganized
3. **Easier Editing**: Claude Code can read/edit smaller focused files
4. **Better Testing**: Individual strategies can be unit tested
5. **Cleaner Git History**: Changes to one strategy don't touch others
6. **Reusability**: Strategy classes could be reused in other bots

---

## Rollback Plan

If anything breaks:
```bash
git checkout master -- scripts/hyperliquid_monster_bot_v2.py
git clean -fd scripts/hyperliquid_monster/
```

---

## Estimated Module Sizes

| Module | Lines | Purpose |
|--------|-------|---------|
| `models.py` | ~70 | Dataclasses for positions, metrics |
| `config.py` | ~150 | Pydantic configuration class |
| `volatility.py` | ~60 | Leverage calculation logic |
| `performance.py` | ~200 | Coin performance tracking |
| `strategies/funding_hunter.py` | ~200 | Funding rate strategy |
| `strategies/grid_trading.py` | ~120 | Grid trading strategy |
| `strategies/momentum.py` | ~150 | RSI momentum strategy |
| `risk.py` | ~50 | Risk management |
| **Main bot (refactored)** | ~250 | Orchestration + status display |
| **Total** | ~1,250 | (vs 1,470 original - some dedup) |
