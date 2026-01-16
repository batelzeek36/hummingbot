# Plan: Coinglass Integration for Funding Strategy + Risk Analysis Script

## Overview

Two enhancements to improve risk management:
1. **Risk Analysis Script** - Combines Hyperliquid volatility + Coinglass liquidation/crowding data
2. **Funding Strategy Coinglass Integration** - Avoid crowded/trapped positions

---

## Part 1: Comprehensive Risk Analysis Script

### Purpose
Create `/scripts/analytics/comprehensive_risk_check.py` that combines:
- Hyperliquid price volatility (what we already ran)
- Coinglass liquidation heatmaps (where are the danger zones?)
- Coinglass L/S ratios (where is the market crowded?)
- Coinglass CVD divergence (who's trapped?)

### File to Create
`/Users/kingkamehameha/Documents/hummingbot/scripts/analytics/comprehensive_risk_check.py`

### Script Structure
```
1. Configuration
   - COINS list (from funding_scan_pairs)
   - Coinglass API key from env or config
   - Thresholds for risk levels

2. Data Fetching
   - Hyperliquid: 7-day OHLCV via ccxt
   - Coinglass: Liquidation heatmap, L/S ratio, CVD divergence

3. Analysis Functions
   - analyze_volatility(coin) → max_move, safe_leverage, buffer
   - analyze_liquidations(coin) → nearest_clusters, magnetic_direction
   - analyze_crowding(coin) → L/S ratio, crowding_score, contrarian_signal
   - analyze_divergence(coin) → spot/perp divergence, trapped traders

4. Combined Risk Score
   - Volatility risk (0-100): Based on max daily move vs liquidation threshold
   - Liquidation risk (0-100): Based on cluster proximity
   - Crowding risk (0-100): Based on L/S imbalance
   - Overall risk = weighted average

5. Output Sections
   - VOLATILITY ANALYSIS (existing data)
   - LIQUIDATION ANALYSIS (Coinglass)
   - CROWDING ANALYSIS (Coinglass)
   - COMBINED RISK MATRIX
   - RECOMMENDATIONS (which coins to trade/avoid)
```

### Output Format
```
================================================================================
  COMPREHENSIVE RISK ANALYSIS - FUNDING COINS
================================================================================
  Time: YYYY-MM-DD HH:MM
  Data Sources: Hyperliquid API, Coinglass API

=== VOLATILITY RISK ===
  Coin     Max Move   Leverage   Liq Threshold   Buffer   Risk
  HYPER    33.4%      2x         50%             16.6%    MEDIUM
  VVV      40.5%      2x         50%             9.5%     HIGH
  ...

=== LIQUIDATION CLUSTERS ===
  Coin     Nearest Long   Nearest Short   Magnet Dir   Risk
  BTC      -3.2%          +2.1%           UP           LOW
  ETH      -1.5%          +4.3%           DOWN         MEDIUM
  ...

=== MARKET CROWDING ===
  Coin     Long%   Short%   Sentiment        Crowding   Contrarian
  BTC      62%     38%      leaning_long     24         neutral
  ETH      71%     29%      crowded_long     42         short
  ...

=== CVD DIVERGENCE ===
  Coin     Perp CVD   Spot CVD   Divergence        Strength   Signal
  BTC      BULLISH    BULLISH    aligned           0          -
  ETH      BULLISH    BEARISH    bearish_div       85%        SHORT
  ...

=== COMBINED RISK MATRIX ===
  Coin     Vol Risk   Liq Risk   Crowd Risk   TOTAL   VERDICT
  BTC      LOW        LOW        LOW          15      SAFE
  HYPER    HIGH       MEDIUM     LOW          55      CAUTION
  VVV      HIGH       HIGH       MEDIUM       78      AVOID
  ...

=== RECOMMENDATIONS ===
  SAFE TO TRADE: BTC, ETH, SOL, DOGE
  USE CAUTION: HYPER, IP, WIF
  AVOID NOW: VVV (crowded + tight buffer)
================================================================================
```

---

## Part 2: Funding Strategy Coinglass Integration

### Purpose
Add Coinglass signals to FundingHunterStrategy to avoid:
1. Entering crowded positions (squeeze risk)
2. Entering against spot/perp divergence (trapped traders)
3. Exiting when liquidation clusters approach

### Files to Modify

#### 1. Config File
`/Users/kingkamehameha/Documents/hummingbot/scripts/hyperliquid_monster/config.py`

Add new config parameters:
```python
# === FUNDING COINGLASS INTEGRATION ===
funding_use_coinglass: bool = Field(
    default=True,
    description="Use Coinglass data for funding entry/exit decisions"
)
funding_coinglass_block_crowded: bool = Field(
    default=True,
    description="Block funding entries when position is crowded (>65% one side)"
)
funding_coinglass_crowding_threshold: Decimal = Field(
    default=Decimal("65"),
    description="L/S ratio % to consider crowded"
)
funding_coinglass_block_divergence: bool = Field(
    default=True,
    description="Block entries against strong spot/perp divergence"
)
funding_coinglass_divergence_threshold: Decimal = Field(
    default=Decimal("70"),
    description="Divergence signal strength to block entry"
)
funding_coinglass_exit_on_squeeze: bool = Field(
    default=True,
    description="Exit early if liquidation clusters approach"
)
funding_liq_proximity_exit_pct: Decimal = Field(
    default=Decimal("3.0"),
    description="Exit if liquidation cluster within this % of price"
)
```

#### 2. Funding Strategy
`/Users/kingkamehameha/Documents/hummingbot/scripts/hyperliquid_monster/strategies/funding_hunter.py`

**Add imports:**
```python
from ..coinglass_api import CoinglassAPI, SpotPerpDivergence, LongShortRatio, LiquidationHeatmap
```

**Add to `__init__`:**
```python
# Coinglass integration
self._coinglass: Optional[CoinglassAPI] = None
self._coinglass_blocks = 0
self._coinglass_exits = 0

if self.config.funding_use_coinglass and self.config.coinglass_enabled:
    try:
        api_key = self.config.coinglass_api_key or None
        self._coinglass = CoinglassAPI(
            api_key=api_key,
            request_interval=float(self.config.coinglass_request_interval),
            logger=self.logger,
        )
        self.logger.info("FUNDING: Coinglass integration enabled")
    except Exception as e:
        self.logger.warning(f"FUNDING: Coinglass init failed - {e}")
```

**Add new method `_check_coinglass_entry()`:**
```python
def _check_coinglass_entry(self, pair: str, direction: str) -> Tuple[bool, str]:
    """Check if entry should be blocked based on Coinglass data."""
    if not self._coinglass:
        return False, ""

    reasons = []

    # Check crowding
    if self.config.funding_coinglass_block_crowded:
        ls_ratio = self._coinglass.get_long_short_ratio(pair)
        if ls_ratio:
            threshold = float(self.config.funding_coinglass_crowding_threshold)
            if direction == "long" and ls_ratio.long_ratio >= threshold:
                reasons.append(f"CROWDED LONG ({ls_ratio.long_ratio:.1f}%)")
            elif direction == "short" and ls_ratio.short_ratio >= threshold:
                reasons.append(f"CROWDED SHORT ({ls_ratio.short_ratio:.1f}%)")

    # Check divergence
    if self.config.funding_coinglass_block_divergence:
        divergence = self._coinglass.get_spot_perp_divergence(pair)
        if divergence:
            threshold = float(self.config.funding_coinglass_divergence_threshold)
            if divergence.signal_strength >= threshold:
                if direction == "long" and divergence.divergence_type == "bearish_divergence":
                    reasons.append(f"BEARISH DIVERGENCE ({divergence.signal_strength:.0f}%)")
                elif direction == "short" and divergence.divergence_type == "bullish_divergence":
                    reasons.append(f"BULLISH DIVERGENCE ({divergence.signal_strength:.0f}%)")

    if reasons:
        return True, " | ".join(reasons)
    return False, ""
```

**Add new method `_check_coinglass_exit()`:**
```python
def _check_coinglass_exit(self, pair: str, position: OpenFundingPosition) -> Tuple[bool, str]:
    """Check if position should exit based on liquidation proximity."""
    if not self._coinglass or not self.config.funding_coinglass_exit_on_squeeze:
        return False, ""

    liq_heatmap = self._coinglass.get_liquidation_heatmap(pair)
    if not liq_heatmap:
        return False, ""

    direction = position.opportunity.direction
    threshold = float(self.config.funding_liq_proximity_exit_pct)

    # For longs, check short liquidations above (squeeze building)
    if direction == "long" and liq_heatmap.nearest_short_liq:
        distance = abs(liq_heatmap.short_clusters[0].distance_pct) if liq_heatmap.short_clusters else 100
        if distance < threshold:
            return True, f"SHORT SQUEEZE BUILDING ({distance:.1f}% away)"

    # For shorts, check long liquidations below
    elif direction == "short" and liq_heatmap.nearest_long_liq:
        distance = abs(liq_heatmap.long_clusters[0].distance_pct) if liq_heatmap.long_clusters else 100
        if distance < threshold:
            return True, f"LONG SQUEEZE BUILDING ({distance:.1f}% away)"

    return False, ""
```

**Modify `_open_best_positions()` - Add after Holy Grail check:**
```python
# === COINGLASS ENTRY CHECK ===
if self._coinglass:
    should_block, reason = self._check_coinglass_entry(opp.pair, opp.direction)
    if should_block:
        self.logger.info(f"FUNDING: {opp.pair} {opp.direction.upper()} blocked - {reason}")
        self._coinglass_blocks += 1
        continue
```

**Modify `_manage_positions()` - Add after OI exit check:**
```python
# === COINGLASS SQUEEZE EXIT ===
if self._coinglass:
    should_exit, reason = self._check_coinglass_exit(pair, position)
    if should_exit:
        self._close_position(pair, f"coinglass_{reason}", current_timestamp)
        self._coinglass_exits += 1
        continue
```

**Update `get_status_info()`:**
```python
if self._coinglass:
    status["Coinglass"] = "ON"
    status["CG Blocks"] = str(self._coinglass_blocks)
    status["CG Exits"] = str(self._coinglass_exits)
```

---

## Implementation Order

1. **Part 1: Risk Analysis Script** (standalone, no risk)
   - Create new file
   - Test independently with `python analytics/comprehensive_risk_check.py`

2. **Part 2: Funding Coinglass Integration**
   - Add config parameters
   - Add methods to funding_hunter.py
   - Integrate into entry/exit flow
   - Test with bot startup

---

## Verification

### Part 1 - Risk Analysis Script
```bash
cd /Users/kingkamehameha/Documents/hummingbot/scripts
python analytics/comprehensive_risk_check.py
```
Expected: Formatted report showing all coins with volatility + Coinglass data

### Part 2 - Funding Strategy
1. Start bot with `funding_use_coinglass: true`
2. Check logs for "FUNDING: Coinglass integration enabled"
3. Trigger a crowded situation (set threshold low temporarily)
4. Verify "blocked" messages appear in logs
5. Check status display shows Coinglass stats

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| New analytics script | None | Standalone, read-only |
| Config additions | Low | New fields with safe defaults |
| Funding entry filter | Low | Can disable via config, graceful fallback |
| Funding exit trigger | Medium | Conservative threshold (3%), config toggle |

---

## Dependencies

- `ccxt` - Already installed (for Hyperliquid OHLCV)
- `requests` - Already installed (for Coinglass API)
- `COINGLASS_API_KEY` environment variable - Required for Coinglass data
