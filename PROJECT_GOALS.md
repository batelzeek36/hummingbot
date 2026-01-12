# DEX Arbitrage Project

## What We're Building

A DEX arbitrage bot that:
- Monitors price discrepancies across decentralized exchanges
- Executes trades when profitable opportunities arise
- Starts with **paper trading** to validate strategies before using real funds
- Runs on smaller chains (lower fees, less competition)

## Why Hummingbot

| Requirement | Hummingbot Solution |
|-------------|---------------------|
| DEX arbitrage | Built-in `amm_arb` strategy |
| Paper trading | `paper_trade` connector |
| Multiple DEXes | Gateway middleware (10+ DEXes) |
| Low barrier | Docker setup, config-based |

---

## Phase 1: Setup (Paper Trading)

### Prerequisites

- [ ] Docker Desktop installed
- [ ] Basic terminal knowledge
- [ ] ~2GB disk space

### Step 1: Start Hummingbot

```bash
cd ~/Documents/hummingbot

# Start the bot
docker compose up -d

# Attach to it
docker attach hummingbot
```

### Step 2: Create a Paper Trading Config

Inside the Hummingbot CLI:

```
create
```

Select `amm_arb` strategy, then configure:

| Parameter | Paper Trading Value | Notes |
|-----------|---------------------|-------|
| connector_1 | `paper_trade` | Simulated exchange |
| market_1 | `ETH-USDT` | Trading pair |
| connector_2 | `paper_trade` | Second simulated exchange |
| market_2 | `ETH-USDT` | Same pair |
| order_amount | `0.1` | Amount per trade |
| min_profitability | `0.5` | 0.5% minimum profit |

### Step 3: Run and Observe

```
start
status
```

Watch for:
- "No arbitrage opportunity" = prices too close
- "Found arbitrage opportunity!" = would execute trade
- Profit/loss tracking in status

---

## Phase 2: Live DEX Testing (Small Amounts)

### Prerequisites

- [ ] Crypto wallet (MetaMask or similar)
- [ ] Small amount of tokens on chosen chain
- [ ] Native gas tokens (ETH, MATIC, etc.)

### Step 1: Enable Gateway

Edit `docker-compose.yml` and uncomment the gateway section:

```yaml
gateway:
  restart: always
  container_name: gateway
  image: hummingbot/gateway:latest
  ports:
    - "15888:15888"
  volumes:
    - "./gateway-files/conf:/home/gateway/conf"
    - "./gateway-files/logs:/home/gateway/logs"
    - "./certs:/home/gateway/certs"
  environment:
    - GATEWAY_PASSPHRASE=admin
    - DEV=true
```

Restart:
```bash
docker compose down
docker compose up -d
```

### Step 2: Connect Wallet

In Hummingbot CLI:
```
gateway connect uniswap
```

Follow prompts to add your wallet private key.

### Step 3: Configure Real DEX Arbitrage

Recommended starting config for Arbitrum:

| Parameter | Value | Why |
|-----------|-------|-----|
| connector_1 | `uniswap_arbitrum_one` | Uniswap on Arbitrum |
| connector_2 | `sushiswap_arbitrum_one` | SushiSwap on Arbitrum |
| market_1 | `WETH-USDC` | Liquid pair |
| market_2 | `WETH-USDC` | Same pair |
| order_amount | `0.01` | Start tiny (~$30) |
| min_profitability | `1.0` | 1% min (conservative) |
| slippage_buffer | `0.5` | 0.5% slippage tolerance |

---

## Recommended Chains (Low Fees)

| Chain | Gas Cost | DEX Options | Notes |
|-------|----------|-------------|-------|
| **Arbitrum** | ~$0.10-0.50 | Uniswap, SushiSwap, Camelot | Good liquidity |
| **Polygon** | ~$0.01-0.05 | QuickSwap, SushiSwap, Uniswap | Very cheap |
| **Base** | ~$0.05-0.20 | Uniswap, Aerodrome | Growing ecosystem |
| **Solana** | ~$0.001 | Jupiter, Raydium, Meteora | Different architecture |

---

## Key Configuration Parameters

### `min_profitability`
Minimum profit % to execute a trade.

```
Too low (0.1%):  Many trades, often unprofitable after gas
Too high (5%):   Rarely find opportunities
Sweet spot:      0.5% - 2% depending on chain/gas
```

### `order_amount`
How much to trade per opportunity.

```
Start small:     0.01 ETH or $20-50 equivalent
Scale up after:  Proven profitable over 50+ trades
```

### `slippage_buffer`
Extra price buffer to account for AMM slippage.

```
Low liquidity pairs:   1-2%
High liquidity pairs:  0.3-0.5%
```

### `concurrent_orders_submission`
Whether to submit both orders simultaneously.

```
True:   Faster, but risk if one fails
False:  Safer, waits for first order to confirm
```

---

## Realistic Expectations

### The Math

```
Scenario: $1000 capital, 1% profit target, Arbitrum

Best case (10 trades/day):
  10 trades × 1% × $100 avg = $10/day
  Minus gas: 10 × $0.30 = $3
  Net: ~$7/day

Realistic (2-3 trades/day):
  3 trades × 0.8% × $100 = $2.40/day
  Minus gas: 3 × $0.30 = $0.90
  Net: ~$1.50/day
```

### Competition Reality

- Professional arbitrage bots run 24/7 with:
  - Colocated nodes (faster than you)
  - Custom MEV strategies
  - Larger capital (better rates)

- Your edge:
  - Smaller trades pros ignore
  - Less liquid pairs
  - Newer/smaller chains
  - Learning and fun > max profit

---

## Troubleshooting

### "No arbitrage opportunity" constantly
- Prices are efficiently arbitraged already
- Try different pairs or DEXes
- Lower `min_profitability` (but watch gas costs)

### Transaction failures
- Increase `slippage_buffer`
- Check gas token balance
- Verify wallet connection

### Gateway connection issues
- Ensure Gateway container is running: `docker ps`
- Check logs: `docker logs gateway`
- Regenerate certs if needed

---

## Files to Know

```
~/Documents/hummingbot/
├── conf/                    # Your strategy configs
├── logs/                    # Bot logs
├── hummingbot/
│   └── strategy/
│       └── amm_arb/         # Arbitrage strategy code
│           ├── amm_arb.py   # Main logic
│           └── amm_arb_config_map.py  # Config options
└── docker-compose.yml       # Container setup
```

---

## ~~Original Plan: AMM Arbitrage~~

**Status: RECONSIDERED**

After analyzing Hummingbot's full capabilities, AMM arbitrage is actually one of the hardest strategies due to:
- MEV competition (professional bots frontrun you)
- Thin margins (opportunities gone before you see them)
- Gateway complexity (extra infrastructure)

---

## NEW PLAN: Dollar-A-Day Market Making

**Goal:** Consistent $1/day profit through conservative market making.

### Why Market Making Instead?

| AMM Arb Problems | Market Making Advantages |
|------------------|--------------------------|
| Racing against MEV bots | Earning spread passively |
| Need perfect timing | Works with any timing |
| Complex Gateway setup | Simple exchange connection |
| Two-sided execution risk | Single exchange |

### The Strategy: `dollar_a_day_pmm.py`

Custom script with safety features:
- **Inventory skew adjustment** - Widens spread when heavy on one side
- **Max inventory limits** - Stops buying/selling when too imbalanced
- **P&L tracking** - See exactly what you're making
- **Conservative defaults** - 0.3% spreads, 30s refresh

### Target Math

| Capital | Daily Return | Profit |
|---------|--------------|--------|
| $100 | 1% | $1/day |
| $200 | 0.5% | $1/day |
| $500 | 0.2% | $1/day |

### Phase 1: Paper Trading (Week 1)

```bash
# Start Hummingbot
docker compose up -d
docker attach hummingbot

# Inside Hummingbot CLI
start --script dollar_a_day_pmm.py
```

**Default config (paper trade):**
- Exchange: `kucoin_paper_trade`
- Pair: `ATOM-USDT`
- Order amount: 5 ATOM (~$50)
- Spreads: 0.3% bid/ask
- Refresh: 30 seconds

### Phase 2: Analyze Results (After Week 1)

Run `status` command to see:
- Total trades executed
- Buy/sell volume
- Average spread captured
- Estimated P&L

**What to look for:**
- Fills per day (target: 5-15)
- Inventory balance (should stay near 50/50)
- Spread captured (should be close to configured spread)

### Phase 3: Real Money (Week 2+)

If paper trading shows profit:

1. Edit config: Change `exchange` from `kucoin_paper_trade` to `kucoin`
2. Fund account: $100-200 in USDT + equivalent in base asset
3. Start small: Reduce `order_amount` initially
4. Monitor closely first 24-48 hours

**Pair Selection for Real Trading:**
- Not BTC or ETH (too competitive)
- $1-10M daily volume (liquid but not crowded)
- Good candidates: ATOM, ALGO, INJ, SEI, TIA, AR, NEAR

---

## Configuration Reference

Edit `scripts/dollar_a_day_pmm.py` to adjust:

```python
# Conservative (start here)
base_bid_spread = 0.003      # 0.3%
base_ask_spread = 0.003      # 0.3%
order_refresh_time = 30      # seconds

# Aggressive (after proven profitable)
base_bid_spread = 0.001      # 0.1%
base_ask_spread = 0.001      # 0.1%
order_refresh_time = 15      # seconds
```

### Safety Settings

```python
# Inventory management
inventory_skew_enabled = True
target_base_ratio = 0.5      # 50/50 balance
max_inventory_ratio = 0.8    # Stop orders if 80%+ one-sided
inventory_skew_intensity = 1.0
```

---

## Quick Reference Commands

```bash
# Start bot
docker compose up -d && docker attach hummingbot

# In Hummingbot CLI
start --script dollar_a_day_pmm.py  # Start strategy
status                               # Check P&L and orders
stop                                 # Stop strategy
exit                                 # Exit bot
```

---

## Files

```
~/Documents/hummingbot/
├── scripts/
│   └── dollar_a_day_pmm.py    # <-- YOUR STRATEGY
├── conf/                       # Saved configs
├── logs/                       # Bot logs
└── docker-compose.yml          # Container setup
```

---

## Next Steps

1. [x] ~~Research strategies~~ (Done - chose PMM over AMM arb)
2. [x] ~~Create custom strategy~~ (Done - dollar_a_day_pmm.py)
3. [ ] Start Docker and run paper trading
4. [ ] Run for 1 week, check `status` daily
5. [ ] Analyze: fills, spreads, inventory, P&L
6. [ ] If profitable: switch to real exchange with $100-200
7. [ ] Scale up if consistently profitable

---

## Resources

- [Hummingbot Docs](https://hummingbot.org)
- [Script Strategy Guide](https://hummingbot.org/scripts/)
- [Discord Community](https://discord.gg/hummingbot)

---

*Created: January 2026*
*Updated: Strategy changed from AMM Arb to Market Making*
*Goal: Consistent $1/day through conservative market making*
