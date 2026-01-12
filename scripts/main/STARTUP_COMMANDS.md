# 🚀 Hummingbot Multi-Strategy Bot - Startup Commands

## Quick Start (Copy & Paste)

```bash
cd /Users/kingkamehameha/Documents/hummingbot && docker compose down && docker compose up -d && docker attach hummingbot
```

### Step 1: Navigate to Hummingbot folder
```bash
cd /Users/kingkamehameha/Documents/hummingbot
```

### Step 2: Full Docker Restart (Recommended)
```bash
docker compose down && docker compose up -d && docker attach hummingbot
```

### Step 3: Inside Hummingbot - Start the bot
```
start --script multi_strategy_bot.py
```

### Step 4: Check status (wait 30-60 seconds first)
```
status
```

---

## One-Liner Full Restart (from anywhere)

```bash
cd /Users/kingkamehameha/Documents/hummingbot && docker compose down && docker compose up -d && docker attach hummingbot
```

---

## Useful Hummingbot Commands

| Command | What It Does |
|---------|--------------|
| `status` | Show bot dashboard |
| `history` | Show recent trades |
| `balance` | Show current balances |
| `stop` | Stop the bot |
| `exit` | Exit Hummingbot |
| `logs` | View logs (scroll with arrows, `q` to quit) |

---

## If "Invalid Strategy" Error

Full Docker reset:
```bash
cd /Users/kingkamehameha/Documents/hummingbot
docker compose down
docker compose up -d
docker attach hummingbot
```

Then start bot:
```
start --script multi_strategy_bot.py
```

---

## If "Invalid Nonce" Error from Kraken

1. Go to **Kraken.com** → Security → API
2. Edit your API key
3. Set **Nonce Window** to `10` or higher
4. Save

---

## Generate Performance Report (after trading)

```bash
cd /Users/kingkamehameha/Documents/hummingbot
python scripts/analytics/performance_tracker.py --days 7 --capital 100 --print
```

---

## Emergency Stop

Inside Hummingbot:
```
stop
```

Or kill Docker entirely:
```bash
docker compose down
```

---

## Check Performance (After Trading)

```bash
python scripts/analytics/performance_tracker.py --days 1 --capital 113 --print
```

Options:
- `--days 7` - Last 7 days of data
- `--capital 250` - If you upgraded your capital
- `--csv report.csv` - Export to CSV file

## Full XRP Performance Report

```bash
cd /Users/kingkamehameha/Documents/hummingbot && python scripts/analytics/performance_tracker.py --days 1 --capital 113 --print
```

---

*Dollar-A-Day Project* 💰
