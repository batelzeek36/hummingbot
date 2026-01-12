# Hummingbot Project Overview

## Purpose
Hummingbot is an open-source framework for designing and deploying automated trading strategies (bots) that can run on centralized and decentralized exchanges. Users have generated over $34 billion in trading volume across 140+ venues.

## License
Apache 2.0

## Tech Stack
- **Language**: Python 3 with Cython (`.pyx`) for performance-critical components
- **Build System**: setuptools with Cython compilation
- **Dependencies**: aiohttp, pandas, numpy, web3, pydantic, sqlalchemy, etc.
- **Deployment**: Docker (primary), source installation
- **Gateway**: TypeScript-based middleware for DEX connections (separate repo)

## Key Components
- `hummingbot/` - Main source code
  - `client/` - CLI interface
  - `connector/` - Exchange connectors (CEX and DEX)
  - `strategy/` - V1 trading strategies (Cython-based)
  - `strategy_v2/` - V2 modern framework (Python-based)
  - `core/` - Core functionality, events, data types
  - `data_feed/` - Price feeds and candle data
- `scripts/` - Python-based strategy scripts (easiest to customize)
- `controllers/` - V2 strategy controllers
- `test/` - Unit tests
- `conf/` - Configuration files
- `logs/` - Log files

## Exchange Support
- 26+ spot exchange connectors
- 12+ perpetual/futures connectors
- AMM DEX via Gateway middleware
- Paper trading for testing

## Strategy Types
1. **V1 Strategies** - Production-grade Cython (pure_market_making, amm_arb, etc.)
2. **V2 Framework** - Modern Python with Controllers and Executors
3. **Script Strategies** - Simple Python scripts (ScriptStrategyBase)
