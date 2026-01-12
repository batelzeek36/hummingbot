"""
Hyperliquid Strategy Optimizer
Analyzes ALL Hyperliquid perpetual coins to find optimal choices for each strategy:
- Funding Harvesting: Best funding rate opportunities
- Grid Trading: Best liquidity + moderate volatility
- Momentum/Directional: Best trending coins with manageable risk

Data Sources:
- Hyperliquid API (direct): Funding rates, open interest, mark prices
- CCXT Hyperliquid: OHLCV data for volatility analysis

Run: python scripts/analytics/hyperliquid_strategy_optimizer.py
"""

import ccxt
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Hyperliquid API
HYPERLIQUID_INFO_API = "https://api.hyperliquid.xyz/info"

# Leverage safety thresholds (max daily move that's safe)
LEVERAGE_THRESHOLDS = {
    8: 10.0,   # 8x safe if max daily < 10%
    5: 15.0,   # 5x safe if max daily < 15%
    3: 25.0,   # 3x safe if max daily < 25%
    2: 40.0,   # 2x safe if max daily < 40%
}


@dataclass
class CoinAnalysis:
    """Complete analysis for a single coin."""
    symbol: str

    # Funding data
    funding_rate: float = 0.0
    funding_apr: float = 0.0
    funding_direction: str = ""  # LONG or SHORT to receive

    # Market data
    mark_price: float = 0.0
    open_interest: float = 0.0
    volume_24h: float = 0.0

    # Volatility data
    avg_daily_volatility: float = 0.0
    max_daily_move: float = 0.0
    avg_hourly_range: float = 0.0

    # Derived metrics
    safe_leverage: int = 2
    effective_funding_apr: float = 0.0
    grid_score: float = 0.0
    momentum_score: float = 0.0
    funding_score: float = 0.0

    # Trend data
    trend_direction: str = ""  # UP, DOWN, SIDEWAYS
    rsi_14: float = 50.0


def fetch_hyperliquid_meta() -> Tuple[List[str], Dict]:
    """Fetch all available coins and their metadata from Hyperliquid."""
    try:
        response = requests.post(
            HYPERLIQUID_INFO_API,
            json={"type": "metaAndAssetCtxs"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        coins = []
        coin_data = {}

        if len(data) >= 2:
            meta = data[0]
            asset_ctxs = data[1]

            for i, asset in enumerate(meta.get('universe', [])):
                coin = asset['name']
                coins.append(coin)

                if i < len(asset_ctxs):
                    ctx = asset_ctxs[i]
                    coin_data[coin] = {
                        'funding_rate': float(ctx.get('funding', 0)),
                        'mark_price': float(ctx.get('markPx', 0)),
                        'open_interest': float(ctx.get('openInterest', 0)),
                        'volume_24h': float(ctx.get('dayNtlVlm', 0)),
                    }

        return coins, coin_data
    except Exception as e:
        print(f"Error fetching Hyperliquid meta: {e}")
        return [], {}


def fetch_ohlcv_ccxt(symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data using ccxt Hyperliquid."""
    try:
        exchange = ccxt.hyperliquid({'enableRateLimit': True})

        # Hyperliquid uses format like "BTC/USDC:USDC"
        market_symbol = f"{symbol}/USDC:USDC"

        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        ohlcv = exchange.fetch_ohlcv(market_symbol, '1h', since=since, limit=500)

        if not ohlcv:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df
    except Exception as e:
        # Silently fail for coins that don't exist or have issues
        return None


def calculate_volatility_metrics(df: pd.DataFrame) -> Dict:
    """Calculate volatility metrics from OHLCV data."""
    if df is None or len(df) < 24:
        return {}

    # Hourly range
    df['hourly_range'] = (df['high'] - df['low']) / df['low'] * 100

    # Daily aggregation
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    if len(daily) < 2:
        return {}

    daily['daily_range'] = (daily['high'] - daily['low']) / daily['low'] * 100
    daily['daily_return'] = daily['close'].pct_change() * 100

    # Calculate RSI
    delta = daily['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Trend detection
    if len(daily) >= 5:
        recent_return = (daily['close'].iloc[-1] - daily['close'].iloc[-5]) / daily['close'].iloc[-5] * 100
        if recent_return > 5:
            trend = "UP"
        elif recent_return < -5:
            trend = "DOWN"
        else:
            trend = "SIDEWAYS"
    else:
        trend = "UNKNOWN"

    return {
        'avg_daily_volatility': daily['daily_range'].mean(),
        'max_daily_move': daily['daily_range'].max(),
        'avg_hourly_range': df['hourly_range'].mean(),
        'rsi_14': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0,
        'trend_direction': trend,
    }


def determine_safe_leverage(max_daily_move: float) -> int:
    """Determine safe leverage based on max daily move."""
    for leverage, threshold in sorted(LEVERAGE_THRESHOLDS.items(), reverse=True):
        if max_daily_move < threshold:
            return leverage
    return 2  # Default to safest


def calculate_grid_score(analysis: CoinAnalysis) -> float:
    """
    Grid trading score: want moderate volatility + high liquidity.
    - Too low volatility = no profit opportunities
    - Too high volatility = stop losses triggered
    - Need good volume for fills
    """
    # Ideal daily volatility is 2-8%
    vol = analysis.avg_daily_volatility
    if vol < 1:
        vol_score = vol * 30  # Low vol penalized
    elif vol <= 8:
        vol_score = 50 + (vol - 1) * 7  # Sweet spot
    else:
        vol_score = max(0, 100 - (vol - 8) * 10)  # High vol penalized

    # Volume score (log scale, normalized)
    vol_24h = analysis.volume_24h
    if vol_24h > 0:
        volume_score = min(100, np.log10(vol_24h + 1) * 15)
    else:
        volume_score = 0

    # Open interest score
    oi = analysis.open_interest
    if oi > 0:
        oi_score = min(100, np.log10(oi + 1) * 20)
    else:
        oi_score = 0

    # Sideways trend is best for grid
    trend_score = 100 if analysis.trend_direction == "SIDEWAYS" else 50

    # Weighted combination
    return (vol_score * 0.35 + volume_score * 0.30 + oi_score * 0.20 + trend_score * 0.15)


def calculate_momentum_score(analysis: CoinAnalysis) -> float:
    """
    Momentum trading score: want trending coins with manageable risk.
    - Clear trend direction
    - RSI at extremes (oversold/overbought)
    - Good liquidity for execution
    - Not too volatile (can't handle 8x)
    """
    # Trend strength
    if analysis.trend_direction in ["UP", "DOWN"]:
        trend_score = 80
    else:
        trend_score = 30

    # RSI extremes (oversold < 30, overbought > 70)
    rsi = analysis.rsi_14
    if rsi < 25 or rsi > 75:
        rsi_score = 100
    elif rsi < 30 or rsi > 70:
        rsi_score = 70
    elif rsi < 40 or rsi > 60:
        rsi_score = 40
    else:
        rsi_score = 20

    # Leverage score (higher safe leverage = better for momentum)
    leverage_score = analysis.safe_leverage * 12

    # Volume for execution
    vol_24h = analysis.volume_24h
    if vol_24h > 0:
        volume_score = min(100, np.log10(vol_24h + 1) * 15)
    else:
        volume_score = 0

    return (trend_score * 0.30 + rsi_score * 0.25 + leverage_score * 0.25 + volume_score * 0.20)


def calculate_funding_score(analysis: CoinAnalysis) -> float:
    """
    Funding score: effective APR considering leverage constraints.
    """
    return analysis.effective_funding_apr


def analyze_all_coins(coins: List[str], coin_data: Dict, verbose: bool = True) -> List[CoinAnalysis]:
    """Analyze all coins for strategy optimization."""
    results = []

    total = len(coins)
    for i, coin in enumerate(coins):
        if verbose and (i + 1) % 10 == 0:
            print(f"  Analyzing {i+1}/{total}...", end='\r')

        analysis = CoinAnalysis(symbol=coin)

        # Funding data
        if coin in coin_data:
            data = coin_data[coin]
            analysis.funding_rate = data['funding_rate']
            analysis.funding_apr = abs(data['funding_rate']) * 8760 * 100
            analysis.funding_direction = "LONG" if data['funding_rate'] > 0 else "SHORT"
            analysis.mark_price = data['mark_price']
            analysis.open_interest = data['open_interest']
            analysis.volume_24h = data['volume_24h']

        # Volatility data (from ccxt)
        df = fetch_ohlcv_ccxt(coin, days=7)
        if df is not None:
            vol_metrics = calculate_volatility_metrics(df)
            if vol_metrics:
                analysis.avg_daily_volatility = vol_metrics.get('avg_daily_volatility', 0)
                analysis.max_daily_move = vol_metrics.get('max_daily_move', 0)
                analysis.avg_hourly_range = vol_metrics.get('avg_hourly_range', 0)
                analysis.rsi_14 = vol_metrics.get('rsi_14', 50)
                analysis.trend_direction = vol_metrics.get('trend_direction', 'UNKNOWN')

        # Derived metrics
        if analysis.max_daily_move > 0:
            analysis.safe_leverage = determine_safe_leverage(analysis.max_daily_move)

        analysis.effective_funding_apr = analysis.funding_apr * analysis.safe_leverage

        # Strategy scores
        analysis.grid_score = calculate_grid_score(analysis)
        analysis.momentum_score = calculate_momentum_score(analysis)
        analysis.funding_score = calculate_funding_score(analysis)

        results.append(analysis)

    if verbose:
        print(f"  Analyzed {total} coins.              ")

    return results


def print_optimization_report(results: List[CoinAnalysis]):
    """Print comprehensive optimization report."""
    print("\n" + "=" * 100)
    print("  HYPERLIQUID STRATEGY OPTIMIZER - COMPREHENSIVE ANALYSIS")
    print("=" * 100)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Coins Analyzed: {len(results)}")
    print("=" * 100)

    # Filter out coins with no data
    valid_results = [r for r in results if r.volume_24h > 0]

    # =========================================================================
    # FUNDING HARVESTING RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("  STRATEGY 1: FUNDING HARVESTING")
    print("  Goal: Maximize funding payments with safe leverage")
    print("=" * 100)

    funding_sorted = sorted(valid_results, key=lambda x: x.effective_funding_apr, reverse=True)[:15]

    print(f"\n  {'Rank':<6} {'Coin':<10} {'APR':<10} {'Direction':<10} {'Safe Lev':<10} {'Eff APR':<12} {'Max Move':<10}")
    print("-" * 100)

    for i, r in enumerate(funding_sorted, 1):
        indicator = ">>>" if r.effective_funding_apr > 100 else "   "
        print(f"  {indicator}{i:<3} {r.symbol:<10} {r.funding_apr:>7.1f}%   {r.funding_direction:<10} {r.safe_leverage}x         {r.effective_funding_apr:>8.1f}%    {r.max_daily_move:>7.1f}%")

    # Current bot coins check
    current_funding_coins = ["ANIME", "VVV", "HYPE", "PURR", "JEFF", "MOG", "WIF", "PEPE", "BONK", "DOGE", "ETH"]
    print(f"\n  CURRENT BOT FUNDING COINS: {', '.join(current_funding_coins)}")

    better_options = [r for r in funding_sorted[:10] if r.symbol not in current_funding_coins]
    if better_options:
        print(f"  POTENTIALLY BETTER OPTIONS NOT IN CONFIG:")
        for r in better_options[:5]:
            print(f"    - {r.symbol}: {r.effective_funding_apr:.1f}% effective APR")
    else:
        print("  Current config looks optimal for funding!")

    # =========================================================================
    # GRID TRADING RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("  STRATEGY 2: GRID TRADING")
    print("  Goal: Moderate volatility + high liquidity for range trading")
    print("=" * 100)

    grid_sorted = sorted(valid_results, key=lambda x: x.grid_score, reverse=True)[:15]

    print(f"\n  {'Rank':<6} {'Coin':<10} {'Score':<10} {'Avg Vol':<10} {'Max Move':<10} {'Volume 24h':<15} {'Trend':<10}")
    print("-" * 100)

    for i, r in enumerate(grid_sorted, 1):
        vol_str = f"${r.volume_24h/1e6:.1f}M" if r.volume_24h >= 1e6 else f"${r.volume_24h/1e3:.0f}K"
        indicator = ">>>" if r.grid_score > 70 else "   "
        print(f"  {indicator}{i:<3} {r.symbol:<10} {r.grid_score:>6.1f}     {r.avg_daily_volatility:>6.1f}%    {r.max_daily_move:>7.1f}%   {vol_str:<15} {r.trend_direction:<10}")

    current_grid = "SOL"
    current_in_list = [r for r in grid_sorted[:10] if r.symbol == current_grid]
    if current_in_list:
        rank = grid_sorted.index(current_in_list[0]) + 1
        print(f"\n  CURRENT GRID PAIR: {current_grid} (Rank #{rank})")
    else:
        print(f"\n  CURRENT GRID PAIR: {current_grid} (NOT in top 10)")

    if grid_sorted[0].symbol != current_grid:
        print(f"  CONSIDER SWITCHING TO: {grid_sorted[0].symbol} (Score: {grid_sorted[0].grid_score:.1f})")

    # =========================================================================
    # MOMENTUM TRADING RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("  STRATEGY 3: MOMENTUM/DIRECTIONAL")
    print("  Goal: Trending coins with good risk/reward at high leverage")
    print("=" * 100)

    momentum_sorted = sorted(valid_results, key=lambda x: x.momentum_score, reverse=True)[:15]

    print(f"\n  {'Rank':<6} {'Coin':<10} {'Score':<10} {'RSI':<10} {'Trend':<12} {'Safe Lev':<10} {'Max Move':<10}")
    print("-" * 100)

    for i, r in enumerate(momentum_sorted, 1):
        indicator = ">>>" if r.momentum_score > 60 else "   "
        rsi_indicator = "OS" if r.rsi_14 < 30 else ("OB" if r.rsi_14 > 70 else "  ")
        print(f"  {indicator}{i:<3} {r.symbol:<10} {r.momentum_score:>6.1f}     {r.rsi_14:>5.1f} {rsi_indicator}  {r.trend_direction:<12} {r.safe_leverage}x         {r.max_daily_move:>7.1f}%")

    current_momentum = "BTC"
    current_in_list = [r for r in momentum_sorted[:10] if r.symbol == current_momentum]
    if current_in_list:
        rank = momentum_sorted.index(current_in_list[0]) + 1
        print(f"\n  CURRENT MOMENTUM PAIR: {current_momentum} (Rank #{rank})")
    else:
        print(f"\n  CURRENT MOMENTUM PAIR: {current_momentum} (NOT in top 10)")

    # Coins with RSI extremes right now
    oversold = [r for r in valid_results if r.rsi_14 < 30]
    overbought = [r for r in valid_results if r.rsi_14 > 70]

    if oversold:
        print(f"\n  OVERSOLD COINS (RSI < 30) - Potential LONG:")
        for r in sorted(oversold, key=lambda x: x.rsi_14)[:5]:
            print(f"    - {r.symbol}: RSI {r.rsi_14:.1f}, Safe leverage {r.safe_leverage}x")

    if overbought:
        print(f"\n  OVERBOUGHT COINS (RSI > 70) - Potential SHORT:")
        for r in sorted(overbought, key=lambda x: x.rsi_14, reverse=True)[:5]:
            print(f"    - {r.symbol}: RSI {r.rsi_14:.1f}, Safe leverage {r.safe_leverage}x")

    # =========================================================================
    # SUMMARY RECOMMENDATIONS
    # =========================================================================
    print("\n" + "=" * 100)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 100)

    print("\n  FUNDING HUNTER:")
    top_funding = funding_sorted[:3]
    print(f"    Top 3: {', '.join([f'{r.symbol} ({r.effective_funding_apr:.0f}%)' for r in top_funding])}")

    print("\n  GRID TRADING:")
    top_grid = grid_sorted[:3]
    print(f"    Top 3: {', '.join([f'{r.symbol} ({r.grid_score:.0f})' for r in top_grid])}")

    print("\n  MOMENTUM:")
    top_momentum = momentum_sorted[:3]
    print(f"    Top 3: {', '.join([f'{r.symbol} ({r.momentum_score:.0f})' for r in top_momentum])}")

    # Compare to current config
    print("\n" + "-" * 100)
    print("  COMPARISON TO CURRENT BOT CONFIG")
    print("-" * 100)

    print("\n  | Strategy | Current | Optimal | Change Recommended? |")
    print("  |----------|---------|---------|---------------------|")

    # Funding
    current_best_funding = max([r for r in valid_results if r.symbol in current_funding_coins],
                               key=lambda x: x.effective_funding_apr, default=None)
    optimal_funding = funding_sorted[0]
    funding_change = "YES" if optimal_funding.symbol not in current_funding_coins else "NO"
    if current_best_funding:
        print(f"  | Funding  | {current_best_funding.symbol} ({current_best_funding.effective_funding_apr:.0f}%) | {optimal_funding.symbol} ({optimal_funding.effective_funding_apr:.0f}%) | {funding_change} |")

    # Grid
    current_grid_data = next((r for r in valid_results if r.symbol == "SOL"), None)
    optimal_grid = grid_sorted[0]
    grid_change = "YES" if optimal_grid.symbol != "SOL" else "NO"
    if current_grid_data:
        print(f"  | Grid     | SOL ({current_grid_data.grid_score:.0f}) | {optimal_grid.symbol} ({optimal_grid.grid_score:.0f}) | {grid_change} |")

    # Momentum
    current_momentum_data = next((r for r in valid_results if r.symbol == "BTC"), None)
    optimal_momentum = momentum_sorted[0]
    momentum_change = "YES" if optimal_momentum.symbol != "BTC" else "NO"
    if current_momentum_data:
        print(f"  | Momentum | BTC ({current_momentum_data.momentum_score:.0f}) | {optimal_momentum.symbol} ({optimal_momentum.momentum_score:.0f}) | {momentum_change} |")

    print("\n" + "=" * 100)


def main():
    print("\n" + "=" * 60)
    print("  HYPERLIQUID STRATEGY OPTIMIZER")
    print("=" * 60)

    print("\n[1/3] Fetching Hyperliquid metadata...")
    coins, coin_data = fetch_hyperliquid_meta()
    print(f"  Found {len(coins)} coins on Hyperliquid")

    print("\n[2/3] Analyzing coins (this may take a few minutes)...")
    results = analyze_all_coins(coins, coin_data, verbose=True)

    print("\n[3/3] Generating optimization report...")
    print_optimization_report(results)


if __name__ == "__main__":
    main()
