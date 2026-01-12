"""
Quick Hyperliquid Strategy Check
Analyzes all Hyperliquid coins with sufficient liquidity (>$1M daily volume).

Run: python scripts/analytics/quick_strategy_check.py

Options:
  --min-volume 500000    Set minimum 24h volume threshold (default: $1M)
  --include ANIME,PEPE   Force include specific coins regardless of volume
"""

import ccxt
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

HYPERLIQUID_INFO_API = "https://api.hyperliquid.xyz/info"

# Default minimum volume threshold ($1M)
DEFAULT_MIN_VOLUME = 1_000_000

# Current bot config
CURRENT_FUNDING_COINS = ["ANIME", "VVV", "HYPE", "PURR", "JEFF", "MOG", "WIF", "PEPE", "BONK", "DOGE", "ETH"]
CURRENT_GRID = "SOL"
CURRENT_MOMENTUM = "BTC"

# Always include these coins even if below volume threshold (for comparison)
ALWAYS_INCLUDE = ["BTC", "ETH", "SOL", "ANIME", "HYPE", "VVV"]


def fetch_all_coin_data() -> List[Dict]:
    """Fetch all coin data from Hyperliquid."""
    try:
        response = requests.post(
            HYPERLIQUID_INFO_API,
            json={"type": "metaAndAssetCtxs"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        data = response.json()

        results = []
        if len(data) >= 2:
            meta = data[0]
            ctxs = data[1]

            for i, asset in enumerate(meta.get('universe', [])):
                if i < len(ctxs):
                    ctx = ctxs[i]
                    coin = asset['name']
                    funding = float(ctx.get('funding', 0))
                    apr = abs(funding) * 8760 * 100

                    results.append({
                        'coin': coin,
                        'funding_rate': funding,
                        'funding_apr': apr,
                        'direction': 'LONG' if funding > 0 else 'SHORT',
                        'mark_price': float(ctx.get('markPx', 0)),
                        'open_interest': float(ctx.get('openInterest', 0)),
                        'volume_24h': float(ctx.get('dayNtlVlm', 0)),
                    })

        return results
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_volatility(coin: str) -> Dict:
    """Get volatility metrics for a coin using ccxt."""
    try:
        exchange = ccxt.hyperliquid({'enableRateLimit': True})
        symbol = f"{coin}/USDC:USDC"
        since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=200)
        if not ohlcv or len(ohlcv) < 24:
            return {}

        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])

        # Hourly range
        df['range'] = (df['h'] - df['l']) / df['l'] * 100

        # Daily aggregation
        df['date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
        daily = df.groupby('date').agg({'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last'})
        daily['range'] = (daily['h'] - daily['l']) / daily['l'] * 100
        daily['ret'] = daily['c'].pct_change() * 100

        # RSI
        delta = daily['c'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Trend
        if len(daily) >= 5:
            ret = (daily['c'].iloc[-1] - daily['c'].iloc[-5]) / daily['c'].iloc[-5] * 100
            trend = "UP" if ret > 5 else ("DOWN" if ret < -5 else "SIDEWAYS")
        else:
            trend = "?"

        return {
            'avg_daily_vol': daily['range'].mean(),
            'max_daily_move': daily['range'].max(),
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'trend': trend,
        }
    except:
        return {}


def safe_leverage(max_move: float) -> int:
    """Determine safe leverage."""
    if max_move < 10: return 8
    if max_move < 15: return 5
    if max_move < 25: return 3
    return 2


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperliquid Strategy Optimizer')
    parser.add_argument('--min-volume', type=float, default=DEFAULT_MIN_VOLUME,
                        help=f'Minimum 24h volume threshold (default: ${DEFAULT_MIN_VOLUME/1e6:.1f}M)')
    parser.add_argument('--include', type=str, default='',
                        help='Comma-separated list of coins to always include')
    args = parser.parse_args()

    min_volume = args.min_volume
    extra_includes = [c.strip().upper() for c in args.include.split(',') if c.strip()]
    force_include = set(ALWAYS_INCLUDE + extra_includes + CURRENT_FUNDING_COINS)

    print("\n" + "=" * 90)
    print("  HYPERLIQUID STRATEGY OPTIMIZER")
    print("=" * 90)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Min Volume: ${min_volume/1e6:.1f}M")

    print("\n[1/2] Fetching Hyperliquid data...")
    all_coins = fetch_all_coin_data()
    print(f"  Found {len(all_coins)} total coins")

    # Filter by volume threshold OR force-include list
    filtered_coins = [
        c for c in all_coins
        if c['volume_24h'] >= min_volume or c['coin'] in force_include
    ]

    # Sort by volume for consistent ordering
    filtered_coins.sort(key=lambda x: x['volume_24h'], reverse=True)

    above_threshold = len([c for c in filtered_coins if c['volume_24h'] >= min_volume])
    force_included = len(filtered_coins) - above_threshold

    print(f"  Above ${min_volume/1e6:.1f}M volume: {above_threshold} coins")
    print(f"  Force-included (config): {force_included} coins")
    print(f"  Total to analyze: {len(filtered_coins)} coins")

    print(f"\n[2/2] Analyzing {len(filtered_coins)} coins...")

    results = []
    total = len(filtered_coins)
    for i, c in enumerate(filtered_coins):
        print(f"  {i+1}/{total}: {c['coin']}...".ljust(30), end='\r')
        vol = get_volatility(c['coin'])
        if vol:
            c.update(vol)
            c['safe_lev'] = safe_leverage(vol.get('max_daily_move', 50))
            c['eff_apr'] = c['funding_apr'] * c['safe_lev']
        else:
            c['avg_daily_vol'] = 0
            c['max_daily_move'] = 50
            c['rsi'] = 50
            c['trend'] = '?'
            c['safe_lev'] = 2
            c['eff_apr'] = c['funding_apr'] * 2
        results.append(c)

    print("  Done!                    ")

    # ===== FUNDING ANALYSIS =====
    print("\n" + "=" * 90)
    print("  FUNDING HARVESTING ANALYSIS")
    print("=" * 90)

    funding_sorted = sorted(results, key=lambda x: x['eff_apr'], reverse=True)

    print(f"\n  {'#':<4} {'Coin':<8} {'APR':<10} {'Dir':<8} {'Lev':<6} {'Eff APR':<12} {'In Config?':<12}")
    print("-" * 90)

    for i, c in enumerate(funding_sorted[:15], 1):
        in_config = "YES" if c['coin'] in CURRENT_FUNDING_COINS else "NO"
        flag = ">>>" if c['eff_apr'] > 100 and in_config == "NO" else "   "
        print(f"  {flag}{i:<1} {c['coin']:<8} {c['funding_apr']:>7.1f}%  {c['direction']:<8} {c['safe_lev']}x     {c['eff_apr']:>8.1f}%     {in_config}")

    # Missing opportunities
    missing = [c for c in funding_sorted[:10] if c['coin'] not in CURRENT_FUNDING_COINS]
    if missing:
        print(f"\n  MISSING FROM CONFIG (consider adding):")
        for c in missing:
            print(f"    + {c['coin']}: {c['eff_apr']:.1f}% effective APR")

    # ===== GRID ANALYSIS =====
    print("\n" + "=" * 90)
    print("  GRID TRADING ANALYSIS")
    print("=" * 90)

    # Grid score: moderate volatility (2-8% ideal) + high volume + sideways trend
    for c in results:
        vol = c.get('avg_daily_vol', 0)
        if vol < 1:
            vol_score = vol * 30
        elif vol <= 8:
            vol_score = 50 + (vol - 1) * 7
        else:
            vol_score = max(0, 100 - (vol - 8) * 10)

        vol_24h = c['volume_24h']
        volume_score = min(100, np.log10(vol_24h + 1) * 15) if vol_24h > 0 else 0

        trend_score = 100 if c.get('trend') == 'SIDEWAYS' else 50

        c['grid_score'] = vol_score * 0.4 + volume_score * 0.4 + trend_score * 0.2

    grid_sorted = sorted(results, key=lambda x: x['grid_score'], reverse=True)

    print(f"\n  {'#':<4} {'Coin':<8} {'Score':<8} {'Avg Vol':<10} {'Max Move':<10} {'Trend':<10} {'Current?':<10}")
    print("-" * 90)

    for i, c in enumerate(grid_sorted[:10], 1):
        current = "<<<" if c['coin'] == CURRENT_GRID else ""
        print(f"  {i:<4} {c['coin']:<8} {c['grid_score']:>5.1f}   {c.get('avg_daily_vol', 0):>6.1f}%    {c.get('max_daily_move', 0):>7.1f}%   {c.get('trend', '?'):<10} {current}")

    sol_data = next((c for c in results if c['coin'] == 'SOL'), None)
    best_grid = grid_sorted[0]
    if sol_data and best_grid['coin'] != 'SOL':
        print(f"\n  Current: SOL (score {sol_data['grid_score']:.1f})")
        print(f"  Better:  {best_grid['coin']} (score {best_grid['grid_score']:.1f})")

    # ===== MOMENTUM ANALYSIS =====
    print("\n" + "=" * 90)
    print("  MOMENTUM TRADING ANALYSIS")
    print("=" * 90)

    # Momentum score: trend + RSI extremes + safe leverage
    for c in results:
        trend_score = 80 if c.get('trend') in ['UP', 'DOWN'] else 30
        rsi = c.get('rsi', 50)
        rsi_score = 100 if (rsi < 25 or rsi > 75) else (70 if (rsi < 30 or rsi > 70) else 30)
        lev_score = c['safe_lev'] * 12
        c['mom_score'] = trend_score * 0.35 + rsi_score * 0.35 + lev_score * 0.30

    mom_sorted = sorted(results, key=lambda x: x['mom_score'], reverse=True)

    print(f"\n  {'#':<4} {'Coin':<8} {'Score':<8} {'RSI':<10} {'Trend':<10} {'Safe Lev':<10} {'Current?':<10}")
    print("-" * 90)

    for i, c in enumerate(mom_sorted[:10], 1):
        current = "<<<" if c['coin'] == CURRENT_MOMENTUM else ""
        rsi = c.get('rsi', 50)
        rsi_flag = " OS" if rsi < 30 else (" OB" if rsi > 70 else "")
        print(f"  {i:<4} {c['coin']:<8} {c['mom_score']:>5.1f}   {rsi:>5.1f}{rsi_flag:<4} {c.get('trend', '?'):<10} {c['safe_lev']}x         {current}")

    # RSI extremes
    oversold = [c for c in results if c.get('rsi', 50) < 30]
    overbought = [c for c in results if c.get('rsi', 50) > 70]

    if oversold:
        print(f"\n  OVERSOLD (RSI < 30) - potential LONG:")
        for c in sorted(oversold, key=lambda x: x.get('rsi', 50))[:5]:
            print(f"    {c['coin']}: RSI {c.get('rsi', 50):.1f}")

    if overbought:
        print(f"\n  OVERBOUGHT (RSI > 70) - potential SHORT:")
        for c in sorted(overbought, key=lambda x: x.get('rsi', 50), reverse=True)[:5]:
            print(f"    {c['coin']}: RSI {c.get('rsi', 50):.1f}")

    # ===== SUMMARY =====
    print("\n" + "=" * 90)
    print("  SUMMARY - CHANGES RECOMMENDED?")
    print("=" * 90)

    print("\n  | Strategy | Current | Best Available | Action |")
    print("  |----------|---------|----------------|--------|")

    # Funding
    best_funding = funding_sorted[0]
    current_funding_in = [c for c in funding_sorted if c['coin'] in CURRENT_FUNDING_COINS]
    if current_funding_in:
        best_current = current_funding_in[0]
        action = "OK" if best_funding['coin'] in CURRENT_FUNDING_COINS else f"Add {best_funding['coin']}"
        print(f"  | Funding  | {best_current['coin']} ({best_current['eff_apr']:.0f}%) | {best_funding['coin']} ({best_funding['eff_apr']:.0f}%) | {action} |")

    # Grid
    action = "OK" if best_grid['coin'] == CURRENT_GRID else f"Consider {best_grid['coin']}"
    sol_score = sol_data['grid_score'] if sol_data else 0
    print(f"  | Grid     | {CURRENT_GRID} ({sol_score:.0f}) | {best_grid['coin']} ({best_grid['grid_score']:.0f}) | {action} |")

    # Momentum
    btc_data = next((c for c in results if c['coin'] == 'BTC'), None)
    best_mom = mom_sorted[0]
    btc_score = btc_data['mom_score'] if btc_data else 0
    action = "OK" if best_mom['coin'] == CURRENT_MOMENTUM else f"Consider {best_mom['coin']}"
    print(f"  | Momentum | {CURRENT_MOMENTUM} ({btc_score:.0f}) | {best_mom['coin']} ({best_mom['mom_score']:.0f}) | {action} |")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
