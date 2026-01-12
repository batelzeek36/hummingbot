"""
Volatility Analysis for Hyperliquid Monster Bot V2
Analyzes historical price data to determine safe leverage levels.

Checks:
1. Max daily move (would 8x get liquidated?)
2. Max hourly move (flash crash risk)
3. How often 12.5%+ moves occur (8x liquidation events)
4. Recommended leverage per coin
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Coins to analyze
# Some memecoins may not be on Binance - we'll handle that
COINS_TO_ANALYZE = {
    # Coin: (exchange, symbol)
    'BTC': ('binance', 'BTC/USDT'),
    'SOL': ('binance', 'SOL/USDT'),
    'DOGE': ('binance', 'DOGE/USDT'),
    'PEPE': ('binance', 'PEPE/USDT'),
    'BONK': ('binance', 'BONK/USDT'),
    'WIF': ('binance', 'WIF/USDT'),
    'ANIME': ('bybit', 'ANIME/USDT'),  # Try bybit for newer memecoins
    'HYPE': ('bybit', 'HYPE/USDT'),
    'MOG': ('bybit', 'MOG/USDT'),
}

# Liquidation thresholds by leverage
LIQUIDATION_THRESHOLDS = {
    3: 33.3,   # 3x = ~33% adverse move
    5: 20.0,   # 5x = ~20% adverse move
    8: 12.5,   # 8x = ~12.5% adverse move
    10: 10.0,  # 10x = ~10% adverse move
}

def get_ohlcv_data(exchange_id: str, symbol: str, timeframe: str = '1h', days: int = 30):
    """Fetch OHLCV data from exchange."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})

        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df
    except Exception as e:
        print(f"  Error fetching {symbol} from {exchange_id}: {e}")
        return None

def analyze_volatility(df: pd.DataFrame, coin: str):
    """Analyze volatility metrics for a coin."""
    if df is None or len(df) < 24:
        return None

    # Calculate returns
    df['returns'] = df['close'].pct_change() * 100

    # Hourly metrics
    df['hourly_range'] = (df['high'] - df['low']) / df['low'] * 100

    # Daily aggregation
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    daily['daily_range'] = (daily['high'] - daily['low']) / daily['low'] * 100
    daily['daily_return'] = daily['close'].pct_change() * 100

    # Calculate metrics
    metrics = {
        'coin': coin,
        'data_days': len(daily),

        # Daily volatility
        'avg_daily_range': daily['daily_range'].mean(),
        'max_daily_range': daily['daily_range'].max(),
        'std_daily_return': daily['daily_return'].std(),

        # Hourly volatility (flash crash risk)
        'avg_hourly_range': df['hourly_range'].mean(),
        'max_hourly_range': df['hourly_range'].max(),

        # Extreme move counts
        'days_over_10pct': (daily['daily_range'] > 10).sum(),
        'days_over_12.5pct': (daily['daily_range'] > 12.5).sum(),
        'days_over_20pct': (daily['daily_range'] > 20).sum(),
        'hours_over_5pct': (df['hourly_range'] > 5).sum(),
        'hours_over_10pct': (df['hourly_range'] > 10).sum(),
    }

    # Calculate safe leverage
    # Rule: Max daily range should be < liquidation threshold
    max_range = metrics['max_daily_range']

    if max_range < 10:
        metrics['recommended_leverage'] = 8
        metrics['risk_level'] = 'LOW'
    elif max_range < 15:
        metrics['recommended_leverage'] = 5
        metrics['risk_level'] = 'MEDIUM'
    elif max_range < 25:
        metrics['recommended_leverage'] = 3
        metrics['risk_level'] = 'HIGH'
    else:
        metrics['recommended_leverage'] = 2
        metrics['risk_level'] = 'EXTREME'

    # 8x safety assessment
    metrics['8x_safe'] = max_range < 12.5
    metrics['5x_safe'] = max_range < 20
    metrics['3x_safe'] = max_range < 33

    return metrics

def print_report(all_metrics: list):
    """Print formatted volatility report."""
    print("\n" + "=" * 80)
    print("  VOLATILITY ANALYSIS REPORT - HYPERLIQUID MONSTER BOT V2")
    print("=" * 80)
    print(f"  Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Data Period: Last 30 days")
    print("=" * 80)

    # Sort by risk level
    risk_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'EXTREME': 3}
    all_metrics.sort(key=lambda x: risk_order.get(x.get('risk_level', 'EXTREME'), 4))

    print("\n" + "-" * 80)
    print("  SUMMARY BY COIN")
    print("-" * 80)
    print(f"  {'Coin':<8} {'Max Daily':<12} {'Avg Daily':<12} {'Risk':<10} {'Rec Lev':<10} {'8x Safe?':<10}")
    print("-" * 80)

    for m in all_metrics:
        safe_8x = "YES" if m.get('8x_safe', False) else "NO"
        print(f"  {m['coin']:<8} {m['max_daily_range']:>8.1f}%    {m['avg_daily_range']:>8.1f}%    {m['risk_level']:<10} {m['recommended_leverage']}x         {safe_8x}")

    print("\n" + "-" * 80)
    print("  EXTREME MOVE FREQUENCY (Last 30 Days)")
    print("-" * 80)
    print(f"  {'Coin':<8} {'Days >10%':<12} {'Days >12.5%':<14} {'Days >20%':<12} {'Hours >5%':<12}")
    print("-" * 80)

    for m in all_metrics:
        print(f"  {m['coin']:<8} {m['days_over_10pct']:>6}       {m['days_over_12.5pct']:>6}         {m['days_over_20pct']:>6}       {m['hours_over_5pct']:>6}")

    print("\n" + "-" * 80)
    print("  LIQUIDATION RISK ASSESSMENT")
    print("-" * 80)

    unsafe_8x = [m for m in all_metrics if not m.get('8x_safe', False)]
    safe_8x = [m for m in all_metrics if m.get('8x_safe', False)]

    if safe_8x:
        print(f"\n  SAFE for 8x leverage:")
        for m in safe_8x:
            print(f"    - {m['coin']}: Max move {m['max_daily_range']:.1f}% (under 12.5% threshold)")

    if unsafe_8x:
        print(f"\n  UNSAFE for 8x leverage:")
        for m in unsafe_8x:
            print(f"    - {m['coin']}: Max move {m['max_daily_range']:.1f}% (EXCEEDS 12.5% threshold)")
            print(f"      Recommended: {m['recommended_leverage']}x leverage")

    print("\n" + "-" * 80)
    print("  RECOMMENDATIONS")
    print("-" * 80)

    # Group by recommended leverage
    by_leverage = {}
    for m in all_metrics:
        lev = m['recommended_leverage']
        if lev not in by_leverage:
            by_leverage[lev] = []
        by_leverage[lev].append(m['coin'])

    print("\n  Suggested leverage configuration:")
    for lev in sorted(by_leverage.keys(), reverse=True):
        coins = by_leverage[lev]
        print(f"    {lev}x leverage: {', '.join(coins)}")

    print("\n" + "=" * 80)
    print("  CONCLUSION")
    print("=" * 80)

    high_risk_count = len([m for m in all_metrics if m['risk_level'] in ['HIGH', 'EXTREME']])
    total = len(all_metrics)

    if high_risk_count == 0:
        print("  All coins are suitable for aggressive leverage (5-8x).")
    elif high_risk_count < total / 2:
        print(f"  {high_risk_count}/{total} coins are HIGH/EXTREME volatility.")
        print("  Consider using dynamic leverage based on coin type.")
    else:
        print(f"  WARNING: {high_risk_count}/{total} coins are HIGH/EXTREME volatility!")
        print("  8x leverage on memecoins is VERY RISKY.")
        print("  Strongly recommend: BTC/SOL at 8x, memecoins at 2-3x")

    print("\n" + "=" * 80)

def main():
    print("\nFetching historical data...")
    print("(This may take a minute)\n")

    all_metrics = []

    for coin, (exchange_id, symbol) in COINS_TO_ANALYZE.items():
        print(f"  Analyzing {coin}...", end=" ")

        df = get_ohlcv_data(exchange_id, symbol, timeframe='1h', days=30)

        if df is not None and len(df) > 0:
            metrics = analyze_volatility(df, coin)
            if metrics:
                all_metrics.append(metrics)
                print(f"OK ({len(df)} data points)")
            else:
                print("Failed (insufficient data)")
        else:
            print("Failed (no data)")

    if all_metrics:
        print_report(all_metrics)
    else:
        print("No data retrieved. Check network connection and exchange availability.")

if __name__ == "__main__":
    main()
