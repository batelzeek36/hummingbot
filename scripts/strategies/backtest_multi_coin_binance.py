"""
Multi-Coin Strategy Backtester v2.0 - BINANCE VERSION (via NordVPN SOCKS5)

Uses Binance API for best possible historical data (1-hour candles).
Routes traffic through NordVPN SOCKS5 proxy to bypass geo-blocking.

SETUP:
1. Get your NordVPN service credentials from:
   https://my.nordaccount.com/dashboard/nordvpn/ -> Advanced configuration
2. Set the credentials below (NORD_USERNAME, NORD_PASSWORD)
3. Run: python3 scripts/strategies/backtest_multi_coin_binance.py

Author: Dollar-A-Day Project
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
import time
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

# =============================================================================
# NORDVPN SOCKS5 CONFIGURATION (loaded from .env)
# =============================================================================
NORD_USERNAME = os.getenv("NORD_USERNAME", "")
NORD_PASSWORD = os.getenv("NORD_PASSWORD", "")

# NordVPN SOCKS5 servers - NON-US only (Binance.com blocks US IPs)
# Full list: https://nordvpn.com/servers/tools/
NORD_SERVERS = [
    "amsterdam.nl.socks.nordhold.net",   # Netherlands
    "stockholm.se.socks.nordhold.net",   # Sweden
    "zurich.ch.socks.nordhold.net",      # Switzerland
    "frankfurt.de.socks.nordhold.net",   # Germany
    "london.uk.socks.nordhold.net",      # UK
    "singapore.sg.socks.nordhold.net",   # Singapore
    "tokyo.jp.socks.nordhold.net",       # Japan
]
NORD_PORT = 1080

# =============================================================================
# BINANCE API CONFIGURATION
# =============================================================================
# Using Binance.com (international) - works with non-US VPN exit
BINANCE_BASE_URL = "https://api.binance.com"

# Coin mappings (Kraken pairs -> Binance symbols)
COIN_MAPPING = {
    "DOGE-USD": {"binance": "DOGEUSDT", "coingecko": "dogecoin"},
    "DOT-USD": {"binance": "DOTUSDT", "coingecko": "polkadot"},
    "ADA-USD": {"binance": "ADAUSDT", "coingecko": "cardano"},
}


def get_proxy_config(server_index=0):
    """Get SOCKS5 proxy configuration for NordVPN."""
    if not NORD_USERNAME or not NORD_PASSWORD:
        return None  # Not configured, will fall back to CoinGecko

    if server_index >= len(NORD_SERVERS):
        return None

    server = NORD_SERVERS[server_index]
    proxy_url = f"socks5://{NORD_USERNAME}:{NORD_PASSWORD}@{server}:{NORD_PORT}"
    return {
        "http": proxy_url,
        "https": proxy_url
    }, server


def test_proxy_connection():
    """Test if the SOCKS5 proxy is working, trying multiple servers."""
    if not NORD_USERNAME or not NORD_PASSWORD:
        print("  NordVPN not configured - will use CoinGecko fallback")
        return False, None

    print("  Testing NordVPN SOCKS5 connection...")
    print(f"  Username: {NORD_USERNAME[:4]}...{NORD_USERNAME[-4:]}")

    for i, server in enumerate(NORD_SERVERS):
        print(f"  Trying server: {server}...")
        proxy_url = f"socks5://{NORD_USERNAME}:{NORD_PASSWORD}@{server}:{NORD_PORT}"
        proxies = {"http": proxy_url, "https": proxy_url}

        try:
            response = requests.get(
                "https://api.binance.com/api/v3/ping",
                proxies=proxies,
                timeout=20
            )
            if response.status_code == 200:
                print(f"  SUCCESS: Connected via {server}")
                return True, proxies
            else:
                print(f"    Binance returned status {response.status_code}")
        except requests.exceptions.ConnectTimeout:
            print(f"    Timeout on {server}")
        except Exception as e:
            error_msg = str(e)[:80]
            print(f"    Failed: {error_msg}")

    print("  All servers failed - falling back to CoinGecko")
    return False, None


def fetch_binance_klines(symbol: str, interval: str = "1h", days: int = 30, proxies: dict = None) -> pd.DataFrame:
    """Fetch OHLCV data from Binance via SOCKS5 proxy."""

    url = f"{BINANCE_BASE_URL}/api/v3/klines"

    # Calculate start time
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }

    print(f"  Fetching {symbol} from Binance...")

    try:
        response = requests.get(url, params=params, proxies=proxies, timeout=30)

        if response.status_code == 451:
            print(f"  ERROR: Binance geo-blocked (HTTP 451). Check VPN connection.")
            return None
        elif response.status_code != 200:
            print(f"  ERROR: Binance returned status {response.status_code}")
            return None

        data = response.json()

        # Binance klines format: [open_time, open, high, low, close, volume, ...]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        print(f"  Got {len(df)} candles for {symbol} (1-hour)")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    except requests.exceptions.ProxyError as e:
        print(f"  Proxy error: {e}")
        return None
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None


def fetch_coingecko_ohlc(coin_id: str, days: int = 30) -> pd.DataFrame:
    """Fallback: Fetch OHLC data from CoinGecko (no VPN needed)."""

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}

    print(f"  Fetching {coin_id} from CoinGecko (fallback)...")
    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        print(f"  Error fetching {coin_id}: {response.status_code}")
        return None

    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['volume'] = 0  # CoinGecko OHLC doesn't include volume

    print(f"  Got {len(df)} candles for {coin_id} (4-hour)")
    return df


def fetch_data(pair: str, days: int, use_binance: bool, proxies: dict) -> pd.DataFrame:
    """Fetch data from Binance (if VPN works) or CoinGecko (fallback)."""

    mapping = COIN_MAPPING.get(pair)
    if not mapping:
        print(f"  Unknown pair: {pair}")
        return None

    if use_binance and proxies:
        df = fetch_binance_klines(mapping["binance"], "1h", days, proxies)
        if df is not None:
            return df
        print(f"  Binance failed, falling back to CoinGecko...")

    # Fallback to CoinGecko
    return fetch_coingecko_ohlc(mapping["coingecko"], days)


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Normalized ATR (volatility indicator)."""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    natr = (atr / df['close']) * 100
    return natr


def backtest_market_making_dynamic(df: pd.DataFrame, order_amount: float,
                                    base_spread_bps: int, fee_bps: int) -> dict:
    """Simulate market making with DYNAMIC spreads based on volatility."""
    df = df.copy()

    df['rsi'] = calculate_rsi(df, period=14)
    df['natr'] = calculate_natr(df, period=14)

    spread_base = 0.8
    df['spread_multiplier'] = df['natr'] / spread_base
    df['spread_multiplier'] = df['spread_multiplier'].clip(0.5, 3.0)

    df['price_shift'] = ((50 - df['rsi']) / 50) * (df['natr'] / 100)

    dynamic_spread = base_spread_bps / 10000 * df['spread_multiplier']
    df['bid'] = df['open'] * (1 + df['price_shift']) * (1 - dynamic_spread)
    df['ask'] = df['open'] * (1 + df['price_shift']) * (1 + dynamic_spread)

    df['buy_fill'] = df['low'] <= df['bid']
    df['sell_fill'] = df['high'] >= df['ask']

    df = df.dropna()

    buys = df['buy_fill'].sum()
    sells = df['sell_fill'].sum()
    matched = min(buys, sells)
    avg_price = df['close'].mean()
    avg_spread = dynamic_spread.mean()

    spread_profit = matched * order_amount * avg_price * (avg_spread * 2)
    fees = matched * 2 * order_amount * avg_price * (fee_bps / 10000)
    pnl = spread_profit - fees

    return {
        'buys': int(buys),
        'sells': int(sells),
        'matched': int(matched),
        'avg_spread_bps': round(avg_spread * 10000, 1),
        'pnl': pnl,
        'fees': fees
    }


def backtest_grid_multi_level(df: pd.DataFrame, order_amount: float,
                               levels: int, spacing_bps: int, fee_bps: int) -> dict:
    """Simulate multi-level grid trading."""
    price_high = df['high'].max()
    price_low = df['low'].min()
    price_mid = (price_high + price_low) / 2

    spacing = spacing_bps / 10000
    grid_levels = []
    for i in range(-levels, levels + 1):
        if i != 0:
            grid_levels.append(price_mid * (1 + spacing * i))

    grid_trips = 0
    total_profit = 0.0
    total_fees = 0.0

    buy_triggered = {level: False for level in grid_levels if level < price_mid}
    sell_triggered = {level: False for level in grid_levels if level > price_mid}

    for _, row in df.iterrows():
        for level in list(buy_triggered.keys()):
            if not buy_triggered[level] and row['low'] <= level:
                buy_triggered[level] = True
                sell_level = level * (1 + spacing)
                if row['high'] >= sell_level:
                    profit = order_amount * level * spacing
                    fee = order_amount * level * 2 * (fee_bps / 10000)
                    total_profit += profit - fee
                    total_fees += fee
                    grid_trips += 1
                    buy_triggered[level] = False

        for level in list(sell_triggered.keys()):
            if not sell_triggered[level] and row['high'] >= level:
                sell_triggered[level] = True
                buy_level = level * (1 - spacing)
                if row['low'] <= buy_level:
                    profit = order_amount * level * spacing
                    fee = order_amount * level * 2 * (fee_bps / 10000)
                    total_profit += profit - fee
                    total_fees += fee
                    grid_trips += 1
                    sell_triggered[level] = False

        if abs(row['close'] - price_mid) / price_mid > 0.05:
            price_mid = row['close']
            buy_triggered = {level: False for level in grid_levels if level < price_mid}
            sell_triggered = {level: False for level in grid_levels if level > price_mid}

    return {
        'grid_trips': grid_trips,
        'levels': levels * 2,
        'pnl': total_profit,
        'fees': total_fees
    }


def backtest_momentum_rsi(df: pd.DataFrame, order_amount: float,
                          tp_pct: float, sl_pct: float, fee_bps: int,
                          rsi_oversold: int = 30, rsi_overbought: int = 70) -> dict:
    """Simulate RSI-based momentum strategy."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df, period=7)

    position = None
    entry_price = 0
    trades = []
    total_pnl = 0.0
    total_fees = 0.0

    for i in range(14, len(df)):
        row = df.iloc[i]
        rsi = row['rsi']

        if pd.isna(rsi):
            continue

        if position is None and rsi < rsi_oversold:
            position = 'long'
            entry_price = row['close']

        elif position == 'long':
            pnl_pct = (row['close'] - entry_price) / entry_price * 100

            exit_trade = False
            exit_reason = ''

            if pnl_pct >= tp_pct:
                exit_trade = True
                exit_reason = 'TP'
            elif pnl_pct <= -sl_pct:
                exit_trade = True
                exit_reason = 'SL'
            elif rsi > rsi_overbought:
                exit_trade = True
                exit_reason = 'RSI_OB'

            if exit_trade:
                trade_pnl = order_amount * (row['close'] - entry_price)
                fees = order_amount * (entry_price + row['close']) * (fee_bps / 10000)
                trade_pnl -= fees
                total_pnl += trade_pnl
                total_fees += fees
                trades.append({'pnl': trade_pnl, 'reason': exit_reason, 'pnl_pct': pnl_pct})
                position = None

    wins = len([t for t in trades if t['pnl'] > 0])
    winning_trades = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing_trades = [t['pnl'] for t in trades if t['pnl'] <= 0]
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = abs(sum(losing_trades) / len(losing_trades)) if losing_trades else 0

    return {
        'trades': len(trades),
        'wins': wins,
        'losses': len(trades) - wins,
        'win_rate': (wins / len(trades) * 100) if trades else 0,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pnl': total_pnl,
        'fees': total_fees,
        'exit_reasons': {r: len([t for t in trades if t['reason'] == r]) for r in ['TP', 'SL', 'RSI_OB']}
    }


def main():
    print("=" * 70)
    print("  MULTI-COIN STRATEGY BACKTESTER v2.0 - BINANCE EDITION")
    print("  Using NordVPN SOCKS5 Proxy for Best Data Quality")
    print("=" * 70)
    print()

    # Configuration
    total_capital = 112.0
    days = 30

    # Strategy parameters (improved v2.0)
    mm_order_amount = 15.0
    mm_base_spread_bps = 50
    mm_fee_bps = 26

    grid_order_amount = 0.5
    grid_levels = 4
    grid_spacing_bps = 80
    grid_fee_bps = 26

    mom_order_amount = 5.0
    mom_tp = 1.5
    mom_sl = 0.75
    mom_fee_bps = 26

    # Test VPN connection
    print("Checking data source...")
    use_binance, proxies = test_proxy_connection()

    data_source = "Binance (1h candles)" if use_binance else "CoinGecko (4h candles)"
    print(f"\nData source: {data_source}")
    print()

    print(f"Fetching {days} days of historical data...")
    print()

    # Fetch data
    df_doge = fetch_data("DOGE-USD", days, use_binance, proxies)
    time.sleep(0.5)
    df_dot = fetch_data("DOT-USD", days, use_binance, proxies)
    time.sleep(0.5)
    df_ada = fetch_data("ADA-USD", days, use_binance, proxies)

    if df_doge is None or df_dot is None or df_ada is None:
        print("\nError fetching data. Please check your VPN connection or try again later.")
        return

    print()
    print("Running backtests...")
    print()

    # Run backtests
    mm_results = backtest_market_making_dynamic(df_doge, mm_order_amount, mm_base_spread_bps, mm_fee_bps)
    grid_results = backtest_grid_multi_level(df_dot, grid_order_amount, grid_levels, grid_spacing_bps, grid_fee_bps)
    mom_results = backtest_momentum_rsi(df_ada, mom_order_amount, mom_tp, mom_sl, mom_fee_bps)

    # Calculate totals
    total_pnl = mm_results['pnl'] + grid_results['pnl'] + mom_results['pnl']
    total_fees = mm_results['fees'] + grid_results['fees'] + mom_results['fees']
    return_pct = (total_pnl / total_capital) * 100

    # Display results
    print("=" * 70)
    print(f"  BACKTEST RESULTS (30 days) - {data_source}")
    print("=" * 70)
    print(f"  Starting Capital: ${total_capital:.2f}")
    print(f"  Data Points: {len(df_doge)} DOGE, {len(df_dot)} DOT, {len(df_ada)} ADA candles")
    print()

    print("-" * 70)
    print("  MARKET MAKING (DOGE) - Dynamic Spreads")
    print("-" * 70)
    print(f"  Buys: {mm_results['buys']}  |  Sells: {mm_results['sells']}  |  Matched: {mm_results['matched']}")
    print(f"  Avg Dynamic Spread: {mm_results['avg_spread_bps']:.1f} bps")
    print(f"  P&L: ${mm_results['pnl']:.2f}")
    print()

    print("-" * 70)
    print(f"  GRID TRADING (DOT) - {grid_results['levels']} Levels")
    print("-" * 70)
    print(f"  Grid Trips: {grid_results['grid_trips']}")
    print(f"  P&L: ${grid_results['pnl']:.2f}")
    print()

    print("-" * 70)
    print("  MOMENTUM RSI (ADA) - Oversold/Overbought")
    print("-" * 70)
    print(f"  Trades: {mom_results['trades']}  |  Wins: {mom_results['wins']}  |  Losses: {mom_results['losses']}")
    print(f"  Win Rate: {mom_results['win_rate']:.1f}%")
    print(f"  Exits: TP={mom_results['exit_reasons'].get('TP', 0)} | SL={mom_results['exit_reasons'].get('SL', 0)} | RSI={mom_results['exit_reasons'].get('RSI_OB', 0)}")
    if mom_results['avg_win'] > 0:
        print(f"  Avg Win: ${mom_results['avg_win']:.2f}  |  Avg Loss: ${mom_results['avg_loss']:.2f}")
    print(f"  P&L: ${mom_results['pnl']:.2f}")
    print()

    print("=" * 70)
    print("  COMBINED RESULTS")
    print("=" * 70)
    print(f"  MM (DOGE):      ${mm_results['pnl']:>8.2f}")
    print(f"  Grid (DOT):     ${grid_results['pnl']:>8.2f}")
    print(f"  Momentum (ADA): ${mom_results['pnl']:>8.2f}")
    print(f"  Total Fees:     ${total_fees:>8.2f}")
    print()

    sign = "+" if total_pnl >= 0 else ""
    print(f"  TOTAL P&L:      {sign}${total_pnl:.2f}")
    print(f"  RETURN:         {sign}{return_pct:.2f}%")
    print("=" * 70)

    # Save to CSV
    results = {
        'Strategy': ['MM (DOGE)', 'Grid (DOT)', 'Momentum (ADA)', 'TOTAL'],
        'P&L': [f'${mm_results["pnl"]:.2f}', f'${grid_results["pnl"]:.2f}',
                f'${mom_results["pnl"]:.2f}', f'${total_pnl:.2f}'],
        'Return %': [f'{mm_results["pnl"]/total_capital*100:.2f}%',
                     f'{grid_results["pnl"]/total_capital*100:.2f}%',
                     f'{mom_results["pnl"]/total_capital*100:.2f}%',
                     f'{return_pct:.2f}%']
    }

    df_results = pd.DataFrame(results)
    csv_path = "backtest_results_binance.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    if not use_binance:
        print("\n" + "=" * 70)
        print("  NOTE: Using CoinGecko fallback (4h candles)")
        print("  For better accuracy, configure NordVPN SOCKS5 credentials at top of script")
        print("=" * 70)


if __name__ == "__main__":
    main()
