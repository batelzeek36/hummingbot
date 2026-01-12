"""
Multi-Coin Strategy Backtester (CoinGecko Version) - IMPROVED PARAMETERS

Uses CoinGecko API for historical data (not geo-blocked).
Tests strategy logic for DOGE (MM), DOT (Grid), ADA (Momentum).

Parameters based on community scripts:
- pmm_with_shifted_mid_dynamic_spreads.py (dynamic spreads)
- fixed_grid.py (multi-level grid)
- simple_rsi_no_config.py (RSI-based momentum)

This is a STANDALONE backtester - run it outside of hummingbot:
    python3 scripts/strategies/backtest_multi_coin.py

Author: Dollar-A-Day Project
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
import time


def fetch_coingecko_ohlc(coin_id: str, days: int = 30) -> pd.DataFrame:
    """Fetch OHLC data from CoinGecko."""

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {
        "vs_currency": "usd",
        "days": days
    }

    print(f"  Fetching {coin_id} data...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"  Error fetching {coin_id}: {response.status_code}")
        return None

    data = response.json()

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    print(f"  Got {len(df)} candles for {coin_id}")
    return df


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
    natr = (atr / df['close']) * 100  # Normalized as percentage
    return natr


def backtest_market_making_dynamic(df: pd.DataFrame, order_amount: float,
                                    base_spread_bps: int, fee_bps: int) -> dict:
    """
    Simulate market making with DYNAMIC spreads based on volatility.
    Based on pmm_with_shifted_mid_dynamic_spreads.py

    - Uses NATR for spread adjustment (wider spreads in volatile markets)
    - Uses RSI for price shifting (buy more aggressively when oversold)
    """
    df = df.copy()

    # Calculate indicators
    df['rsi'] = calculate_rsi(df, period=14)
    df['natr'] = calculate_natr(df, period=14)

    # Dynamic spread multiplier based on NATR
    # Base spread = 0.8%, if NATR = 0.8% then multiplier = 1
    # If NATR = 1.6%, multiplier = 2 (wider spreads)
    spread_base = 0.8  # 0.8% base
    df['spread_multiplier'] = df['natr'] / spread_base
    df['spread_multiplier'] = df['spread_multiplier'].clip(0.5, 3.0)  # Cap at 0.5x to 3x

    # Price shift based on RSI (shift mid-price toward buying when oversold)
    # RSI=30 -> shift +0.5%, RSI=70 -> shift -0.5%
    df['price_shift'] = ((50 - df['rsi']) / 50) * (df['natr'] / 100)

    # Calculate dynamic bid/ask
    dynamic_spread = base_spread_bps / 10000 * df['spread_multiplier']
    df['bid'] = df['open'] * (1 + df['price_shift']) * (1 - dynamic_spread)
    df['ask'] = df['open'] * (1 + df['price_shift']) * (1 + dynamic_spread)

    df['buy_fill'] = df['low'] <= df['bid']
    df['sell_fill'] = df['high'] >= df['ask']

    # Skip rows where indicators aren't ready
    df = df.dropna()

    buys = df['buy_fill'].sum()
    sells = df['sell_fill'].sum()
    matched = min(buys, sells)
    avg_price = df['close'].mean()
    avg_spread = dynamic_spread.mean()

    # Profit from spread capture minus fees
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
    """
    Simulate multi-level grid trading.
    Based on fixed_grid.py - uses multiple grid levels simultaneously.

    - More levels = more opportunities
    - Rebalances grid when price moves significantly
    """
    # Calculate grid based on price range
    price_high = df['high'].max()
    price_low = df['low'].min()
    price_mid = (price_high + price_low) / 2
    price_range = price_high - price_low

    # Create grid levels around the mid price
    spacing = spacing_bps / 10000
    grid_levels = []
    for i in range(-levels, levels + 1):
        if i != 0:  # Skip the mid price
            grid_levels.append(price_mid * (1 + spacing * i))

    grid_trips = 0
    total_profit = 0.0
    total_fees = 0.0

    # Track which levels have been triggered
    buy_triggered = {level: False for level in grid_levels if level < price_mid}
    sell_triggered = {level: False for level in grid_levels if level > price_mid}

    for _, row in df.iterrows():
        # Check buy levels (below mid)
        for level in list(buy_triggered.keys()):
            if not buy_triggered[level] and row['low'] <= level:
                # Buy triggered, now look for sell opportunity
                buy_triggered[level] = True

                # Check if we can sell at the next level up
                sell_level = level * (1 + spacing)
                if row['high'] >= sell_level:
                    profit = order_amount * level * spacing
                    fee = order_amount * level * 2 * (fee_bps / 10000)
                    total_profit += profit - fee
                    total_fees += fee
                    grid_trips += 1
                    buy_triggered[level] = False  # Reset for next trip

        # Check sell levels (above mid)
        for level in list(sell_triggered.keys()):
            if not sell_triggered[level] and row['high'] >= level:
                # Sell triggered, now look for buy opportunity
                sell_triggered[level] = True

                # Check if we can buy at the next level down
                buy_level = level * (1 - spacing)
                if row['low'] <= buy_level:
                    profit = order_amount * level * spacing
                    fee = order_amount * level * 2 * (fee_bps / 10000)
                    total_profit += profit - fee
                    total_fees += fee
                    grid_trips += 1
                    sell_triggered[level] = False  # Reset for next trip

        # Reset grid if price moved too far (rebalance)
        if abs(row['close'] - price_mid) / price_mid > 0.05:  # 5% move triggers rebalance
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
    """
    Simulate RSI-based momentum strategy.
    Based on simple_rsi_no_config.py

    - Buy when RSI < 30 (oversold)
    - Sell when RSI > 70 (overbought)
    - Tight stop loss, reasonable take profit
    - Better risk:reward ratio
    """
    df = df.copy()
    df['rsi'] = calculate_rsi(df, period=7)  # Shorter period for more signals

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

        # Entry: Buy when oversold
        if position is None and rsi < rsi_oversold:
            position = 'long'
            entry_price = row['close']

        # Exit conditions
        elif position == 'long':
            pnl_pct = (row['close'] - entry_price) / entry_price * 100

            exit_trade = False
            exit_reason = ''

            # Take profit
            if pnl_pct >= tp_pct:
                exit_trade = True
                exit_reason = 'TP'
            # Stop loss
            elif pnl_pct <= -sl_pct:
                exit_trade = True
                exit_reason = 'SL'
            # RSI reversal (overbought = sell)
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

    # Calculate average win/loss
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
    print("=" * 60)
    print("  MULTI-COIN STRATEGY BACKTESTER v2.0")
    print("  Using IMPROVED Parameters from Community Scripts")
    print("=" * 60)
    print()

    # Configuration - IMPROVED based on community scripts
    total_capital = 112.0
    days = 30

    # MM settings (DOGE) - Dynamic spreads like pmm_with_shifted_mid_dynamic_spreads.py
    mm_order_amount = 15.0
    mm_base_spread_bps = 50  # Wider base spread (was 30)
    mm_fee_bps = 26

    # Grid settings (DOT) - Multi-level like fixed_grid.py
    grid_order_amount = 0.5
    grid_levels = 4  # More levels (was 2)
    grid_spacing_bps = 80  # Wider spacing (was 50)
    grid_fee_bps = 26

    # Momentum settings (ADA) - RSI-based like simple_rsi_no_config.py
    mom_order_amount = 5.0
    mom_tp = 1.5  # Tighter TP (was 4.0)
    mom_sl = 0.75  # Tighter SL (was 2.0) - better risk:reward
    mom_fee_bps = 26

    print(f"Fetching {days} days of historical data...")
    print()

    # Fetch data (rate limit: wait between requests)
    df_doge = fetch_coingecko_ohlc("dogecoin", days)
    time.sleep(1)  # Rate limit
    df_dot = fetch_coingecko_ohlc("polkadot", days)
    time.sleep(1)
    df_ada = fetch_coingecko_ohlc("cardano", days)

    if df_doge is None or df_dot is None or df_ada is None:
        print("Error fetching data. Try again later.")
        return

    print()
    print("Running IMPROVED backtests...")
    print()

    # Run improved backtests
    mm_results = backtest_market_making_dynamic(df_doge, mm_order_amount, mm_base_spread_bps, mm_fee_bps)
    grid_results = backtest_grid_multi_level(df_dot, grid_order_amount, grid_levels, grid_spacing_bps, grid_fee_bps)
    mom_results = backtest_momentum_rsi(df_ada, mom_order_amount, mom_tp, mom_sl, mom_fee_bps)

    # Calculate totals
    total_pnl = mm_results['pnl'] + grid_results['pnl'] + mom_results['pnl']
    total_fees = mm_results['fees'] + grid_results['fees'] + mom_results['fees']
    return_pct = (total_pnl / total_capital) * 100

    # Display results
    print("=" * 60)
    print("  IMPROVED BACKTEST RESULTS (30 days)")
    print("=" * 60)
    print(f"  Starting Capital: ${total_capital:.2f}")
    print()

    print("-" * 60)
    print("  MARKET MAKING (DOGE) - Dynamic Spreads")
    print("-" * 60)
    print(f"  Buys: {mm_results['buys']}  |  Sells: {mm_results['sells']}  |  Matched: {mm_results['matched']}")
    print(f"  Avg Dynamic Spread: {mm_results['avg_spread_bps']:.1f} bps")
    print(f"  P&L: ${mm_results['pnl']:.2f}")
    print()

    print("-" * 60)
    print(f"  GRID TRADING (DOT) - {grid_results['levels']} Levels")
    print("-" * 60)
    print(f"  Grid Trips: {grid_results['grid_trips']}")
    print(f"  P&L: ${grid_results['pnl']:.2f}")
    print()

    print("-" * 60)
    print("  MOMENTUM RSI (ADA) - Oversold/Overbought")
    print("-" * 60)
    print(f"  Trades: {mom_results['trades']}  |  Wins: {mom_results['wins']}  |  Losses: {mom_results['losses']}")
    print(f"  Win Rate: {mom_results['win_rate']:.1f}%")
    print(f"  Exits: TP={mom_results['exit_reasons'].get('TP', 0)} | SL={mom_results['exit_reasons'].get('SL', 0)} | RSI={mom_results['exit_reasons'].get('RSI_OB', 0)}")
    if mom_results['avg_win'] > 0:
        print(f"  Avg Win: ${mom_results['avg_win']:.2f}  |  Avg Loss: ${mom_results['avg_loss']:.2f}")
    print(f"  P&L: ${mom_results['pnl']:.2f}")
    print()

    print("=" * 60)
    print("  COMBINED RESULTS")
    print("=" * 60)
    print(f"  MM (DOGE):      ${mm_results['pnl']:>8.2f}")
    print(f"  Grid (DOT):     ${grid_results['pnl']:>8.2f}")
    print(f"  Momentum (ADA): ${mom_results['pnl']:>8.2f}")
    print(f"  Total Fees:     ${total_fees:>8.2f}")
    print()

    sign = "+" if total_pnl >= 0 else ""
    print(f"  TOTAL P&L:      {sign}${total_pnl:.2f}")
    print(f"  RETURN:         {sign}{return_pct:.2f}%")
    print("=" * 60)

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
    csv_path = "backtest_results_v2.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Show comparison
    print("\n" + "=" * 60)
    print("  PARAMETER CHANGES FROM v1")
    print("=" * 60)
    print("  Market Making:")
    print("    - Added NATR-based dynamic spread (0.5x to 3x multiplier)")
    print("    - Added RSI-based price shifting")
    print("    - Base spread: 30 bps -> 50 bps")
    print()
    print("  Grid Trading:")
    print("    - Levels: 2 -> 4 (8 total grid lines)")
    print("    - Spacing: 50 bps -> 80 bps")
    print("    - Added 5% rebalance trigger")
    print()
    print("  Momentum:")
    print("    - Changed from momentum% to RSI-based signals")
    print("    - Entry: RSI < 30 (oversold)")
    print("    - Exit: RSI > 70 (overbought) or TP/SL")
    print("    - TP: 4% -> 1.5%, SL: 2% -> 0.75%")
    print("    - Better risk:reward ratio (2:1)")
    print("=" * 60)


if __name__ == "__main__":
    main()
