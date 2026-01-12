"""
Hyperliquid Funding Rate Checker
Fetches current funding rates to verify which coins are best for the funding hunter strategy.

Run: python scripts/analytics/funding_rate_checker.py
"""

import requests
from datetime import datetime
from typing import Dict, List, Tuple

# Hyperliquid API endpoint
HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"

# Coins we care about (matching the bot config)
TARGET_COINS = [
    "ETH", "BTC", "SOL", "ANIME", "VVV", "HYPE", "PURR",
    "JEFF", "MOG", "WIF", "PEPE", "BONK", "DOGE"
]

# Volatility classifications from the bot
VOLATILITY_CLASS = {
    "BTC": ("SAFE", 8),
    "ETH": ("SAFE", 8),
    "SOL": ("MEDIUM", 5),
    "DOGE": ("HIGH", 3),
    "HYPE": ("HIGH", 3),
    "MOG": ("HIGH", 3),
    "VVV": ("HIGH", 3),
    "PURR": ("HIGH", 3),
    "JEFF": ("HIGH", 3),
    "PEPE": ("EXTREME", 2),
    "BONK": ("EXTREME", 2),
    "WIF": ("EXTREME", 2),
    "ANIME": ("EXTREME", 2),
}


def fetch_funding_rates() -> List[Dict]:
    """Fetch all funding rates from Hyperliquid."""
    try:
        response = requests.post(
            HYPERLIQUID_API,
            json={"type": "metaAndAssetCtxs"},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()

        # Extract asset contexts (contains funding info)
        if len(data) >= 2:
            asset_ctxs = data[1]
            meta = data[0]

            results = []
            for i, ctx in enumerate(asset_ctxs):
                if i < len(meta.get('universe', [])):
                    coin = meta['universe'][i]['name']
                    funding_rate = float(ctx.get('funding', 0))
                    mark_price = float(ctx.get('markPx', 0))
                    open_interest = float(ctx.get('openInterest', 0))

                    results.append({
                        'coin': coin,
                        'funding_rate': funding_rate,
                        'mark_price': mark_price,
                        'open_interest': open_interest
                    })

            return results
        return []
    except Exception as e:
        print(f"Error fetching funding rates: {e}")
        return []


def calculate_apr(hourly_rate: float) -> float:
    """Convert hourly funding rate to APR."""
    # Hyperliquid uses hourly funding (8760 hours/year)
    return hourly_rate * 8760 * 100


def calculate_effective_apr(apr: float, leverage: int) -> float:
    """Calculate effective APR with leverage."""
    return apr * leverage


def print_funding_report(rates: List[Dict]):
    """Print formatted funding rate report."""
    print("\n" + "=" * 90)
    print("  HYPERLIQUID FUNDING RATE CHECKER")
    print("=" * 90)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # Filter to target coins and sort by APR
    filtered = []
    for r in rates:
        if r['coin'] in TARGET_COINS:
            apr = calculate_apr(r['funding_rate'])
            vol_class, max_lev = VOLATILITY_CLASS.get(r['coin'], ("HIGH", 3))
            effective_apr = calculate_effective_apr(abs(apr), max_lev)
            direction = "LONG" if r['funding_rate'] > 0 else "SHORT"

            filtered.append({
                **r,
                'apr': apr,
                'effective_apr': effective_apr,
                'direction': direction,
                'vol_class': vol_class,
                'max_leverage': max_lev
            })

    # Sort by effective APR (what matters for profit)
    filtered.sort(key=lambda x: x['effective_apr'], reverse=True)

    print("\n" + "-" * 90)
    print("  TOP FUNDING OPPORTUNITIES (sorted by Effective APR)")
    print("-" * 90)
    print(f"  {'Coin':<8} {'Rate/hr':<12} {'APR':<12} {'Direction':<10} {'Vol Class':<10} {'Leverage':<10} {'Eff APR':<12}")
    print("-" * 90)

    for r in filtered:
        rate_str = f"{r['funding_rate']*100:.4f}%"
        apr_str = f"{abs(r['apr']):.1f}%"
        eff_apr_str = f"{r['effective_apr']:.1f}%"

        # Highlight top opportunities
        indicator = "***" if r['effective_apr'] > 100 else "   "

        print(f"  {indicator}{r['coin']:<5} {rate_str:<12} {apr_str:<12} {r['direction']:<10} {r['vol_class']:<10} {r['max_leverage']}x         {eff_apr_str}")

    print("\n" + "-" * 90)
    print("  ANALYSIS")
    print("-" * 90)

    # Best opportunities above 30% APR threshold
    viable = [r for r in filtered if abs(r['apr']) >= 30]

    if viable:
        print(f"\n  Coins meeting 30% APR threshold: {len(viable)}")
        for r in viable:
            print(f"    - {r['coin']}: {abs(r['apr']):.1f}% APR ({r['direction']}) -> {r['effective_apr']:.1f}% effective @ {r['max_leverage']}x")
    else:
        print("\n  WARNING: No coins currently meet the 30% APR threshold!")
        print("  Consider lowering min_funding_apr in config or waiting for better rates.")

    # Compare memecoin vs majors
    print("\n  MEMECOIN vs MAJOR COMPARISON:")

    majors = [r for r in filtered if r['coin'] in ['BTC', 'ETH', 'SOL']]
    memes = [r for r in filtered if r['coin'] not in ['BTC', 'ETH', 'SOL']]

    if majors:
        best_major = max(majors, key=lambda x: x['effective_apr'])
        print(f"    Best major: {best_major['coin']} @ {best_major['effective_apr']:.1f}% effective APR")

    if memes:
        best_meme = max(memes, key=lambda x: x['effective_apr'])
        print(f"    Best meme:  {best_meme['coin']} @ {best_meme['effective_apr']:.1f}% effective APR")

    if majors and memes:
        if best_meme['effective_apr'] > best_major['effective_apr']:
            diff = best_meme['effective_apr'] - best_major['effective_apr']
            print(f"\n    -> Memecoins winning by {diff:.1f}% effective APR")
        else:
            diff = best_major['effective_apr'] - best_meme['effective_apr']
            print(f"\n    -> Majors winning by {diff:.1f}% effective APR (consider focusing on majors)")

    print("\n" + "-" * 90)
    print("  RECOMMENDATIONS")
    print("-" * 90)

    top3 = filtered[:3]
    if top3:
        print(f"\n  Top 3 funding opportunities right now:")
        for i, r in enumerate(top3, 1):
            print(f"    {i}. {r['coin']}: Go {r['direction']} to receive {abs(r['apr']):.1f}% APR ({r['effective_apr']:.1f}% @ {r['max_leverage']}x)")

    print("\n" + "=" * 90)


def main():
    print("\nFetching funding rates from Hyperliquid...")
    rates = fetch_funding_rates()

    if rates:
        print_funding_report(rates)
    else:
        print("Failed to fetch funding rates. Check network connection.")


if __name__ == "__main__":
    main()
