"""
Comprehensive Risk Analysis Script for Funding Strategy

Combines multiple data sources to provide a holistic risk assessment:
1. Hyperliquid price volatility (via ccxt)
2. Coinglass liquidation heatmaps
3. Coinglass L/S ratios (market crowding)
4. Coinglass CVD divergence (trapped traders)

Risk Scoring:
- Each category scores 0-100 (0=safe, 100=dangerous)
- Final score = weighted average: 30% volatility, 25% liquidation, 25% crowding, 20% divergence
- SAFE: <30, CAUTION: 30-60, AVOID: >60
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from hyperliquid_monster.coinglass import CoinglassAPI
except ImportError:
    CoinglassAPI = None
    print("WARNING: CoinglassAPI not available. Coinglass data will be skipped.")


# === CONFIGURATION ===

# Coins to analyze (from funding_scan_pairs)
COINS_TO_ANALYZE = {
    # Coin: (exchange_for_ohlcv, symbol)
    "BTC-USD": ("binance", "BTC/USDT"),
    "ETH-USD": ("binance", "ETH/USDT"),
    "SOL-USD": ("binance", "SOL/USDT"),
    "DOGE-USD": ("binance", "DOGE/USDT"),
    "TAO-USD": ("binance", "TAO/USDT"),
    "HYPE-USD": ("bybit", "HYPE/USDT"),
    "AVNT-USD": None,  # May not be available on major exchanges
    "kPEPE-USD": ("binance", "1000PEPE/USDT"),
    "kBONK-USD": ("binance", "1000BONK/USDT"),
    "WIF-USD": ("binance", "WIF/USDT"),
    "VVV-USD": None,  # May not be available
    "HYPER-USD": None,  # May not be available
    "IP-USD": None,  # May not be available
}

# Risk weights for final score
RISK_WEIGHTS = {
    "volatility": 0.30,
    "liquidation": 0.25,
    "crowding": 0.25,
    "divergence": 0.20,
}

# Risk thresholds
RISK_THRESHOLDS = {
    "SAFE": 30,
    "CAUTION": 60,
    # Above CAUTION = AVOID
}


# === VOLATILITY ANALYSIS ===

def get_ohlcv_data(exchange_id: str, symbol: str, timeframe: str = "1h", days: int = 7) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from exchange."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"enableRateLimit": True})
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        return None


def analyze_volatility(pair: str, exchange_info: Optional[Tuple[str, str]]) -> Dict:
    """
    Analyze price volatility for a coin.

    Returns dict with:
    - max_daily_move: Maximum daily range in %
    - avg_daily_move: Average daily range in %
    - volatility_score: 0-100 risk score
    - safe_leverage: Recommended leverage
    """
    result = {
        "pair": pair,
        "max_daily_move": None,
        "avg_daily_move": None,
        "volatility_score": 50,  # Default if no data
        "safe_leverage": 3,
        "data_available": False,
    }

    if not exchange_info:
        return result

    exchange_id, symbol = exchange_info
    df = get_ohlcv_data(exchange_id, symbol, "1h", days=7)

    if df is None or len(df) < 24:
        return result

    result["data_available"] = True

    # Calculate daily metrics
    daily = df.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna()

    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["low"] * 100

    result["max_daily_move"] = daily["range_pct"].max()
    result["avg_daily_move"] = daily["range_pct"].mean()

    # Calculate volatility score (0-100)
    # <5% avg = 0 (very safe), >20% avg = 100 (very risky)
    avg_move = result["avg_daily_move"]
    max_move = result["max_daily_move"]

    # Score based on both average and max moves
    avg_score = min(100, max(0, (avg_move - 3) / 17 * 100))
    max_score = min(100, max(0, (max_move - 8) / 32 * 100))
    result["volatility_score"] = int(avg_score * 0.6 + max_score * 0.4)

    # Determine safe leverage
    if max_move < 10:
        result["safe_leverage"] = 8
    elif max_move < 15:
        result["safe_leverage"] = 5
    elif max_move < 25:
        result["safe_leverage"] = 3
    else:
        result["safe_leverage"] = 2

    return result


# === COINGLASS ANALYSIS ===

def analyze_liquidations(coinglass: "CoinglassAPI", pair: str) -> Dict:
    """
    Analyze liquidation clusters for a coin.

    Returns dict with:
    - nearest_long_dist: Distance to nearest long liquidation cluster (%)
    - nearest_short_dist: Distance to nearest short liquidation cluster (%)
    - magnetic_direction: "up" or "down"
    - liquidation_score: 0-100 risk score
    """
    result = {
        "pair": pair,
        "nearest_long_dist": None,
        "nearest_short_dist": None,
        "magnetic_direction": "unknown",
        "liquidation_score": 50,  # Default
        "data_available": False,
    }

    if not coinglass:
        return result

    try:
        heatmap = coinglass.get_liquidation_heatmap(pair)
        if not heatmap:
            return result

        result["data_available"] = True
        result["magnetic_direction"] = heatmap.magnetic_direction

        # Get distances to nearest clusters
        if heatmap.long_clusters:
            result["nearest_long_dist"] = abs(heatmap.long_clusters[0].distance_pct)
        if heatmap.short_clusters:
            result["nearest_short_dist"] = abs(heatmap.short_clusters[0].distance_pct)

        # Calculate liquidation risk score
        # Closer clusters = higher risk
        min_dist = 100
        if result["nearest_long_dist"]:
            min_dist = min(min_dist, result["nearest_long_dist"])
        if result["nearest_short_dist"]:
            min_dist = min(min_dist, result["nearest_short_dist"])

        # <2% away = 100 (very dangerous), >10% away = 0 (safe)
        result["liquidation_score"] = int(max(0, min(100, (10 - min_dist) / 8 * 100)))

    except Exception as e:
        pass

    return result


def analyze_crowding(coinglass: "CoinglassAPI", pair: str) -> Dict:
    """
    Analyze market crowding via L/S ratios.

    Returns dict with:
    - long_ratio: % of longs
    - short_ratio: % of shorts
    - sentiment: "crowded_long", "crowded_short", "balanced"
    - crowding_score: 0-100 risk score
    - contrarian_signal: "long", "short", "neutral"
    """
    result = {
        "pair": pair,
        "long_ratio": None,
        "short_ratio": None,
        "sentiment": "unknown",
        "crowding_score": 50,  # Default
        "contrarian_signal": "neutral",
        "data_available": False,
    }

    if not coinglass:
        return result

    try:
        ls_ratio = coinglass.get_long_short_ratio(pair)
        if not ls_ratio:
            return result

        result["data_available"] = True
        result["long_ratio"] = ls_ratio.long_ratio
        result["short_ratio"] = ls_ratio.short_ratio
        result["sentiment"] = ls_ratio.sentiment
        result["crowding_score"] = int(ls_ratio.crowding_score)
        result["contrarian_signal"] = ls_ratio.contrarian_signal

    except Exception as e:
        pass

    return result


def analyze_divergence(coinglass: "CoinglassAPI", pair: str) -> Dict:
    """
    Analyze spot/perp CVD divergence.

    Returns dict with:
    - spot_direction: CVD direction for spot
    - perp_direction: CVD direction for perp
    - divergence_type: "bullish_divergence", "bearish_divergence", "aligned"
    - signal_strength: 0-100
    - divergence_score: 0-100 risk score
    """
    result = {
        "pair": pair,
        "spot_direction": "unknown",
        "perp_direction": "unknown",
        "divergence_type": "unknown",
        "signal_strength": 0,
        "divergence_score": 50,  # Default
        "data_available": False,
    }

    if not coinglass:
        return result

    try:
        divergence = coinglass.get_spot_perp_divergence(pair)
        if not divergence:
            return result

        result["data_available"] = True
        result["spot_direction"] = divergence.spot_direction.value if divergence.spot_direction else "unknown"
        result["perp_direction"] = divergence.perp_direction.value if divergence.perp_direction else "unknown"
        result["divergence_type"] = divergence.divergence_type
        result["signal_strength"] = divergence.signal_strength

        # Divergence is risky - traders may be trapped
        if divergence.divergence_type in ("bullish_divergence", "bearish_divergence"):
            result["divergence_score"] = int(divergence.signal_strength)
        else:
            result["divergence_score"] = 0  # Aligned = no trapped traders

    except Exception as e:
        pass

    return result


# === COMBINED RISK CALCULATION ===

def calculate_combined_risk(volatility: Dict, liquidation: Dict, crowding: Dict, divergence: Dict) -> Dict:
    """
    Calculate combined risk score using weighted average.
    """
    scores = {
        "volatility": volatility["volatility_score"],
        "liquidation": liquidation["liquidation_score"],
        "crowding": crowding["crowding_score"],
        "divergence": divergence["divergence_score"],
    }

    # Calculate weighted average
    total_weight = 0
    weighted_sum = 0

    for category, weight in RISK_WEIGHTS.items():
        score = scores.get(category, 50)
        weighted_sum += score * weight
        total_weight += weight

    combined_score = weighted_sum / total_weight if total_weight > 0 else 50

    # Determine verdict
    if combined_score < RISK_THRESHOLDS["SAFE"]:
        verdict = "SAFE"
    elif combined_score < RISK_THRESHOLDS["CAUTION"]:
        verdict = "CAUTION"
    else:
        verdict = "AVOID"

    return {
        "volatility_score": scores["volatility"],
        "liquidation_score": scores["liquidation"],
        "crowding_score": scores["crowding"],
        "divergence_score": scores["divergence"],
        "combined_score": int(combined_score),
        "verdict": verdict,
    }


# === REPORT OUTPUT ===

def print_report(all_results: List[Dict]):
    """Print formatted risk analysis report."""
    print("\n" + "=" * 90)
    print("  COMPREHENSIVE RISK ANALYSIS - FUNDING COINS")
    print("=" * 90)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Data Sources: Hyperliquid API (via ccxt), Coinglass API")
    print("=" * 90)

    # === VOLATILITY RISK ===
    print("\n" + "-" * 90)
    print("  VOLATILITY RISK")
    print("-" * 90)
    print(f"  {'Coin':<12} {'Max Move':<12} {'Avg Move':<12} {'Leverage':<10} {'Risk Score':<12} {'Status'}")
    print("-" * 90)

    for r in all_results:
        v = r["volatility"]
        max_move = f"{v['max_daily_move']:.1f}%" if v["max_daily_move"] else "N/A"
        avg_move = f"{v['avg_daily_move']:.1f}%" if v["avg_daily_move"] else "N/A"
        score = v["volatility_score"]
        status = "LOW" if score < 30 else ("MEDIUM" if score < 60 else "HIGH")
        print(f"  {r['pair']:<12} {max_move:<12} {avg_move:<12} {v['safe_leverage']}x         {score:<12} {status}")

    # === LIQUIDATION CLUSTERS ===
    print("\n" + "-" * 90)
    print("  LIQUIDATION CLUSTERS")
    print("-" * 90)
    print(f"  {'Coin':<12} {'Near Long':<12} {'Near Short':<12} {'Magnet':<10} {'Risk Score':<12} {'Status'}")
    print("-" * 90)

    for r in all_results:
        liq = r["liquidation"]
        near_long = f"{liq['nearest_long_dist']:.1f}%" if liq["nearest_long_dist"] else "N/A"
        near_short = f"{liq['nearest_short_dist']:.1f}%" if liq["nearest_short_dist"] else "N/A"
        magnet = liq["magnetic_direction"].upper() if liq["magnetic_direction"] != "unknown" else "N/A"
        score = liq["liquidation_score"]
        status = "LOW" if score < 30 else ("MEDIUM" if score < 60 else "HIGH")
        print(f"  {r['pair']:<12} {near_long:<12} {near_short:<12} {magnet:<10} {score:<12} {status}")

    # === MARKET CROWDING ===
    print("\n" + "-" * 90)
    print("  MARKET CROWDING")
    print("-" * 90)
    print(f"  {'Coin':<12} {'Long%':<10} {'Short%':<10} {'Sentiment':<16} {'Score':<10} {'Contrarian'}")
    print("-" * 90)

    for r in all_results:
        c = r["crowding"]
        long_pct = f"{c['long_ratio']:.1f}%" if c["long_ratio"] else "N/A"
        short_pct = f"{c['short_ratio']:.1f}%" if c["short_ratio"] else "N/A"
        sentiment = c["sentiment"] if c["sentiment"] != "unknown" else "N/A"
        score = c["crowding_score"]
        contrarian = c["contrarian_signal"].upper()
        print(f"  {r['pair']:<12} {long_pct:<10} {short_pct:<10} {sentiment:<16} {score:<10} {contrarian}")

    # === CVD DIVERGENCE ===
    print("\n" + "-" * 90)
    print("  CVD DIVERGENCE")
    print("-" * 90)
    print(f"  {'Coin':<12} {'Perp CVD':<14} {'Spot CVD':<14} {'Divergence':<18} {'Strength':<10} {'Signal'}")
    print("-" * 90)

    for r in all_results:
        d = r["divergence"]
        perp = d["perp_direction"].upper() if d["perp_direction"] != "unknown" else "N/A"
        spot = d["spot_direction"].upper() if d["spot_direction"] != "unknown" else "N/A"
        div_type = d["divergence_type"] if d["divergence_type"] != "unknown" else "N/A"
        strength = f"{d['signal_strength']:.0f}%" if d["signal_strength"] > 0 else "-"

        # Determine signal based on divergence type
        if d["divergence_type"] == "bearish_divergence" and d["signal_strength"] >= 50:
            signal = "SHORT"
        elif d["divergence_type"] == "bullish_divergence" and d["signal_strength"] >= 50:
            signal = "LONG"
        else:
            signal = "-"

        print(f"  {r['pair']:<12} {perp:<14} {spot:<14} {div_type:<18} {strength:<10} {signal}")

    # === COMBINED RISK MATRIX ===
    print("\n" + "-" * 90)
    print("  COMBINED RISK MATRIX")
    print("-" * 90)
    print(f"  {'Coin':<12} {'Vol':<8} {'Liq':<8} {'Crowd':<8} {'Div':<8} {'TOTAL':<8} {'VERDICT'}")
    print("-" * 90)

    # Sort by combined score (safest first)
    sorted_results = sorted(all_results, key=lambda x: x["combined"]["combined_score"])

    for r in sorted_results:
        c = r["combined"]
        print(f"  {r['pair']:<12} {c['volatility_score']:<8} {c['liquidation_score']:<8} {c['crowding_score']:<8} {c['divergence_score']:<8} {c['combined_score']:<8} {c['verdict']}")

    # === RECOMMENDATIONS ===
    print("\n" + "-" * 90)
    print("  RECOMMENDATIONS")
    print("-" * 90)

    safe = [r["pair"] for r in sorted_results if r["combined"]["verdict"] == "SAFE"]
    caution = [r["pair"] for r in sorted_results if r["combined"]["verdict"] == "CAUTION"]
    avoid = [r["pair"] for r in sorted_results if r["combined"]["verdict"] == "AVOID"]

    if safe:
        print(f"\n  SAFE TO TRADE: {', '.join(safe)}")
    if caution:
        print(f"  USE CAUTION:   {', '.join(caution)}")
    if avoid:
        print(f"  AVOID NOW:     {', '.join(avoid)}")

    # Print specific warnings for avoid coins
    if avoid:
        print("\n  Reasons to avoid:")
        for r in sorted_results:
            if r["combined"]["verdict"] == "AVOID":
                reasons = []
                if r["combined"]["volatility_score"] >= 60:
                    reasons.append("high volatility")
                if r["combined"]["liquidation_score"] >= 60:
                    reasons.append("liquidation cluster nearby")
                if r["combined"]["crowding_score"] >= 60:
                    reasons.append("crowded position")
                if r["combined"]["divergence_score"] >= 60:
                    reasons.append("trapped traders")
                print(f"    - {r['pair']}: {', '.join(reasons)}")

    print("\n" + "=" * 90)
    print("  SCORING GUIDE")
    print("=" * 90)
    print("  Risk Score: 0-100 (0=safe, 100=dangerous)")
    print("  Weights: Volatility 30%, Liquidation 25%, Crowding 25%, Divergence 20%")
    print("  SAFE: <30 | CAUTION: 30-60 | AVOID: >60")
    print("=" * 90 + "\n")


# === MAIN ===

def main():
    print("\n" + "=" * 90)
    print("  INITIALIZING COMPREHENSIVE RISK ANALYSIS")
    print("=" * 90)

    # Initialize Coinglass API
    coinglass = None
    api_key = os.environ.get("COINGLASS_API_KEY")

    if CoinglassAPI and api_key:
        try:
            coinglass = CoinglassAPI(api_key=api_key, request_interval=2.5)
            print("  Coinglass API: Connected")
        except Exception as e:
            print(f"  Coinglass API: Failed to connect - {e}")
    elif not api_key:
        print("  Coinglass API: No API key found (set COINGLASS_API_KEY)")
    else:
        print("  Coinglass API: Module not available")

    print("\n  Analyzing coins...\n")

    all_results = []

    for pair, exchange_info in COINS_TO_ANALYZE.items():
        print(f"  {pair}...", end=" ", flush=True)

        # Get volatility data
        volatility = analyze_volatility(pair, exchange_info)

        # Get Coinglass data
        liquidation = analyze_liquidations(coinglass, pair)
        crowding = analyze_crowding(coinglass, pair)
        divergence = analyze_divergence(coinglass, pair)

        # Calculate combined risk
        combined = calculate_combined_risk(volatility, liquidation, crowding, divergence)

        all_results.append({
            "pair": pair,
            "volatility": volatility,
            "liquidation": liquidation,
            "crowding": crowding,
            "divergence": divergence,
            "combined": combined,
        })

        status_parts = []
        if volatility["data_available"]:
            status_parts.append("vol")
        if liquidation["data_available"]:
            status_parts.append("liq")
        if crowding["data_available"]:
            status_parts.append("ls")
        if divergence["data_available"]:
            status_parts.append("cvd")

        if status_parts:
            print(f"OK ({', '.join(status_parts)})")
        else:
            print("(no data)")

    # Print report
    print_report(all_results)


if __name__ == "__main__":
    main()
