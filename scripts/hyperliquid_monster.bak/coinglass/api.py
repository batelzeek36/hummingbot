"""
Coinglass API - Main Interface

Composes all domain modules into a unified API client.
"""

import time
from typing import Dict, Tuple

from .client import CoinglassClient
from .cvd import CVDMixin
from .divergence import DivergenceMixin
from .liquidations import LiquidationMixin
from .models import CVDDirection
from .ratios import RatioMixin


class CoinglassAPI(
    CVDMixin,
    DivergenceMixin,
    LiquidationMixin,
    RatioMixin,
    CoinglassClient
):
    """
    Coinglass API client for aggregated market data.

    Provides:
    - CVD (Cumulative Volume Delta) - Aggregated across exchanges
    - Spot vs Perp CVD Divergence - The killer reversal signal
    - Liquidation Heatmap - Know where liquidation clusters sit
    - Long/Short Ratios - Market positioning data

    Rate limit: 30 requests/min for Hobbyist plan.
    """

    def get_combined_signal(self, pair: str) -> Dict:
        """
        Get all Coinglass signals combined for a trading pair.

        Args:
            pair: Trading pair

        Returns:
            Dict with all signals and overall recommendation
        """
        futures_cvd = self.get_futures_cvd(pair)
        divergence = self.get_spot_perp_divergence(pair)
        liquidation = self.get_liquidation_heatmap(pair)
        ls_ratio = self.get_long_short_ratio(pair)

        signals = {
            "pair": pair,
            "timestamp": time.time(),
            "futures_cvd": {
                "direction": futures_cvd.direction.value if futures_cvd else "unknown",
                "cvd_change": futures_cvd.cvd_change if futures_cvd else 0,
            } if futures_cvd else None,
            "spot_perp_divergence": {
                "type": divergence.divergence_type if divergence else "unknown",
                "strength": divergence.signal_strength if divergence else 0,
                "warnings": divergence.warnings if divergence else [],
            } if divergence else None,
            "liquidation": {
                "magnetic_direction": liquidation.magnetic_direction if liquidation else "unknown",
                "nearest_long_liq": liquidation.nearest_long_liq if liquidation else None,
                "nearest_short_liq": liquidation.nearest_short_liq if liquidation else None,
            } if liquidation else None,
            "long_short_ratio": {
                "sentiment": ls_ratio.sentiment if ls_ratio else "unknown",
                "crowding_score": ls_ratio.crowding_score if ls_ratio else 0,
                "contrarian_signal": ls_ratio.contrarian_signal if ls_ratio else "neutral",
            } if ls_ratio else None,
        }

        # Calculate overall recommendation
        bullish_signals, bearish_signals = _count_signals(
            futures_cvd, divergence, liquidation, ls_ratio
        )

        # Overall recommendation
        if bullish_signals > bearish_signals + 1:
            signals["recommendation"] = "LONG"
            signals["confidence"] = min(95, bullish_signals * 20)
        elif bearish_signals > bullish_signals + 1:
            signals["recommendation"] = "SHORT"
            signals["confidence"] = min(95, bearish_signals * 20)
        else:
            signals["recommendation"] = "NEUTRAL"
            signals["confidence"] = 50

        return signals

    def should_block_entry(self, pair: str, direction: str) -> Tuple[bool, str]:
        """
        Check if entry should be blocked based on Coinglass data.

        Args:
            pair: Trading pair
            direction: "long" or "short"

        Returns:
            Tuple of (should_block, reason)
        """
        divergence = self.get_spot_perp_divergence(pair)
        ls_ratio = self.get_long_short_ratio(pair)

        reasons = []

        # Check divergence
        if divergence and divergence.signal_strength >= 70:
            if direction == "long" and divergence.divergence_type == "bearish_divergence":
                reasons.append(
                    f"SPOT/PERP BEARISH DIVERGENCE ({divergence.signal_strength:.0f}%): "
                    "Perp longs are trapped - don't join them"
                )
            elif direction == "short" and divergence.divergence_type == "bullish_divergence":
                reasons.append(
                    f"SPOT/PERP BULLISH DIVERGENCE ({divergence.signal_strength:.0f}%): "
                    "Perp shorts are trapped - don't join them"
                )

        # Check long/short ratio
        if ls_ratio and ls_ratio.crowding_score >= 70:
            if direction == "long" and ls_ratio.sentiment == "crowded_long":
                reasons.append(
                    f"CROWDED LONG ({ls_ratio.long_ratio:.1f}%): "
                    "Too many longs - squeeze risk"
                )
            elif direction == "short" and ls_ratio.sentiment == "crowded_short":
                reasons.append(
                    f"CROWDED SHORT ({ls_ratio.short_ratio:.1f}%): "
                    "Too many shorts - squeeze risk"
                )

        if reasons:
            return True, " | ".join(reasons)
        return False, ""


def _count_signals(futures_cvd, divergence, liquidation, ls_ratio) -> Tuple[int, int]:
    """Count bullish and bearish signals from all sources."""
    bullish_signals = 0
    bearish_signals = 0

    if futures_cvd:
        if futures_cvd.direction in (CVDDirection.BULLISH, CVDDirection.STRONG_BULLISH):
            bullish_signals += 1
        elif futures_cvd.direction in (CVDDirection.BEARISH, CVDDirection.STRONG_BEARISH):
            bearish_signals += 1

    if divergence:
        if divergence.divergence_type == "bullish_divergence":
            bullish_signals += 2  # Weighted more heavily
        elif divergence.divergence_type == "bearish_divergence":
            bearish_signals += 2

    if liquidation:
        if liquidation.magnetic_direction == "up":
            bullish_signals += 1
        elif liquidation.magnetic_direction == "down":
            bearish_signals += 1

    if ls_ratio:
        if ls_ratio.contrarian_signal == "long":
            bullish_signals += 1
        elif ls_ratio.contrarian_signal == "short":
            bearish_signals += 1

    return bullish_signals, bearish_signals
