"""
Coinglass Liquidation Heatmap Module

Handles fetching and processing of liquidation data from Coinglass API.
"""

import time
from typing import TYPE_CHECKING, Optional

from .models import LiquidationCluster, LiquidationHeatmap

if TYPE_CHECKING:
    from .client import CoinglassClient


class LiquidationMixin:
    """Mixin providing liquidation-related methods for CoinglassAPI."""

    def get_liquidation_heatmap(self: "CoinglassClient", pair: str) -> Optional[LiquidationHeatmap]:
        """
        Get liquidation heatmap showing where liquidation clusters sit.

        Price is "magnetically attracted" to liquidation clusters because
        whales hunt them for liquidity.

        Args:
            pair: Trading pair

        Returns:
            LiquidationHeatmap or None
        """
        symbol = self._to_coinglass_symbol(pair)
        cache_key = f"liq_heatmap_{symbol}"

        if self._is_cache_valid(cache_key) and symbol in self._liquidation_cache:
            return self._liquidation_cache[symbol]

        data = self._request(
            "/api/futures/liquidation-heatmap",
            params={"symbol": symbol, "interval": "h24"}
        )

        if not data:
            return None

        # Parse heatmap data
        current_price = float(data.get("price", 0))
        liq_data = data.get("data", [])

        long_clusters, short_clusters = _parse_liquidation_clusters(liq_data, current_price)

        # Sort by distance
        long_clusters.sort(key=lambda x: abs(x.distance_pct))
        short_clusters.sort(key=lambda x: abs(x.distance_pct))

        # Find nearest clusters
        nearest_long = long_clusters[0].price if long_clusters else None
        nearest_short = short_clusters[0].price if short_clusters else None

        # Determine magnetic direction
        magnetic_direction, interpretation = _calculate_magnetic_direction(
            long_clusters, short_clusters
        )

        heatmap = LiquidationHeatmap(
            symbol=symbol,
            current_price=current_price,
            long_clusters=long_clusters[:5],  # Top 5
            short_clusters=short_clusters[:5],
            nearest_long_liq=nearest_long,
            nearest_short_liq=nearest_short,
            magnetic_direction=magnetic_direction,
            interpretation=interpretation,
        )

        self._liquidation_cache[symbol] = heatmap
        self._cache_timestamps[cache_key] = time.time()

        return heatmap


def _parse_liquidation_clusters(
    liq_data: list,
    current_price: float
) -> tuple[list[LiquidationCluster], list[LiquidationCluster]]:
    """Parse liquidation data into long and short clusters."""
    long_clusters = []
    short_clusters = []

    for level in liq_data:
        price = float(level.get("price", 0))
        liq_amount = float(level.get("liqAmount", level.get("liquidation", 0)))

        if price == 0 or current_price == 0:
            continue

        distance_pct = ((price - current_price) / current_price) * 100

        # Determine risk level based on size and distance
        if liq_amount > 10_000_000:  # $10M+
            risk_level = "high"
        elif liq_amount > 1_000_000:  # $1M+
            risk_level = "medium"
        else:
            risk_level = "low"

        cluster = LiquidationCluster(
            price=price,
            side="long" if price < current_price else "short",
            estimated_size_usd=liq_amount,
            distance_pct=distance_pct,
            risk_level=risk_level,
        )

        if price < current_price:
            long_clusters.append(cluster)
        else:
            short_clusters.append(cluster)

    return long_clusters, short_clusters


def _calculate_magnetic_direction(
    long_clusters: list[LiquidationCluster],
    short_clusters: list[LiquidationCluster]
) -> tuple[str, str]:
    """
    Determine magnetic direction based on liquidation imbalance.

    Price tends toward larger liquidation pools as whales hunt liquidity.
    """
    long_total = sum(c.estimated_size_usd for c in long_clusters[:3])
    short_total = sum(c.estimated_size_usd for c in short_clusters[:3])

    if long_total > short_total * 1.5:
        magnetic_direction = "down"
        interpretation = (
            f"More liquidation liquidity below (${long_total/1e6:.1f}M long liqs). "
            "Price may be drawn down to hunt long liquidations."
        )
    elif short_total > long_total * 1.5:
        magnetic_direction = "up"
        interpretation = (
            f"More liquidation liquidity above (${short_total/1e6:.1f}M short liqs). "
            "Price may be drawn up to hunt short liquidations."
        )
    else:
        magnetic_direction = "neutral"
        interpretation = "Liquidation clusters relatively balanced above and below."

    return magnetic_direction, interpretation
