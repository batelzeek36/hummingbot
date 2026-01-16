"""
Coinglass Spot vs Perp Divergence Module

Detects divergence between spot and perpetual CVD - THE killer reversal signal.
"""

from typing import TYPE_CHECKING, Optional

from .models import CVDDirection, SpotPerpDivergence

if TYPE_CHECKING:
    from .api import CoinglassAPI


class DivergenceMixin:
    """Mixin providing divergence detection methods for CoinglassAPI."""

    def get_spot_perp_divergence(self: "CoinglassAPI", pair: str) -> Optional[SpotPerpDivergence]:
        """
        Detect Spot vs Perpetual CVD divergence - THE KILLER SIGNAL.

        When perp CVD diverges from spot CVD, leverage traders are trapped:
        - Perp CVD up + Spot CVD flat/down = Longs trapped, SHORT signal
        - Perp CVD down + Spot CVD flat/up = Shorts trapped, LONG signal

        Args:
            pair: Trading pair

        Returns:
            SpotPerpDivergence analysis or None
        """
        futures_cvd = self.get_futures_cvd(pair)
        spot_cvd = self.get_spot_cvd(pair)

        if not futures_cvd or not spot_cvd:
            return None

        symbol = self._to_coinglass_symbol(pair)

        # Analyze divergence
        divergence_type, signal_strength, interpretation, warnings = _analyze_divergence(
            futures_cvd.direction,
            spot_cvd.direction
        )

        return SpotPerpDivergence(
            symbol=symbol,
            spot_cvd=spot_cvd.cvd,
            perp_cvd=futures_cvd.cvd,
            spot_direction=spot_cvd.direction,
            perp_direction=futures_cvd.direction,
            divergence_type=divergence_type,
            signal_strength=signal_strength,
            interpretation=interpretation,
            warnings=warnings,
        )


def _analyze_divergence(
    perp_direction: CVDDirection,
    spot_direction: CVDDirection
) -> tuple[str, float, str, list[str]]:
    """
    Analyze divergence between perp and spot CVD directions.

    Returns:
        Tuple of (divergence_type, signal_strength, interpretation, warnings)
    """
    warnings = []
    divergence_type = "aligned"
    signal_strength = 0.0
    interpretation = ""

    # Strong divergence: perp bullish, spot bearish/neutral
    if perp_direction in (CVDDirection.BULLISH, CVDDirection.STRONG_BULLISH):
        if spot_direction in (CVDDirection.BEARISH, CVDDirection.STRONG_BEARISH):
            divergence_type = "bearish_divergence"
            signal_strength = 90.0 if perp_direction == CVDDirection.STRONG_BULLISH else 70.0
            interpretation = (
                "BEARISH DIVERGENCE: Perp CVD rising but spot CVD falling. "
                "Leveraged longs are buying, but real money is selling. "
                "LONGS ARE TRAPPED - expect reversal down."
            )
            warnings.append("HIGH CONFIDENCE SHORT SIGNAL")
        elif spot_direction == CVDDirection.NEUTRAL:
            divergence_type = "bearish_divergence"
            signal_strength = 60.0
            interpretation = (
                "Mild bearish divergence: Perp CVD rising, spot CVD flat. "
                "Leverage is leading without spot follow-through."
            )

    # Strong divergence: perp bearish, spot bullish/neutral
    elif perp_direction in (CVDDirection.BEARISH, CVDDirection.STRONG_BEARISH):
        if spot_direction in (CVDDirection.BULLISH, CVDDirection.STRONG_BULLISH):
            divergence_type = "bullish_divergence"
            signal_strength = 90.0 if perp_direction == CVDDirection.STRONG_BEARISH else 70.0
            interpretation = (
                "BULLISH DIVERGENCE: Perp CVD falling but spot CVD rising. "
                "Leveraged shorts are selling, but real money is buying. "
                "SHORTS ARE TRAPPED - expect reversal up."
            )
            warnings.append("HIGH CONFIDENCE LONG SIGNAL")
        elif spot_direction == CVDDirection.NEUTRAL:
            divergence_type = "bullish_divergence"
            signal_strength = 60.0
            interpretation = (
                "Mild bullish divergence: Perp CVD falling, spot CVD flat. "
                "Leverage is leading without spot follow-through."
            )

    # Aligned - both moving same direction
    else:
        if perp_direction == spot_direction:
            interpretation = (
                f"Aligned: Both perp and spot CVD are {perp_direction.value}. "
                "No divergence - follow the trend."
            )
        else:
            interpretation = "Mixed signals - no clear divergence."

    return divergence_type, signal_strength, interpretation, warnings
