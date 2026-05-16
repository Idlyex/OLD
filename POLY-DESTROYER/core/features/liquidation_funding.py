"""Block 4: Liquidation & Funding Features (8 features).

- Distance to major liquidation levels (long/short clusters)
- Liquidation heat (by time)
- Funding rate + funding rate acceleration (1m/5m change)
- OI change rate + OI-weighted price
- CVD vs Funding divergence
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

from core.utils.helpers import safe_div


class LiquidationFundingFeatures:
    """Liquidation cascade and funding rate feature computation."""

    def __init__(self):
        # Track liquidation clusters for level estimation
        self._liq_clusters: Dict[str, Dict] = {}

    def compute(
        self,
        liquidation_heat: Optional[Dict] = None,
        funding_rate: float = 0.0,
        funding_accel_1m: float = 0.0,
        funding_accel_5m: float = 0.0,
        cvd_value: float = 0.0,
        current_price: float = 0.0,
        recent_liqs: Optional[List[Dict]] = None,
        symbol: str = "SOLUSDT",
    ) -> Dict[str, float]:
        """Compute all 8 liquidation & funding features."""
        features = {}

        # === Distance to major liquidation levels ===
        long_dist, short_dist = self._estimate_liq_levels(
            symbol, recent_liqs, current_price
        )
        features["liq_dist_long"] = long_dist  # distance to nearest long liq cluster
        features["liq_dist_short"] = short_dist  # distance to nearest short liq cluster

        # === Liquidation heat ===
        if liquidation_heat:
            features["liq_heat_total"] = float(liquidation_heat.get("total_count", 0))
            features["liq_heat_long"] = float(liquidation_heat.get("long_liq_notional", 0))
            features["liq_heat_short"] = float(liquidation_heat.get("short_liq_notional", 0))
            # Normalized ratio
            total_notional = (
                liquidation_heat.get("long_liq_notional", 0)
                + liquidation_heat.get("short_liq_notional", 0)
            )
            features["liq_imbalance"] = safe_div(
                liquidation_heat.get("long_liq_notional", 0)
                - liquidation_heat.get("short_liq_notional", 0),
                total_notional + 1,
            )
        else:
            features["liq_heat_total"] = 0.0
            features["liq_heat_long"] = 0.0
            features["liq_heat_short"] = 0.0
            features["liq_imbalance"] = 0.0

        # === Funding rate ===
        features["funding_rate"] = funding_rate
        features["funding_accel_1m"] = funding_accel_1m
        features["funding_accel_5m"] = funding_accel_5m

        # === OI change rate proxy ===
        # OI-weighted price movement: large funding + volume spike = OI change
        features["oi_change_proxy"] = abs(funding_rate) * 10000  # normalized

        # === CVD vs Funding divergence ===
        # Positive funding + negative CVD = divergence (potential reversal)
        features["cvd_funding_divergence"] = self._compute_cvd_funding_divergence(
            cvd_value, funding_rate
        )

        return features

    def _estimate_liq_levels(
        self, symbol: str, recent_liqs: Optional[List[Dict]], current_price: float
    ) -> tuple:
        """Estimate distance to nearest long/short liquidation clusters."""
        if not recent_liqs or current_price <= 0:
            return 0.0, 0.0

        # Cluster liquidations by price level
        long_liqs = [l for l in recent_liqs if l.get("side") == "SELL"]  # long liq
        short_liqs = [l for l in recent_liqs if l.get("side") == "BUY"]  # short liq

        long_dist = 0.0
        short_dist = 0.0

        if long_liqs:
            # Long liquidation levels are below current price
            long_prices = [l["price"] for l in long_liqs]
            nearest_long = max(long_prices)  # closest from below
            long_dist = safe_div(current_price - nearest_long, current_price)

        if short_liqs:
            # Short liquidation levels are above current price
            short_prices = [l["price"] for l in short_liqs]
            nearest_short = min(short_prices)  # closest from above
            short_dist = safe_div(nearest_short - current_price, current_price)

        return long_dist, short_dist

    def _compute_cvd_funding_divergence(
        self, cvd_value: float, funding_rate: float
    ) -> float:
        """
        CVD vs Funding divergence.
        Positive funding (longs pay) + negative CVD (selling pressure) = bearish divergence.
        Negative funding (shorts pay) + positive CVD (buying pressure) = bullish divergence.
        """
        if abs(funding_rate) < 1e-8:
            return 0.0

        # Normalize CVD to a sign
        cvd_sign = 1.0 if cvd_value > 0 else -1.0 if cvd_value < 0 else 0.0
        funding_sign = 1.0 if funding_rate > 0 else -1.0

        # Divergence when signs oppose
        if cvd_sign != funding_sign:
            return -funding_sign * abs(funding_rate) * 10000  # bearish if funding+/cvd-
        return 0.0
