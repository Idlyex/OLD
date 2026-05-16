"""Feature Engine — orchestrates all 82 features across 6 blocks.

Aggregates: PriceVolume(18) + Technical(14) + Microstructure(22) +
            LiquidationFunding(8) + OnChain(10) + Regime(10) = 82 features.
"""

import time
import numpy as np
from typing import Dict, Optional, Any

from core.features.price_volume import PriceVolumeFeatures
from core.features.technical import TechnicalFeatures
from core.features.microstructure import MicrostructureFeatures
from core.features.liquidation_funding import LiquidationFundingFeatures
from core.features.regime import RegimeFeatures
from core.utils.logger import log


class FeatureEngine:
    """Master feature generator — computes all 82 features from raw data."""

    def __init__(self):
        self.price_volume = PriceVolumeFeatures()
        self.technical = TechnicalFeatures()
        self.microstructure = MicrostructureFeatures()
        self.liquidation_funding = LiquidationFundingFeatures()
        self.regime = RegimeFeatures()

        self._feature_names = None
        self._compute_count = 0
        self._total_time_ms = 0.0

    def compute_all(
        self,
        symbol: str,
        cex_collector=None,
        onchain_collector=None,
        current_price: float = 0.0,
    ) -> Dict[str, float]:
        """Compute all 82 features for a symbol.

        Args:
            symbol: Trading pair (e.g. 'SOLUSDT')
            cex_collector: CEXCollector instance with live data buffers
            onchain_collector: OnchainCollector instance
            current_price: Current mark/last price
        """
        t0 = time.perf_counter()
        features = {}

        if current_price <= 0 and cex_collector:
            current_price = cex_collector.get_latest_price(symbol)

        # ── Block 1: Price & Volume (18) ──
        ohlcv_1m = cex_collector.get_ohlcv_arrays(symbol, "1m", 60) if cex_collector else None
        ohlcv_5m = cex_collector.get_ohlcv_arrays(symbol, "5m", 30) if cex_collector else None
        recent_trades = cex_collector.get_recent_trades(symbol, 300) if cex_collector else []

        pv = self.price_volume.compute(
            ohlcv_1m=ohlcv_1m,
            ohlcv_5m=ohlcv_5m,
            recent_trades=recent_trades,
            current_price=current_price,
            symbol=symbol,
        )
        features.update(pv)

        # ── Block 2: Advanced Technical (14) ──
        tech = self.technical.compute(
            ohlcv=ohlcv_1m,
            current_price=current_price,
        )
        features.update(tech)

        # ── Block 3: Microstructure (22) ──
        depth = cex_collector.get_depth_snapshot(symbol) if cex_collector else None
        cvd_state = cex_collector.get_cvd(symbol) if cex_collector else None

        micro = self.microstructure.compute(
            depth=depth,
            recent_trades=recent_trades,
            cvd_state=cvd_state,
            current_price=current_price,
            symbol=symbol,
        )
        features.update(micro)

        # ── Block 4: Liquidation & Funding (8) ──
        liq_heat = cex_collector.get_liquidation_heat(symbol, 60) if cex_collector else None
        funding_rate = cex_collector.get_funding_rate(symbol) if cex_collector else 0.0
        funding_accel_1m = cex_collector.get_funding_acceleration(symbol, 1) if cex_collector else 0.0
        funding_accel_5m = cex_collector.get_funding_acceleration(symbol, 5) if cex_collector else 0.0
        recent_liqs = list(cex_collector.liquidations.get(symbol, [])) if cex_collector else []

        lf = self.liquidation_funding.compute(
            liquidation_heat=liq_heat,
            funding_rate=funding_rate,
            funding_accel_1m=funding_accel_1m,
            funding_accel_5m=funding_accel_5m,
            cvd_value=cvd_state.get("value", 0) if cvd_state else 0.0,
            current_price=current_price,
            recent_liqs=recent_liqs,
            symbol=symbol,
        )
        features.update(lf)

        # ── Block 5: On-Chain Solana (10) ──
        if onchain_collector:
            onchain = onchain_collector.get_all_features()
            features.update(onchain)
        else:
            features.update({
                "onchain_large_transfers_60s": 0.0,
                "onchain_whale_activity": 0.0,
                "onchain_dex_volume_spike": 1.0,
                "onchain_jupiter_accel": 1.0,
                "onchain_mev_bundles": 0.0,
                "onchain_priority_fee_pressure": 1.0,
                "onchain_token_creation_rate": 0.0,
                "onchain_large_transfers_300s": 0.0,
                "onchain_dex_volume_spike_300s": 1.0,
                "onchain_jupiter_accel_300s": 1.0,
            })

        # ── Block 6: Regime & Statistical (10) ──
        returns = None
        close_prices = None
        if ohlcv_1m and len(ohlcv_1m.get("close", [])) >= 20:
            close_arr = np.array(ohlcv_1m["close"])
            returns = np.diff(np.log(close_arr[close_arr > 0]))
            close_prices = close_arr

        regime = self.regime.compute(
            returns=returns,
            volatility=features.get("vol_garman_klass", 0.0),
            close_prices=close_prices,
        )
        features.update(regime)

        # ── Metadata ──
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._compute_count += 1
        self._total_time_ms += elapsed_ms

        if self._feature_names is None:
            self._feature_names = sorted(features.keys())
            log.info(f"FeatureEngine: {len(self._feature_names)} features initialized ({elapsed_ms:.1f}ms)")

        return features

    def get_feature_names(self):
        """Get ordered list of all feature names."""
        return self._feature_names or []

    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to ordered numpy array for model input."""
        if not self._feature_names:
            self._feature_names = sorted(features.keys())
        return np.array([features.get(k, 0.0) for k in self._feature_names], dtype=np.float64)

    def get_stats(self) -> Dict:
        """Performance stats."""
        return {
            "total_computes": self._compute_count,
            "avg_time_ms": self._total_time_ms / max(self._compute_count, 1),
            "feature_count": len(self._feature_names or []),
        }
