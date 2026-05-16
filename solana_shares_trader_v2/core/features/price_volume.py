"""Block 1: Price & Volume Multi-TF Features (18 features).

- OHLCV on 3s, 5s, 15s, 30s, 1m, 3m, 5m, 15m
- Returns (log, simple) on all TF
- Volatility (Garman-Klass, Parkinson, Rogers-Satchell) on all TF
- Volume delta, Volume ratio (current / mean 20 periods)
- VWAP deviation
- Session volume profile (POC, VAH, VAL)
"""

import numpy as np
import math
from typing import Dict, Optional
from collections import deque

from core.utils.helpers import (
    log_return, simple_return, garman_klass_vol,
    parkinson_vol, rogers_satchell_vol, safe_div, ema,
)


class PriceVolumeFeatures:
    """Computes price, return, volatility, volume, and VWAP features."""

    TIMEFRAMES = ["3s", "5s", "15s", "30s", "1m", "3m", "5m", "15m"]

    def __init__(self):
        # Running VWAP state per symbol
        self._vwap: Dict[str, Dict] = {}
        # Volume profile buckets
        self._volume_profile: Dict[str, Dict] = {}

    def compute(
        self,
        ohlcv_1m: Optional[Dict] = None,
        ohlcv_5m: Optional[Dict] = None,
        recent_trades: list = None,
        current_price: float = 0.0,
        symbol: str = "SOLUSDT",
    ) -> Dict[str, float]:
        """Compute all 18 price/volume features."""
        features = {}

        # === Returns on multiple timeframes ===
        if ohlcv_1m and len(ohlcv_1m.get("close", [])) >= 2:
            close = ohlcv_1m["close"]
            high = ohlcv_1m["high"]
            low = ohlcv_1m["low"]
            open_ = ohlcv_1m["open"]
            volume = ohlcv_1m["volume"]

            # Log returns at different lookbacks (simulating sub-minute from 1m data)
            features["ret_log_1m"] = log_return(close[-1], close[-2])
            features["ret_simple_1m"] = simple_return(close[-1], close[-2])

            if len(close) >= 3:
                features["ret_log_3m"] = log_return(close[-1], close[-3])
                features["ret_simple_3m"] = simple_return(close[-1], close[-3])
            else:
                features["ret_log_3m"] = 0.0
                features["ret_simple_3m"] = 0.0

            if len(close) >= 5:
                features["ret_log_5m"] = log_return(close[-1], close[-5])
                features["ret_simple_5m"] = simple_return(close[-1], close[-5])
            else:
                features["ret_log_5m"] = 0.0
                features["ret_simple_5m"] = 0.0

            if len(close) >= 15:
                features["ret_log_15m"] = log_return(close[-1], close[-15])
                features["ret_simple_15m"] = simple_return(close[-1], close[-15])
            else:
                features["ret_log_15m"] = 0.0
                features["ret_simple_15m"] = 0.0

            # === Volatility estimators (last 20 bars) ===
            window = min(20, len(close))
            h = high[-window:]
            l = low[-window:]
            o = open_[-window:]
            c = close[-window:]

            features["vol_garman_klass"] = garman_klass_vol(h, l, o, c)
            features["vol_parkinson"] = parkinson_vol(h, l)
            features["vol_rogers_satchell"] = rogers_satchell_vol(h, l, o, c)

            # === Volume features ===
            vol_arr = volume[-window:]
            vol_mean = np.mean(vol_arr) if len(vol_arr) > 0 else 1.0
            features["volume_ratio"] = safe_div(volume[-1], vol_mean) if len(volume) > 0 else 1.0

            # Volume delta (taker buy - taker sell proxy)
            taker_buy = ohlcv_1m.get("taker_buy_volume", np.zeros(1))
            if len(taker_buy) > 0 and len(volume) > 0:
                features["volume_delta"] = float(taker_buy[-1] - (volume[-1] - taker_buy[-1]))
            else:
                features["volume_delta"] = 0.0

        else:
            # Defaults when no 1m data
            for k in ["ret_log_1m", "ret_simple_1m", "ret_log_3m", "ret_simple_3m",
                       "ret_log_5m", "ret_simple_5m", "ret_log_15m", "ret_simple_15m",
                       "vol_garman_klass", "vol_parkinson", "vol_rogers_satchell",
                       "volume_ratio", "volume_delta"]:
                features[k] = 0.0

        # === VWAP deviation ===
        features["vwap_deviation"] = self._compute_vwap_deviation(
            symbol, recent_trades, current_price
        )

        # === Session Volume Profile (POC, VAH, VAL) ===
        poc, vah, val = self._compute_volume_profile(symbol, recent_trades, current_price)
        features["vol_profile_poc_dist"] = safe_div(current_price - poc, poc) if poc > 0 else 0.0
        features["vol_profile_vah_dist"] = safe_div(current_price - vah, vah) if vah > 0 else 0.0
        features["vol_profile_val_dist"] = safe_div(current_price - val, val) if val > 0 else 0.0

        # Sub-minute return proxies from trade data
        if recent_trades and len(recent_trades) >= 2:
            prices = [t["price"] for t in recent_trades]
            # 5s, 15s, 30s returns from trade-level data
            now_ms = recent_trades[-1]["ts"]
            for label, window_ms in [("3s", 3000), ("5s", 5000), ("15s", 15000), ("30s", 30000)]:
                cutoff = now_ms - window_ms
                in_window = [t["price"] for t in recent_trades if t["ts"] >= cutoff]
                if len(in_window) >= 2:
                    features[f"ret_log_{label}"] = log_return(in_window[-1], in_window[0])
                else:
                    features[f"ret_log_{label}"] = 0.0
        else:
            for label in ["3s", "5s", "15s", "30s"]:
                features[f"ret_log_{label}"] = 0.0

        return features

    def _compute_vwap_deviation(
        self, symbol: str, trades: list, current_price: float
    ) -> float:
        """VWAP = sum(price * volume) / sum(volume). Deviation from current."""
        if not trades:
            return 0.0

        if symbol not in self._vwap:
            self._vwap[symbol] = {"pv_sum": 0.0, "v_sum": 0.0}

        state = self._vwap[symbol]
        for t in trades[-50:]:  # last 50 trades for efficiency
            state["pv_sum"] += t["price"] * t["qty"]
            state["v_sum"] += t["qty"]

        vwap = safe_div(state["pv_sum"], state["v_sum"])
        if vwap <= 0:
            return 0.0
        return (current_price - vwap) / vwap

    def _compute_volume_profile(
        self, symbol: str, trades: list, current_price: float
    ) -> tuple:
        """Compute POC (Point of Control), VAH, VAL from trade data."""
        if not trades or len(trades) < 10:
            return current_price, current_price, current_price

        prices = [t["price"] for t in trades]
        volumes = [t["qty"] for t in trades]
        min_p, max_p = min(prices), max(prices)

        if max_p <= min_p:
            return current_price, current_price, current_price

        n_bins = 50
        bin_edges = np.linspace(min_p, max_p, n_bins + 1)
        vol_profile = np.zeros(n_bins)

        for p, v in zip(prices, volumes):
            idx = min(int((p - min_p) / (max_p - min_p) * n_bins), n_bins - 1)
            vol_profile[idx] += v

        # POC = price level with max volume
        poc_idx = np.argmax(vol_profile)
        poc = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

        # Value Area (70% of volume)
        total_vol = vol_profile.sum()
        target = total_vol * 0.70
        sorted_idx = np.argsort(-vol_profile)
        cumulative = 0.0
        va_indices = []
        for idx in sorted_idx:
            va_indices.append(idx)
            cumulative += vol_profile[idx]
            if cumulative >= target:
                break

        va_indices.sort()
        val = (bin_edges[va_indices[0]] + bin_edges[va_indices[0] + 1]) / 2
        vah = (bin_edges[va_indices[-1]] + bin_edges[va_indices[-1] + 1]) / 2

        return poc, vah, val
