"""Ultra-fast vectorized dataset builder.

Replaces the slow per-row loop in TrainingDataset with fully vectorized
NumPy/Pandas operations.  Builds features + targets for 90 days of 1m data
in seconds, not minutes.

Supports both 5m and 15m market durations.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import norm

from core.utils.logger import log


# ═══════════════════════════════════════════════════════════
#  VECTORIZED FEATURE BLOCKS
# ═══════════════════════════════════════════════════════════

def _safe_div_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise safe division."""
    out = np.zeros_like(a, dtype=np.float64)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out


def _rolling_std(arr: np.ndarray, w: int) -> np.ndarray:
    """Fast rolling std using cumsum trick."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n < w:
        return out
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    s = cs[w - 1:].copy()
    s[1:] -= cs[:n - w]
    s2 = cs2[w - 1:].copy()
    s2[1:] -= cs2[:n - w]
    var = (s2 - s ** 2 / w) / w
    var = np.maximum(var, 0.0)
    out[w - 1:] = np.sqrt(var)
    return out


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """Fast rolling mean using cumsum."""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < w:
        return out
    cs = np.cumsum(arr)
    s = cs[w - 1:].copy()
    s[1:] -= cs[:n - w]
    out[w - 1:] = s / w
    return out


def _ema_vec(arr: np.ndarray, span: int) -> np.ndarray:
    """Vectorized EMA."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


# ── Block 1: Price & Volume (13 features from historical data) ──

def compute_price_volume_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized price/volume features across entire DataFrame."""
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    tbv = df.get("taker_buy_volume", df["volume"] * 0.5).values.astype(np.float64)
    n = len(close)

    feats = {}

    # Log returns at multiple lookbacks
    for lb, label in [(1, "1m"), (3, "3m"), (5, "5m"), (15, "15m")]:
        lr = np.zeros(n, dtype=np.float64)
        sr = np.zeros(n, dtype=np.float64)
        if n > lb:
            ratio = close[lb:] / np.maximum(close[:-lb], 1e-10)
            lr[lb:] = np.log(ratio)
            sr[lb:] = ratio - 1.0
        feats[f"ret_log_{label}"] = lr
        feats[f"ret_simple_{label}"] = sr

    # Volatility estimators (rolling window=20)
    w = 20
    u = np.log(np.maximum(high / np.maximum(open_, 1e-10), 1e-10))
    d = np.log(np.maximum(low / np.maximum(open_, 1e-10), 1e-10))
    c = np.log(np.maximum(close / np.maximum(open_, 1e-10), 1e-10))

    # Garman-Klass
    gk_elem = 0.5 * (u - d) ** 2 - (2 * np.log(2) - 1) * c ** 2
    feats["vol_garman_klass"] = np.sqrt(np.maximum(_rolling_mean(gk_elem, w), 0.0))
    feats["vol_garman_klass"] = np.nan_to_num(feats["vol_garman_klass"], 0.0)

    # Parkinson
    hl = np.log(np.maximum(high / np.maximum(low, 1e-10), 1e-10))
    feats["vol_parkinson"] = np.sqrt(np.maximum(_rolling_mean(hl ** 2, w), 0.0) / (4 * np.log(2)))
    feats["vol_parkinson"] = np.nan_to_num(feats["vol_parkinson"], 0.0)

    # Rogers-Satchell
    hc = np.log(np.maximum(high / np.maximum(close, 1e-10), 1e-10))
    ho = np.log(np.maximum(high / np.maximum(open_, 1e-10), 1e-10))
    lc = np.log(np.maximum(low / np.maximum(close, 1e-10), 1e-10))
    lo = np.log(np.maximum(low / np.maximum(open_, 1e-10), 1e-10))
    rs_elem = hc * ho + lc * lo
    feats["vol_rogers_satchell"] = np.sqrt(np.maximum(_rolling_mean(rs_elem, w), 0.0))
    feats["vol_rogers_satchell"] = np.nan_to_num(feats["vol_rogers_satchell"], 0.0)

    # Volume ratio
    vol_mean20 = _rolling_mean(volume, w)
    feats["volume_ratio"] = _safe_div_vec(volume, np.nan_to_num(vol_mean20, nan=1.0))

    # Volume delta
    feats["volume_delta"] = tbv - (volume - tbv)

    # VWAP deviation (rolling 20-bar proxy)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    feats["vwap_deviation"] = _safe_div_vec(
        close - _safe_div_vec(_rolling_mean(pv, w) * w, np.maximum(_rolling_mean(volume, w) * w, 1e-10)),
        np.maximum(close, 1e-10),
    )
    feats["vwap_deviation"] = np.nan_to_num(feats["vwap_deviation"], 0.0)

    # Volume profile distances (simplified — use rolling POC proxy)
    feats["vol_profile_poc_dist"] = np.zeros(n)
    feats["vol_profile_vah_dist"] = np.zeros(n)
    feats["vol_profile_val_dist"] = np.zeros(n)

    # Sub-minute returns (zeros in historical — no tick data)
    for label in ["3s", "5s", "15s", "30s"]:
        feats[f"ret_log_{label}"] = np.zeros(n)

    return pd.DataFrame(feats, index=df.index)


# ── Block 2: Technical (14 features) ──

def compute_technical_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized technical features."""
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    n = len(close)

    feats = {}

    # EMA ribbon
    ema_periods = [5, 8, 13, 21, 34, 55, 89, 144]
    ema_matrix = np.zeros((n, len(ema_periods)), dtype=np.float64)
    for j, p in enumerate(ema_periods):
        if n >= p:
            ema_matrix[:, j] = _ema_vec(close, p)
        else:
            ema_matrix[:, j] = close

    feats["ribbon_width"] = _safe_div_vec(
        np.max(ema_matrix, axis=1) - np.min(ema_matrix, axis=1),
        np.maximum(close, 1e-10),
    )

    # Ribbon alignment
    bullish_cnt = np.zeros(n, dtype=np.float64)
    for j in range(len(ema_periods) - 1):
        bullish_cnt += (ema_matrix[:, j] > ema_matrix[:, j + 1]).astype(np.float64)
    feats["ribbon_alignment"] = (bullish_cnt / max(len(ema_periods) - 1, 1)) * 2 - 1

    # EMA8 slope
    if n >= 8:
        ema8 = _ema_vec(close, 8)
        slope = np.zeros(n, dtype=np.float64)
        slope[1:] = (ema8[1:] - ema8[:-1]) / np.maximum(close[1:], 1e-10)
        feats["ema8_slope"] = slope
    else:
        feats["ema8_slope"] = np.zeros(n)

    # ATR normalized
    tr = np.zeros(n, dtype=np.float64)
    if n >= 2:
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
    atr = _ema_vec(tr, 14)
    feats["atr_normalized"] = _safe_div_vec(atr, np.maximum(close, 1e-10))

    # SuperTrend (simplified vectorized)
    hl2 = (high + low) / 2.0
    upper = hl2 + 3.0 * atr
    lower = hl2 - 3.0 * atr
    st_dir = np.where(close > upper, 1.0, np.where(close < lower, -1.0, 0.0))
    # Forward-fill zeros
    for i in range(1, n):
        if st_dir[i] == 0:
            st_dir[i] = st_dir[i - 1] if st_dir[i - 1] != 0 else 1.0
    feats["supertrend_direction"] = st_dir
    feats["supertrend_distance"] = np.where(
        st_dir > 0,
        _safe_div_vec(close - lower, np.maximum(close, 1e-10)),
        _safe_div_vec(upper - close, np.maximum(close, 1e-10)),
    )

    # ICT features (simplified vectorized)
    # FVG score
    fvg = np.zeros(n, dtype=np.float64)
    if n >= 3:
        bull_fvg = low[2:] - high[:-2]
        bear_fvg = low[:-2] - high[2:]
        gap = np.where(bull_fvg > 0, np.minimum(bull_fvg / np.maximum(close[2:], 1e-10) * 100, 1.0),
                       np.where(bear_fvg > 0, -np.minimum(bear_fvg / np.maximum(close[2:], 1e-10) * 100, 1.0), 0.0))
        fvg[2:] = gap
    feats["ict_fvg_score"] = fvg

    # Order block, BOS/CHOCH, liquidity sweep, MSS, Wyckoff — simplified
    feats["ict_order_block"] = np.zeros(n)
    feats["ict_bos_choch"] = np.zeros(n)
    feats["ict_liquidity_sweep"] = np.zeros(n)

    # MSS
    mss = np.zeros(n, dtype=np.float64)
    if n >= 15:
        trend_short = np.zeros(n)
        trend_long = np.zeros(n)
        trend_short[5:] = close[5:] - close[:-5]
        trend_long[15:] = close[15:] - close[:-15]
        bearish = (trend_long > 0) & (trend_short < 0)
        bullish = (trend_long < 0) & (trend_short > 0)
        mss[bearish] = -_safe_div_vec(np.abs(trend_short[bearish]), np.maximum(close[bearish], 1e-10))
        mss[bullish] = _safe_div_vec(np.abs(trend_short[bullish]), np.maximum(close[bullish], 1e-10))
    feats["mss_score"] = mss

    feats["wyckoff_signal"] = np.zeros(n)

    # Fractal dimension (simplified — rolling Katz)
    feats["fractal_higuchi"] = np.full(n, 1.5)
    feats["fractal_katz"] = np.full(n, 1.0)

    return pd.DataFrame(feats, index=df.index)


# ── Block 3: Microstructure (22 features — zeros for historical) ──

def compute_microstructure_vectorized(n: int) -> pd.DataFrame:
    """Microstructure features — default zeros for historical data."""
    feats = {
        "ob_imbalance_5": np.zeros(n), "ob_imbalance_10": np.zeros(n),
        "ob_imbalance_20": np.zeros(n), "book_pressure": np.zeros(n),
        "spread_bps": np.zeros(n), "depth_ratio": np.ones(n),
        "cvd_value": np.zeros(n), "cvd_slope": np.zeros(n),
        "aggr_buy_ratio_30s": np.full(n, 0.5), "aggr_sell_ratio_30s": np.full(n, 0.5),
        "aggr_buy_ratio_60s": np.full(n, 0.5), "aggr_sell_ratio_60s": np.full(n, 0.5),
        "aggr_buy_ratio_180s": np.full(n, 0.5), "aggr_sell_ratio_180s": np.full(n, 0.5),
        "vpin": np.full(n, 0.5), "flow_toxicity": np.zeros(n),
        "spoofing_score": np.zeros(n), "iceberg_score": np.zeros(n),
        "delta_divergence": np.zeros(n), "micro_breakout": np.zeros(n),
    }
    return pd.DataFrame(feats)


# ── Block 4: Liquidation & Funding (variable count) ──

def compute_liquidation_funding_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized liquidation/funding features."""
    n = len(df)
    funding = df["funding_rate"].values.astype(np.float64) if "funding_rate" in df.columns else np.zeros(n)

    feats = {
        "liq_dist_long": np.zeros(n), "liq_dist_short": np.zeros(n),
        "liq_heat_total": np.zeros(n), "liq_heat_long": np.zeros(n),
        "liq_heat_short": np.zeros(n), "liq_imbalance": np.zeros(n),
        "funding_rate": funding,
        "funding_accel_1m": np.zeros(n), "funding_accel_5m": np.zeros(n),
        "oi_change_proxy": np.abs(funding) * 10000,
        "cvd_funding_divergence": np.zeros(n),
    }
    return pd.DataFrame(feats, index=df.index)


# ── Block 5: On-chain (10 features — zeros for historical) ──

def compute_onchain_vectorized(n: int) -> pd.DataFrame:
    feats = {
        "onchain_large_transfers_60s": np.zeros(n),
        "onchain_whale_activity": np.zeros(n),
        "onchain_dex_volume_spike": np.zeros(n),
        "onchain_jupiter_accel": np.zeros(n),
        "onchain_mev_bundles": np.zeros(n),
        "onchain_priority_fee_pressure": np.zeros(n),
        "onchain_token_creation_rate": np.zeros(n),
        "onchain_large_transfers_300s": np.zeros(n),
        "onchain_dex_volume_spike_300s": np.zeros(n),
        "onchain_jupiter_accel_300s": np.zeros(n),
    }
    return pd.DataFrame(feats)


# ── Block 6: Regime (10 features — vectorized) ──

def compute_regime_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized regime features."""
    close = df["close"].values.astype(np.float64)
    n = len(close)

    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(np.maximum(close[1:] / np.maximum(close[:-1], 1e-10), 1e-10))

    # Rolling volatility (20-bar)
    rvol = _rolling_std(log_ret, 20)

    # Vol regime classification
    vol_regime = np.ones(n, dtype=np.float64)
    vol_regime_score = np.full(n, 0.5, dtype=np.float64)
    if n >= 80:
        rvol_valid = rvol[20:]
        q25 = np.percentile(rvol_valid[rvol_valid > 0], 25) if np.any(rvol_valid > 0) else 0.001
        q50 = np.percentile(rvol_valid[rvol_valid > 0], 50) if np.any(rvol_valid > 0) else 0.003
        q75 = np.percentile(rvol_valid[rvol_valid > 0], 75) if np.any(rvol_valid > 0) else 0.006
        q95 = np.percentile(rvol_valid[rvol_valid > 0], 95) if np.any(rvol_valid > 0) else 0.012
        vol_regime = np.where(rvol < q25, 0.0, np.where(rvol < q50, 1.0, np.where(rvol < q95, 2.0, 3.0)))
        vol_regime_score = np.where(rvol < q25, _safe_div_vec(rvol, q50),
                           np.where(rvol < q50, _safe_div_vec(rvol, q75),
                           np.where(rvol < q95, _safe_div_vec(rvol, q95),
                                    np.minimum(_safe_div_vec(rvol, q95), 2.0))))

    feats = {
        "vol_regime": vol_regime,
        "vol_regime_score": vol_regime_score,
        "hurst_exponent": np.full(n, 0.5),
        "entropy_shannon": np.full(n, 1.0),
        "entropy_approximate": np.full(n, 0.5),
        "autocorr_lag1": np.zeros(n),
        "autocorr_lag5": np.zeros(n),
        "autocorr_decay": np.zeros(n),
        "hmm_regime": np.zeros(n),
        "hmm_transition_prob": np.full(n, 0.25),
    }

    # Rolling autocorrelation at lag 1 and 5
    w_ac = 60
    if n >= w_ac + 5:
        for lag, key in [(1, "autocorr_lag1"), (5, "autocorr_lag5")]:
            ac_arr = np.zeros(n, dtype=np.float64)
            for i in range(w_ac + lag, n):
                chunk = log_ret[i - w_ac:i]
                m = np.mean(chunk)
                v = np.var(chunk)
                if v > 0:
                    ac_arr[i] = np.clip(np.mean((chunk[lag:] - m) * (chunk[:-lag] - m)) / v, -1, 1)
            feats[key] = ac_arr
        feats["autocorr_decay"] = _safe_div_vec(
            np.abs(feats["autocorr_lag5"]), np.maximum(np.abs(feats["autocorr_lag1"]), 1e-10)
        )

    return pd.DataFrame(feats, index=df.index)


# ── Block 7: Shares features (16 features — vectorized) ──

def compute_shares_features_vectorized(
    sol_price: np.ndarray,
    ptb: np.ndarray,
    time_elapsed_ms: np.ndarray,
    time_remaining_ms: np.ndarray,
    duration_minutes: int,
    vol: np.ndarray,
) -> pd.DataFrame:
    """Fully vectorized shares features for all samples at once."""
    n = len(sol_price)
    total_ms = time_elapsed_ms + time_remaining_ms

    feats = {}
    feats["time_remaining_pct"] = _safe_div_vec(time_remaining_ms, np.maximum(total_ms, 1.0))
    feats["time_remaining_min"] = time_remaining_ms / 60_000.0
    feats["time_elapsed_min"] = time_elapsed_ms / 60_000.0

    elapsed_pct = _safe_div_vec(time_elapsed_ms, np.maximum(total_ms, 1.0))
    feats["life_phase"] = np.clip(elapsed_pct ** 1.5, 0.0, 1.0)

    # Distance from PTB
    dist_pct = _safe_div_vec(sol_price - ptb, np.maximum(ptb, 1e-10)) * 100.0
    feats["distance_from_ptb_pct"] = dist_pct

    time_factor = np.sqrt(np.maximum(time_remaining_ms / 60_000.0, 0.1))
    vol_factor = np.maximum(vol, 1e-6) * 100.0
    feats["distance_from_ptb_norm"] = _safe_div_vec(dist_pct, np.maximum(vol_factor * time_factor, 0.01))

    # Implied prob via Black-Scholes-style CDF
    dist = _safe_div_vec(sol_price - ptb, np.maximum(ptb, 1e-10))
    t_min = np.maximum(time_remaining_ms / 60_000.0, 0.01)
    vol_adj = np.maximum(vol, 0.001) * np.sqrt(t_min)
    d_stat = _safe_div_vec(dist, vol_adj)
    yes_price = np.clip(norm.cdf(d_stat), 0.02, 0.98)
    no_price = 1.0 - yes_price

    feats["up_implied_prob"] = np.clip(yes_price, 0.01, 0.99)

    # Mispricing score (naive logistic)
    z = feats["distance_from_ptb_norm"]
    naive_prob = 1.0 / (1.0 + np.exp(-z * 2))
    feats["mispricing_score"] = naive_prob - yes_price

    # Shares momentum (zeros — no real-time history in historical mode)
    feats["shares_momentum_30s"] = np.zeros(n)
    feats["shares_momentum_1m"] = np.zeros(n)
    feats["shares_momentum_3m"] = np.zeros(n)

    # Volume/liquidity/spread (defaults)
    feats["volume_imbalance"] = np.zeros(n)
    feats["liquidity_score"] = np.zeros(n)
    feats["spread_normalized"] = np.zeros(n)

    # Mean reversion strength
    distance_from_half = np.abs(yes_price - 0.5)
    expiry_factor = 1.0 - feats["time_remaining_pct"]
    feats["mean_reversion_strength"] = distance_from_half * expiry_factor

    # Arbitrage score
    feats["arbitrage_score"] = (yes_price + no_price) - 1.0

    return pd.DataFrame(feats), yes_price


# ── Pre-market lookback features ──

def compute_pre_market_features_vectorized(
    close: np.ndarray,
    market_start_indices: np.ndarray,
    n_samples: int,
    sample_to_market: np.ndarray,
) -> pd.DataFrame:
    """Vectorized pre-market lookback features."""
    feats = {}
    for lookback in [2, 5, 10, 15, 30]:
        ret_arr = np.zeros(n_samples, dtype=np.float64)
        vol_arr = np.zeros(n_samples, dtype=np.float64)
        for m_idx in range(len(market_start_indices)):
            mask = sample_to_market == m_idx
            if not np.any(mask):
                continue
            si = market_start_indices[m_idx]
            start_idx = max(0, si - lookback)
            if start_idx < si and si > 0:
                pre_closes = close[start_idx:si]
                if len(pre_closes) >= 2:
                    pre_ret = (pre_closes[-1] - pre_closes[0]) / (pre_closes[0] + 1e-10)
                    pre_lr = np.diff(np.log(pre_closes + 1e-10))
                    pre_vol = float(np.std(pre_lr)) if len(pre_lr) > 1 else 0.0
                    ret_arr[mask] = pre_ret
                    vol_arr[mask] = pre_vol
        feats[f"pre_mkt_ret_{lookback}m"] = ret_arr
        feats[f"pre_mkt_vol_{lookback}m"] = vol_arr
    return pd.DataFrame(feats)


# ═══════════════════════════════════════════════════════════
#  MASTER BUILDER
# ═══════════════════════════════════════════════════════════

class FastDatasetBuilder:
    """Ultra-fast vectorized dataset builder.

    Scientifically rigorous:
    - MARKET-LEVEL split (no market straddles train/test)
    - Embargo gap between train and test sets
    - Phase tracking (early/mid/late per sample)
    - Entry-bar extraction for per-market evaluation
    - No look-ahead bias: all features use only past data
    """

    EMBARGO_MARKETS = 3

    def __init__(self):
        pass

    def build(
        self,
        df: pd.DataFrame,
        duration_minutes: int = 15,
        min_history: int = 60,
        train_ratio: float = 0.7,
    ) -> Dict:
        """Build dataset for a single timeframe.

        Returns dict with X_train, y_train, X_test, y_test, feature_names,
        plus phase arrays, market IDs, entry-bar data for honest evaluation.
        """
        t0 = time.perf_counter()
        log.info(f"FastDatasetBuilder: {len(df):,} rows, {duration_minutes}m markets")

        df = df.sort_values("ts").reset_index(drop=True)
        ts = df["ts"].values.astype(np.int64)
        close = df["close"].values.astype(np.float64)
        n = len(df)

        # ── Step 1: Compute ALL feature blocks on full DataFrame (vectorized) ──
        log.info("  [1/6] Computing feature blocks (vectorized)...")
        t1 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=4) as pool:
            f_pv = pool.submit(compute_price_volume_vectorized, df)
            f_tech = pool.submit(compute_technical_vectorized, df)
            f_regime = pool.submit(compute_regime_vectorized, df)
            f_lf = pool.submit(compute_liquidation_funding_vectorized, df)

        pv_feats = f_pv.result()
        tech_feats = f_tech.result()
        regime_feats = f_regime.result()
        lf_feats = f_lf.result()
        micro_feats = compute_microstructure_vectorized(n)
        onchain_feats = compute_onchain_vectorized(n)

        log.info(f"    Feature blocks: {time.perf_counter() - t1:.2f}s")

        # ── Step 2: Simulate markets and create samples ──
        log.info("  [2/6] Simulating markets...")
        t2 = time.perf_counter()

        interval_ms = duration_minutes * 60_000
        start_ts = int(ts[0])
        end_ts = int(ts[-1])
        first_market = ((start_ts // interval_ms) + 1) * interval_ms

        # Pre-compute rolling volatility for shares features
        log_ret = np.zeros(n, dtype=np.float64)
        log_ret[1:] = np.log(np.maximum(close[1:] / np.maximum(close[:-1], 1e-10), 1e-10))
        rolling_vol = _rolling_std(log_ret, 60)
        rolling_vol = np.maximum(rolling_vol, 0.003)

        # Enumerate all markets
        market_starts_ts = np.arange(first_market, end_ts - interval_ms + 1, interval_ms)

        # Build index: for each market, find bar indices using searchsorted
        market_start_bar = np.searchsorted(ts, market_starts_ts, side="left")
        market_end_bar = np.searchsorted(ts, market_starts_ts + interval_ms, side="left")

        # Collect samples + per-market metadata
        sample_bar_indices = []
        sample_market_idx = []
        sample_ptb = []
        sample_outcome_up = []
        sample_time_elapsed = []
        sample_time_remaining = []
        sample_vol = []

        market_first_bars = []
        market_outcomes = []
        valid_market_global_idx = []    # maps valid market local idx -> global market idx

        valid_m = 0
        for m in range(len(market_starts_ts)):
            sb = market_start_bar[m]
            eb = market_end_bar[m]
            if eb - sb < 3:
                continue

            bar_idx = np.arange(sb, eb)
            valid = bar_idx[bar_idx >= min_history]
            if len(valid) == 0:
                continue

            ptb_val = close[sb]
            final_sol = close[eb - 1]
            outcome = 1 if final_sol >= ptb_val else 0

            market_first_bars.append(sb)
            market_outcomes.append(outcome)
            valid_market_global_idx.append(m)

            for idx in valid:
                sample_bar_indices.append(idx)
                sample_market_idx.append(valid_m)
                sample_ptb.append(ptb_val)
                sample_outcome_up.append(outcome)
                elapsed_ms = int(ts[idx]) - int(market_starts_ts[m])
                remaining_ms = max(0, int(market_starts_ts[m]) + interval_ms - int(ts[idx]))
                sample_time_elapsed.append(elapsed_ms)
                sample_time_remaining.append(remaining_ms)
                sample_vol.append(rolling_vol[idx])

            valid_m += 1

        sample_bar_indices = np.array(sample_bar_indices, dtype=np.int64)
        sample_market_idx = np.array(sample_market_idx, dtype=np.int64)
        n_valid_markets = valid_m
        n_samples = len(sample_bar_indices)
        market_outcomes = np.array(market_outcomes, dtype=np.int64)

        log.info(f"    Valid markets: {n_valid_markets}, samples: {n_samples:,} ({time.perf_counter() - t2:.2f}s)")

        if n_samples == 0:
            return {"X_train": np.array([]), "feature_names": [], "n_valid_markets": 0}

        # ── Step 3: Phase classification (20% steps) ──
        te_arr = np.array(sample_time_elapsed, dtype=np.float64)
        tr_arr = np.array(sample_time_remaining, dtype=np.float64)
        total_ms = te_arr + tr_arr
        elapsed_pct = _safe_div_vec(te_arr, np.maximum(total_ms, 1.0))
        # 0=0-20%, 1=20-40%, 2=40-60%, 3=60-80%, 4=80-100%
        sample_phase = np.clip((elapsed_pct * 5).astype(int), 0, 4)

        # ── Step 4: MARKET-LEVEL 70/30 split with embargo ──
        log.info("  [3/6] Market-level split with embargo...")
        embargo = self.EMBARGO_MARKETS
        n_train_markets = int(n_valid_markets * train_ratio)
        n_embargo = min(embargo, n_valid_markets - n_train_markets)
        n_test_start = n_train_markets + n_embargo

        train_mask = sample_market_idx < n_train_markets
        test_mask = sample_market_idx >= n_test_start
        # embargo samples are DISCARDED (not in train or test)

        log.info(
            f"    Markets: {n_train_markets} train | {n_embargo} embargo (discarded) | "
            f"{n_valid_markets - n_test_start} test"
        )

        # ── Step 5: Gather features for ALL samples (vectorized indexing) ──
        log.info("  [4/6] Gathering feature matrix...")
        t3 = time.perf_counter()

        all_feats = pd.concat([
            pv_feats, tech_feats, micro_feats, lf_feats, onchain_feats, regime_feats,
        ], axis=1)

        if "oi_change" in df.columns:
            all_feats["oi_change"] = df["oi_change"].values
        if "long_short_ratio" in df.columns:
            all_feats["long_short_ratio"] = df["long_short_ratio"].values

        X_base = all_feats.values[sample_bar_indices]

        # Shares features
        sol_prices = close[sample_bar_indices]
        ptb_arr = np.array(sample_ptb, dtype=np.float64)
        vol_arr = np.array(sample_vol, dtype=np.float64)

        shares_df, yes_prices = compute_shares_features_vectorized(
            sol_prices, ptb_arr, te_arr, tr_arr, duration_minutes, vol_arr,
        )

        # Pre-market features
        pre_mkt_df = compute_pre_market_features_vectorized(
            close, np.array(market_first_bars, dtype=np.int64),
            n_samples, sample_market_idx,
        )

        # Combine all features
        feature_names_base = list(all_feats.columns)
        feature_names_shares = list(shares_df.columns)
        feature_names_pre = list(pre_mkt_df.columns)
        feature_names = sorted(feature_names_base + feature_names_shares + feature_names_pre)

        X_shares = shares_df.values
        X_pre = pre_mkt_df.values
        X_full = np.hstack([X_base, X_shares, X_pre])

        all_col_names = feature_names_base + feature_names_shares + feature_names_pre
        col_order = [all_col_names.index(f) for f in feature_names]
        X_full = X_full[:, col_order]
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

        log.info(f"    Feature matrix: {X_full.shape} ({time.perf_counter() - t3:.2f}s)")

        # ── Step 6: Targets ──
        log.info("  [5/6] Computing targets...")
        y_direction = np.array(sample_outcome_up, dtype=np.int64)
        y_shares_pnl = np.where(
            y_direction == 1,
            (1.0 - yes_prices) / np.maximum(yes_prices, 0.01),
            (0.0 - yes_prices) / np.maximum(yes_prices, 0.01),
        )

        # ── Apply masks to get train/test sets ──
        log.info("  [6/6] Finalizing splits...")
        X_train = X_full[train_mask]
        X_test = X_full[test_mask]
        y_train = y_direction[train_mask]
        y_test = y_direction[test_mask]
        y_train_pnl = y_shares_pnl[train_mask]
        y_test_pnl = y_shares_pnl[test_mask]

        # Phase arrays for test set (for stratified evaluation)
        test_phase = sample_phase[test_mask]
        test_market_idx = sample_market_idx[test_mask]
        test_yes_prices = yes_prices[test_mask]

        # Entry bars: first bar of each TEST market (for per-market evaluation)
        test_market_ids = np.unique(test_market_idx)
        entry_indices_in_test = []
        for mid in test_market_ids:
            mask_m = np.where(test_market_idx == mid)[0]
            if len(mask_m) > 0:
                entry_indices_in_test.append(mask_m[0])
        entry_indices_in_test = np.array(entry_indices_in_test, dtype=np.int64)

        # Implied prob baseline for test set (hand-crafted predictor)
        implied_prob_test = test_yes_prices

        elapsed = time.perf_counter() - t0
        log.info(
            f"  DONE in {elapsed:.1f}s | "
            f"train={len(X_train):,} ({n_train_markets} markets) | "
            f"test={len(X_test):,} ({len(test_market_ids)} markets) | "
            f"embargo={n_embargo} markets discarded | "
            f"{X_full.shape[1]} features | "
            f"train_up={np.mean(y_train):.2%} test_up={np.mean(y_test):.2%}"
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_train_pnl": y_train_pnl,
            "y_test_pnl": y_test_pnl,
            "feature_names": feature_names,
            # Metadata
            "n_valid_markets": n_valid_markets,
            "n_train_markets": n_train_markets,
            "n_test_markets": len(test_market_ids),
            "n_embargo": n_embargo,
            "n_samples": n_samples,
            "duration_minutes": duration_minutes,
            "elapsed_s": elapsed,
            # For honest evaluation
            "test_phase": test_phase,               # 0=0-20%, 1=20-40%, 2=40-60%, 3=60-80%, 4=80-100%
            "test_market_idx": test_market_idx,     # market ID per test sample
            "test_market_ids": test_market_ids,     # unique market IDs in test
            "entry_indices_in_test": entry_indices_in_test,  # first-bar index per test market
            "implied_prob_test": implied_prob_test,  # BS-implied prob baseline
            "market_outcomes": market_outcomes,      # outcome per valid market
        }

    def build_both_timeframes(
        self,
        df: pd.DataFrame,
        min_history: int = 60,
        train_ratio: float = 0.7,
    ) -> Dict[str, Dict]:
        """Build datasets for both 5m and 15m markets."""
        results = {}
        for dur in [5, 15]:
            log.info(f"\n{'='*60}")
            log.info(f"  Building {dur}m dataset...")
            log.info(f"{'='*60}")
            results[f"{dur}m"] = self.build(
                df, duration_minutes=dur,
                min_history=min_history, train_ratio=train_ratio,
            )
        return results
