"""Dataset construction — transforms processed OHLCV data into ML-ready feature matrices.

Creates:
  - Feature matrix X (n_samples, n_features) — CEX features + Shares features
  - Sequence data X_seq (n_samples, seq_len, n_features) for Transformer
  - Multi-task targets:
    * y_direction: SOL direction (1=up, 0=down)
    * y_return: log return of SOL
    * y_shares_change_1m/3m/5m: predicted UP shares price change
    * y_early_exit_prob: probability of profitable early exit
    * y_optimal_exit_min: optimal exit time in minutes
  - Purged K-Fold splits with embargo

Handles both historical (parquet) and live (collector) data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from pathlib import Path

from core.features.price_volume import PriceVolumeFeatures
from core.features.technical import TechnicalFeatures
from core.features.microstructure import MicrostructureFeatures
from core.features.liquidation_funding import LiquidationFundingFeatures
from core.features.regime import RegimeFeatures
from core.features.shares import compute_shares_features, SHARES_FEATURE_NAMES
from core.utils.logger import log
from core.utils.helpers import safe_div
from config import config


class TrainingDataset:
    """Constructs ML-ready datasets from processed market data."""

    def __init__(self):
        self.pv = PriceVolumeFeatures()
        self.tech = TechnicalFeatures()
        self.micro = MicrostructureFeatures()
        self.liq = LiquidationFundingFeatures()
        self.regime = RegimeFeatures()

        self._seq_len = config.get("models", {}).get("primary", {}).get("sequence_length", 60)
        self._feature_names: Optional[List[str]] = None

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        forward_minutes: int = 5,
        min_history: int = 60,
    ) -> Dict[str, np.ndarray]:
        """Build feature matrix + targets from processed OHLCV DataFrame.

        Args:
            df: DataFrame with ts, open, high, low, close, volume, + extras
            forward_minutes: lookahead for target computation
            min_history: minimum bars of history before generating features

        Returns:
            Dict with keys: X, X_seq, y_direction, y_return, y_reversal, y_hold_time, timestamps
        """
        log.info(f"Building dataset: {len(df)} rows, forward={forward_minutes}m")

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by timestamp
        if "ts" in df.columns:
            df = df.sort_values("ts").reset_index(drop=True)

        n = len(df)
        feature_rows = []
        targets_dir = []
        targets_ret = []
        targets_rev = []
        targets_hold = []
        timestamps = []

        log.info(f"  Computing features for {n - min_history - forward_minutes} samples...")

        for i in range(min_history, n - forward_minutes):
            # Build OHLCV window
            window = df.iloc[max(0, i - 60):i + 1]
            ohlcv = {
                "open": window["open"].values.astype(np.float64),
                "high": window["high"].values.astype(np.float64),
                "low": window["low"].values.astype(np.float64),
                "close": window["close"].values.astype(np.float64),
                "volume": window["volume"].values.astype(np.float64),
                "taker_buy_volume": window.get("taker_buy_volume", window["volume"] * 0.5).values.astype(np.float64),
            }
            current_price = float(df.iloc[i]["close"])

            # ── Compute features ──
            features = {}

            # Block 1: Price & Volume
            features.update(self.pv.compute(ohlcv_1m=ohlcv, current_price=current_price))

            # Block 2: Technical
            features.update(self.tech.compute(ohlcv=ohlcv, current_price=current_price))

            # Block 3: Microstructure (limited in historical mode)
            micro_feats = self.micro.compute(current_price=current_price)
            features.update(micro_feats)

            # Block 4: Liquidation & Funding
            funding_rate = float(df.iloc[i].get("funding_rate", 0) if "funding_rate" in df.columns else 0)
            liq_feats = self.liq.compute(
                funding_rate=funding_rate,
                current_price=current_price,
            )
            features.update(liq_feats)

            # Block 5: On-chain defaults (not available historically)
            for k in [
                "onchain_large_transfers_60s", "onchain_whale_activity",
                "onchain_dex_volume_spike", "onchain_jupiter_accel",
                "onchain_mev_bundles", "onchain_priority_fee_pressure",
                "onchain_token_creation_rate", "onchain_large_transfers_300s",
                "onchain_dex_volume_spike_300s", "onchain_jupiter_accel_300s",
            ]:
                features[k] = 0.0

            # Block 6: Regime
            close_arr = ohlcv["close"]
            returns = np.diff(np.log(close_arr[close_arr > 0])) if len(close_arr) > 10 else None
            regime_feats = self.regime.compute(returns=returns, close_prices=close_arr)
            features.update(regime_feats)

            # Extra derived features from processed data
            if "oi_change" in df.columns:
                features["oi_change"] = float(df.iloc[i]["oi_change"])
            if "long_short_ratio" in df.columns:
                features["long_short_ratio"] = float(df.iloc[i]["long_short_ratio"])
            if "funding_rate_change" in df.columns:
                features["funding_rate_change_ext"] = float(df.iloc[i]["funding_rate_change"])

            # Stabilize feature names
            if self._feature_names is None:
                self._feature_names = sorted(features.keys())

            fv = np.array([features.get(k, 0.0) for k in self._feature_names], dtype=np.float64)
            # Replace inf/nan
            fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
            feature_rows.append(fv)

            # ── Compute targets ──
            future = df.iloc[i + 1:i + 1 + forward_minutes]
            future_close = future["close"].values
            if len(future_close) < forward_minutes:
                targets_dir.append(0)
                targets_ret.append(0.0)
                targets_rev.append(0.0)
                targets_hold.append(forward_minutes * 60)
            else:
                # Direction: 1 = up, 0 = down
                final_price = future_close[-1]
                direction = 1 if final_price > current_price else 0
                targets_dir.append(direction)

                # Expected return (log return)
                expected_ret = np.log(final_price / current_price) if current_price > 0 else 0
                targets_ret.append(float(expected_ret))

                # Reversal probability: did price reverse significantly within window?
                max_adverse = np.min(future_close) / current_price - 1 if direction == 1 else 1 - np.max(future_close) / current_price
                reversal = 1.0 if max_adverse < -0.005 else 0.0  # >0.5% adverse move
                targets_rev.append(reversal)

                # Optimal hold time: when was max favorable price reached
                if direction == 1:
                    best_idx = np.argmax(future_close)
                else:
                    best_idx = np.argmin(future_close)
                optimal_hold = (best_idx + 1) * 60  # seconds
                targets_hold.append(float(optimal_hold))

            timestamps.append(int(df.iloc[i].get("ts", i)))

            if (i - min_history) % 5000 == 0 and i > min_history:
                log.info(f"  {i - min_history}/{n - min_history - forward_minutes} samples computed")

        X = np.array(feature_rows, dtype=np.float64)

        # Build sequence data for Transformer
        X_seq = self._build_sequences(X, self._seq_len)

        result = {
            "X": X,
            "X_seq": X_seq,
            "y_direction": np.array(targets_dir, dtype=np.int64),
            "y_return": np.array(targets_ret, dtype=np.float64),
            "y_reversal": np.array(targets_rev, dtype=np.float64),
            "y_hold_time": np.array(targets_hold, dtype=np.float64),
            "timestamps": np.array(timestamps, dtype=np.int64),
            "feature_names": self._feature_names,
        }

        log.info(
            f"  ✅ Dataset: X={X.shape}, X_seq={X_seq.shape}, "
            f"dir_balance={np.mean(targets_dir):.2%} up, "
            f"avg_return={np.mean(targets_ret):.5f}"
        )

        return result

    def _build_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray:
        """Build overlapping sequences for Transformer input."""
        n, d = X.shape
        if n < seq_len:
            # Pad with zeros
            padded = np.zeros((seq_len, d))
            padded[-n:] = X
            return padded.reshape(1, seq_len, d)

        sequences = []
        for i in range(n):
            start = max(0, i - seq_len + 1)
            seq = X[start:i + 1]
            if len(seq) < seq_len:
                pad = np.zeros((seq_len - len(seq), d))
                seq = np.vstack([pad, seq])
            sequences.append(seq)

        return np.array(sequences, dtype=np.float64)

    def build_shares_dataset(
        self,
        sol_data: pd.DataFrame,
        duration_minutes: int = 15,
        min_history: int = 60,
    ) -> Dict[str, np.ndarray]:
        """Build dataset for shares prediction — simulates markets from SOL data.

        For each simulated market window:
          - Computes CEX features (82) + shares features (16) = 98 features
          - At each bar, creates a sample with targets:
            * y_direction: will SOL be above PTB at expiry? (1/0)
            * y_shares_pnl: profit from buying UP shares at current price and holding to expiry
            * y_early_exit_ok: could we exit profitably within next 3 bars? (1/0)
            * y_optimal_exit_min: minutes to best exit point

        Args:
            sol_data: DataFrame with ts, open, high, low, close, volume
            duration_minutes: market duration (5, 15, 60)
            min_history: bars needed before feature computation

        Returns:
            Dict with X, y_direction, y_shares_pnl, y_early_exit_ok, y_optimal_exit_min, feature_names
        """
        from scipy.stats import norm

        log.info(f"Building shares dataset: {len(sol_data)} rows, {duration_minutes}m markets")

        sol_data = sol_data.sort_values("ts").reset_index(drop=True)
        ts_col = sol_data["ts"].values
        interval_ms = duration_minutes * 60_000

        # Align markets
        start_ts = int(ts_col[0])
        end_ts = int(ts_col[-1])
        first_market = (start_ts // interval_ms + 1) * interval_ms

        feature_rows = []
        t_direction = []
        t_shares_pnl = []
        t_early_exit = []
        t_opt_exit = []
        timestamps = []

        market_count = 0
        current_start = first_market

        while current_start + interval_ms <= end_ts:
            market_end = current_start + interval_ms
            mask = (ts_col >= current_start) & (ts_col < market_end)
            bar_indices = np.where(mask)[0]

            if len(bar_indices) < 3:
                current_start += interval_ms
                continue

            ptb = float(sol_data.iloc[bar_indices[0]]["close"])
            final_sol = float(sol_data.iloc[bar_indices[-1]]["close"])
            outcome_up = final_sol >= ptb

            # Volatility for this window
            closes = sol_data.iloc[bar_indices]["close"].values
            log_rets = np.diff(np.log(closes + 1e-10))
            vol = float(np.std(log_rets)) if len(log_rets) > 1 else 0.003

            # ── Pre-market lookback features (computed once per market) ──
            first_bar_idx = bar_indices[0]
            pre_market_feats = {}
            for lookback in [2, 5, 10, 15, 30]:
                start_idx = max(0, first_bar_idx - lookback)
                if start_idx < first_bar_idx and first_bar_idx > 0:
                    pre_closes = sol_data.iloc[start_idx:first_bar_idx]["close"].values
                    if len(pre_closes) >= 2:
                        pre_ret = (pre_closes[-1] - pre_closes[0]) / (pre_closes[0] + 1e-10)
                        pre_log_rets = np.diff(np.log(pre_closes + 1e-10))
                        pre_vol = float(np.std(pre_log_rets)) if len(pre_log_rets) > 1 else 0.0
                        pre_market_feats[f"pre_mkt_ret_{lookback}m"] = float(pre_ret)
                        pre_market_feats[f"pre_mkt_vol_{lookback}m"] = pre_vol
                    else:
                        pre_market_feats[f"pre_mkt_ret_{lookback}m"] = 0.0
                        pre_market_feats[f"pre_mkt_vol_{lookback}m"] = 0.0
                else:
                    pre_market_feats[f"pre_mkt_ret_{lookback}m"] = 0.0
                    pre_market_feats[f"pre_mkt_vol_{lookback}m"] = 0.0

            for j, idx in enumerate(bar_indices):
                # Need some history for CEX features
                if idx < min_history:
                    continue

                row = sol_data.iloc[idx]
                sol_price = float(row["close"])
                time_elapsed_ms = int(ts_col[idx]) - current_start
                time_remaining_ms = max(0, market_end - int(ts_col[idx]))
                total_ms = time_elapsed_ms + time_remaining_ms

                # ── CEX features ──
                window = sol_data.iloc[max(0, idx - 60):idx + 1]
                ohlcv = {
                    "open": window["open"].values.astype(np.float64),
                    "high": window["high"].values.astype(np.float64),
                    "low": window["low"].values.astype(np.float64),
                    "close": window["close"].values.astype(np.float64),
                    "volume": window["volume"].values.astype(np.float64),
                    "taker_buy_volume": window.get(
                        "taker_buy_volume", window["volume"] * 0.5
                    ).values.astype(np.float64),
                }

                features = {}
                features.update(self.pv.compute(ohlcv_1m=ohlcv, current_price=sol_price))
                features.update(self.tech.compute(ohlcv=ohlcv, current_price=sol_price))
                features.update(self.micro.compute(current_price=sol_price))

                funding = float(row.get("funding_rate", 0) if "funding_rate" in sol_data.columns else 0)
                features.update(self.liq.compute(funding_rate=funding, current_price=sol_price))

                for k in [
                    "onchain_large_transfers_60s", "onchain_whale_activity",
                    "onchain_dex_volume_spike", "onchain_jupiter_accel",
                    "onchain_mev_bundles", "onchain_priority_fee_pressure",
                    "onchain_token_creation_rate", "onchain_large_transfers_300s",
                    "onchain_dex_volume_spike_300s", "onchain_jupiter_accel_300s",
                ]:
                    features[k] = 0.0

                close_arr = ohlcv["close"]
                rets = np.diff(np.log(close_arr[close_arr > 0])) if len(close_arr) > 10 else None
                features.update(self.regime.compute(returns=rets, close_prices=close_arr))

                # ── Shares features ──
                # Estimate current shares prices
                dist = (sol_price - ptb) / ptb if ptb > 0 else 0
                t_min = max(time_remaining_ms / 60_000, 0.01)
                vol_adj = max(vol, 0.001) * np.sqrt(t_min)
                d = dist / vol_adj
                yes_price = float(np.clip(norm.cdf(d), 0.02, 0.98))
                no_price = 1.0 - yes_price

                shares_feats = compute_shares_features(
                    sol_price=sol_price,
                    price_to_beat=ptb,
                    yes_price=yes_price,
                    no_price=no_price,
                    time_remaining_ms=time_remaining_ms,
                    time_elapsed_ms=time_elapsed_ms,
                    duration_minutes=duration_minutes,
                    sol_volatility=vol,
                )
                features.update(shares_feats)

                # ── Pre-market lookback (per-market, same for all bars) ──
                features.update(pre_market_feats)

                if "oi_change" in sol_data.columns:
                    features["oi_change"] = float(row.get("oi_change", 0))
                if "long_short_ratio" in sol_data.columns:
                    features["long_short_ratio"] = float(row.get("long_short_ratio", 0))

                # Stabilize feature names
                if self._feature_names is None:
                    self._feature_names = sorted(features.keys())

                fv = np.array([features.get(k, 0.0) for k in self._feature_names], dtype=np.float64)
                fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
                feature_rows.append(fv)

                # ── Targets ──
                # 1. Direction at expiry
                t_direction.append(1 if outcome_up else 0)

                # 2. Shares PnL if buy UP now and hold to expiry
                if outcome_up:
                    shares_pnl = (1.0 - yes_price) / max(yes_price, 0.01)
                else:
                    shares_pnl = (0.0 - yes_price) / max(yes_price, 0.01)
                t_shares_pnl.append(float(shares_pnl))

                # 3. Early exit: check if shares price improves in next 3 bars
                future_indices = bar_indices[j + 1:j + 4]
                early_profit = False
                best_exit_idx = 0
                best_exit_pnl = 0.0
                for fi_idx, fi in enumerate(future_indices):
                    future_sol = float(sol_data.iloc[fi]["close"])
                    future_dist = (future_sol - ptb) / ptb if ptb > 0 else 0
                    future_t_ms = max(0, market_end - int(ts_col[fi]))
                    future_t_min = max(future_t_ms / 60_000, 0.01)
                    future_vol_adj = max(vol, 0.001) * np.sqrt(future_t_min)
                    future_d = future_dist / future_vol_adj
                    future_yes = float(np.clip(norm.cdf(future_d), 0.02, 0.98))
                    exit_pnl = (future_yes - yes_price) / max(yes_price, 0.01)
                    if exit_pnl > 0.02:  # > 2% profit
                        early_profit = True
                    if exit_pnl > best_exit_pnl:
                        best_exit_pnl = exit_pnl
                        best_exit_idx = fi_idx + 1

                t_early_exit.append(1 if early_profit else 0)

                # 4. Optimal exit: bars to best exit point
                remaining_bars = bar_indices[j + 1:]
                if len(remaining_bars) > 0:
                    best_exit_min = best_exit_idx  # minutes (1m bars)
                else:
                    best_exit_min = duration_minutes
                t_opt_exit.append(float(best_exit_min))

                timestamps.append(int(ts_col[idx]))

            market_count += 1
            if market_count % 100 == 0:
                log.info(f"  {market_count} markets processed, {len(feature_rows)} samples")

            current_start += interval_ms

        if not feature_rows:
            log.warning("No samples generated")
            return {"X": np.array([]), "feature_names": []}

        X = np.array(feature_rows, dtype=np.float64)
        X_seq = self._build_sequences(X, self._seq_len)

        result = {
            "X": X,
            "X_seq": X_seq,
            "y_direction": np.array(t_direction, dtype=np.int64),
            "y_shares_pnl": np.array(t_shares_pnl, dtype=np.float64),
            "y_early_exit_ok": np.array(t_early_exit, dtype=np.int64),
            "y_optimal_exit_min": np.array(t_opt_exit, dtype=np.float64),
            # Keep legacy targets for compatibility
            "y_return": np.array(t_shares_pnl, dtype=np.float64),
            "y_reversal": np.array(t_early_exit, dtype=np.float64),
            "y_hold_time": np.array(t_opt_exit, dtype=np.float64) * 60,
            "timestamps": np.array(timestamps, dtype=np.int64),
            "feature_names": self._feature_names,
        }

        log.info(
            f"  ✅ Shares Dataset: X={X.shape}, {market_count} markets, "
            f"dir_balance={np.mean(t_direction):.2%} up, "
            f"avg_shares_pnl={np.mean(t_shares_pnl):.3f}"
        )

        return result

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names or []

    @property
    def n_features(self) -> int:
        return len(self._feature_names) if self._feature_names else 0


class PurgedKFold:
    """Purged K-Fold cross-validation with embargo.

    Prevents data leakage in time-series by:
    1. Purging: removing samples from train that overlap with test labels
    2. Embargo: adding a gap between train and test sets
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self, X: np.ndarray, timestamps: np.ndarray = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged K-Fold indices."""
        n = len(X)
        embargo_size = max(1, int(n * self.embargo_pct))
        fold_size = n // self.n_splits
        splits = []

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, n)

            # Train indices: everything outside test + embargo
            train_end = max(0, test_start - embargo_size)
            train_start_after = min(n, test_end + embargo_size)

            train_idx = np.concatenate([
                np.arange(0, train_end),
                np.arange(train_start_after, n),
            ]).astype(int)

            test_idx = np.arange(test_start, test_end).astype(int)

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits


class WalkForwardSplit:
    """Walk-forward optimization splits.

    Rolling window: train on [t-window, t], test on [t, t+step].
    """

    def __init__(
        self,
        train_size: int = 5000,
        test_size: int = 1000,
        step: int = 500,
        min_train: int = 1000,
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.min_train = min(min_train, train_size)  # can't require more than train_size

    def split(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        splits = []
        train_start = 0

        while True:
            train_end = train_start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size

            if test_end > n:
                break

            if train_end - train_start < self.min_train:
                train_start += self.step
                continue

            train_idx = np.arange(train_start, train_end).astype(int)
            test_idx = np.arange(test_start, test_end).astype(int)
            splits.append((train_idx, test_idx))

            train_start += self.step

        log.info(f"WalkForward: {len(splits)} folds, train={self.train_size}, test={self.test_size}, step={self.step}")
        return splits
