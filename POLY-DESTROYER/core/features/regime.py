"""Block 6: Regime & Statistical Features (10 features).

- Hidden Markov Model regimes (4-6 states)
- Volatility regime (low/med/high/extreme)
- Hurst exponent
- Entropy (Shannon + Approximate)
- Autocorrelation decay
- Regime transition probability
"""

import numpy as np
import math
from typing import Dict, Optional

from core.utils.helpers import safe_div


class RegimeFeatures:
    """Regime detection, statistical properties, and market state classification."""

    def __init__(self):
        self._hmm_model = None
        self._hmm_fitted = False
        self._n_states = 4
        self._min_samples_hmm = 100

        # Regime labels
        self.REGIMES = ["low_vol", "trending", "mean_revert", "extreme"]

    def compute(
        self,
        returns: Optional[np.ndarray] = None,
        volatility: float = 0.0,
        close_prices: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute all 10 regime & statistical features."""
        features = {}

        if returns is None or len(returns) < 20:
            return self._defaults()

        # === Volatility Regime Classification ===
        vol_regime, vol_regime_score = self._classify_vol_regime(returns)
        features["vol_regime"] = vol_regime  # 0=low, 1=med, 2=high, 3=extreme
        features["vol_regime_score"] = vol_regime_score

        # === Hurst Exponent ===
        features["hurst_exponent"] = self._compute_hurst(returns)

        # === Shannon Entropy ===
        features["entropy_shannon"] = self._shannon_entropy(returns)

        # === Approximate Entropy ===
        features["entropy_approximate"] = self._approximate_entropy(returns)

        # === Autocorrelation Decay ===
        features["autocorr_lag1"] = self._autocorrelation(returns, 1)
        features["autocorr_lag5"] = self._autocorrelation(returns, 5)
        features["autocorr_decay"] = self._autocorrelation_decay(returns)

        # === HMM Regime ===
        hmm_state, transition_prob = self._hmm_regime(returns)
        features["hmm_regime"] = hmm_state
        features["hmm_transition_prob"] = transition_prob

        return features

    def _defaults(self) -> Dict[str, float]:
        return {
            "vol_regime": 1.0, "vol_regime_score": 0.5,
            "hurst_exponent": 0.5, "entropy_shannon": 1.0,
            "entropy_approximate": 0.5, "autocorr_lag1": 0.0,
            "autocorr_lag5": 0.0, "autocorr_decay": 0.0,
            "hmm_regime": 0.0, "hmm_transition_prob": 0.25,
        }

    def _classify_vol_regime(self, returns: np.ndarray) -> tuple:
        """Classify volatility regime into 4 states."""
        vol = np.std(returns)
        # Use rolling quantiles for adaptive thresholds
        if len(returns) >= 60:
            vol_history = np.array([
                np.std(returns[max(0, i - 20):i])
                for i in range(20, len(returns))
            ])
            q25 = np.percentile(vol_history, 25)
            q50 = np.percentile(vol_history, 50)
            q75 = np.percentile(vol_history, 75)
            q95 = np.percentile(vol_history, 95)
        else:
            q25, q50, q75, q95 = 0.001, 0.003, 0.006, 0.012

        if vol < q25:
            return 0.0, vol / q50 if q50 > 0 else 0.0  # low
        elif vol < q50:
            return 1.0, vol / q75 if q75 > 0 else 0.0  # medium
        elif vol < q95:
            return 2.0, vol / q95 if q95 > 0 else 0.0  # high
        else:
            return 3.0, min(vol / q95, 2.0) if q95 > 0 else 1.0  # extreme

    def _compute_hurst(self, series: np.ndarray, max_lag: int = 20) -> float:
        """Hurst exponent via R/S analysis.
        H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending.
        """
        n = len(series)
        if n < max_lag * 2:
            return 0.5

        try:
            lags = range(2, min(max_lag + 1, n // 2))
            rs_values = []

            for lag in lags:
                chunks = [series[i:i + lag] for i in range(0, n - lag, lag)]
                rs_per_chunk = []
                for chunk in chunks:
                    if len(chunk) < 2:
                        continue
                    mean = np.mean(chunk)
                    deviations = np.cumsum(chunk - mean)
                    R = np.max(deviations) - np.min(deviations)
                    S = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 0 else 1e-10
                    rs_per_chunk.append(R / S)

                if rs_per_chunk:
                    rs_values.append((lag, np.mean(rs_per_chunk)))

            if len(rs_values) < 3:
                return 0.5

            log_lags = np.log([r[0] for r in rs_values])
            log_rs = np.log([r[1] for r in rs_values])

            coeffs = np.polyfit(log_lags, log_rs, 1)
            hurst = float(np.clip(coeffs[0], 0.0, 1.0))
            return hurst

        except Exception:
            return 0.5

    def _shannon_entropy(self, returns: np.ndarray, bins: int = 20) -> float:
        """Shannon entropy of return distribution."""
        if len(returns) < 10:
            return 1.0

        try:
            hist, _ = np.histogram(returns, bins=bins, density=True)
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(bins)
            return float(entropy / max_entropy) if max_entropy > 0 else 0.0
        except Exception:
            return 1.0

    def _approximate_entropy(
        self, series: np.ndarray, m: int = 2, r_mult: float = 0.2
    ) -> float:
        """Approximate entropy (ApEn) — measures regularity/predictability."""
        n = len(series)
        if n < 50:
            return 0.5

        try:
            # Subsample for speed
            if n > 200:
                series = series[-200:]
                n = 200

            r = r_mult * np.std(series)
            if r == 0:
                return 0.0

            def _count_matches(template_length):
                templates = np.array([
                    series[i:i + template_length]
                    for i in range(n - template_length + 1)
                ])
                count = 0
                total = len(templates)
                for i in range(total):
                    diffs = np.max(np.abs(templates - templates[i]), axis=1)
                    count += np.sum(diffs <= r) - 1  # exclude self
                return count / (total * (total - 1)) if total > 1 else 0

            phi_m = np.log(_count_matches(m)) if _count_matches(m) > 0 else 0
            phi_m1 = np.log(_count_matches(m + 1)) if _count_matches(m + 1) > 0 else 0

            return float(abs(phi_m - phi_m1))
        except Exception:
            return 0.5

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Autocorrelation at specific lag."""
        n = len(series)
        if n <= lag + 1:
            return 0.0

        try:
            mean = np.mean(series)
            var = np.var(series)
            if var == 0:
                return 0.0
            ac = np.mean((series[lag:] - mean) * (series[:-lag] - mean)) / var
            return float(np.clip(ac, -1, 1))
        except Exception:
            return 0.0

    def _autocorrelation_decay(self, series: np.ndarray, max_lag: int = 10) -> float:
        """Rate at which autocorrelation decays — fast decay = random, slow = trending."""
        if len(series) < max_lag + 5:
            return 0.0

        acs = [abs(self._autocorrelation(series, lag)) for lag in range(1, max_lag + 1)]
        if not acs or acs[0] == 0:
            return 0.0

        # Fit exponential decay: ac(lag) ≈ a * exp(-b * lag)
        # Simple: ratio of ac[5] / ac[1]
        if len(acs) >= 5:
            return float(safe_div(acs[4], acs[0]))
        return float(safe_div(acs[-1], acs[0]))

    def _hmm_regime(self, returns: np.ndarray) -> tuple:
        """HMM regime classification using hmmlearn."""
        if len(returns) < self._min_samples_hmm:
            return 0.0, 0.25

        try:
            if not self._hmm_fitted:
                from hmmlearn.hmm import GaussianHMM

                self._hmm_model = GaussianHMM(
                    n_components=self._n_states,
                    covariance_type="full",
                    n_iter=50,
                    random_state=42,
                )
                X = returns.reshape(-1, 1)
                self._hmm_model.fit(X)
                self._hmm_fitted = True

            X = returns[-50:].reshape(-1, 1)
            states = self._hmm_model.predict(X)
            current_state = float(states[-1])

            # Transition probability from current state
            transmat = self._hmm_model.transmat_
            state_idx = int(states[-1])
            stay_prob = float(transmat[state_idx, state_idx])
            transition_prob = 1.0 - stay_prob

            return current_state, transition_prob

        except Exception:
            return 0.0, 0.25
