"""Block 2: Advanced Technical & Fractal Features (14 features).

- EMA/SMA ribbon (8 lines)
- SuperTrend + ATR trailing
- ICT Concepts: FVG, Order Blocks, BOS, CHOCH, Liquidity sweeps
- Market Structure Shift (MSS)
- Wyckoff + Spring/Upthrust detection
- Fractal dimension (Higuchi + Katz)
"""

import numpy as np
from typing import Dict, Optional, List

from core.utils.helpers import ema, sma, safe_div


class TechnicalFeatures:
    """Advanced technical indicators and fractal analysis."""

    # EMA ribbon periods
    EMA_PERIODS = [5, 8, 13, 21, 34, 55, 89, 144]
    ATR_PERIOD = 14
    SUPERTREND_MULT = 3.0

    def compute(
        self,
        ohlcv: Optional[Dict] = None,
        current_price: float = 0.0,
    ) -> Dict[str, float]:
        """Compute all 14 technical features."""
        features = {}

        if not ohlcv or len(ohlcv.get("close", [])) < 20:
            return self._defaults()

        close = np.array(ohlcv["close"], dtype=np.float64)
        high = np.array(ohlcv["high"], dtype=np.float64)
        low = np.array(ohlcv["low"], dtype=np.float64)
        n = len(close)

        # === EMA/SMA Ribbon (8 lines) → ribbon width + slope ===
        ema_values = []
        for period in self.EMA_PERIODS:
            if n >= period:
                e = ema(close, period)
                ema_values.append(e[-1])
            else:
                ema_values.append(close[-1])

        # Ribbon width: (max EMA - min EMA) / price
        ribbon_max = max(ema_values)
        ribbon_min = min(ema_values)
        features["ribbon_width"] = safe_div(ribbon_max - ribbon_min, close[-1])

        # Ribbon alignment: how many EMAs are in order (bullish/bearish)
        bullish_count = sum(
            1 for i in range(len(ema_values) - 1)
            if ema_values[i] > ema_values[i + 1]
        )
        features["ribbon_alignment"] = (bullish_count / max(len(ema_values) - 1, 1)) * 2 - 1  # -1 to 1

        # Short EMA slope
        if n >= 8:
            ema8 = ema(close, 8)
            features["ema8_slope"] = (ema8[-1] - ema8[-2]) / close[-1] if n >= 2 else 0.0
        else:
            features["ema8_slope"] = 0.0

        # === SuperTrend + ATR ===
        atr = self._compute_atr(high, low, close, self.ATR_PERIOD)
        features["atr_normalized"] = atr / close[-1] if close[-1] > 0 else 0.0

        supertrend_dir, st_dist = self._compute_supertrend(
            high, low, close, self.ATR_PERIOD, self.SUPERTREND_MULT
        )
        features["supertrend_direction"] = supertrend_dir  # 1=bullish, -1=bearish
        features["supertrend_distance"] = st_dist  # distance from ST line

        # === ICT Concepts ===
        fvg_score = self._detect_fvg(high, low, close)
        features["ict_fvg_score"] = fvg_score

        ob_score = self._detect_order_blocks(high, low, close, ohlcv.get("volume", np.ones(n)))
        features["ict_order_block"] = ob_score

        bos_choch = self._detect_bos_choch(high, low, close)
        features["ict_bos_choch"] = bos_choch

        sweep_score = self._detect_liquidity_sweep(high, low, close)
        features["ict_liquidity_sweep"] = sweep_score

        # === Market Structure Shift ===
        features["mss_score"] = self._detect_mss(high, low, close)

        # === Wyckoff Spring/Upthrust ===
        features["wyckoff_signal"] = self._detect_wyckoff(high, low, close, ohlcv.get("volume", np.ones(n)))

        # === Fractal Dimension ===
        features["fractal_higuchi"] = self._higuchi_fd(close, kmax=10)
        features["fractal_katz"] = self._katz_fd(close)

        return features

    def _defaults(self) -> Dict[str, float]:
        return {
            "ribbon_width": 0.0, "ribbon_alignment": 0.0, "ema8_slope": 0.0,
            "atr_normalized": 0.0, "supertrend_direction": 0.0, "supertrend_distance": 0.0,
            "ict_fvg_score": 0.0, "ict_order_block": 0.0, "ict_bos_choch": 0.0,
            "ict_liquidity_sweep": 0.0, "mss_score": 0.0, "wyckoff_signal": 0.0,
            "fractal_higuchi": 1.5, "fractal_katz": 1.0,
        }

    def _compute_atr(self, high, low, close, period: int) -> float:
        """Average True Range."""
        n = len(close)
        if n < 2:
            return 0.0
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        if len(tr) < period:
            return float(np.mean(tr))
        atr_arr = ema(tr, period)
        return float(atr_arr[-1])

    def _compute_supertrend(self, high, low, close, period: int, mult: float):
        """SuperTrend indicator — returns (direction, distance_pct)."""
        n = len(close)
        if n < period + 1:
            return 0.0, 0.0

        atr = self._compute_atr(high, low, close, period)
        hl2 = (high + low) / 2

        upper_band = hl2[-1] + mult * atr
        lower_band = hl2[-1] - mult * atr

        # Simple direction: price above lower band = bullish
        if close[-1] > upper_band:
            direction = 1.0
            distance = safe_div(close[-1] - lower_band, close[-1])
        elif close[-1] < lower_band:
            direction = -1.0
            distance = safe_div(upper_band - close[-1], close[-1])
        else:
            direction = 1.0 if close[-1] > close[-2] else -1.0
            distance = 0.0

        return direction, distance

    def _detect_fvg(self, high, low, close) -> float:
        """Fair Value Gap detection. Returns score -1 to 1."""
        n = len(close)
        if n < 3:
            return 0.0

        score = 0.0
        # Bullish FVG: candle[i-2].high < candle[i].low (gap up)
        if low[-1] > high[-3]:
            gap = (low[-1] - high[-3]) / close[-1]
            score = min(gap * 100, 1.0)
        # Bearish FVG: candle[i-2].low > candle[i].high (gap down)
        elif high[-1] < low[-3]:
            gap = (low[-3] - high[-1]) / close[-1]
            score = -min(gap * 100, 1.0)

        return score

    def _detect_order_blocks(self, high, low, close, volume) -> float:
        """Order block detection. Returns proximity score -1 to 1."""
        n = len(close)
        if n < 10:
            return 0.0

        # Find strong rejection candles with high volume
        score = 0.0
        for i in range(-10, -1):
            body = abs(close[i] - close[i - 1]) if i > -n else 0
            wick_up = high[i] - max(close[i], close[i - 1]) if i > -n else 0
            wick_dn = min(close[i], close[i - 1]) - low[i] if i > -n else 0
            vol_ratio = volume[i] / np.mean(volume[-20:]) if np.mean(volume[-20:]) > 0 else 1

            # Bullish OB: strong down candle followed by reversal
            if close[i] < close[i - 1] and vol_ratio > 1.5 and i < -1:
                if close[-1] > close[i]:  # price returned above
                    proximity = 1.0 - min(abs(close[-1] - high[i]) / close[-1], 1.0)
                    score = max(score, proximity * 0.5)

        return score

    def _detect_bos_choch(self, high, low, close) -> float:
        """Break of Structure / Change of Character. Returns -1 to 1."""
        n = len(close)
        if n < 20:
            return 0.0

        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        for i in range(2, n - 2):
            if high[i] > high[i - 1] and high[i] > high[i - 2] and high[i] > high[i + 1] and high[i] > high[i + 2]:
                swing_highs.append((i, high[i]))
            if low[i] < low[i - 1] and low[i] < low[i - 2] and low[i] < low[i + 1] and low[i] < low[i + 2]:
                swing_lows.append((i, low[i]))

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 0.0

        # BOS: higher high (bullish) or lower low (bearish)
        last_sh = swing_highs[-1][1]
        prev_sh = swing_highs[-2][1]
        last_sl = swing_lows[-1][1]
        prev_sl = swing_lows[-2][1]

        if last_sh > prev_sh and close[-1] > last_sh:
            return 1.0  # Bullish BOS
        elif last_sl < prev_sl and close[-1] < last_sl:
            return -1.0  # Bearish BOS
        # CHOCH: break in opposite direction
        elif last_sh < prev_sh and close[-1] > prev_sh:
            return 0.5  # Bullish CHOCH
        elif last_sl > prev_sl and close[-1] < prev_sl:
            return -0.5  # Bearish CHOCH

        return 0.0

    def _detect_liquidity_sweep(self, high, low, close) -> float:
        """Detect liquidity sweeps (stop hunts). Returns -1 to 1."""
        n = len(close)
        if n < 20:
            return 0.0

        # Look for wick beyond recent range followed by reversal
        recent_high = np.max(high[-20:-1])
        recent_low = np.min(low[-20:-1])

        score = 0.0
        # Upside sweep: wick above recent high, close back below
        if high[-1] > recent_high and close[-1] < recent_high:
            sweep_size = (high[-1] - recent_high) / close[-1]
            score = -min(sweep_size * 50, 1.0)  # bearish (false breakout)

        # Downside sweep: wick below recent low, close back above
        if low[-1] < recent_low and close[-1] > recent_low:
            sweep_size = (recent_low - low[-1]) / close[-1]
            score = min(sweep_size * 50, 1.0)  # bullish (spring)

        return score

    def _detect_mss(self, high, low, close) -> float:
        """Market Structure Shift detection."""
        n = len(close)
        if n < 15:
            return 0.0

        # Trend detection: compare recent closes
        trend_short = close[-1] - close[-5] if n >= 5 else 0
        trend_long = close[-1] - close[-15] if n >= 15 else 0

        # MSS: short-term reverses long-term
        if trend_long > 0 and trend_short < 0:
            return -safe_div(abs(trend_short), close[-1])  # Bearish shift
        elif trend_long < 0 and trend_short > 0:
            return safe_div(abs(trend_short), close[-1])  # Bullish shift
        return 0.0

    def _detect_wyckoff(self, high, low, close, volume) -> float:
        """Wyckoff Spring/Upthrust detection."""
        n = len(close)
        if n < 20:
            return 0.0

        recent_low = np.min(low[-20:-1])
        recent_high = np.max(high[-20:-1])
        avg_vol = np.mean(volume[-20:])

        # Spring: price dips below support then recovers, ideally on low volume
        if low[-1] < recent_low and close[-1] > recent_low:
            vol_confirm = volume[-1] < avg_vol * 0.8  # low volume spring
            return 1.0 if vol_confirm else 0.5

        # Upthrust: price spikes above resistance then falls, on high volume
        if high[-1] > recent_high and close[-1] < recent_high:
            vol_confirm = volume[-1] > avg_vol * 1.2
            return -1.0 if vol_confirm else -0.5

        return 0.0

    def _higuchi_fd(self, series: np.ndarray, kmax: int = 10) -> float:
        """Higuchi fractal dimension."""
        n = len(series)
        if n < kmax * 4:
            return 1.5

        try:
            lk = []
            x = np.arange(1, kmax + 1)
            y = np.zeros(kmax)

            for k in range(1, kmax + 1):
                lm_sum = 0.0
                for m in range(1, k + 1):
                    ll = 0.0
                    idx = np.arange(1, int((n - m) / k) + 1)
                    if len(idx) == 0:
                        continue
                    for i in idx:
                        ll += abs(series[m + i * k - 1] - series[m + (i - 1) * k - 1])
                    norm = (n - 1) / (int((n - m) / k) * k)
                    ll = ll * norm / k
                    lm_sum += ll
                y[k - 1] = np.log(lm_sum / k) if lm_sum > 0 else 0

            # Linear regression on log-log
            valid = y > 0
            if np.sum(valid) < 2:
                return 1.5
            log_x = np.log(1.0 / x[valid])
            log_y = y[valid]
            coeffs = np.polyfit(log_x, log_y, 1)
            return float(np.clip(coeffs[0], 1.0, 2.0))
        except Exception:
            return 1.5

    def _katz_fd(self, series: np.ndarray) -> float:
        """Katz fractal dimension."""
        n = len(series)
        if n < 10:
            return 1.0

        try:
            diffs = np.abs(np.diff(series))
            L = np.sum(diffs)  # total path length
            d = np.max(np.abs(series - series[0]))  # max distance from start
            a = np.mean(diffs)  # average step

            if d == 0 or a == 0:
                return 1.0

            n_steps = n - 1
            return float(np.log10(n_steps) / (np.log10(n_steps) + np.log10(d / L)))
        except Exception:
            return 1.0
