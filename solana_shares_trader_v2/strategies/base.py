"""Base strategy interface and built-in strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def evaluate(self, features: Dict[str, float], market_state: Dict) -> Optional[Dict[str, Any]]:
        """Evaluate features and return a signal or None.

        Returns:
            Signal dict with keys: direction, confidence, reason
            or None if no signal.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class MLHybridStrategy(BaseStrategy):
    """Primary strategy: uses the hybrid ML model for signal generation."""

    def __init__(self, model=None):
        self.model = model

    def name(self) -> str:
        return "MLHybrid"

    def evaluate(self, features: Dict[str, float], market_state: Dict) -> Optional[Dict[str, Any]]:
        """Generate signal from ML model prediction."""
        if not self.model:
            return None

        import numpy as np
        fv = np.array(list(features.values()), dtype=np.float64)
        prediction = self.model.predict(fv)

        if not prediction.get("should_take", False):
            return None

        return {
            "direction": prediction["direction"],
            "confidence": prediction["confidence"],
            "expected_return": prediction.get("expected_return", 0),
            "reversal_prob": prediction.get("reversal_prob", 0.5),
            "hold_time": prediction.get("hold_time", 60),
            "reason": f"ML conf={prediction['confidence']:.3f} lgbm={prediction['lgbm_score']:.3f} cb={prediction['catboost_score']:.3f}",
        }


class MicrostructureStrategy(BaseStrategy):
    """Strategy based purely on order flow / microstructure signals."""

    def name(self) -> str:
        return "Microstructure"

    def evaluate(self, features: Dict[str, float], market_state: Dict) -> Optional[Dict[str, Any]]:
        """Generate signal from microstructure features only."""
        imb_20 = features.get("ob_imbalance_20", 0)
        pressure = features.get("book_pressure", 0)
        cvd_slope = features.get("cvd_slope", 0)
        aggr_buy = features.get("aggr_buy_ratio_30s", 0.5)
        vpin = features.get("vpin", 0.5)

        # Strong buy signal
        score = 0.0
        if imb_20 > 0.3:
            score += 0.2
        if pressure > 0.3:
            score += 0.2
        if cvd_slope > 0.1:
            score += 0.2
        if aggr_buy > 0.65:
            score += 0.2
        if vpin < 0.4:
            score += 0.2

        if score >= 0.6:
            return {
                "direction": "UP",
                "confidence": score,
                "reason": f"micro imb={imb_20:.2f} press={pressure:.2f} cvd_s={cvd_slope:.2f}",
            }

        # Strong sell signal (inverse)
        score_sell = 0.0
        if imb_20 < -0.3:
            score_sell += 0.2
        if pressure < -0.3:
            score_sell += 0.2
        if cvd_slope < -0.1:
            score_sell += 0.2
        if aggr_buy < 0.35:
            score_sell += 0.2
        if vpin > 0.6:
            score_sell += 0.2

        if score_sell >= 0.6:
            return {
                "direction": "DOWN",
                "confidence": score_sell,
                "reason": f"micro imb={imb_20:.2f} press={pressure:.2f} cvd_s={cvd_slope:.2f}",
            }

        return None


class RegimeAwareStrategy(BaseStrategy):
    """Strategy that adapts based on detected market regime."""

    def name(self) -> str:
        return "RegimeAware"

    def evaluate(self, features: Dict[str, float], market_state: Dict) -> Optional[Dict[str, Any]]:
        """Adapt signal generation to current regime."""
        regime = features.get("vol_regime", 1)
        hurst = features.get("hurst_exponent", 0.5)

        # In trending regime (hurst > 0.5): follow momentum
        if hurst > 0.55 and regime <= 2:
            ret_5m = features.get("ret_log_5m", 0)
            ribbon = features.get("ribbon_alignment", 0)

            if ret_5m > 0.001 and ribbon > 0.3:
                return {
                    "direction": "UP",
                    "confidence": min(hurst, 0.9),
                    "reason": f"trend H={hurst:.2f} ret5m={ret_5m:.4f} ribbon={ribbon:.2f}",
                }
            elif ret_5m < -0.001 and ribbon < -0.3:
                return {
                    "direction": "DOWN",
                    "confidence": min(hurst, 0.9),
                    "reason": f"trend H={hurst:.2f} ret5m={ret_5m:.4f} ribbon={ribbon:.2f}",
                }

        # In mean-reverting regime (hurst < 0.45): fade extremes
        elif hurst < 0.45:
            vwap_dev = features.get("vwap_deviation", 0)
            if vwap_dev > 0.002:
                return {
                    "direction": "DOWN",
                    "confidence": 0.6 * (1 - hurst),
                    "reason": f"mean_rev H={hurst:.2f} vwap_dev={vwap_dev:.4f}",
                }
            elif vwap_dev < -0.002:
                return {
                    "direction": "UP",
                    "confidence": 0.6 * (1 - hurst),
                    "reason": f"mean_rev H={hurst:.2f} vwap_dev={vwap_dev:.4f}",
                }

        return None
