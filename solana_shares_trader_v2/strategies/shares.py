"""Shares Trading Strategies — Polymarket prediction market focused.

Strategies:
  1. SharesMispricingStrategy — buy when shares are mispriced vs CEX-derived probability
  2. SharesMomentumStrategy — buy in direction of recent shares momentum
  3. SharesHybridStrategy — ML model + mispricing + momentum combined
"""

from typing import Dict, Optional, Any
from strategies.base import BaseStrategy


class SharesMispricingStrategy(BaseStrategy):
    """Buy shares when market price diverges from estimated real probability.

    Core idea:
      - Compute real probability that SOL > PriceToBeat at expiry
        using distance_from_ptb_norm (normalized by vol * sqrt(T))
      - Compare with market-implied probability (yes_price)
      - If mispricing > threshold → buy underpriced side
    """

    def __init__(self, min_mispricing: float = 0.03, min_time_pct: float = 0.15):
        self.min_mispricing = min_mispricing
        self.min_time_pct = min_time_pct  # don't enter with < 15% time left

    def name(self) -> str:
        return "SharesMispricing"

    def evaluate(self, features: Dict[str, float], market_state: Dict) -> Optional[Dict[str, Any]]:
        mispricing = features.get("mispricing_score", 0)
        time_left_pct = features.get("time_remaining_pct", 0)
        spread_norm = features.get("spread_normalized", 1)
        up_prob = features.get("up_implied_prob", 0.5)

        # Gate: enough time and reasonable spread
        if time_left_pct < self.min_time_pct:
            return None
        if spread_norm > 0.10:  # spread > 10% of price
            return None

        # UP mispriced cheap (model says higher prob than market)
        if mispricing > self.min_mispricing and up_prob <= 0.55:
            return {
                "direction": "UP",
                "confidence": min(abs(mispricing) * 8, 1.0),
                "reason": f"mispricing={mispricing:+.3f} up={up_prob:.2f}",
            }

        # DOWN mispriced cheap (model says SOL will drop, but market says UP)
        if mispricing < -self.min_mispricing and up_prob >= 0.45:
            return {
                "direction": "DOWN",
                "confidence": min(abs(mispricing) * 8, 1.0),
                "reason": f"mispricing={mispricing:+.3f} up={up_prob:.2f}",
            }

        return None

    def should_exit(self, features: Dict, market_state: Dict, position: Dict) -> Optional[Dict]:
        """Exit if mispricing reversed or near expiry."""
        mispricing = features.get("mispricing_score", 0)
        time_left_pct = features.get("time_remaining_pct", 0)
        pnl = position.get("current_pnl_pct", 0)

        # Take profit if mispricing closed
        if position["direction"] == "UP" and mispricing < 0.02 and pnl > 5:
            return {"reason": "mispricing_closed"}
        if position["direction"] == "DOWN" and mispricing > -0.02 and pnl > 5:
            return {"reason": "mispricing_closed"}

        return None


class SharesMomentumStrategy(BaseStrategy):
    """Follow recent shares price momentum.

    Core idea:
      - If UP shares rising fast → momentum buy UP
      - If DOWN shares rising fast → momentum buy DOWN
      - Only enter when price is in favorable range (< max_share_price)
    """

    def __init__(self, min_momentum: float = 0.03, min_time_pct: float = 0.20):
        self.min_momentum = min_momentum  # 3 cent momentum minimum
        self.min_time_pct = min_time_pct

    def name(self) -> str:
        return "SharesMomentum"

    def evaluate(self, features: Dict[str, float], market_state: Dict) -> Optional[Dict[str, Any]]:
        mom_1m = features.get("shares_momentum_1m", 0)
        mom_30s = features.get("shares_momentum_30s", 0)
        time_left_pct = features.get("time_remaining_pct", 0)
        up_prob = features.get("up_implied_prob", 0.5)
        vol_imbalance = features.get("volume_imbalance", 0)

        if time_left_pct < self.min_time_pct:
            return None

        # Strong upward momentum in UP shares
        if mom_1m > self.min_momentum and mom_30s > 0 and up_prob <= 0.40:
            conf = min(mom_1m * 10, 1.0)
            if vol_imbalance > 0:
                conf *= 1.2
            return {
                "direction": "UP",
                "confidence": min(conf, 1.0),
                "reason": f"mom_1m={mom_1m:+.3f} mom_30s={mom_30s:+.3f}",
            }

        # Strong downward momentum (UP shares falling → buy DOWN)
        if mom_1m < -self.min_momentum and mom_30s < 0 and up_prob >= 0.60:
            conf = min(abs(mom_1m) * 10, 1.0)
            if vol_imbalance < 0:
                conf *= 1.2
            return {
                "direction": "DOWN",
                "confidence": min(conf, 1.0),
                "reason": f"mom_1m={mom_1m:+.3f} mom_30s={mom_30s:+.3f}",
            }

        return None

    def should_exit(self, features: Dict, market_state: Dict, position: Dict) -> Optional[Dict]:
        """Exit on momentum reversal."""
        mom_30s = features.get("shares_momentum_30s", 0)
        pnl = position.get("current_pnl_pct", 0)

        # Momentum reversed against position
        if position["direction"] == "UP" and mom_30s < -0.02 and pnl > 0:
            return {"reason": "momentum_reversal"}
        if position["direction"] == "DOWN" and mom_30s > 0.02 and pnl > 0:
            return {"reason": "momentum_reversal"}

        return None


class SharesHybridStrategy(BaseStrategy):
    """Combined strategy: ML model + mispricing + momentum.

    Decision tree:
      1. Check ML model prediction (if available)
      2. Check mispricing edge
      3. Check momentum confirmation
      4. Score = weighted average of all signals
    """

    def __init__(self, model=None, min_confidence: float = 0.55):
        self.model = model
        self.min_confidence = min_confidence
        self._mispricing = SharesMispricingStrategy(min_mispricing=0.03)
        self._momentum = SharesMomentumStrategy(min_momentum=0.02)

    def name(self) -> str:
        return "SharesHybrid"

    def evaluate(self, features: Dict[str, float], market_state: Dict) -> Optional[Dict[str, Any]]:
        signals = []

        # ML model signal (weight=0.5)
        if self.model:
            import numpy as np
            fv = np.array(list(features.values()), dtype=np.float64)
            pred = self.model.predict(fv)
            if pred.get("should_take"):
                signals.append({
                    "direction": pred["direction"],
                    "confidence": pred["confidence"],
                    "weight": 0.5,
                    "source": "ml",
                })

        # Mispricing signal (weight=0.3)
        misp_signal = self._mispricing.evaluate(features, market_state)
        if misp_signal:
            signals.append({
                "direction": misp_signal["direction"],
                "confidence": misp_signal["confidence"],
                "weight": 0.3,
                "source": "mispricing",
            })

        # Momentum signal (weight=0.2)
        mom_signal = self._momentum.evaluate(features, market_state)
        if mom_signal:
            signals.append({
                "direction": mom_signal["direction"],
                "confidence": mom_signal["confidence"],
                "weight": 0.2,
                "source": "momentum",
            })

        if not signals:
            return None

        # Majority vote on direction
        up_score = sum(s["confidence"] * s["weight"] for s in signals if s["direction"] == "UP")
        down_score = sum(s["confidence"] * s["weight"] for s in signals if s["direction"] == "DOWN")

        if max(up_score, down_score) < self.min_confidence * 0.3:  # scaled by max weight
            return None

        direction = "UP" if up_score > down_score else "DOWN"
        confidence = max(up_score, down_score) / sum(s["weight"] for s in signals)
        sources = "+".join(s["source"] for s in signals)

        return {
            "direction": direction,
            "confidence": min(confidence, 1.0),
            "reason": f"{sources} up={up_score:.2f} down={down_score:.2f}",
        }

    def should_exit(self, features: Dict, market_state: Dict, position: Dict) -> Optional[Dict]:
        # Delegate to sub-strategies
        for strat in (self._mispricing, self._momentum):
            sig = strat.should_exit(features, market_state, position)
            if sig:
                return sig
        return None
