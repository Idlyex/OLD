"""Risk Manager — Dynamic stops, Kelly sizing, hedge module, drawdown model.

Philosophy: No hard stops. Everything is model-driven and volatility-adjusted.
- Dynamic Volatility Stop = base_stop * predicted_vol / avg_vol
- Max Drawdown Exit: if P(further DD) > 68% → exit
- Hedge Module: at -12% open 30-40% counter-position
- Position Size = Kelly * f(confidence, regime, vol)
"""

import math
import numpy as np
from typing import Dict, Optional, Any
from collections import deque

from core.utils.helpers import kelly_fraction, clip, safe_div
from core.utils.logger import log
from config import config

_risk_cfg = config.get("risk", {})
_trading_cfg = config.get("trading", {})


class RiskManager:
    """Handles position sizing, dynamic stops, hedging, and drawdown management."""

    def __init__(self):
        # Config
        self.base_stop_pct = _risk_cfg.get("dynamic_stop", {}).get("base_stop_pct", 5.0)
        self.vol_lookback = _risk_cfg.get("dynamic_stop", {}).get("vol_lookback", 100)
        self.dd_threshold = _risk_cfg.get("max_dd_exit", {}).get("dd_probability_threshold", 0.68)
        self.hedge_trigger = _risk_cfg.get("hedge", {}).get("trigger_loss_pct", -12.0)
        self.hedge_size_pct = _risk_cfg.get("hedge", {}).get("hedge_size_pct", 35.0)
        self.max_position_pct = _risk_cfg.get("position_sizing", {}).get("max_position_pct", 15.0)
        self.kelly_frac = _risk_cfg.get("position_sizing", {}).get("kelly_fraction", 0.25)
        self.min_confidence_full = _risk_cfg.get("position_sizing", {}).get("min_confidence_for_full", 0.80)

        # State
        self._vol_history = deque(maxlen=1000)
        self._avg_vol = 0.001
        self._equity_curve = deque(maxlen=10000)
        self._peak_equity = 0.0
        self._hedge_active: Dict[str, bool] = {}

    def compute_position_size(
        self,
        capital: float,
        confidence: float,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
        current_vol: float,
        regime: float = 1.0,
    ) -> float:
        """Kelly-based position sizing adjusted for confidence, regime, volatility.

        Returns: position size in USD.
        """
        if capital <= 0 or confidence < 0.1:
            return 0.0

        # Base Kelly fraction
        win_loss_ratio = safe_div(abs(avg_win), abs(avg_loss), default=1.0)
        base_kelly = kelly_fraction(win_prob, win_loss_ratio)

        # Scale by quarter-Kelly for safety
        kelly_size = base_kelly * self.kelly_frac

        # Confidence scaling: linear from 0 to 1
        confidence_factor = clip(confidence, 0.0, 1.0)
        if confidence < self.min_confidence_full:
            confidence_factor *= confidence / self.min_confidence_full

        # Regime scaling: reduce in extreme regimes
        regime_factor = 1.0
        if regime >= 3.0:  # extreme
            regime_factor = 0.3
        elif regime >= 2.0:  # high vol
            regime_factor = 0.6

        # Volatility scaling: inverse vol
        vol_factor = safe_div(self._avg_vol, max(current_vol, 1e-8), default=1.0)
        vol_factor = clip(vol_factor, 0.3, 2.0)

        # Final size
        position_pct = kelly_size * confidence_factor * regime_factor * vol_factor
        position_pct = clip(position_pct, 0.0, self.max_position_pct / 100)

        size_usd = capital * position_pct
        min_size = _trading_cfg.get("order_size_usd", 2.0)
        max_size = capital * (self.max_position_pct / 100)

        return clip(size_usd, min_size, max_size)

    def compute_dynamic_stop(self, current_vol: float) -> float:
        """Dynamic stop level based on current vs average volatility.
        Stop = base_stop * (predicted_vol / avg_vol)
        """
        if self._avg_vol <= 0:
            return self.base_stop_pct

        vol_ratio = safe_div(current_vol, self._avg_vol, default=1.0)
        vol_ratio = clip(vol_ratio, 0.5, 3.0)

        dynamic_stop = self.base_stop_pct * vol_ratio
        return clip(dynamic_stop, 2.0, 20.0)  # never less than 2%, never more than 20%

    def should_exit_drawdown(
        self, current_pnl_pct: float, peak_pnl_pct: float,
        model_dd_prob: float = 0.0
    ) -> bool:
        """Max Drawdown Exit: if P(further drawdown) > threshold → exit."""
        if not _risk_cfg.get("max_dd_exit", {}).get("enabled", True):
            return False

        # Current drawdown from peak
        drawdown = peak_pnl_pct - current_pnl_pct

        # Model-based: if model predicts high probability of further DD
        if model_dd_prob > self.dd_threshold:
            log.info(
                f"🛑 DD Model: P(further DD)={model_dd_prob:.1%} > {self.dd_threshold:.1%} → EXIT"
            )
            return True

        # Rule-based fallback: if DD is very large and accelerating
        if drawdown > 15 and current_pnl_pct < -8:
            return True

        return False

    def should_hedge(self, current_pnl_pct: float, pos_key: str) -> Optional[Dict]:
        """Hedge Module: at trigger loss, open counter-position.

        Returns hedge parameters or None.
        """
        if not _risk_cfg.get("hedge", {}).get("enabled", True):
            return None

        if self._hedge_active.get(pos_key, False):
            return None  # already hedged

        if current_pnl_pct <= self.hedge_trigger:
            self._hedge_active[pos_key] = True
            log.info(
                f"🛡️ HEDGE TRIGGER: {pos_key} at {current_pnl_pct:.1f}% → "
                f"opening {self.hedge_size_pct:.0f}% counter-position"
            )
            return {
                "action": "hedge",
                "size_pct": self.hedge_size_pct,
                "reason": f"DD {current_pnl_pct:.1f}%",
            }

        return None

    def update_volatility(self, vol: float):
        """Track volatility for dynamic stop computation."""
        self._vol_history.append(vol)
        if len(self._vol_history) >= 10:
            self._avg_vol = float(np.mean(list(self._vol_history)))

    def update_equity(self, equity: float):
        """Track equity for drawdown monitoring."""
        self._equity_curve.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

    def get_max_drawdown(self) -> float:
        """Current max drawdown from peak equity."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._equity_curve[-1]) / self._peak_equity * 100 if self._equity_curve else 0.0

    def clear_hedge(self, pos_key: str):
        """Clear hedge state for a position."""
        self._hedge_active.pop(pos_key, None)

    def get_risk_metrics(self) -> Dict[str, float]:
        """Current risk metrics snapshot."""
        equity_arr = np.array(list(self._equity_curve)) if self._equity_curve else np.array([0.0])
        returns = np.diff(equity_arr) / (equity_arr[:-1] + 1e-10) if len(equity_arr) > 1 else np.array([0.0])

        sharpe = 0.0
        if len(returns) > 10 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4))

        sortino = 0.0
        downside = returns[returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = float(np.mean(returns) / np.std(downside) * np.sqrt(252 * 24 * 4))

        return {
            "max_drawdown_pct": self.get_max_drawdown(),
            "avg_volatility": self._avg_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "peak_equity": self._peak_equity,
            "current_equity": float(self._equity_curve[-1]) if self._equity_curve else 0.0,
            "active_hedges": sum(1 for v in self._hedge_active.values() if v),
        }
