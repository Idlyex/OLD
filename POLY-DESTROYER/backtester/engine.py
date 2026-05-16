"""Backtester Engine — replay historical data through the full pipeline.

Features:
- Realistic slippage model (0.8-2.5 bps, volatility-adjusted, sqrt impact)
- Full feature computation on historical data
- Position-level P&L tracking
- Equity curve generation
- Per-trade logging
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field

from core.features.engine import FeatureEngine
from core.models.hybrid_model import HybridModel
from core.risk.manager import RiskManager
from core.utils.logger import log
from core.utils.helpers import safe_div, format_pnl, clip
from config import config

_bt_cfg = config.get("backtester", {})
_slippage_cfg = _bt_cfg.get("slippage", {})


@dataclass
class BacktestTrade:
    """Single backtest trade record."""
    entry_ts: int
    exit_ts: int = 0
    direction: str = "UP"
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: float = 0.0
    size_usd: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    slippage_bps: float = 0.0
    exit_reason: str = ""
    confidence: float = 0.0
    hold_time_s: float = 0.0
    peak_pnl_pct: float = 0.0
    features_at_entry: dict = field(default_factory=dict)


class SlippageModel:
    """Realistic slippage model: sqrt impact, volatility-adjusted."""

    def __init__(self):
        self.base_bps = _slippage_cfg.get("base_bps", 0.8)
        self.vol_multiplier = _slippage_cfg.get("vol_multiplier", 2.5)
        self.impact_model = _slippage_cfg.get("impact_model", "sqrt")

    def compute(self, size_usd: float, volatility: float, avg_vol: float = 0.003) -> float:
        """Compute slippage in bps.

        Slippage = base_bps * vol_factor * size_impact
        """
        # Volatility factor: higher vol → more slippage
        vol_ratio = safe_div(volatility, avg_vol, default=1.0)
        vol_factor = clip(vol_ratio, 0.5, 3.0)

        # Size impact
        if self.impact_model == "sqrt":
            # Square root market impact model
            size_factor = np.sqrt(max(size_usd, 1.0)) / np.sqrt(10.0)  # normalized to $10
        else:
            size_factor = max(size_usd, 1.0) / 10.0

        size_factor = clip(size_factor, 0.5, 3.0)

        slippage_bps = self.base_bps * vol_factor * size_factor
        return clip(slippage_bps, self.base_bps, self.vol_multiplier)


class Backtester:
    """Full backtest engine with feature replay, model evaluation, and analytics."""

    def __init__(self):
        self.feature_engine = FeatureEngine()
        self.model = HybridModel()
        self.risk_manager = RiskManager()
        self.slippage = SlippageModel()

        self.initial_capital = _bt_cfg.get("initial_capital", 100.0)
        self.commission_bps = _bt_cfg.get("commission_bps", 0.0)

        # Results
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[Dict] = []
        self.capital = self.initial_capital

    def run(
        self,
        data: pd.DataFrame,
        strategy=None,
        verbose: bool = True,
    ) -> Dict:
        """Run backtest on historical data.

        Args:
            data: DataFrame with columns: ts, open, high, low, close, volume + any extra
            strategy: Strategy instance for signal generation
            verbose: Print progress

        Returns:
            Results dict with trades, metrics, equity curve
        """
        log.info(f"═══ Backtester starting: {len(data)} bars, ${self.initial_capital:.2f} capital ═══")
        t0 = time.perf_counter()

        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        open_position: Optional[Dict] = None
        feature_history = deque(maxlen=config.get("models", {}).get("primary", {}).get("sequence_length", 60))

        for i in range(20, len(data)):
            row = data.iloc[i]
            ts = int(row.get("ts", i))
            close = float(row["close"])

            # Build OHLCV window
            window = data.iloc[max(0, i - 60):i + 1]
            ohlcv = {
                "open": window["open"].values.astype(np.float64),
                "high": window["high"].values.astype(np.float64),
                "low": window["low"].values.astype(np.float64),
                "close": window["close"].values.astype(np.float64),
                "volume": window["volume"].values.astype(np.float64),
                "taker_buy_volume": window.get("taker_buy_volume", window["volume"] * 0.5).values.astype(np.float64),
            }

            # Compute features (simplified for backtest — no live collectors)
            features = {}
            features.update(self.feature_engine.price_volume.compute(
                ohlcv_1m=ohlcv, current_price=close,
            ))
            features.update(self.feature_engine.technical.compute(
                ohlcv=ohlcv, current_price=close,
            ))
            # Microstructure, liquidation, onchain default to zeros in backtest
            features.update(self.feature_engine.microstructure.compute(current_price=close))
            features.update(self.feature_engine.liquidation_funding.compute(current_price=close))
            features.update(self.feature_engine.regime.compute(
                returns=np.diff(np.log(ohlcv["close"][ohlcv["close"] > 0])) if len(ohlcv["close"]) > 10 else None,
                close_prices=ohlcv["close"],
            ))
            # Onchain defaults
            for k in ["onchain_large_transfers_60s", "onchain_whale_activity",
                       "onchain_dex_volume_spike", "onchain_jupiter_accel",
                       "onchain_mev_bundles", "onchain_priority_fee_pressure",
                       "onchain_token_creation_rate", "onchain_large_transfers_300s",
                       "onchain_dex_volume_spike_300s", "onchain_jupiter_accel_300s"]:
                features.setdefault(k, 0.0)

            fv = self.feature_engine.get_feature_vector(features)
            feature_history.append(fv)

            # Update risk vol tracking
            vol = features.get("vol_garman_klass", 0.003)
            self.risk_manager.update_volatility(vol)

            # ── Monitor open position ──
            if open_position:
                pos = open_position
                pnl_pct = safe_div(close - pos["entry_price"], pos["entry_price"]) * 100
                if pos["direction"] == "DOWN":
                    pnl_pct = -pnl_pct  # inverse for shorts
                pos["peak_pnl"] = max(pos.get("peak_pnl", 0), pnl_pct)
                pos["age_bars"] += 1

                # Dynamic stop
                dyn_stop = self.risk_manager.compute_dynamic_stop(vol)
                should_exit = False
                exit_reason = ""

                if pnl_pct <= -dyn_stop:
                    should_exit = True
                    exit_reason = f"dyn_stop_{dyn_stop:.1f}"
                elif pos["peak_pnl"] > 30 and pnl_pct < pos["peak_pnl"] * 0.65:
                    should_exit = True
                    exit_reason = f"trailing_{pos['peak_pnl']:.0f}pk"
                elif pos["age_bars"] >= 15:  # ~15 min timeout
                    should_exit = True
                    exit_reason = "timeout"
                elif self.risk_manager.should_exit_drawdown(pnl_pct, pos["peak_pnl"]):
                    should_exit = True
                    exit_reason = "dd_model"

                if should_exit:
                    # Apply slippage on exit
                    slip = self.slippage.compute(pos["size"], vol)
                    exit_price = close * (1 - slip / 10000) if pos["direction"] == "UP" else close * (1 + slip / 10000)

                    final_pnl_pct = safe_div(exit_price - pos["entry_price"], pos["entry_price"]) * 100
                    if pos["direction"] == "DOWN":
                        final_pnl_pct = -final_pnl_pct

                    pnl_usd = final_pnl_pct / 100 * pos["size"]
                    self.capital += pnl_usd

                    trade = BacktestTrade(
                        entry_ts=pos["entry_ts"],
                        exit_ts=ts,
                        direction=pos["direction"],
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        shares=pos.get("shares", 1),
                        size_usd=pos["size"],
                        pnl_pct=final_pnl_pct,
                        pnl_usd=pnl_usd,
                        slippage_bps=slip,
                        exit_reason=exit_reason,
                        confidence=pos.get("confidence", 0),
                        hold_time_s=pos["age_bars"] * 60,
                        peak_pnl_pct=pos["peak_pnl"],
                    )
                    self.trades.append(trade)
                    open_position = None

            # ── Evaluate new entry ──
            if open_position is None and strategy:
                signal = strategy.evaluate(features, {"price": close})

                if signal and signal.get("direction") in ("UP", "DOWN"):
                    # Position sizing
                    size = self.risk_manager.compute_position_size(
                        capital=self.capital,
                        confidence=signal.get("confidence", 0.5),
                        win_prob=signal.get("confidence", 0.5),
                        avg_win=0.10, avg_loss=0.05,
                        current_vol=vol,
                        regime=features.get("vol_regime", 1),
                    )
                    size = max(size, 2.0)

                    # Apply slippage on entry
                    slip = self.slippage.compute(size, vol)
                    entry_price = close * (1 + slip / 10000) if signal["direction"] == "UP" else close * (1 - slip / 10000)

                    open_position = {
                        "entry_ts": ts,
                        "entry_price": entry_price,
                        "direction": signal["direction"],
                        "size": size,
                        "shares": size / entry_price,
                        "confidence": signal.get("confidence", 0),
                        "peak_pnl": 0,
                        "age_bars": 0,
                    }

            # Equity tracking
            unrealized = 0
            if open_position:
                unrealized_pnl = safe_div(close - open_position["entry_price"], open_position["entry_price"])
                if open_position["direction"] == "DOWN":
                    unrealized_pnl = -unrealized_pnl
                unrealized = unrealized_pnl * open_position["size"]

            self.equity_curve.append({
                "ts": ts,
                "bar": i,
                "equity": self.capital + unrealized,
                "capital": self.capital,
                "unrealized": unrealized,
                "price": close,
            })
            self.risk_manager.update_equity(self.capital + unrealized)

        elapsed = time.perf_counter() - t0
        results = self._compute_results()
        results["elapsed_s"] = elapsed

        if verbose:
            self._print_results(results)

        return results

    def _compute_results(self) -> Dict:
        """Compute backtest metrics."""
        if not self.trades:
            return {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                "total_pnl_usd": 0, "total_pnl_pct": 0, "avg_pnl_pct": 0,
                "avg_win_pct": 0, "avg_loss_pct": 0, "max_drawdown_pct": 0,
                "sharpe_ratio": 0, "profit_factor": 0, "avg_hold_time_s": 0,
                "avg_slippage_bps": 0, "initial_capital": self.initial_capital,
                "final_equity": self.capital,
            }

        wins = [t for t in self.trades if t.pnl_pct > 0]
        losses = [t for t in self.trades if t.pnl_pct <= 0]

        equity = [e["equity"] for e in self.equity_curve]
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / (peak + 1e-10) * 100

        returns = np.diff(equity) / (np.array(equity[:-1]) + 1e-10)
        sharpe = 0.0
        if len(returns) > 10 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 4))

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": safe_div(len(wins), len(self.trades)) * 100,
            "total_pnl_usd": sum(t.pnl_usd for t in self.trades),
            "total_pnl_pct": safe_div(self.capital - self.initial_capital, self.initial_capital) * 100,
            "avg_pnl_pct": float(np.mean([t.pnl_pct for t in self.trades])),
            "avg_win_pct": float(np.mean([t.pnl_pct for t in wins])) if wins else 0,
            "avg_loss_pct": float(np.mean([t.pnl_pct for t in losses])) if losses else 0,
            "max_drawdown_pct": float(np.max(drawdown)) if len(drawdown) > 0 else 0,
            "sharpe_ratio": sharpe,
            "profit_factor": safe_div(
                sum(t.pnl_usd for t in wins),
                abs(sum(t.pnl_usd for t in losses)),
            ) if losses else float("inf"),
            "avg_hold_time_s": float(np.mean([t.hold_time_s for t in self.trades])),
            "avg_slippage_bps": float(np.mean([t.slippage_bps for t in self.trades])),
            "initial_capital": self.initial_capital,
            "final_equity": self.capital,
        }

    def _print_results(self, r: Dict):
        """Print formatted backtest results."""
        log.info("═══════════════════════════════════════════════════")
        log.info("           BACKTEST RESULTS")
        log.info("═══════════════════════════════════════════════════")
        log.info(f"  Trades:       {r['total_trades']} (W:{r['wins']} L:{r['losses']})")
        log.info(f"  Win Rate:     {r['win_rate']:.1f}%")
        log.info(f"  Total PnL:    ${r['total_pnl_usd']:+.2f} ({r['total_pnl_pct']:+.1f}%)")
        log.info(f"  Avg Win:      {r['avg_win_pct']:+.2f}%")
        log.info(f"  Avg Loss:     {r['avg_loss_pct']:+.2f}%")
        log.info(f"  Max DD:       {r['max_drawdown_pct']:.1f}%")
        log.info(f"  Sharpe:       {r['sharpe_ratio']:.2f}")
        log.info(f"  Profit Factor:{r['profit_factor']:.2f}")
        log.info(f"  Avg Hold:     {r['avg_hold_time_s']:.0f}s")
        log.info(f"  Avg Slippage: {r['avg_slippage_bps']:.2f} bps")
        log.info(f"  Capital:      ${r['initial_capital']:.2f} → ${r['final_equity']:.2f}")
        log.info(f"  Time:         {r.get('elapsed_s', 0):.1f}s")
        log.info("═══════════════════════════════════════════════════")

    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame([{
            "entry_ts": t.entry_ts, "exit_ts": t.exit_ts,
            "direction": t.direction, "entry_price": t.entry_price,
            "exit_price": t.exit_price, "pnl_pct": t.pnl_pct,
            "pnl_usd": t.pnl_usd, "slippage_bps": t.slippage_bps,
            "exit_reason": t.exit_reason, "confidence": t.confidence,
            "hold_time_s": t.hold_time_s, "peak_pnl_pct": t.peak_pnl_pct,
        } for t in self.trades])
