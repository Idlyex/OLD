"""Replay Backtester — honest backtesting on real recorded Polymarket data.

Loads parquet snapshots from data/recorded/shares/ and replays them tick-by-tick.
Each snapshot = one data point with real bid/ask/mid/spread/volume.

Key differences from shares_engine.py (synthetic):
  - All prices are REAL (recorded from CLOB orderbooks)
  - Spreads are REAL (not modeled)
  - Liquidity is REAL (not assumed)
  - Slippage = spread-based (buy at ask, sell at bid)
  - Market outcomes determined by real resolution (or SOL price vs PTB)

Usage:
  python main.py --mode backtest --replay --date 2025-05-03 --market-duration 15
"""

import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from core.utils.logger import log
from config import config

RECORD_DIR = Path("data/recorded/shares")

_bt_cfg = config.get("backtester", {})
_shares_cfg = config.get("shares", {})


# ═══════════════════════════════════════════════════════════
#  TRADE DATA
# ═══════════════════════════════════════════════════════════

@dataclass
class ReplayTrade:
    """A single shares trade from replay backtesting."""
    market_slug: str = ""
    entry_ts: float = 0
    exit_ts: float = 0
    direction: str = ""       # "UP" or "DOWN"
    entry_price: float = 0    # actual entry (ask for buy)
    exit_price: float = 0     # actual exit (bid for sell, or 0/1 at expiry)
    shares_count: float = 0
    size_usd: float = 0
    pnl_pct: float = 0
    pnl_usd: float = 0
    slippage_entry: float = 0  # mid - entry_price
    slippage_exit: float = 0   # exit_price - mid at exit
    exit_reason: str = ""
    confidence: float = 0
    hold_time_s: float = 0
    peak_pnl_pct: float = 0
    price_to_beat: float = 0
    sol_price_at_entry: float = 0
    sol_price_at_exit: float = 0
    duration_minutes: int = 0
    time_remaining_at_entry_pct: float = 0
    # Recorded features at entry
    up_mid_at_entry: float = 0
    up_spread_at_entry: float = 0
    liquidity_at_entry: float = 0
    momentum_30s_at_entry: float = 0


# ═══════════════════════════════════════════════════════════
#  REPLAY BACKTESTER
# ═══════════════════════════════════════════════════════════

class ReplayBacktester:
    """Replays real recorded Polymarket data for honest backtesting."""

    def __init__(self, initial_capital: float = None):
        self.capital = initial_capital or _bt_cfg.get("initial_capital", 100.0)
        self.initial_capital = self.capital
        self.trades: List[ReplayTrade] = []
        self.equity_curve: List[Dict] = []

        self.max_position_size = config.get("trading", {}).get("order_size_usd", 2.0)
        self.max_open_positions = config.get("trading", {}).get("max_open_positions", 3)
        self.max_share_price = _shares_cfg.get("max_share_price", 0.55)

    def run(
        self,
        data: pd.DataFrame = None,
        dates: List[str] = None,
        duration_minutes: int = 15,
        strategy=None,
        verbose: bool = True,
    ) -> Dict:
        """Run replay backtest.

        Args:
            data: Pre-loaded DataFrame of snapshots (optional)
            dates: List of date strings to load from recorded dir (optional)
            duration_minutes: Market duration to filter (5, 15, 60)
            strategy: Strategy instance for entry/exit decisions
            verbose: Print progress

        Returns:
            Results dict with metrics
        """
        # Load data
        if data is None:
            data = self._load_recorded_data(dates, duration_minutes)

        if data.empty:
            log.error("No recorded data to replay. Run recorder first.")
            return self._empty_results()

        # Group by market (slug)
        markets = data.groupby("slug")
        n_markets = len(markets)

        if verbose:
            log.info(f"📊 Replaying {n_markets} recorded markets ({duration_minutes}m) | "
                     f"capital=${self.capital:.2f} | {len(data):,} snapshots")

        open_positions: List[Dict] = []
        market_idx = 0

        for slug, market_df in markets:
            market_idx += 1
            market_df = market_df.sort_values("ts").reset_index(drop=True)

            if len(market_df) < 2:
                continue

            # Market metadata from first snapshot
            first = market_df.iloc[0]
            ptb = first.get("price_to_beat")
            dur_min = int(first.get("duration_min", duration_minutes))
            creation_time = first.get("creation_time", "")
            expiration_time = first.get("expiration_time", "")

            # Iterate through snapshots (ticks)
            for tick_idx in range(len(market_df)):
                snap = market_df.iloc[tick_idx]
                ts = snap["ts"]

                up_mid = snap.get("up_mid_price", 0.5)
                dn_mid = snap.get("dn_mid_price", 0.5)
                up_bid = snap.get("up_best_bid", 0)
                up_ask = snap.get("up_best_ask", 1)
                dn_bid = snap.get("dn_best_bid", 0)
                dn_ask = snap.get("dn_best_ask", 1)
                sol_price = snap.get("sol_price", 0)
                time_remaining_sec = snap.get("time_remaining_sec", 0)
                time_remaining_pct = snap.get("time_remaining_pct", 0)

                # Build features dict from snapshot columns
                features = {
                    "up_mid_price": up_mid,
                    "dn_mid_price": dn_mid,
                    "up_implied_prob": up_mid,
                    "time_remaining_pct": time_remaining_pct,
                    "time_remaining_min": time_remaining_sec / 60,
                    "time_elapsed_min": snap.get("time_elapsed_sec", 0) / 60,
                    "spread_normalized": snap.get("up_spread_pct", 0),
                    "distance_from_ptb_pct": (sol_price - ptb) / ptb * 100 if ptb and sol_price else 0,
                    "volume_imbalance": snap.get("volume_imbalance", 0),
                    "liquidity_score": snap.get("liquidity_score", 0),
                    "shares_momentum_30s": snap.get("shares_momentum_30s", 0),
                    "shares_momentum_2m": snap.get("shares_momentum_2m", 0),
                    "shares_acceleration": snap.get("shares_acceleration", 0),
                    "volume_spike": snap.get("volume_spike", 1),
                    "up_ob_imbalance": snap.get("up_ob_imbalance", 0),
                    "dn_ob_imbalance": snap.get("dn_ob_imbalance", 0),
                    "up_spread": snap.get("up_spread", 0),
                    "dn_spread": snap.get("dn_spread", 0),
                    "sol_price": sol_price,
                    "price_to_beat": ptb or 0,
                }

                # Compute mispricing from recorded data
                if ptb and sol_price and time_remaining_sec > 0:
                    dist = (sol_price - ptb) / ptb
                    vol = 0.003
                    t_min = max(time_remaining_sec / 60, 0.01)
                    d_score = dist / (max(vol, 0.001) * math.sqrt(t_min))
                    # Simple logistic estimate
                    naive_prob = 1.0 / (1.0 + math.exp(-d_score * 2))
                    features["mispricing_score"] = naive_prob - up_mid
                    features["distance_from_ptb_norm"] = d_score
                else:
                    features["mispricing_score"] = 0
                    features["distance_from_ptb_norm"] = 0

                market_state = {
                    "slug": slug,
                    "sol_price": sol_price,
                    "price_to_beat": ptb,
                    "yes_price": up_mid,
                    "no_price": dn_mid,
                    "time_remaining_ms": time_remaining_sec * 1000,
                    "time_elapsed_ms": snap.get("time_elapsed_sec", 0) * 1000,
                    "duration_minutes": dur_min,
                    "open_positions": [p for p in open_positions if p["slug"] == slug],
                }

                # ── CHECK EXITS ──
                positions_to_close = []
                for pos in open_positions:
                    if pos["slug"] != slug:
                        continue

                    # Update current price from REAL bid
                    if pos["direction"] == "UP":
                        pos["current_price"] = up_bid  # sell at bid
                        pos["current_mid"] = up_mid
                    else:
                        pos["current_price"] = dn_bid
                        pos["current_mid"] = dn_mid

                    pos["current_pnl_pct"] = (
                        (pos["current_price"] - pos["entry_price"])
                        / max(pos["entry_price"], 0.01) * 100
                    )
                    pos["peak_pnl_pct"] = max(pos["peak_pnl_pct"], pos["current_pnl_pct"])
                    pos["sol_price"] = sol_price

                    exit_signal = self._check_exit(pos, features, market_state, strategy)
                    if exit_signal:
                        positions_to_close.append((pos, exit_signal))

                for pos, exit_signal in positions_to_close:
                    self._close_position(pos, exit_signal, ts)
                    open_positions.remove(pos)

                # ── CHECK ENTRIES ──
                if strategy and len(open_positions) < self.max_open_positions:
                    if time_remaining_sec > 30:  # don't enter in last 30s
                        signal = strategy.evaluate(features, market_state)
                        if signal:
                            if signal["direction"] == "UP":
                                entry_price = up_ask  # buy at ask
                                entry_mid = up_mid
                            else:
                                entry_price = dn_ask
                                entry_mid = dn_mid

                            if (entry_price <= self.max_share_price
                                    and entry_price > 0.01
                                    and self.capital >= self.max_position_size):
                                pos = self._open_position(
                                    signal, entry_price, entry_mid, ts, slug, ptb,
                                    sol_price, time_remaining_pct, dur_min, snap,
                                )
                                open_positions.append(pos)

            # ── MARKET EXPIRY ──
            last_snap = market_df.iloc[-1]
            final_sol = last_snap.get("sol_price", 0)
            outcome = last_snap.get("outcome")

            # Try to determine outcome from SOL price vs PTB
            if not outcome and ptb and final_sol:
                outcome = "Up" if final_sol >= ptb else "Down"

            positions_to_resolve = [p for p in open_positions if p["slug"] == slug]
            for pos in positions_to_resolve:
                if outcome:
                    pos_won = (
                        (pos["direction"] == "UP" and outcome in ("Up", "UP")) or
                        (pos["direction"] == "DOWN" and outcome in ("Down", "DOWN"))
                    )
                    pos["current_price"] = 1.0 if pos_won else 0.0
                    exit_reason = "expiry_win" if pos_won else "expiry_loss"
                else:
                    # Unknown outcome — use last known bid
                    exit_reason = "expiry_unknown"

                self._close_position(
                    pos, {"reason": exit_reason},
                    last_snap["ts"],
                )
                open_positions.remove(pos)

            # Equity checkpoint
            self.equity_curve.append({
                "ts": last_snap["ts"],
                "equity": self.capital,
                "market_idx": market_idx,
            })

            if verbose and market_idx % max(1, n_markets // 10) == 0:
                log.info(
                    f"  Market {market_idx}/{n_markets}: "
                    f"equity=${self.capital:.2f} trades={len(self.trades)}"
                )

        # Final results
        results = self._compute_results()
        if verbose:
            self._print_results(results)
        return results

    # ═══════════════════════════════════════════════════════════
    #  POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def _open_position(
        self, signal, entry_price, entry_mid, ts, slug, ptb,
        sol_price, time_remaining_pct, dur_min, snap,
    ) -> Dict:
        size_usd = min(self.max_position_size, self.capital)
        shares = size_usd / entry_price
        self.capital -= size_usd
        slippage = entry_price - entry_mid  # positive = paid more than mid

        return {
            "slug": slug,
            "direction": signal["direction"],
            "entry_price": entry_price,
            "entry_mid": entry_mid,
            "shares": shares,
            "size_usd": size_usd,
            "entry_ts": ts,
            "confidence": signal.get("confidence", 0),
            "price_to_beat": ptb,
            "sol_price_at_entry": sol_price,
            "time_remaining_at_entry_pct": time_remaining_pct,
            "duration_minutes": dur_min,
            "slippage_entry": slippage,
            "current_price": entry_price,
            "current_mid": entry_mid,
            "current_pnl_pct": 0,
            "peak_pnl_pct": 0,
            "sol_price": sol_price,
            # Record snapshot features at entry
            "up_mid_at_entry": snap.get("up_mid_price", 0.5),
            "up_spread_at_entry": snap.get("up_spread", 0),
            "liquidity_at_entry": snap.get("liquidity_score", 0),
            "momentum_30s_at_entry": snap.get("shares_momentum_30s", 0),
        }

    def _close_position(self, pos: Dict, exit_signal: Dict, exit_ts: float):
        exit_price = pos["current_price"]
        pnl_pct = (exit_price - pos["entry_price"]) / max(pos["entry_price"], 0.01) * 100
        pnl_usd = pos["shares"] * (exit_price - pos["entry_price"])
        slippage_exit = pos["current_mid"] - exit_price  # positive = got less than mid

        self.capital += pos["shares"] * exit_price

        trade = ReplayTrade(
            market_slug=pos["slug"],
            entry_ts=pos["entry_ts"],
            exit_ts=exit_ts,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            shares_count=pos["shares"],
            size_usd=pos["size_usd"],
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            slippage_entry=pos["slippage_entry"],
            slippage_exit=slippage_exit,
            exit_reason=exit_signal["reason"],
            confidence=pos["confidence"],
            hold_time_s=exit_ts - pos["entry_ts"],
            peak_pnl_pct=pos["peak_pnl_pct"],
            price_to_beat=pos.get("price_to_beat", 0) or 0,
            sol_price_at_entry=pos.get("sol_price_at_entry", 0),
            sol_price_at_exit=pos.get("sol_price", 0),
            duration_minutes=pos.get("duration_minutes", 0),
            time_remaining_at_entry_pct=pos.get("time_remaining_at_entry_pct", 0),
            up_mid_at_entry=pos.get("up_mid_at_entry", 0),
            up_spread_at_entry=pos.get("up_spread_at_entry", 0),
            liquidity_at_entry=pos.get("liquidity_at_entry", 0),
            momentum_30s_at_entry=pos.get("momentum_30s_at_entry", 0),
        )
        self.trades.append(trade)

    def _check_exit(self, pos, features, market_state, strategy) -> Optional[Dict]:
        pnl = pos["current_pnl_pct"]
        peak = pos["peak_pnl_pct"]
        time_left_sec = market_state.get("time_remaining_ms", 0) / 1000

        # Hard stop loss
        stop_loss = _shares_cfg.get("stop_loss_pct", -15)
        if pnl < stop_loss:
            return {"reason": "stop_loss"}

        # Dead share (price collapsed)
        dead_price = _shares_cfg.get("dead_share_price", 0.02)
        if pos["current_price"] < dead_price:
            return {"reason": "dead_share"}

        # Trailing stop
        trail_peak = _shares_cfg.get("trailing_stop_peak", 30)
        trail_drop = _shares_cfg.get("trailing_stop_drop", 0.60)
        if peak > trail_peak and pnl < peak * trail_drop:
            return {"reason": "trailing_stop"}

        # Expiry lock
        lock_pct = _shares_cfg.get("expiry_lock_pct", 10)
        if pnl > lock_pct and time_left_sec < 30:
            return {"reason": "expiry_lock"}

        # Strategy-driven exit
        if strategy and hasattr(strategy, "should_exit"):
            exit_sig = strategy.should_exit(features, market_state, pos)
            if exit_sig:
                return exit_sig

        return None

    # ═══════════════════════════════════════════════════════════
    #  DATA LOADING
    # ═══════════════════════════════════════════════════════════

    def _load_recorded_data(self, dates: List[str], duration_minutes: int) -> pd.DataFrame:
        """Load recorded snapshots from disk."""
        if not RECORD_DIR.exists():
            return pd.DataFrame()

        all_frames = []

        # Find all date directories
        if dates:
            date_dirs = [RECORD_DIR / d for d in dates if (RECORD_DIR / d).exists()]
        else:
            date_dirs = sorted(RECORD_DIR.iterdir())
            date_dirs = [d for d in date_dirs if d.is_dir() and d.name != "__pycache__"]

        for date_dir in date_dirs:
            dur_dir = date_dir / f"{duration_minutes}m"
            snap_file = dur_dir / "snapshots.parquet"
            if snap_file.exists():
                df = pd.read_parquet(snap_file)
                all_frames.append(df)
                log.info(f"  Loaded {len(df):,} snapshots from {snap_file}")

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.sort_values("ts").reset_index(drop=True)
        log.info(f"  Total: {len(combined):,} snapshots, {combined['slug'].nunique()} markets")
        return combined

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    def _empty_results(self) -> Dict:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "avg_pnl_pct": 0, "total_pnl_usd": 0, "final_equity": self.capital,
            "max_drawdown_pct": 0, "sharpe_ratio": 0, "profit_factor": 0,
            "expiry_wins": 0, "expiry_losses": 0, "early_exits": 0,
            "up_trades": 0, "down_trades": 0,
            "avg_entry_price": 0, "avg_hold_time_s": 0,
            "avg_slippage_entry": 0, "avg_slippage_exit": 0,
        }

    def _compute_results(self) -> Dict:
        if not self.trades:
            return self._empty_results()

        pnls = [t.pnl_pct for t in self.trades]
        pnl_usds = [t.pnl_usd for t in self.trades]
        wins = [t for t in self.trades if t.pnl_pct > 0]
        losses = [t for t in self.trades if t.pnl_pct <= 0]

        # Drawdown
        equity_vals = [e["equity"] for e in self.equity_curve] if self.equity_curve else [self.capital]
        peak_eq = equity_vals[0]
        max_dd = 0
        for eq in equity_vals:
            peak_eq = max(peak_eq, eq)
            dd = (eq - peak_eq) / peak_eq * 100 if peak_eq > 0 else 0
            max_dd = min(max_dd, dd)

        # Sharpe
        pnl_arr = np.array(pnls)
        sharpe = float(np.mean(pnl_arr) / max(np.std(pnl_arr), 1e-9)) * np.sqrt(252) if len(pnl_arr) > 1 else 0

        # Profit factor
        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        # Exit reasons
        expiry_wins = sum(1 for t in self.trades if t.exit_reason == "expiry_win")
        expiry_losses = sum(1 for t in self.trades if t.exit_reason == "expiry_loss")
        early_exits = sum(1 for t in self.trades if t.exit_reason not in ("expiry_win", "expiry_loss", "expiry_unknown"))

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(len(self.trades), 1) * 100,
            "avg_pnl_pct": float(np.mean(pnls)),
            "total_pnl_usd": sum(pnl_usds),
            "final_equity": self.capital,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "profit_factor": profit_factor,
            "expiry_wins": expiry_wins,
            "expiry_win_rate": expiry_wins / max(expiry_wins + expiry_losses, 1) * 100,
            "expiry_losses": expiry_losses,
            "early_exits": early_exits,
            "up_trades": sum(1 for t in self.trades if t.direction == "UP"),
            "down_trades": sum(1 for t in self.trades if t.direction == "DOWN"),
            "avg_entry_price": float(np.mean([t.entry_price for t in self.trades])),
            "avg_hold_time_s": float(np.mean([t.hold_time_s for t in self.trades])),
            "avg_slippage_entry": float(np.mean([t.slippage_entry for t in self.trades])),
            "avg_slippage_exit": float(np.mean([t.slippage_exit for t in self.trades])),
        }

    def _print_results(self, r: Dict):
        log.info("")
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║      REPLAY BACKTEST RESULTS (REAL DATA)        ║")
        log.info("╠══════════════════════════════════════════════════╣")
        log.info(f"║  Total Trades:        {r['total_trades']}")
        log.info(f"║  Win Rate:           {r['win_rate']:.1f}%")
        log.info(f"║  Avg PnL:          {r['avg_pnl_pct']:+.2f}%")
        log.info(f"║  Total PnL:        ${r['total_pnl_usd']:+.2f}")
        log.info(f"║  Final Equity:     ${r['final_equity']:8.2f}")
        log.info(f"║  Max Drawdown:      {r['max_drawdown_pct']:.1f}%")
        log.info(f"║  Sharpe Ratio:      {r['sharpe_ratio']:.2f}")
        log.info(f"║  Profit Factor:     {r['profit_factor']:.2f}")
        log.info(f"║  ──────────────────────────────────────────────")
        log.info(f"║  Expiry Wins:         {r['expiry_wins']}  ({r.get('expiry_win_rate', 0):.1f}%)")
        log.info(f"║  Expiry Losses:       {r['expiry_losses']}")
        log.info(f"║  Early Exits:         {r['early_exits']}")
        log.info(f"║  UP / DOWN:        {r['up_trades']} / {r['down_trades']}")
        log.info(f"║  Avg Entry Price:  ${r['avg_entry_price']:.3f}")
        log.info(f"║  Avg Hold Time:    {r['avg_hold_time_s']:.0f}s")
        log.info(f"║  Avg Slippage In:  ${r['avg_slippage_entry']:.4f}")
        log.info(f"║  Avg Slippage Out: ${r['avg_slippage_exit']:.4f}")
        log.info("╚══════════════════════════════════════════════════╝")

    # ═══════════════════════════════════════════════════════════
    #  DATAFRAMES
    # ═══════════════════════════════════════════════════════════

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([asdict(t) for t in self.trades])

    def get_equity_df(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)
