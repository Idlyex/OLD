"""Shares Backtester — simulates trading UP/DOWN shares on prediction markets.

Key differences from CEX backtester:
  - Markets have fixed lifetimes (5m, 15m, 1h)
  - New markets created every interval (aligned to wall clock)
  - Entry: buy UP or DOWN shares at current market price
  - Exit: sell shares early (at market price) OR hold to expiry (payout 1.0 or 0.0)
  - Slippage depends on shares liquidity, not SOL volatility
  - PnL = (exit_price - entry_price) / entry_price
  - Each market has PriceToBeat — known at entry

Simulation:
  For each historical market (from Polymarket data):
    1. Synthesize shares price curve from SOL price data + market metadata
    2. Evaluate strategy at each timestep
    3. If entry signal → buy shares
    4. Monitor position: early exit on signal OR hold to expiry
    5. At expiry: shares worth 1.0 (correct side) or 0.0 (wrong side)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.utils.logger import log
from core.features.shares import compute_shares_features
from config import config

_bt_cfg = config.get("backtester", {})
_slippage_cfg = _bt_cfg.get("slippage", {})


@dataclass
class SharesTrade:
    """Record of a single shares trade."""
    market_slug: str = ""
    entry_ts: int = 0
    exit_ts: int = 0
    direction: str = "UP"  # UP or DOWN
    entry_price: float = 0.0  # shares price at entry (0.01 - 0.99)
    exit_price: float = 0.0   # shares price at exit (or 0/1 at expiry)
    shares_count: float = 0.0
    size_usd: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    slippage_bps: float = 0.0
    exit_reason: str = ""     # "early_exit", "expiry_win", "expiry_loss", "stop_loss"
    confidence: float = 0.0
    hold_time_s: float = 0.0
    peak_pnl_pct: float = 0.0
    price_to_beat: float = 0.0
    sol_price_at_entry: float = 0.0
    sol_price_at_exit: float = 0.0
    time_remaining_at_entry_pct: float = 0.0
    duration_minutes: int = 15
    features_at_entry: dict = field(default_factory=dict)


class SharesSlippageModel:
    """Slippage model for shares markets — depends on liquidity and spread."""

    def __init__(self):
        self.base_bps = _slippage_cfg.get("base_bps", 1.0)

    def compute(self, size_usd: float, spread: float, liquidity: float) -> float:
        """Compute slippage in price units (not bps).

        On thin markets, slippage = half-spread + impact.
        On liquid markets, slippage ≈ 0.
        """
        # Half spread component
        half_spread = spread / 2

        # Impact: sqrt model, normalized by liquidity
        if liquidity > 0:
            impact = 0.001 * np.sqrt(size_usd / max(liquidity, 1.0))
        else:
            impact = 0.005  # default impact for unknown liquidity

        return half_spread + impact


class SharesBacktester:
    """Backtester for prediction market shares trading.

    Inputs:
      - sol_data: DataFrame with 1m SOL OHLCV (from Binance)
      - markets_data: DataFrame of historical Polymarket markets
      - strategy: callable(features, market_state) → signal or None

    The backtester:
      1. Iterates through time, creating simulated markets at each interval
      2. For each market, synthesizes UP/DOWN shares prices from SOL price
      3. Calls strategy at each bar within the market's lifetime
      4. Manages positions (entry, early exit, hold to expiry)
    """

    def __init__(self, initial_capital: float = None):
        self.capital = initial_capital or _bt_cfg.get("initial_capital", 100.0)
        self.initial_capital = self.capital
        self.trades: List[SharesTrade] = []
        self.equity_curve: List[Dict] = []
        self.slippage_model = SharesSlippageModel()
        self.max_position_size = config.get("trading", {}).get("order_size_usd", 2.0)
        self.max_open_positions = config.get("trading", {}).get("max_open_positions", 3)
        self.max_share_price = config.get("shares", {}).get("max_share_price", 0.55)

    def run(
        self,
        sol_data: pd.DataFrame,
        markets_data: pd.DataFrame = None,
        strategy=None,
        duration_minutes: int = 15,
        verbose: bool = True,
    ) -> Dict:
        """Run the shares backtester.

        If markets_data is None, synthesizes markets from sol_data every `duration_minutes`.

        Args:
            sol_data: DataFrame with columns [ts, open, high, low, close, volume, ...]
            markets_data: Optional DataFrame of historical Polymarket markets
            strategy: Strategy object with .evaluate(features, market_state) method
            duration_minutes: Market lifetime in minutes
            verbose: Log progress

        Returns:
            Results dict with metrics
        """
        if sol_data.empty:
            log.error("No SOL data to backtest")
            return self._empty_results()

        self.trades = []
        self.equity_curve = []
        self.capital = self.initial_capital

        # Ensure sorted by timestamp
        sol_data = sol_data.sort_values("ts").reset_index(drop=True)

        # Generate simulated markets if no real market data
        if markets_data is None or markets_data.empty:
            markets = self._synthesize_markets(sol_data, duration_minutes)
        else:
            markets = self._prepare_markets(markets_data, sol_data)

        if not markets:
            log.warning("No markets to backtest")
            return self._empty_results()

        log.info(f"📊 Backtesting {len(markets)} markets ({duration_minutes}m) | capital=${self.capital:.2f}")

        # Simulate each market
        open_positions: List[Dict] = []
        total_entries = 0
        total_exits = 0

        for market_idx, market in enumerate(markets):
            market_bars = market["bars"]  # List of bar dicts within this market
            ptb = market["price_to_beat"]
            market_slug = market.get("slug", f"market_{market_idx}")
            dur_min = market.get("duration_minutes", duration_minutes)
            market_start_ts = market["start_ts"]
            market_end_ts = market["end_ts"]

            for bar_idx, bar in enumerate(market_bars):
                ts = bar["ts"]
                sol_price = bar["close"]
                time_elapsed_ms = (ts - market_start_ts)
                time_remaining_ms = max(0, market_end_ts - ts)
                total_ms = time_elapsed_ms + time_remaining_ms

                # Synthesize shares prices from SOL price dynamics
                yes_price, no_price = self._estimate_shares_price(
                    sol_price, ptb, time_remaining_ms, total_ms,
                    bar.get("volatility", 0.003),
                )

                # Update open positions
                for pos in open_positions:
                    if pos["market_slug"] == market_slug:
                        if pos["direction"] == "UP":
                            pos["current_price"] = yes_price
                        else:
                            pos["current_price"] = no_price
                        pos["current_pnl_pct"] = (pos["current_price"] - pos["entry_price"]) / max(pos["entry_price"], 0.01) * 100
                        pos["peak_pnl_pct"] = max(pos["peak_pnl_pct"], pos["current_pnl_pct"])
                        pos["sol_price"] = sol_price
                        pos["time_remaining_ms"] = time_remaining_ms

                # Build market state for strategy
                market_state = {
                    "slug": market_slug,
                    "sol_price": sol_price,
                    "price_to_beat": ptb,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "time_remaining_ms": time_remaining_ms,
                    "time_elapsed_ms": time_elapsed_ms,
                    "duration_minutes": dur_min,
                    "bar": bar,
                    "open_positions": [p for p in open_positions if p["market_slug"] == market_slug],
                }

                # Compute shares features
                shares_feat = compute_shares_features(
                    sol_price=sol_price,
                    price_to_beat=ptb,
                    yes_price=yes_price,
                    no_price=no_price,
                    time_remaining_ms=time_remaining_ms,
                    time_elapsed_ms=time_elapsed_ms,
                    duration_minutes=dur_min,
                    sol_volatility=bar.get("volatility", 0.003),
                )

                # Merge with CEX features from bar
                all_features = {**bar.get("features", {}), **shares_feat}

                # ── CHECK EXITS ──
                positions_to_close = []
                for pos in open_positions:
                    if pos["market_slug"] != market_slug:
                        continue
                    exit_signal = self._check_exit(pos, all_features, market_state, strategy)
                    if exit_signal:
                        positions_to_close.append((pos, exit_signal))

                for pos, exit_signal in positions_to_close:
                    self._close_position(pos, exit_signal, ts)
                    open_positions.remove(pos)
                    total_exits += 1

                # ── CHECK ENTRIES ──
                if strategy and len(open_positions) < self.max_open_positions:
                    # Don't enter in last 30s
                    if time_remaining_ms > 30_000:
                        signal = strategy.evaluate(all_features, market_state)
                        if signal:
                            entry_price = yes_price if signal["direction"] == "UP" else no_price
                            if entry_price <= self.max_share_price and self.capital >= self.max_position_size:
                                pos = self._open_position(
                                    signal, entry_price, ts, market_slug, ptb,
                                    sol_price, time_remaining_ms / max(total_ms, 1),
                                    dur_min, all_features,
                                )
                                open_positions.append(pos)
                                total_entries += 1

            # ── MARKET EXPIRY — resolve all positions ──
            final_sol = market_bars[-1]["close"] if market_bars else ptb
            outcome = "UP" if final_sol >= ptb else "DOWN"

            positions_to_resolve = [p for p in open_positions if p["market_slug"] == market_slug]
            for pos in positions_to_resolve:
                if pos["direction"] == outcome:
                    pos["current_price"] = 1.0
                    exit_reason = "expiry_win"
                else:
                    pos["current_price"] = 0.0
                    exit_reason = "expiry_loss"

                self._close_position(pos, {"reason": exit_reason}, market_bars[-1]["ts"] if market_bars else market_end_ts)
                open_positions.remove(pos)
                total_exits += 1

            # Track equity after each market
            self.equity_curve.append({
                "ts": market_end_ts,
                "equity": self.capital,
                "market_idx": market_idx,
            })

            if verbose and (market_idx + 1) % 50 == 0:
                log.info(f"  Market {market_idx+1}/{len(markets)}: equity=${self.capital:.2f} trades={len(self.trades)}")

        results = self._compute_results()
        if verbose:
            self._print_results(results)
        return results

    # ─── Position Management ─────────────────────────────────

    def _open_position(
        self, signal: Dict, entry_price: float, ts: int,
        market_slug: str, ptb: float, sol_price: float,
        time_remaining_pct: float, dur_min: int, features: Dict,
    ) -> Dict:
        size_usd = min(self.max_position_size, self.capital * 0.1)
        slippage = self.slippage_model.compute(size_usd, 0.02, 100)
        adjusted_price = entry_price + slippage
        shares = size_usd / adjusted_price

        self.capital -= size_usd

        return {
            "market_slug": market_slug,
            "direction": signal["direction"],
            "entry_price": adjusted_price,
            "entry_ts": ts,
            "shares": shares,
            "size_usd": size_usd,
            "current_price": entry_price,
            "current_pnl_pct": 0.0,
            "peak_pnl_pct": 0.0,
            "confidence": signal.get("confidence", 0),
            "price_to_beat": ptb,
            "sol_price_at_entry": sol_price,
            "sol_price": sol_price,
            "time_remaining_at_entry_pct": time_remaining_pct,
            "duration_minutes": dur_min,
            "features_at_entry": {k: float(v) for k, v in list(features.items())[:20]},
            "slippage_bps": slippage * 10000,
            "time_remaining_ms": 0,
        }

    def _close_position(self, pos: Dict, exit_signal: Dict, exit_ts: int):
        exit_price = pos["current_price"]
        slippage = self.slippage_model.compute(pos["size_usd"], 0.02, 100)

        # Don't apply slippage on expiry resolution (settled, not traded)
        if exit_signal["reason"] not in ("expiry_win", "expiry_loss"):
            exit_price = max(0.01, exit_price - slippage)

        pnl_pct = (exit_price - pos["entry_price"]) / max(pos["entry_price"], 0.01) * 100
        pnl_usd = pos["shares"] * (exit_price - pos["entry_price"])

        self.capital += pos["shares"] * exit_price

        trade = SharesTrade(
            market_slug=pos["market_slug"],
            entry_ts=pos["entry_ts"],
            exit_ts=exit_ts,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=exit_price,
            shares_count=pos["shares"],
            size_usd=pos["size_usd"],
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            slippage_bps=pos["slippage_bps"],
            exit_reason=exit_signal["reason"],
            confidence=pos["confidence"],
            hold_time_s=(exit_ts - pos["entry_ts"]) / 1000,
            peak_pnl_pct=pos["peak_pnl_pct"],
            price_to_beat=pos["price_to_beat"],
            sol_price_at_entry=pos["sol_price_at_entry"],
            sol_price_at_exit=pos.get("sol_price", 0),
            time_remaining_at_entry_pct=pos["time_remaining_at_entry_pct"],
            duration_minutes=pos["duration_minutes"],
            features_at_entry=pos["features_at_entry"],
        )
        self.trades.append(trade)

    def _check_exit(self, pos: Dict, features: Dict, market_state: Dict, strategy) -> Optional[Dict]:
        """Check if position should be closed early."""
        pnl = pos["current_pnl_pct"]
        peak = pos["peak_pnl_pct"]
        time_left_ms = pos.get("time_remaining_ms", 0)

        # Hard stop loss: -15%
        if pnl < -15:
            return {"reason": "stop_loss"}

        # Dead shares: price < 0.02
        if pos["current_price"] < 0.02:
            return {"reason": "dead_share"}

        # Trailing stop: if peak > 30% and dropped 40% from peak
        if peak > 30 and pnl < peak * 0.6:
            return {"reason": "trailing_stop"}

        # Profit lock near expiry: if pnl > 10% and < 30s left
        if pnl > 10 and time_left_ms < 30_000:
            return {"reason": "expiry_lock"}

        # Strategy exit signal
        if strategy and hasattr(strategy, "should_exit"):
            exit_sig = strategy.should_exit(features, market_state, pos)
            if exit_sig:
                return {"reason": f"strategy_{exit_sig.get('reason', 'exit')}"}

        return None

    # ─── Market Synthesis ────────────────────────────────────

    def _synthesize_markets(self, sol_data: pd.DataFrame, duration_minutes: int) -> List[Dict]:
        """Create simulated shares markets from SOL price data.

        Every `duration_minutes` minutes, a new market opens with:
          - PriceToBeat = SOL price at market open
          - Duration = duration_minutes
          - Shares prices estimated from SOL price dynamics
        """
        interval_ms = duration_minutes * 60_000
        ts_col = sol_data["ts"].values
        start_ts = int(ts_col[0])
        end_ts = int(ts_col[-1])

        # Align to interval
        first_market_start = (start_ts // interval_ms + 1) * interval_ms

        markets = []
        current_start = first_market_start

        while current_start + interval_ms <= end_ts:
            market_end = current_start + interval_ms

            # Get bars within this market window
            mask = (ts_col >= current_start) & (ts_col < market_end)
            bar_indices = np.where(mask)[0]

            if len(bar_indices) < 2:
                current_start += interval_ms
                continue

            # PriceToBeat = close at market open
            ptb = float(sol_data.iloc[bar_indices[0]]["close"])

            # Compute volatility for this window
            closes = sol_data.iloc[bar_indices]["close"].values
            returns = np.diff(np.log(closes + 1e-10))
            vol = float(np.std(returns)) if len(returns) > 1 else 0.003

            bars = []
            for idx in bar_indices:
                row = sol_data.iloc[idx]
                bars.append({
                    "ts": int(row["ts"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0)),
                    "volatility": vol,
                    "features": {},  # CEX features can be pre-computed
                })

            markets.append({
                "slug": f"sim-sol-{duration_minutes}m-{current_start}",
                "price_to_beat": ptb,
                "start_ts": current_start,
                "end_ts": market_end,
                "duration_minutes": duration_minutes,
                "bars": bars,
            })

            current_start += interval_ms

        return markets

    def _prepare_markets(self, markets_data: pd.DataFrame, sol_data: pd.DataFrame) -> List[Dict]:
        """Convert real Polymarket market history + SOL data into backtestable format."""
        markets = []
        ts_col = sol_data["ts"].values

        for _, row in markets_data.iterrows():
            start_ts = int(row.get("start_ts", 0)) * 1000  # convert to ms
            dur_min = int(row.get("duration_minutes", 15))
            end_ts = start_ts + dur_min * 60_000
            ptb = float(row.get("price_to_beat", 0))

            if ptb == 0 or start_ts == 0:
                continue

            # Find SOL bars in this window
            mask = (ts_col >= start_ts) & (ts_col < end_ts)
            bar_indices = np.where(mask)[0]

            if len(bar_indices) < 2:
                continue

            closes = sol_data.iloc[bar_indices]["close"].values
            returns = np.diff(np.log(closes + 1e-10))
            vol = float(np.std(returns)) if len(returns) > 1 else 0.003

            bars = []
            for idx in bar_indices:
                r = sol_data.iloc[idx]
                bars.append({
                    "ts": int(r["ts"]),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r.get("volume", 0)),
                    "volatility": vol,
                    "features": {},
                })

            markets.append({
                "slug": row.get("slug", f"pm-{start_ts}"),
                "price_to_beat": ptb,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "duration_minutes": dur_min,
                "bars": bars,
            })

        return markets

    @staticmethod
    def _estimate_shares_price(
        sol_price: float,
        price_to_beat: float,
        time_remaining_ms: int,
        total_ms: int,
        volatility: float = 0.003,
    ) -> Tuple[float, float]:
        """Estimate UP/DOWN shares prices from SOL price dynamics.

        Uses simplified Black-Scholes-like model with market noise:
          P(UP) ≈ Φ(d) where d = (sol - ptb) / (vol * sqrt(T))
          + random noise ±3% to simulate market inefficiency

        Near expiry (T→0), prices converge to 0 or 1.
        """
        from scipy.stats import norm

        if price_to_beat <= 0 or total_ms <= 0:
            return 0.5, 0.5

        # Distance from PTB in %
        distance = (sol_price - price_to_beat) / price_to_beat

        # Time to expiry in minutes
        t_min = max(time_remaining_ms / 60_000, 0.01)

        # d-score: distance normalized by vol * sqrt(time)
        vol_adj = max(volatility, 0.001) * np.sqrt(t_min)
        d = distance / vol_adj

        # CDF gives probability
        up_prob = float(norm.cdf(d))

        # Add market noise — real markets are not perfectly efficient
        # Noise decreases near expiry (prices converge to 0/1)
        time_pct = time_remaining_ms / max(total_ms, 1)
        noise_scale = 0.05 * time_pct  # up to ±5% noise early, less near expiry
        # Deterministic noise based on price (reproducible)
        noise = np.sin(sol_price * 1000 + time_remaining_ms * 0.001) * noise_scale
        up_prob = up_prob + noise

        up_prob = float(np.clip(up_prob, 0.02, 0.98))

        yes_price = up_prob
        no_price = 1.0 - up_prob

        return round(yes_price, 4), round(no_price, 4)

    # ─── Results ─────────────────────────────────────────────

    def _compute_results(self) -> Dict:
        if not self.trades:
            return self._empty_results()

        wins = [t for t in self.trades if t.pnl_pct > 0]
        losses = [t for t in self.trades if t.pnl_pct <= 0]
        pnls = [t.pnl_pct for t in self.trades]
        pnl_usds = [t.pnl_usd for t in self.trades]

        expiry_wins = [t for t in self.trades if t.exit_reason == "expiry_win"]
        expiry_losses = [t for t in self.trades if t.exit_reason == "expiry_loss"]
        early_exits = [t for t in self.trades if t.exit_reason not in ("expiry_win", "expiry_loss")]

        # Equity curve for drawdown
        equity = [self.initial_capital]
        for t in self.trades:
            equity.append(equity[-1] + t.pnl_usd)

        peak_equity = np.maximum.accumulate(equity)
        drawdowns = (np.array(equity) - peak_equity) / np.maximum(peak_equity, 1) * 100

        # Sharpe (daily)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / max(np.std(pnls), 1e-6) * np.sqrt(252)
        else:
            sharpe = 0

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(len(self.trades), 1) * 100,
            "total_pnl_pct": sum(pnls),
            "total_pnl_usd": sum(pnl_usds),
            "avg_pnl_pct": np.mean(pnls),
            "median_pnl_pct": np.median(pnls),
            "max_win_pct": max(pnls) if pnls else 0,
            "max_loss_pct": min(pnls) if pnls else 0,
            "avg_win_pct": np.mean([t.pnl_pct for t in wins]) if wins else 0,
            "avg_loss_pct": np.mean([t.pnl_pct for t in losses]) if losses else 0,
            "final_equity": self.capital,
            "max_drawdown_pct": float(np.min(drawdowns)),
            "sharpe_ratio": sharpe,
            "profit_factor": abs(sum(t.pnl_usd for t in wins)) / max(abs(sum(t.pnl_usd for t in losses)), 0.01),
            "expiry_wins": len(expiry_wins),
            "expiry_losses": len(expiry_losses),
            "early_exits": len(early_exits),
            "expiry_win_rate": len(expiry_wins) / max(len(expiry_wins) + len(expiry_losses), 1) * 100,
            "avg_hold_time_s": np.mean([t.hold_time_s for t in self.trades]),
            "avg_entry_price": np.mean([t.entry_price for t in self.trades]),
            "avg_confidence": np.mean([t.confidence for t in self.trades]),
            "up_trades": len([t for t in self.trades if t.direction == "UP"]),
            "down_trades": len([t for t in self.trades if t.direction == "DOWN"]),
        }

    def _empty_results(self) -> Dict:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl_pct": 0, "total_pnl_usd": 0, "avg_pnl_pct": 0,
            "median_pnl_pct": 0, "max_win_pct": 0, "max_loss_pct": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0,
            "final_equity": self.capital, "max_drawdown_pct": 0,
            "sharpe_ratio": 0, "profit_factor": 0,
            "expiry_wins": 0, "expiry_losses": 0, "early_exits": 0,
            "expiry_win_rate": 0, "avg_hold_time_s": 0,
            "avg_entry_price": 0, "avg_confidence": 0,
            "up_trades": 0, "down_trades": 0,
        }

    def _print_results(self, r: Dict):
        log.info("")
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║          SHARES BACKTEST RESULTS                ║")
        log.info("╠══════════════════════════════════════════════════╣")
        log.info(f"║  Total Trades:     {r['total_trades']:>6}")
        log.info(f"║  Win Rate:         {r['win_rate']:>6.1f}%")
        log.info(f"║  Avg PnL:          {r['avg_pnl_pct']:>+6.2f}%")
        log.info(f"║  Total PnL:        ${r['total_pnl_usd']:>+6.2f}")
        log.info(f"║  Final Equity:     ${r['final_equity']:>8.2f}")
        log.info(f"║  Max Drawdown:     {r['max_drawdown_pct']:>6.1f}%")
        log.info(f"║  Sharpe Ratio:     {r['sharpe_ratio']:>6.2f}")
        log.info(f"║  Profit Factor:    {r['profit_factor']:>6.2f}")
        log.info("║  ─────────────────────────────────────────────")
        log.info(f"║  Expiry Wins:      {r['expiry_wins']:>6}  ({r['expiry_win_rate']:.1f}%)")
        log.info(f"║  Expiry Losses:    {r['expiry_losses']:>6}")
        log.info(f"║  Early Exits:      {r['early_exits']:>6}")
        log.info(f"║  UP / DOWN:        {r['up_trades']} / {r['down_trades']}")
        log.info(f"║  Avg Entry Price:  ${r['avg_entry_price']:.3f}")
        log.info(f"║  Avg Hold Time:    {r['avg_hold_time_s']:.0f}s")
        log.info("╚══════════════════════════════════════════════════╝")

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {k: v for k, v in t.__dict__.items() if k != "features_at_entry"}
            for t in self.trades
        ])

    def get_equity_df(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)
