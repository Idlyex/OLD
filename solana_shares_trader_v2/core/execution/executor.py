"""Execution Engine — manages positions, entry/exit decisions, order placement.
Coordinates between models, risk manager, and exchange client.
"""

import asyncio
import time
from typing import Dict, Optional, List, Any
from collections import OrderedDict
from dataclasses import dataclass, field

from core.utils.logger import log
from core.utils.helpers import safe_div, ts_now_ms, format_pnl
from core.risk.manager import RiskManager
from config import config

_trading_cfg = config.get("trading", {})


@dataclass
class Position:
    """Live position state."""
    pos_key: str
    slug: str
    token_id: str
    yes_token_id: str
    direction: str  # UP / DOWN
    entry_price: float
    shares: float
    cost: float
    entry_ts: int
    end_date: str
    condition_id: str
    price_to_beat: Optional[float] = None

    # State
    current_price: float = 0.0
    peak_pnl_pct: float = 0.0
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    age_s: float = 0.0

    # Model predictions at entry
    confidence: float = 0.0
    expected_return: float = 0.0
    reversal_prob: float = 0.0
    optimal_hold_time: float = 60.0

    # Tracking
    snapshots: list = field(default_factory=list)
    hedge_active: bool = False
    exit_reason: str = ""


class ExecutionEngine:
    """Manages the full trade lifecycle: signal evaluation → entry → monitoring → exit."""

    def __init__(self, clob_client, risk_manager: RiskManager, model=None):
        self.clob = clob_client
        self.risk = risk_manager
        self.model = model
        self.positions: OrderedDict[str, Position] = OrderedDict()
        self.trade_history: List[Dict] = []

        # Config
        self.max_positions = _trading_cfg.get("max_open_positions", 3)
        self.dry_run = config.get("dry_run", True)
        self.order_size = _trading_cfg.get("order_size_usd", 2.0)
        self.max_share_price = _trading_cfg.get("max_share_price", 0.40)

        # Cooldowns
        self._entry_cooldowns: Dict[str, float] = {}
        self._cooldown_ms = 20000

        # Stats
        self._total_trades = 0
        self._wins = 0
        self._losses = 0
        self._total_pnl = 0.0

    async def evaluate_entry(
        self,
        signal: Dict[str, Any],
        features: Dict[str, float],
        feature_sequence: Any = None,
        capital: float = 100.0,
    ) -> Optional[Position]:
        """Evaluate a trading signal and potentially enter a position.

        Args:
            signal: Market signal with direction, slug, token IDs, etc.
            features: Current feature vector
            feature_sequence: Historical feature sequence for transformer
            capital: Available capital
        """
        direction = signal.get("direction")
        slug = signal.get("slug", "")
        if not direction or direction == "NEUTRAL":
            return None

        # Position limits
        if len(self.positions) >= self.max_positions:
            return None

        # Cooldown check
        cooldown_key = f"{slug}_{direction}"
        if time.time() * 1000 - self._entry_cooldowns.get(cooldown_key, 0) < self._cooldown_ms:
            return None

        # Price check
        share_price = signal.get("share_price", 0)
        if share_price <= 0.03 or share_price > self.max_share_price:
            return None

        # Model prediction
        prediction = {}
        if self.model:
            import numpy as np
            fv = np.array(list(features.values()), dtype=np.float64)
            seq = feature_sequence if feature_sequence is not None else None
            prediction = self.model.predict(fv, seq)

            if not prediction.get("should_take", False):
                return None

        # Risk-adjusted position size
        vol = features.get("vol_garman_klass", 0.003)
        regime = features.get("vol_regime", 1.0)
        confidence = prediction.get("confidence", signal.get("confidence", 0.5))

        position_size = self.risk.compute_position_size(
            capital=capital,
            confidence=confidence,
            win_prob=confidence,
            avg_win=0.10,
            avg_loss=0.05,
            current_vol=vol,
            regime=regime,
        )

        if position_size < self.order_size:
            position_size = self.order_size

        # Execute order
        token_id = signal.get("token_id", "")
        buy_result = await self.clob.buy_shares(
            token_id=token_id,
            price=share_price,
            amount_usd=position_size,
            condition_id=signal.get("condition_id", ""),
            dry_run=self.dry_run,
        )

        if not buy_result:
            return None

        actual_shares = buy_result.get("shares", 0)
        actual_price = buy_result.get("entry_price", share_price)

        if actual_shares < 0.5:
            return None

        # Create position
        pos_key = f"{slug}_{direction}_{len(self.positions)}"
        pos = Position(
            pos_key=pos_key,
            slug=slug,
            token_id=token_id,
            yes_token_id=signal.get("yes_token_id", token_id),
            direction=direction,
            entry_price=actual_price,
            shares=actual_shares,
            cost=actual_shares * actual_price,
            entry_ts=ts_now_ms(),
            end_date=signal.get("end_date", ""),
            condition_id=signal.get("condition_id", ""),
            price_to_beat=signal.get("price_to_beat"),
            confidence=confidence,
            expected_return=prediction.get("expected_return", 0),
            reversal_prob=prediction.get("reversal_prob", 0.5),
            optimal_hold_time=prediction.get("hold_time", 60),
        )

        self.positions[pos_key] = pos
        self._entry_cooldowns[cooldown_key] = time.time() * 1000
        self._total_trades += 1

        sym = slug.split("-")[0].upper()
        log.info(
            f"💰 ENTRY | {direction} {sym} | {actual_shares:.1f}sh @ ${actual_price:.4f} "
            f"(${pos.cost:.2f}) conf={confidence:.2f}"
        )

        return pos

    async def monitor_positions(
        self,
        price_getter,
        features_getter=None,
    ):
        """Monitor all open positions and evaluate exits.

        Args:
            price_getter: callable(token_id) -> current_price
            features_getter: callable(symbol) -> features dict
        """
        now = ts_now_ms()
        to_close = []

        for pos_key, pos in self.positions.items():
            current_price = price_getter(pos.token_id) if price_getter else pos.entry_price
            if current_price <= 0:
                current_price = pos.entry_price

            pos.current_price = current_price
            pos.age_s = (now - pos.entry_ts) / 1000

            pnl = current_price - pos.entry_price
            pos.pnl_pct = safe_div(pnl, pos.entry_price) * 100
            pos.pnl_usd = pnl * pos.shares

            if pos.pnl_pct > pos.peak_pnl_pct:
                pos.peak_pnl_pct = pos.pnl_pct

            # ── Dynamic volatility stop ──
            vol = 0.003  # default
            if features_getter:
                sym = pos.slug.split("-")[0].upper() + "USDT"
                feat = features_getter(sym)
                if feat:
                    vol = feat.get("vol_garman_klass", 0.003)

            dynamic_stop = self.risk.compute_dynamic_stop(vol)

            if pos.pnl_pct <= -dynamic_stop:
                pos.exit_reason = f"dynamic_stop_{dynamic_stop:.1f}pct"
                to_close.append(pos_key)
                continue

            # ── Drawdown exit ──
            exit_model_prob = 0.0
            if self.model and features_getter:
                import numpy as np
                sym = pos.slug.split("-")[0].upper() + "USDT"
                feat = features_getter(sym)
                if feat:
                    fv = np.array(list(feat.values()), dtype=np.float64)
                    pos_state = np.array([
                        pos.pnl_pct / 100, pos.peak_pnl_pct / 100,
                        pos.age_s / 900, pos.reversal_prob, pos.confidence,
                    ])
                    exit_model_prob = self.model.predict_exit(fv, pos_state)

            if self.risk.should_exit_drawdown(pos.pnl_pct, pos.peak_pnl_pct, exit_model_prob):
                pos.exit_reason = f"dd_model_prob_{exit_model_prob:.2f}"
                to_close.append(pos_key)
                continue

            # ── Hedge check ──
            hedge = self.risk.should_hedge(pos.pnl_pct, pos_key)
            if hedge and not pos.hedge_active:
                pos.hedge_active = True
                log.info(f"🛡️ HEDGE: {pos_key} → {hedge['size_pct']:.0f}% counter")

            # ── Expiry ──
            if pos.end_date:
                from datetime import datetime, timezone
                try:
                    end_ms = int(datetime.fromisoformat(pos.end_date.replace("Z", "+00:00")).timestamp() * 1000)
                    if now > end_ms + 60_000:
                        pos.exit_reason = "expired"
                        to_close.append(pos_key)
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Model-based exit ──
            if exit_model_prob > 0.75 and pos.age_s > 30:
                pos.exit_reason = f"exit_model_{exit_model_prob:.2f}"
                to_close.append(pos_key)
                continue

            # ── Trailing stop on profits ──
            if pos.peak_pnl_pct > 30:
                trail = pos.peak_pnl_pct * 0.65
                if pos.pnl_pct < trail:
                    pos.exit_reason = f"trailing_{pos.peak_pnl_pct:.0f}pk"
                    to_close.append(pos_key)
                    continue

        # Execute closes
        for pos_key in to_close:
            await self._close_position(pos_key)

    async def _close_position(self, pos_key: str):
        """Close a position and record the trade."""
        pos = self.positions.get(pos_key)
        if not pos:
            return

        # Execute sell
        sell_result = await self.clob.sell_shares(
            token_id=pos.token_id,
            shares=pos.shares,
            min_price=0.01,
            dry_run=self.dry_run,
        )

        exit_price = pos.current_price
        if sell_result:
            exit_price = sell_result.get("exit_price", pos.current_price)

        # Record trade
        trade = {
            "slug": pos.slug,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "shares": pos.shares,
            "pnl_pct": pos.pnl_pct,
            "pnl_usd": pos.pnl_usd,
            "hold_time_s": pos.age_s,
            "exit_reason": pos.exit_reason,
            "confidence": pos.confidence,
            "peak_pnl_pct": pos.peak_pnl_pct,
            "entry_ts": pos.entry_ts,
            "exit_ts": ts_now_ms(),
        }
        self.trade_history.append(trade)

        if pos.pnl_pct > 0:
            self._wins += 1
        else:
            self._losses += 1
        self._total_pnl += pos.pnl_usd

        sym = pos.slug.split("-")[0].upper()
        icon = "✅" if pos.pnl_pct > 0 else "❌"
        log.info(
            f"{icon} EXIT | {pos.direction} {sym} | "
            f"${pos.entry_price:.4f}→${exit_price:.4f} "
            f"{format_pnl(pos.pnl_pct)} (${pos.pnl_usd:+.2f}) "
            f"| {pos.exit_reason} | {pos.age_s:.0f}s"
        )

        # Cleanup
        self.risk.clear_hedge(pos_key)
        del self.positions[pos_key]

    def get_stats(self) -> Dict:
        """Trading statistics."""
        total = self._wins + self._losses
        wr = safe_div(self._wins, total) * 100 if total > 0 else 0

        avg_win = 0.0
        avg_loss = 0.0
        if self.trade_history:
            wins = [t["pnl_pct"] for t in self.trade_history if t["pnl_pct"] > 0]
            losses = [t["pnl_pct"] for t in self.trade_history if t["pnl_pct"] <= 0]
            avg_win = float(sum(wins) / len(wins)) if wins else 0.0
            avg_loss = float(sum(losses) / len(losses)) if losses else 0.0

        return {
            "total_trades": self._total_trades,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": wr,
            "total_pnl": self._total_pnl,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "open_positions": len(self.positions),
        }
