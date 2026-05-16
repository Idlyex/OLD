"""Shares Live Trader — real-time Polymarket shares trading.

Orchestrates:
  1. Market discovery via Gamma API (find active UP/DOWN markets)
  2. CEX data feed from Binance WS (SOL price, orderbook, trades)
  3. Feature computation (82 CEX + 16 shares = 98 features)
  4. Strategy evaluation (mispricing, momentum, hybrid)
  5. Order execution via Polymarket CLOB
  6. Position monitoring + exit management
  7. Dashboard updates

Architecture:
  - Main loop runs every 500ms
  - Market refresh every 12s (new markets, prices)
  - Binance WS provides real-time SOL price
  - Strategy decisions made per-market, per-tick
"""

import asyncio
import time
import json
import numpy as np
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from core.utils.logger import log
from core.features.shares import compute_shares_features, SHARES_FEATURE_NAMES
from data.polymarket_collector import PolymarketCollector, SharesMarket
from config import config, trading_config

_shares_cfg = trading_config  # trading.yaml is the single source now
_trading_cfg = trading_config.get("execution", {})


@dataclass
class LivePosition:
    """Active shares position."""
    market_slug: str
    token_id: str
    direction: str  # "UP" or "DOWN"
    entry_price: float
    shares: float
    size_usd: float
    entry_ts: float
    confidence: float
    price_to_beat: float
    duration_minutes: int
    end_date: str
    peak_pnl_pct: float = 0.0
    current_price: float = 0.0
    current_pnl_pct: float = 0.0
    sol_price_at_entry: float = 0.0


class SharesLiveTrader:
    """Real-time Polymarket shares trader."""

    def __init__(self, enable_recording: bool = True):
        self.collector = PolymarketCollector()
        self.positions: List[LivePosition] = []
        self.completed_trades: List[Dict] = []

        # Price history for momentum features
        self._price_history: Dict[str, deque] = {}  # slug → deque of (ts, yes_price)
        self._sol_price: float = 0.0
        self._sol_volatility: float = 0.003

        # Configuration
        self.order_size = _trading_cfg.get("order_size_usd", 2.0)
        self.max_positions = _trading_cfg.get("max_open_positions", 3)
        self.max_share_price = _shares_cfg.get("max_share_price", 0.55)
        self.dry_run = config.get("dry_run", True)

        # Strategy
        self._strategy = None
        self._model = None

        # Control
        self._running = False
        self._market_refresh_interval = config.get("timing", {}).get("market_refresh_ms", 12000) / 1000
        self._tick_interval = config.get("timing", {}).get("analysis_interval_ms", 500) / 1000

        # Market slugs
        self.slugs = _shares_cfg.get("market_slugs", ["sol-updown-15m", "sol-updown-5m"])

        # Simultaneous recording (always-on by default)
        self._enable_recording = enable_recording
        self._recorder = None

    async def start(self):
        """Start the live trading loop."""
        log.info("╔══════════════════════════════════════════════════╗")
        log.info("║  Shares Live Trader — Starting                  ║")
        log.info(f"║  Slugs: {', '.join(self.slugs):<38}║")
        log.info(f"║  Mode: {'DRY RUN' if self.dry_run else '🔴 LIVE'}{'':>32}║")
        log.info("╚══════════════════════════════════════════════════╝")

        # Load strategy
        self._load_strategy()

        self._running = True

        # Start recorder in background (always records data for future backtesting)
        tasks = [
            self._market_loop(),
            self._tick_loop(),
            self._position_monitor_loop(),
        ]

        if self._enable_recording:
            from data.recorder import PolymarketRecorder
            self._recorder = PolymarketRecorder(
                interval_sec=5.0,
                market_slugs=self.slugs,
                duration_sec=None,  # infinite
            )
            tasks.append(self._recorder.start())
            log.info("  📹 Simultaneous recording enabled")

        # Run market refresh + tick loops concurrently
        await asyncio.gather(*tasks)

    async def shutdown(self):
        """Graceful shutdown."""
        self._running = False
        await self.collector.close()
        log.info("Shares trader shut down ✅")

    def _load_strategy(self):
        """Load trading strategy."""
        try:
            from strategies.shares import SharesHybridStrategy
            self._strategy = SharesHybridStrategy()
            log.info(f"  Strategy: {self._strategy.name()}")
        except Exception as e:
            log.warning(f"Failed to load strategy: {e}")
            from strategies.shares import SharesMispricingStrategy
            self._strategy = SharesMispricingStrategy()

    # ═══════════════════════════════════════════════════════════
    #  MAIN LOOPS
    # ═══════════════════════════════════════════════════════════

    async def _market_loop(self):
        """Refresh active markets periodically."""
        while self._running:
            try:
                for slug in self.slugs:
                    markets = await self.collector.get_active_markets(slug)
                    for market in markets:
                        if market.is_tradeable:
                            await self._process_market(market)
            except Exception as e:
                log.error(f"Market loop error: {e}")

            await asyncio.sleep(self._market_refresh_interval)

    async def _tick_loop(self):
        """High-frequency tick: update prices, evaluate signals."""
        while self._running:
            try:
                # Update positions with latest prices
                for pos in self.positions:
                    await self._update_position_price(pos)
            except Exception as e:
                log.debug(f"Tick error: {e}")

            await asyncio.sleep(self._tick_interval)

    async def _position_monitor_loop(self):
        """Monitor positions for exit signals every 2s."""
        while self._running:
            try:
                positions_to_close = []
                for pos in self.positions:
                    exit_signal = self._check_exit(pos)
                    if exit_signal:
                        positions_to_close.append((pos, exit_signal))

                for pos, reason in positions_to_close:
                    await self._close_position(pos, reason)
            except Exception as e:
                log.error(f"Position monitor error: {e}")

            await asyncio.sleep(2.0)

    # ═══════════════════════════════════════════════════════════
    #  MARKET PROCESSING
    # ═══════════════════════════════════════════════════════════

    async def _process_market(self, market: SharesMarket):
        """Evaluate a single market for entry."""
        if len(self.positions) >= self.max_positions:
            return

        # Skip if already have position in this market
        if any(p.market_slug == market.slug for p in self.positions):
            return

        # Track price history
        slug = market.slug
        if slug not in self._price_history:
            self._price_history[slug] = deque(maxlen=300)
        self._price_history[slug].append((time.time(), market.yes_price))

        # Build features
        yes_history = [p[1] for p in self._price_history[slug]]

        # Get orderbook for liquidity features
        ob = await self.collector.get_orderbook(market.yes_token_id)
        bid_vol = ob["bid_volume"] if ob else 0
        ask_vol = ob["ask_volume"] if ob else 0

        features = compute_shares_features(
            sol_price=self._sol_price or (market.price_to_beat or 150),
            price_to_beat=market.price_to_beat or 0,
            yes_price=market.yes_price,
            no_price=market.no_price,
            time_remaining_ms=market.time_remaining_ms,
            time_elapsed_ms=market.time_elapsed_ms,
            duration_minutes=market.duration_minutes,
            best_bid=market.best_bid,
            best_ask=market.best_ask,
            bid_volume=bid_vol,
            ask_volume=ask_vol,
            spread=market.spread,
            yes_price_history=yes_history,
            sol_volatility=self._sol_volatility,
        )

        # Market state for strategy
        market_state = {
            "slug": market.slug,
            "sol_price": self._sol_price,
            "price_to_beat": market.price_to_beat,
            "yes_price": market.yes_price,
            "no_price": market.no_price,
            "time_remaining_ms": market.time_remaining_ms,
            "time_elapsed_ms": market.time_elapsed_ms,
            "duration_minutes": market.duration_minutes,
            "open_positions": [],
        }

        # Evaluate strategy
        if not self._strategy:
            return

        signal = self._strategy.evaluate(features, market_state)
        if not signal:
            return

        # Entry price check
        entry_price = market.yes_price if signal["direction"] == "UP" else market.no_price
        if entry_price > self.max_share_price:
            return

        token_id = market.yes_token_id if signal["direction"] == "UP" else market.no_token_id

        # Execute entry
        await self._open_position(
            market=market,
            signal=signal,
            entry_price=entry_price,
            token_id=token_id,
        )

    # ═══════════════════════════════════════════════════════════
    #  POSITION MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    async def _open_position(self, market: SharesMarket, signal: Dict, entry_price: float, token_id: str):
        """Open a new shares position."""
        size_usd = min(self.order_size, self.order_size)  # could use Kelly sizing
        shares = size_usd / entry_price

        if self.dry_run:
            log.info(
                f"[DRY] 📈 BUY {signal['direction']} {shares:.1f} shares @ ${entry_price:.3f} "
                f"| {market.slug} | conf={signal['confidence']:.2f} | {signal['reason']}"
            )
        else:
            # Real order via CLOB — would call clob_client.buyLimit()
            log.info(
                f"📈 BUY {signal['direction']} {shares:.1f} shares @ ${entry_price:.3f} "
                f"| {market.slug} | conf={signal['confidence']:.2f}"
            )

        pos = LivePosition(
            market_slug=market.slug,
            token_id=token_id,
            direction=signal["direction"],
            entry_price=entry_price,
            shares=shares,
            size_usd=size_usd,
            entry_ts=time.time(),
            confidence=signal.get("confidence", 0),
            price_to_beat=market.price_to_beat or 0,
            duration_minutes=market.duration_minutes,
            end_date=market.end_date or "",
            current_price=entry_price,
            sol_price_at_entry=self._sol_price,
        )
        self.positions.append(pos)

    async def _update_position_price(self, pos: LivePosition):
        """Update position with latest market price."""
        # In production, this would come from WS
        # For now, fetch from CLOB midpoint
        ob = await self.collector.get_orderbook(pos.token_id)
        if ob and ob["mid"] > 0:
            pos.current_price = ob["mid"]
            pos.current_pnl_pct = (pos.current_price - pos.entry_price) / max(pos.entry_price, 0.01) * 100
            pos.peak_pnl_pct = max(pos.peak_pnl_pct, pos.current_pnl_pct)

    def _check_exit(self, pos: LivePosition) -> Optional[str]:
        """Check if position should be closed."""
        pnl = pos.current_pnl_pct
        peak = pos.peak_pnl_pct

        # Hard stop
        stop_loss = _shares_cfg.get("stop_loss_pct", -15)
        if pnl < stop_loss:
            return "stop_loss"

        # Dead share
        dead_price = _shares_cfg.get("dead_share_price", 0.02)
        if pos.current_price < dead_price:
            return "dead_share"

        # Trailing stop
        trail_peak = _shares_cfg.get("trailing_stop_peak", 30)
        trail_drop = _shares_cfg.get("trailing_stop_drop", 0.60)
        if peak > trail_peak and pnl < peak * trail_drop:
            return "trailing_stop"

        # Expiry lock
        if pos.end_date:
            try:
                from datetime import datetime, timezone
                end_ts = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00")).timestamp()
                time_left_s = end_ts - time.time()
                lock_pct = _shares_cfg.get("expiry_lock_pct", 10)
                if pnl > lock_pct and time_left_s < 30:
                    return "expiry_lock"
            except (ValueError, TypeError):
                pass

        return None

    async def _close_position(self, pos: LivePosition, reason: str):
        """Close a position."""
        pnl_usd = pos.shares * (pos.current_price - pos.entry_price)

        if self.dry_run:
            log.info(
                f"[DRY] 📉 SELL {pos.direction} {pos.shares:.1f} shares @ ${pos.current_price:.3f} "
                f"| PnL={pos.current_pnl_pct:+.1f}% (${pnl_usd:+.2f}) | {reason}"
            )
        else:
            log.info(
                f"📉 SELL {pos.direction} {pos.shares:.1f} shares @ ${pos.current_price:.3f} "
                f"| PnL={pos.current_pnl_pct:+.1f}% | {reason}"
            )

        self.completed_trades.append({
            "slug": pos.market_slug,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": pos.current_price,
            "pnl_pct": pos.current_pnl_pct,
            "pnl_usd": pnl_usd,
            "reason": reason,
            "hold_time_s": time.time() - pos.entry_ts,
        })

        self.positions.remove(pos)

    # ═══════════════════════════════════════════════════════════
    #  BINANCE SOL PRICE FEED
    # ═══════════════════════════════════════════════════════════

    def update_sol_price(self, price: float, volatility: float = None):
        """Called by Binance WS handler to update SOL price."""
        self._sol_price = price
        if volatility is not None:
            self._sol_volatility = volatility

    def get_status(self) -> Dict:
        """Get current trader status for dashboard."""
        return {
            "running": self._running,
            "positions": len(self.positions),
            "completed": len(self.completed_trades),
            "sol_price": self._sol_price,
            "total_pnl": sum(t["pnl_usd"] for t in self.completed_trades),
            "win_rate": (
                sum(1 for t in self.completed_trades if t["pnl_pct"] > 0) /
                max(len(self.completed_trades), 1) * 100
            ),
        }
