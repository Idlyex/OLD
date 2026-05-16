"""Binance SOL Liquidation Recorder.

Records real-time SOL/USDT forced liquidation events from Binance WebSocket.
Liquidations = forced position closures at specific price levels — predict
large moves when concentrated near current price.

Also computes:
  - Liquidation heatmap: aggregate USD at each $0.10 price level
  - Cumulative long/short liq pressure within $1, $2, $5 of current price
  - Net liquidation imbalance (long_liq - short_liq near price)

Output:
  data/recorded/liquidations/YYYY-MM-DD_liquidations.jsonl   — raw events
  data/recorded/liquidations/YYYY-MM-DD_liq_levels.jsonl     — aggregated per 30s

Usage:
  Runs as async background task inside MLSharesTrader.
"""

import asyncio
import json
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from core.utils.logger import log

LIQUIDATION_DIR = Path("data/recorded/liquidations")


class LiquidationRecorder:
    """Records Binance SOL/USDT forced liquidation events via WebSocket."""

    def __init__(self, symbol: str = "SOLUSDT"):
        self.symbol = symbol.lower()
        self._running = False
        self._events: list = []  # raw liquidation events buffer
        self._liq_levels: dict = defaultdict(lambda: {"long_usd": 0.0, "short_usd": 0.0, "count": 0})
        self._total_events = 0
        self._total_long_usd = 0.0
        self._total_short_usd = 0.0
        self._last_flush_ts = 0
        self._flush_interval = 30  # seconds
        LIQUIDATION_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def stats(self) -> dict:
        """Current liquidation stats for TUI display."""
        return {
            "total_events": self._total_events,
            "total_long_usd": self._total_long_usd,
            "total_short_usd": self._total_short_usd,
            "levels": dict(self._liq_levels),
        }

    def get_pressure(self, sol_price: float, radius: float = 2.0) -> dict:
        """Get liquidation pressure near current price.

        Returns:
            long_usd: Total long liquidation USD within radius of price
            short_usd: Total short liquidation USD within radius of price
            imbalance: long_usd - short_usd (positive = more longs getting liquidated = bearish)
            ratio: long_usd / (long_usd + short_usd)
        """
        long_near = 0.0
        short_near = 0.0
        for level_str, data in self._liq_levels.items():
            try:
                level = float(level_str)
            except (ValueError, TypeError):
                continue
            if abs(level - sol_price) <= radius:
                long_near += data["long_usd"]
                short_near += data["short_usd"]

        total = long_near + short_near
        return {
            "long_usd": long_near,
            "short_usd": short_near,
            "imbalance": long_near - short_near,
            "ratio": long_near / total if total > 0 else 0.5,
            "total_usd": total,
        }

    async def start(self):
        """Start listening to Binance forceOrder WebSocket."""
        self._running = True
        self._last_flush_ts = time.time()
        log.info(f"  💀 Liquidation recorder starting: {self.symbol.upper()}")

        import websockets

        ws_url = f"wss://fstream.binance.com/ws/{self.symbol}@forceOrder"
        reconnect_delay = 1.0

        while self._running:
            ws = None
            try:
                ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=10)
                log.info(f"  ✅ Liquidation WS connected: {self.symbol.upper()}")
                reconnect_delay = 1.0

                async for raw_msg in ws:
                    if not self._running:
                        break

                    try:
                        msg = json.loads(raw_msg)
                        self._process_liquidation(msg)
                    except (json.JSONDecodeError, Exception) as e:
                        log.debug(f"  Liq WS parse error: {e}")

                    # Periodic flush
                    now = time.time()
                    if now - self._last_flush_ts >= self._flush_interval:
                        self._flush()
                        self._last_flush_ts = now

            except Exception as e:
                log.debug(f"  Liquidation WS error: {e}")
            finally:
                if ws:
                    try:
                        await ws.close()
                    except Exception:
                        pass

            if self._running:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)

    def stop(self):
        self._running = False
        self._flush()

    def _process_liquidation(self, msg: dict):
        """Process a single forceOrder event.

        Binance format:
        {
          "e": "forceOrder",
          "o": {
            "s": "SOLUSDT",
            "S": "SELL",  # SELL = long liquidation, BUY = short liquidation
            "o": "LIMIT",
            "f": "IOC",
            "q": "10.0",   # quantity
            "p": "93.50",  # price
            "ap": "93.48", # average price
            "X": "FILLED",
            "l": "10.0",   # last filled qty
            "z": "10.0",   # cumulative filled qty
            "T": 1234567890123  # trade time ms
          }
        }
        """
        order = msg.get("o", {})
        if not order:
            return

        side = order.get("S", "")  # SELL = long liq, BUY = short liq
        price = float(order.get("p", 0))
        qty = float(order.get("z", 0)) or float(order.get("q", 0))
        avg_price = float(order.get("ap", 0)) or price
        trade_time = int(order.get("T", 0))
        usd_value = qty * avg_price

        if price <= 0 or qty <= 0:
            return

        is_long_liq = side == "SELL"

        # Bucket by $0.10 price levels
        level = round(price * 10) / 10  # round to nearest $0.10
        level_key = f"{level:.1f}"

        if is_long_liq:
            self._liq_levels[level_key]["long_usd"] += usd_value
            self._total_long_usd += usd_value
        else:
            self._liq_levels[level_key]["short_usd"] += usd_value
            self._total_short_usd += usd_value
        self._liq_levels[level_key]["count"] += 1

        self._total_events += 1

        event = {
            "ts": round(time.time(), 3),
            "trade_time": trade_time,
            "side": "long" if is_long_liq else "short",
            "price": round(price, 4),
            "avg_price": round(avg_price, 4),
            "qty": round(qty, 4),
            "usd": round(usd_value, 2),
            "level": level_key,
        }
        self._events.append(event)

    def _flush(self):
        """Flush buffered events and level snapshots to JSONL files."""
        if not self._events and not self._liq_levels:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Raw events
        if self._events:
            path = LIQUIDATION_DIR / f"{today}_liquidations.jsonl"
            try:
                with open(path, "a", encoding="utf-8") as f:
                    for ev in self._events:
                        f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except Exception as e:
                log.debug(f"Liq flush error: {e}")
            self._events.clear()

        # Level snapshot (aggregate)
        if self._liq_levels:
            path = LIQUIDATION_DIR / f"{today}_liq_levels.jsonl"
            snapshot = {
                "ts": round(time.time(), 3),
                "total_events": self._total_events,
                "total_long_usd": round(self._total_long_usd, 2),
                "total_short_usd": round(self._total_short_usd, 2),
                "levels": {k: {
                    "long_usd": round(v["long_usd"], 2),
                    "short_usd": round(v["short_usd"], 2),
                    "count": v["count"],
                } for k, v in sorted(self._liq_levels.items())},
            }
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
            except Exception as e:
                log.debug(f"Liq level flush error: {e}")
