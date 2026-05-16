"""Binance Futures WebSocket — klines, aggTrade, markPrice, depth, liquidations.
Async Python port of POLYx binance-ws.js with full reconnect logic.
"""

import asyncio
import json
import time
from typing import Optional, Callable, Dict, List, Any
from collections import defaultdict

import websockets
from core.utils.logger import log
from config import config

_cfg_bn = config.get("infrastructure", {}).get("binance", {})
WS_URL = _cfg_bn.get("ws_url", "wss://fstream.binancefuture.com")
SYMBOLS = [s.lower() for s in _cfg_bn.get("symbols", ["solusdt"])]
KLINE_INTERVALS = _cfg_bn.get("kline_intervals", ["1m", "5m"])
DEPTH_ENABLED = _cfg_bn.get("depth_enabled", True)
LIQUIDATION_ENABLED = _cfg_bn.get("liquidation_enabled", True)


class BinanceWS:
    """Async Binance Futures WebSocket client with auto-reconnect."""

    def __init__(self):
        self.symbols = SYMBOLS
        self.intervals = KLINE_INTERVALS
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._destroyed = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        self._last_message_ts = 0.0
        self._msg_count = 0
        self.last_reconnect_ts = 0.0

        # Callbacks
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Build stream list
        self._streams = self._build_streams()

    def _build_streams(self) -> List[str]:
        streams = []
        for sym in self.symbols:
            for interval in self.intervals:
                streams.append(f"{sym}@kline_{interval}")
            streams.append(f"{sym}@aggTrade")
            streams.append(f"{sym}@markPrice@1s")
            if DEPTH_ENABLED:
                streams.append(f"{sym}@depth20@100ms")
            if LIQUIDATION_ENABLED:
                streams.append(f"{sym}@forceOrder")
        return streams

    def on(self, event: str, callback: Callable):
        """Register event handler. Events: kline, trade, mark_price, orderbook, liquidation"""
        self._handlers[event].append(callback)

    def _emit(self, event: str, data: Any):
        for handler in self._handlers.get(event, []):
            try:
                handler(data)
            except Exception as e:
                log.error(f"BinanceWS handler error [{event}]: {e}")

    async def connect(self):
        """Connect to Binance combined WebSocket stream."""
        while not self._destroyed:
            stream_list = "/".join(self._streams)
            url = f"{WS_URL}/stream?streams={stream_list}"
            log.info(
                f"Binance WS: connecting ({len(self._streams)} streams for {', '.join(s.upper() for s in self.symbols)})"
            )

            try:
                async with websockets.connect(
                    url, ping_interval=120, ping_timeout=30, max_size=10 * 1024 * 1024
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    self.last_reconnect_ts = time.time()
                    self._last_message_ts = time.time()
                    self._msg_count = 0
                    log.info(f"Binance WS: connected ✅ ({len(self._streams)} streams)")
                    self._emit("connected", {})

                    # Health check task
                    health_task = asyncio.create_task(self._health_check())

                    try:
                        async for raw in ws:
                            self._last_message_ts = time.time()
                            self._msg_count += 1
                            try:
                                msg = json.loads(raw)
                                self._process_message(msg)
                            except json.JSONDecodeError:
                                pass
                    finally:
                        health_task.cancel()

            except (
                websockets.ConnectionClosed,
                websockets.InvalidHandshake,
                ConnectionRefusedError,
                OSError,
            ) as e:
                log.warning(f"Binance WS: closed/error: {e}")
            except Exception as e:
                log.error(f"Binance WS: unexpected error: {e}")

            self._emit("disconnected", {})
            if not self._destroyed:
                delay = self._reconnect_delay + (time.time() % 1) * 0.5
                self._reconnect_delay = min(
                    self._reconnect_delay * 1.5, self._max_reconnect_delay
                )
                log.debug(f"Binance WS: reconnecting in {delay:.1f}s")
                await asyncio.sleep(delay)

    async def _health_check(self):
        """Force reconnect if no data for 15s."""
        while True:
            await asyncio.sleep(5)
            silence = time.time() - self._last_message_ts
            if silence > 15 and self._ws:
                log.warning(
                    f"Binance WS: no data for {silence:.0f}s ({self._msg_count} msgs), forcing reconnect"
                )
                await self._ws.close()
                return

    def _process_message(self, msg: dict):
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        if not stream or not data:
            return

        event_type = data.get("e", "")

        if event_type == "kline":
            self._handle_kline(data)
        elif event_type == "aggTrade":
            self._handle_agg_trade(data)
        elif event_type == "markPriceUpdate":
            self._handle_mark_price(data)
        elif event_type == "forceOrder":
            self._handle_liquidation(data)
        elif "@depth" in stream:
            self._handle_depth(stream, data)

    def _handle_kline(self, data: dict):
        k = data.get("k", {})
        self._emit(
            "kline",
            {
                "symbol": data["s"].upper(),
                "interval": k["i"],
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "quote_volume": float(k["q"]),
                "trades": k["n"],
                "taker_buy_volume": float(k["V"]),
                "is_closed": k["x"],
                "open_time": k["t"],
                "close_time": k["T"],
                "ts": data["E"],
            },
        )

    def _handle_agg_trade(self, data: dict):
        self._emit(
            "trade",
            {
                "symbol": data["s"].upper(),
                "price": float(data["p"]),
                "qty": float(data["q"]),
                "is_buyer_maker": data["m"],
                "ts": data["T"],
            },
        )

    def _handle_mark_price(self, data: dict):
        self._emit(
            "mark_price",
            {
                "symbol": data["s"].upper(),
                "mark_price": float(data["p"]),
                "index_price": float(data["i"]),
                "funding_rate": float(data["r"]),
                "next_funding_time": data.get("T"),
                "ts": data["E"],
            },
        )

    def _handle_depth(self, stream: str, data: dict):
        sym = stream.split("@")[0].upper()
        bids = [
            {"price": float(p), "qty": float(q)}
            for p, q in (data.get("bids") or data.get("b") or [])
        ]
        asks = [
            {"price": float(p), "qty": float(q)}
            for p, q in (data.get("asks") or data.get("a") or [])
        ]
        self._emit(
            "orderbook",
            {
                "symbol": sym,
                "bids": bids,
                "asks": asks,
                "levels": max(len(bids), len(asks)),
                "ts": int(time.time() * 1000),
            },
        )

    def _handle_liquidation(self, data: dict):
        o = data.get("o", {})
        if not o:
            return
        self._emit(
            "liquidation",
            {
                "symbol": o["s"].upper(),
                "side": o["S"],  # BUY = short liq, SELL = long liq
                "price": float(o["p"]),
                "qty": float(o["q"]),
                "quote_qty": float(o["p"]) * float(o["q"]),
                "ts": data.get("E", int(time.time() * 1000)),
            },
        )

    def destroy(self):
        self._destroyed = True
        if self._ws:
            asyncio.create_task(self._ws.close())
