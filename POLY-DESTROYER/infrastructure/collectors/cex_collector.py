"""CEX Data Collector — aggregates Binance WS data into structured buffers.
Tracks: klines, trades, mark prices, depth, liquidations, CVD, trade flow.
Port of POLYx market-analyzer.js data collection logic.
"""

import time
import math
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any

from core.utils.logger import log
from core.utils.helpers import ema, safe_div
from config import config

_cfg_bn = config.get("infrastructure", {}).get("binance", {})
MAX_KLINES = _cfg_bn.get("max_klines", 120)
MAX_TRADES = _cfg_bn.get("max_trades", 1000)


class CEXCollector:
    """Aggregates raw Binance WS data into structured time-series buffers."""

    def __init__(self):
        # Kline storage: symbol -> interval -> deque of candles
        self.klines: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=MAX_KLINES))
        )

        # Trade flow: symbol -> deque of {ts, price, qty, is_buy}
        self.trades: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_TRADES)
        )

        # Mark prices: symbol -> {mark_price, index_price, funding_rate, ts}
        self.mark_prices: Dict[str, Dict] = {}

        # Order book depth: symbol -> {bids, asks, ts}
        self.depth: Dict[str, Dict] = {}

        # Liquidations: symbol -> deque of {ts, side, price, qty}
        self.liquidations: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )

        # CVD (Cumulative Volume Delta): symbol -> {value, history}
        self.cvd: Dict[str, Dict] = defaultdict(
            lambda: {"value": 0.0, "history": deque(maxlen=3600)}
        )

        # Funding rate history: symbol -> deque of {ts, rate}
        self.funding_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=300)
        )

        # Trade rate tracking
        self._trade_counts: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=300)
        )

        # OI change tracking (simulated from funding + volume)
        self.oi_proxy: Dict[str, Dict] = defaultdict(
            lambda: {"value": 0.0, "history": deque(maxlen=300)}
        )

    # ── Handlers (called from BinanceWS events) ──

    def on_kline(self, data: Dict):
        """Process incoming kline data."""
        symbol = data["symbol"]
        interval = data["interval"]
        candle = {
            "ts": data["ts"],
            "open_time": data["open_time"],
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "quote_volume": data["quote_volume"],
            "trades": data["trades"],
            "taker_buy_volume": data["taker_buy_volume"],
            "is_closed": data["is_closed"],
        }

        kline_buf = self.klines[symbol][interval]
        if kline_buf and kline_buf[-1]["open_time"] == candle["open_time"]:
            kline_buf[-1] = candle  # update current candle
        else:
            if candle["is_closed"] or not kline_buf:
                kline_buf.append(candle)

    def on_trade(self, data: Dict):
        """Process incoming aggregate trade."""
        symbol = data["symbol"]
        ts = data["ts"]
        price = data["price"]
        qty = data["qty"]
        is_buy = not data["is_buyer_maker"]  # buyer maker = sell aggressor

        self.trades[symbol].append({
            "ts": ts,
            "price": price,
            "qty": qty,
            "is_buy": is_buy,
            "notional": price * qty,
        })

        # Update CVD
        delta = qty if is_buy else -qty
        cvd = self.cvd[symbol]
        cvd["value"] += delta
        cvd["history"].append({"ts": ts, "value": cvd["value"], "delta": delta})

        # Trade count for rate
        self._trade_counts[symbol].append(ts)

    def on_mark_price(self, data: Dict):
        """Process mark price update."""
        symbol = data["symbol"]
        self.mark_prices[symbol] = {
            "mark_price": data["mark_price"],
            "index_price": data["index_price"],
            "funding_rate": data["funding_rate"],
            "next_funding_time": data.get("next_funding_time"),
            "ts": data["ts"],
        }

        # Track funding history
        self.funding_history[symbol].append({
            "ts": data["ts"],
            "rate": data["funding_rate"],
        })

    def on_orderbook(self, data: Dict):
        """Process order book depth snapshot."""
        symbol = data["symbol"]
        self.depth[symbol] = {
            "bids": data["bids"],
            "asks": data["asks"],
            "levels": data["levels"],
            "ts": data["ts"],
        }

    def on_liquidation(self, data: Dict):
        """Process liquidation event."""
        symbol = data["symbol"]
        self.liquidations[symbol].append({
            "ts": data["ts"],
            "side": data["side"],
            "price": data["price"],
            "qty": data["qty"],
            "notional": data["quote_qty"],
        })

    # ── Accessors ──

    def get_latest_price(self, symbol: str) -> float:
        """Get latest mark price for symbol."""
        mp = self.mark_prices.get(symbol)
        return mp["mark_price"] if mp else 0.0

    def get_klines(self, symbol: str, interval: str, limit: int = None) -> List[Dict]:
        """Get recent klines."""
        buf = self.klines.get(symbol, {}).get(interval, deque())
        data = list(buf)
        if limit:
            data = data[-limit:]
        return data

    def get_ohlcv_arrays(self, symbol: str, interval: str, limit: int = None):
        """Get OHLCV as numpy arrays for feature computation."""
        klines = self.get_klines(symbol, interval, limit)
        if not klines:
            return None

        n = len(klines)
        result = {
            "ts": np.array([k["ts"] for k in klines], dtype=np.int64),
            "open": np.array([k["open"] for k in klines], dtype=np.float64),
            "high": np.array([k["high"] for k in klines], dtype=np.float64),
            "low": np.array([k["low"] for k in klines], dtype=np.float64),
            "close": np.array([k["close"] for k in klines], dtype=np.float64),
            "volume": np.array([k["volume"] for k in klines], dtype=np.float64),
            "quote_volume": np.array(
                [k.get("quote_volume", 0) for k in klines], dtype=np.float64
            ),
            "taker_buy_volume": np.array(
                [k.get("taker_buy_volume", 0) for k in klines], dtype=np.float64
            ),
            "trades": np.array(
                [k.get("trades", 0) for k in klines], dtype=np.int64
            ),
        }
        return result

    def get_recent_trades(self, symbol: str, window_s: float = 60) -> List[Dict]:
        """Get trades within the last window_s seconds."""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_s * 1000)
        return [t for t in self.trades.get(symbol, []) if t["ts"] >= cutoff]

    def get_cvd(self, symbol: str) -> Dict:
        """Get current CVD state."""
        return dict(self.cvd.get(symbol, {"value": 0.0, "history": deque()}))

    def get_depth_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get latest order book depth."""
        return self.depth.get(symbol)

    def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate."""
        mp = self.mark_prices.get(symbol)
        return mp["funding_rate"] if mp else 0.0

    def get_funding_acceleration(self, symbol: str, window_m: int = 5) -> float:
        """Compute funding rate change over window minutes."""
        hist = list(self.funding_history.get(symbol, []))
        if len(hist) < 2:
            return 0.0
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_m * 60_000
        recent = [h for h in hist if h["ts"] >= cutoff]
        if len(recent) < 2:
            return 0.0
        return recent[-1]["rate"] - recent[0]["rate"]

    def get_liquidation_heat(self, symbol: str, window_s: float = 60) -> Dict:
        """Compute liquidation heat: count + notional in window."""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_s * 1000)
        liqs = [l for l in self.liquidations.get(symbol, []) if l["ts"] >= cutoff]

        long_liqs = [l for l in liqs if l["side"] == "SELL"]
        short_liqs = [l for l in liqs if l["side"] == "BUY"]

        return {
            "total_count": len(liqs),
            "long_liq_count": len(long_liqs),
            "short_liq_count": len(short_liqs),
            "long_liq_notional": sum(l["notional"] for l in long_liqs),
            "short_liq_notional": sum(l["notional"] for l in short_liqs),
        }

    def get_trade_rate(self, symbol: str, window_s: float = 10) -> float:
        """Trades per second over window."""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_s * 1000)
        counts = self._trade_counts.get(symbol, deque())
        recent = sum(1 for ts in counts if ts >= cutoff)
        return recent / window_s if window_s > 0 else 0.0

    def get_aggressive_volume(self, symbol: str, window_s: float = 60) -> Dict:
        """Compute aggressive buy/sell volume in window."""
        trades = self.get_recent_trades(symbol, window_s)
        buy_vol = sum(t["qty"] for t in trades if t["is_buy"])
        sell_vol = sum(t["qty"] for t in trades if not t["is_buy"])
        total = buy_vol + sell_vol

        return {
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "total_volume": total,
            "buy_ratio": safe_div(buy_vol, total),
            "sell_ratio": safe_div(sell_vol, total),
            "delta": buy_vol - sell_vol,
        }
